"""Prepare, submit, inspect and collect the isolated FreeCus 512 experiment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

from .manifest import PROJECT, load, sha256
from .run import write_json

BASE=Path(__file__).resolve().parent
STATE=BASE/'outputs/modal512-20261112'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def prepare():
    config=read(BASE/'config.512.json');lock=read(BASE/'models.lock.json')
    if config['models']!={k:v['repo'] for k,v in lock.items()}: raise ValueError('Model lock mismatch')
    paths=[*BASE.glob('*.py'),*BASE.glob('runtime_src/*.py'),*BASE.glob('upstream/*.py'),
           *BASE.glob('operations/*.py'),*BASE.glob('project_metrics/*'),
           *BASE.glob('requirements*.txt'),BASE/'constraints.txt',BASE/'UPSTREAM.json']
    paths=[p for p in paths if p.is_file()]
    code={p.relative_to(BASE).as_posix():sha256(p) for p in sorted(paths)}
    digest=hashlib.sha256(json.dumps(code,sort_keys=True).encode()).hexdigest()
    source=STATE/'source'/digest
    for p in paths:
        dst=source/p.relative_to(BASE);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
    plans=[]
    for benchmark,expected in [('dreambench',750),('dreambench_plus',1350)]:
        manifest=BASE/f'manifests/{benchmark}_project_seed42.jsonl'
        rows=load(manifest,PROJECT)
        if len(rows)!=expected or {r['seed'] for r in rows}!={42}: raise ValueError('Invalid canonical cases')
        selected=[r for r in rows if benchmark=='dreambench' or r['category']!='style']
        excluded=[r['case_id'] for r in rows if r not in selected]
        identity=dict(benchmark=benchmark,config=config,lock=lock,code=code,manifest_sha256=sha256(manifest),
                      generation_selection='all' if benchmark=='dreambench' else 'dbpp-no-style',
                      expected_generated=len(selected),excluded_case_ids=excluded,
                      git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip())
        key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
        run_id=f'freecus512-{benchmark}-s42-{key}'
        local=BASE/'results_512'/benchmark/run_id
        local.mkdir(parents=True,exist_ok=True)
        write_json(local/'identity.json',identity)
        shutil.copy2(manifest,local/'manifest.jsonl');shutil.copy2(BASE/'models.lock.json',local/'models.lock.json')
        shutil.copy2(BASE/'config.512.json',local/'config.512.json')
        archive=STATE/'inputs'/f'{run_id}.zip';archive.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(local/'source.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
            for p in source.rglob('*'):
                if p.is_file(): z.write(p,p.relative_to(source))
        files={'manifest.jsonl':manifest.read_bytes(),'config.json':json.dumps(config).encode(),
               'models.lock.json':json.dumps(lock).encode(),'identity.json':json.dumps(identity).encode(),
               'source.zip':(local/'source.zip').read_bytes()}
        for name in sorted({r['reference'] for r in rows}): files['input/'+name]=(PROJECT/name).read_bytes()
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as z:
            for name,data in files.items(): z.writestr(name,data)
            z.writestr('bundle.json',json.dumps(dict(files={n:hashlib.sha256(d).hexdigest() for n,d in files.items()})))
        refs=sorted({r['reference_id'] for r in selected});owner={ref:i%2 for i,ref in enumerate(refs)}
        lanes=[[r['case_id'] for r in selected if owner[r['reference_id']]==i] for i in range(2)]
        plan=dict(run_id=run_id,benchmark=benchmark,lanes=lanes,source=str(source.resolve()),
                  archive=str(archive.resolve()),identity=identity,local=str(local.resolve()))
        write_json(local/'plan.json',plan);plans.append(plan)
    write_json(STATE/'plans.json',plans)
    print(json.dumps([dict(benchmark=p['benchmark'],run_id=p['run_id'],images=sum(map(len,p['lanes']))) for p in plans],indent=2))


def check_code(plans):
    for p in plans:
        if any(not (BASE/name).is_file() or sha256(BASE/name)!=value for name,value in p['identity']['code'].items()):
            raise ValueError('Code changed after prepare; preserve existing submission and prepare only before a new run')


def collect(modal,plans):
    from .remote_512 import RUNS
    from .operations.artifact_bundle import digest,verify
    volume=modal.Volume.from_name(RUNS)
    reports=[]
    for plan in plans:
        run_id=plan['run_id'];local=Path(plan['local'])
        for entry in volume.iterdir(f'metrics/{run_id}',recursive=True):
            if entry.type==1:
                target=local/'metrics'/Path(entry.path).name
                target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('wb') as stream:
                    for chunk in volume.read_file(entry.path): stream.write(chunk)
        try: index=json.loads(b''.join(volume.read_file(f'exports/{run_id}/bundle_index.json')))
        except FileNotFoundError:
            reports.append(dict(run_id=run_id,status='export_pending'));continue
        folder=PROJECT.parent/'AI_PROJECT_images/freecus_512'/plan['benchmark']/run_id
        folder.mkdir(parents=True,exist_ok=True)
        needed=sum(p['size_bytes'] for p in index['parts'] if not (folder/p['file']).exists())
        if shutil.disk_usage(folder).free<needed+200*1024**2: raise OSError('Insufficient local space for ZIP collection')
        for part in index['parts']:
            target=folder/part['file']
            if target.exists() and digest(target)==part['sha256']: continue
            temp=target.with_suffix('.zip.partial')
            print('Downloading '+run_id+'/'+part['file'],flush=True)
            with temp.open('wb') as stream:
                volume._read_file_into_fileobj(f"exports/{run_id}/{part['file']}",stream,concurrency=2)
            if temp.stat().st_size!=part['size_bytes'] or digest(temp)!=part['sha256']: raise ValueError('Download hash mismatch')
            temp.replace(target)
        write_json(folder/'bundle_index.json',index)
        verify(folder,index)
        excluded=set(plan['identity']['excluded_case_ids'])
        verified=dict(verified=True,generated=index['generated_images'],excluded_case_ids=sorted(excluded),
                      missing_case_ids=sorted(set(index['missing_case_ids'])-excluded),
                      archive_folder=str(folder),parts=index['parts'],member_hashes_and_crc_verified=True)
        write_json(local/'bundle_index.json',index);write_json(local/'verification.json',verified)
        write_json(folder/'verification.json',verified);reports.append(verified)
    print(json.dumps(reports,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','deploy','submit','status','collect'])
    parser.add_argument('--profile',default='second')
    args=parser.parse_args()
    STATE.mkdir(parents=True,exist_ok=True)
    if args.command=='prepare': prepare();return
    from .modal_client import connect
    from .remote_512 import APP,RUNS,build
    modal=connect(args.profile)
    plans=read(STATE/'plans.json')
    if args.command=='deploy':
        check_code(plans)
        modal.Volume.from_name(RUNS,create_if_missing=True,version=2).hydrate()
        app,_=build(plans[0]['source'])
        with modal.enable_output(): app.deploy()
        write_json(STATE/'deployment.json',dict(app=APP,app_id=app.app_id,profile=args.profile))
    elif args.command=='submit':
        check_code(plans)
        receipt=STATE/'receipt.json'
        if receipt.exists(): raise ValueError('Already submitted; use status, do not duplicate')
        volume=modal.Volume.from_name(RUNS)
        with volume.batch_upload(force=False) as upload:
            for p in plans: upload.put_file(p['archive'],f"/uploads/{p['run_id']}.zip")
        workflow_id='freecus512-'+hashlib.sha256(''.join(p['run_id'] for p in plans).encode()).hexdigest()[:16]
        record=dict(workflow_id=workflow_id,profile=args.profile,app=APP,status='submitting',
                    runs=[dict(benchmark=p['benchmark'],run_id=p['run_id']) for p in plans])
        write_json(receipt,record)
        call=modal.Function.from_name(APP,'workflow_512').spawn(workflow_id,
            [dict(run_id=p['run_id'],benchmark=p['benchmark'],lanes=p['lanes']) for p in plans])
        record.update(call_id=call.object_id,status='submitted',dashboard=f'https://modal.com/id/{call.object_id}')
        write_json(receipt,record)
        for p in plans: write_json(Path(p['local'])/'receipt.json',record)
        print(json.dumps(record,indent=2))
    elif args.command=='status':
        receipt=read(STATE/'receipt.json')
        result=modal.Function.from_name(APP,'inspect_512').remote(receipt['workflow_id'])
        write_json(STATE/'status.json',result)
        for p in plans: write_json(Path(p['local'])/'workflow-status.json',result)
        print(json.dumps(result,indent=2))
    else: collect(modal,plans)


if __name__=='__main__': main()
