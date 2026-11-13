"""Detached canonical USO 1024 generation, common evaluation and verified exports."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
from freecus.manifest import PROJECT, load, sha256
from freecus.run import write_json

BASE = PROJECT / 'uso'
STATE = BASE / 'outputs/modal1024-s42-sangyun-lee0-20261113'
PROFILE = 'sangyun-lee0'
APP = 'uso-1024-s42-canonical-2100-20261113'
RUNS = APP + '-runs'
CACHE = 'hf-model-cache'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def prepare():
    source_files = [p for p in BASE.rglob('*') if p.is_file()
                    and not any(x in p.relative_to(BASE).parts for x in ['outputs','results','__pycache__','tests'])
                    and p.suffix not in {'.tar', '.pyc'}]
    shared = ['__init__.py','manifest.py','run.py','evaluate.py','modal_cost.py',
              'operations/artifact_bundle.py','project_metrics/benchmark_metrics.py','project_metrics/SOURCE.json']
    source_files += [PROJECT / 'freecus' / name for name in shared]
    code = {p.relative_to(PROJECT).as_posix(): sha256(p) for p in sorted(source_files)}
    key = hashlib.sha256(json.dumps(code,sort_keys=True).encode()).hexdigest()
    source = STATE / 'sources' / key
    for p in source_files:
        target = source / p.relative_to(PROJECT)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,target)
    cfg, lock = read(BASE/'config.json'),read(BASE/'models.lock.json')
    plans = []
    for benchmark, expected in [('dreambench',750),('dreambench_plus',1350)]:
        manifest = PROJECT / f'freecus/manifests/{benchmark}_project_seed42.jsonl'
        rows = load(manifest,PROJECT)
        if len(rows)!=expected or {r['seed'] for r in rows}!={42}:
            raise ValueError('Canonical manifest differs')
        identity = dict(config=cfg, models=lock, code=code, manifest_sha256=sha256(manifest),
                        expected=expected, excluded_case_ids=[], style_included=True,
                        upstream=read(BASE/'UPSTREAM.json'),modal_profile=PROFILE)
        run_id = f'uso1024-{benchmark}-s42-'+hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
        local = BASE / 'results' / benchmark / run_id
        local.mkdir(parents=True,exist_ok=True)
        for p in [BASE/'config.json',BASE/'models.lock.json',manifest]:
            shutil.copy2(p,local/('manifest.jsonl' if p==manifest else p.name))
        write_json(local/'provenance.json',identity)
        with zipfile.ZipFile(local/'generation_source.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
            for p in source.rglob('*'):
                if p.is_file():z.write(p,p.relative_to(source))
        files={'manifest.jsonl':manifest.read_bytes(),'config.json':json.dumps(cfg).encode(),
               'identity.json':json.dumps(identity).encode(),'models.lock.json':json.dumps(lock).encode(),
               'source.zip':(local/'generation_source.zip').read_bytes()}
        for path in sorted({r['reference'] for r in rows}):files['input/'+path]=(PROJECT/path).read_bytes()
        archive=STATE/'inputs'/f'{run_id}.zip'
        archive.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as z:
            for name,data in files.items():z.writestr(name,data)
            z.writestr('bundle.json',json.dumps(dict(files={n:hashlib.sha256(b).hexdigest() for n,b in files.items()})))
        owners={r:i%2 for i,r in enumerate(sorted({r['reference_id'] for r in rows}))}
        lanes=[[r['case_id'] for r in rows if owners[r['reference_id']]==i] for i in range(2)]
        pilot=[]
        for lane in lanes:
            selected=[r for r in rows if r['case_id'] in set(lane)]
            pilot.append([next(r['case_id'] for r in selected if r['category']==c)
                          for c in sorted({r['category'] for r in selected})])
        plan=dict(run_id=run_id,benchmark=benchmark,expected=expected,lanes=lanes,pilot=pilot,
                  source=str(source.resolve()),archive=str(archive.resolve()),local=str(local.resolve()),identity=identity)
        write_json(local/'plan.json',plan);plans.append(plan)
    write_json(STATE/'plans.json',plans)
    print(json.dumps([dict(run_id=p['run_id'],cases=p['expected'],pilot=p['pilot']) for p in plans],indent=2))


def build(source):
    import modal
    runs=modal.Volume.from_name(RUNS,create_if_missing=True,version=2)
    cache=modal.Volume.from_name(CACHE)
    secret=modal.Secret.from_name('huggingface-secret',required_keys=['HF_TOKEN'])
    cpu=(modal.Image.debian_slim(python_version='3.11').pip_install('huggingface-hub==0.29.3','hf_transfer==0.1.9')
         .env(dict(HF_HOME='/cache/huggingface',HF_HUB_ENABLE_HF_TRANSFER='1',PYTHONUNBUFFERED='1'))
         .add_local_dir(source,'/root'))
    gpu=(modal.Image.debian_slim(python_version='3.11').add_local_dir(source,'/root',copy=True)
         .run_commands('pip install --no-cache-dir -r /root/uso/requirements-gpu.txt')
         .env(dict(HF_HOME='/cache/huggingface',PYTHONUNBUFFERED='1')))
    metric=(modal.Image.debian_slim(python_version='3.11').apt_install('git')
            .pip_install('torch==2.7.1','torchvision==0.22.1','numpy==1.26.4','Pillow==11.3.0',
                         'ftfy==6.3.1','regex','tqdm','packaging')
            .pip_install('git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6')
            .env(dict(TORCH_HOME='/cache/eval-torch',CLIP_DOWNLOAD_ROOT='/cache/eval-clip',PYTHONUNBUFFERED='1'))
            .add_local_dir(source,'/root'))
    app=modal.App(APP)

    @app.function(name='stage',serialized=True,image=cpu,secrets=[secret],
                  volumes={'/work':runs,'/cache':cache},cpu=2,memory=8192,timeout=7200,
                  max_containers=1,retries=0,scaledown_window=2)
    def stage(run_id):
        from uso.models import stage as stage_models
        runs.reload();cache.reload()
        folder=Path('/work/bundles')/run_id
        with zipfile.ZipFile(Path('/work/uploads')/(run_id+'.zip')) as z:
            for e in z.infolist():
                if not (folder/e.filename).resolve().is_relative_to(folder.resolve()):raise ValueError('Unsafe archive')
            z.extractall(folder)
        for name,wanted in read(folder/'bundle.json')['files'].items():
            if sha256(folder/name)!=wanted:raise ValueError('Input hash mismatch')
        identity=read(folder/'identity.json')
        for name,wanted in identity['code'].items():
            if sha256(Path('/root')/name)!=wanted:raise ValueError('Code differs from frozen source')
        try:stage_models(read(folder/'models.lock.json'),folder/'resolved_models.json')
        finally:cache.commit()
        # Reuse exact DINO weights already present from earlier experiments.
        target=Path('/cache/eval-torch/hub/checkpoints/dino_deitsmall16_pretrain.pth')
        if not target.exists():
            for found in Path('/cache').rglob(target.name):
                if sha256(found)=='1566d50496f27f52f07fea6094fa29b2fdd6fae89da65bdd3ebc3b24ef6b7eb7':
                    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(found,target);break
        cache.commit();runs.commit()
        return dict(status='ready')

    @app.function(name='generate',serialized=True,image=gpu,secrets=[secret],gpu='A100-80GB',
                  volumes={'/work':runs,'/cache':cache},cpu=4,memory=98304,timeout=21600,
                  max_containers=2,retries=0,scaledown_window=2,single_use_containers=True)
    def generate(run_id,lane,cases,invocation):
        import resource
        import subprocess
        import sys
        import time
        from freecus.modal_cost import rate
        runs.reload();cache.reload()
        folder=Path('/work/results')/run_id/f'lane-{lane}'
        journal=Path('/work/invocations')/invocation/f'lane-{lane}'
        journal.mkdir(parents=True,exist_ok=False)
        command=[sys.executable,'-m','uso.run','--bundle',str(Path('/work/bundles')/run_id),
                 '--output',str(folder),'--case-ids',*cases]
        began=time.monotonic()
        with (journal/'worker.log').open('wb') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        try:
            while child.poll() is None:
                runs.commit()
                if time.monotonic()-began>21000:child.kill();child.wait();break
                time.sleep(20)
        finally:
            if child.poll() is None:child.kill();child.wait()
        elapsed=time.monotonic()-began
        records=[read(p) for p in (folder/'metadata').rglob('*.json')]
        result=dict(status='complete' if child.returncode==0 else 'failed',exit_code=child.returncode,
                    lane=lane,requested=len(cases),generated=len(records),seconds=elapsed,
                    estimated_compute_usd=(elapsed+22)*rate('A100-80GB'),
                    host_peak_gib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/2**20,
                    timing=[r['timing'] for r in records])
        if child.returncode:result['error']=(journal/'worker.log').read_text(errors='replace')[-6000:]
        write_json(journal/'result.json',result);runs.commit();cache.commit();return result

    @app.function(name='evaluate_export',serialized=True,image=metric,gpu='T4',
                  volumes={'/work':runs,'/cache':cache},cpu=2,memory=8192,timeout=7200,
                  max_containers=1,retries=0,scaledown_window=2)
    def evaluate_export(run_id):
        import csv
        from freecus.evaluate import prepare,validate,project_evaluator
        from freecus.run import artifact_paths,complete
        from freecus.operations.artifact_bundle import pack,verify
        runs.reload();cache.reload()
        bundle=Path('/work/bundles')/run_id
        rows=load(bundle/'manifest.jsonl',bundle/'input')
        lanes=sorted(p for p in (Path('/work/results')/run_id).glob('lane-*') if (p/'run.json').exists())
        if not lanes:raise ValueError('No generated lanes')
        target=Path('/work/evaluation_inputs')/run_id
        target.mkdir(parents=True,exist_ok=True)
        shutil.copy2(lanes[0]/'run.json',target/'run.json')
        infos=[read(p/'run.json') for p in lanes]
        for key in ['fingerprint','config','manifest_sha256','code','versions','cuda']:
            if any(i[key]!=infos[0][key] for i in infos):raise ValueError('Mixed generation identity: '+key)
        if any(i['gpu'] not in {'NVIDIA A100 80GB PCIe','NVIDIA A100-SXM4-80GB'} for i in infos):
            raise ValueError('Unexpected GPU model')
        available=[]
        for row in rows:
            matches=[artifact_paths(p,row) for p in lanes if complete(artifact_paths(p,row))]
            if len(matches)>1:raise ValueError('Duplicate case')
            if not matches:continue
            for key,src in matches[0].items():
                dst=artifact_paths(target,row)[key];dst.parent.mkdir(parents=True,exist_ok=True)
                if dst.exists():
                    if sha256(dst)!=sha256(src):raise ValueError('Assembled bytes differ')
                else:
                    try:dst.hardlink_to(src)
                    except OSError:shutil.copy2(src,dst)
            available.append(row['case_id'])
        folder=Path('/work/metrics')/run_id
        coverage=dict(expected=len(rows),generated=len(available),missing_case_ids=sorted({r['case_id'] for r in rows}-set(available)),
                      excluded_case_ids=[],style_included=True,status='complete' if len(available)==len(rows) else 'incomplete')
        write_json(folder/'coverage.json',coverage)
        evaluator=project_evaluator()
        plan=prepare(bundle/'manifest.jsonl',bundle/'input',target,evaluator,'all',available,condition='uso')
        write_json(folder/'plan.json',plan)
        scores=folder/'metrics.json'
        result=read(scores) if scores.exists() else evaluator.evaluate(plan,scores,'cuda',32,validator=validate,
                         checkpoint=folder/'checkpoint.json',commit=lambda _:runs.commit())
        with (folder/'per_image.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            fields=['case_id','reference_id','benchmark','category','condition','CLIP-I','CLIP-T','DINO-I']
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(result['scores'])
        extras={p.name:p for p in [bundle/'config.json',bundle/'models.lock.json',bundle/'identity.json',bundle/'source.zip',folder/'coverage.json']}
        exports=Path('/work/exports')/run_id
        index=pack(lanes,bundle/'manifest.jsonl',bundle/'input',exports,'uso',rows[0]['benchmark'],run_id,len(available),
                   resolution=(1024,1024),allow_gpu_variants=True,provenance_files=extras)
        verify(exports,index)
        report=dict(coverage=coverage,summary=result['summary'],verified=True,parts=index['parts'])
        write_json(folder/'summary.json',report);write_json(exports/'verification.json',report)
        cache.commit();runs.commit();return report

    @app.function(name='workflow',serialized=True,image=cpu,volumes={'/work':runs},
                  timeout=86400,cpu=(.125,.125),memory=(512,512),max_containers=1,retries=0,scaledown_window=2)
    def workflow(workflow_id,plans):
        import time
        import uuid
        import modal
        runs.reload();path=Path('/work/workflows')/(workflow_id+'.json')
        if path.exists():raise ValueError('Duplicate workflow')
        state=dict(workflow_id=workflow_id,status='starting',phases=[],evaluations={},plans=plans,
                   started_at=time.time(),budget_policy='User managed Modal account spending limit')
        def save():state['updated_at']=time.time();write_json(path,state);runs.commit()
        def phase(plan,kind,cases):
            state['status']=kind+'_'+plan['benchmark']
            item=dict(run_id=plan['run_id'],benchmark=plan['benchmark'],kind=kind,calls=[])
            state['phases'].append(item);save();calls=[];invocation=uuid.uuid4().hex
            for lane,ids in enumerate(cases):
                call=modal.Function.from_name(APP,'generate').spawn(plan['run_id'],lane,ids,invocation)
                calls.append(call);item['calls'].append(dict(call_id=call.object_id,lane=lane,cases=len(ids)));save()
            for i,call in enumerate(calls):
                try:item['calls'][i]['result']=call.get()
                except Exception as exc:item['calls'][i]['result']=dict(status='failed',error=repr(exc))
                save()
            return all(c['result']['status']=='complete' for c in item['calls'])
        save()
        try:
            for plan in plans:
                state['status']='staging_'+plan['benchmark'];save()
                call=modal.Function.from_name(APP,'stage').spawn(plan['run_id'])
                state.setdefault('stage_calls',[]).append(dict(run_id=plan['run_id'],call_id=call.object_id));save();call.get()
                if not phase(plan,'pilot',plan['pilot']):state['status']='pilot_failed';save();return state
                successful=phase(plan,'generation',plan['lanes'])
                state['status']='evaluating_'+plan['benchmark'];save()
                call=modal.Function.from_name(APP,'evaluate_export').spawn(plan['run_id'])
                state.setdefault('evaluation_calls',[]).append(dict(run_id=plan['run_id'],call_id=call.object_id));save()
                state['evaluations'][plan['benchmark']]=call.get();save()
                if not successful or state['evaluations'][plan['benchmark']]['coverage']['status']!='complete':
                    state['status']='finished_with_missing_cases';save();return state
            state['status']='complete';save();return state
        except BaseException as exc:state.update(status='failed',error=repr(exc));save();raise

    @app.function(name='inspect',serialized=True,image=cpu,volumes={'/work':runs},
                  timeout=300,cpu=(.125,.125),memory=(512,512),scaledown_window=2)
    def inspect(workflow_id):
        runs.reload();path=Path('/work/workflows')/(workflow_id+'.json')
        if not path.exists():return dict(status='queued')
        state=read(path);state['progress']={}
        for p in state.pop('plans'):
            root=Path('/work/results')/p['run_id']
            state['progress'][p['benchmark']]=dict(expected=p['expected'],generated=sum(1 for _ in root.glob('lane-*/metadata/**/*.json')))
        for phase in state['phases']:
            for call in phase['calls']:
                if 'result' in call:call['result'].pop('timing',None)
        return state
    return app


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','deploy','submit','status','collect'])
    args=parser.parse_args();STATE.mkdir(parents=True,exist_ok=True)
    if args.command=='prepare':prepare();return
    from freecus.modal_client import connect
    modal=connect(PROFILE);plans=read(STATE/'plans.json')
    if args.command in {'deploy','submit'}:
        if any(sha256(PROJECT/n)!=v for n,v in plans[0]['identity']['code'].items()):raise ValueError('Prepared code changed')
    if args.command=='deploy':
        with modal.enable_output():build(plans[0]['source']).deploy()
    elif args.command=='submit':
        receipt=STATE/'receipt.json'
        if receipt.exists():raise ValueError('Already submitted; inspect before resuming')
        volume=modal.Volume.from_name(RUNS)
        with volume.batch_upload(force=False) as upload:
            for p in plans:upload.put_file(p['archive'],f"/uploads/{p['run_id']}.zip")
        workflow_id='uso1024-'+hashlib.sha256(''.join(p['run_id'] for p in plans).encode()).hexdigest()[:16]
        call=modal.Function.from_name(APP,'workflow').spawn(workflow_id,
              [{k:p[k] for k in ['run_id','benchmark','expected','lanes','pilot']} for p in plans])
        record=dict(workflow_id=workflow_id,call_id=call.object_id,profile=PROFILE,app=APP,
                    dashboard='https://modal.com/id/'+call.object_id)
        write_json(receipt,record)
        for p in plans:write_json(Path(p['local'])/'receipt.json',record)
        print(json.dumps(record,indent=2))
    elif args.command=='status':
        receipt=read(STATE/'receipt.json')
        result=modal.Function.from_name(APP,'inspect').remote(receipt['workflow_id'])
        write_json(STATE/'status.json',result)
        print(json.dumps(result,indent=2))
    else:
        from freecus.operations.artifact_bundle import verify,digest
        volume=modal.Volume.from_name(RUNS)
        for plan in plans:
            local=Path(plan['local']);run_id=plan['run_id']
            for entry in volume.iterdir(f'metrics/{run_id}',recursive=True):
                if entry.type==1:(local/Path(entry.path).name).write_bytes(b''.join(volume.read_file(entry.path)))
            index=json.loads(b''.join(volume.read_file(f'exports/{run_id}/bundle_index.json')))
            folder=PROJECT.parent/'AI_PROJECT_images/uso'/plan['benchmark']/run_id
            folder.mkdir(parents=True,exist_ok=True)
            for part in index['parts']:
                dest=folder/part['file']
                if dest.exists() and digest(dest)==part['sha256']:continue
                tmp=dest.with_suffix('.zip.partial')
                with tmp.open('wb') as stream:volume._read_file_into_fileobj(f"exports/{run_id}/{part['file']}",stream,concurrency=2)
                if digest(tmp)!=part['sha256']:raise ValueError('Downloaded ZIP checksum mismatch')
                tmp.replace(dest)
            write_json(folder/'bundle_index.json',index);verify(folder,index)
            report=dict(verified=True,generated=index['generated_images'],parts=index['parts'],archive_folder=str(folder))
            write_json(folder/'verification.json',report);write_json(local/'bundle_index.json',index);write_json(local/'verification.json',report)
            print(json.dumps(report,indent=2))


if __name__=='__main__':main()
