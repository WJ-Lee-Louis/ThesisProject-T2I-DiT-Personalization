"""Detached UNO 512 queue, gated on completed FreeCus 512 evaluation and export."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile

from freecus.manifest import PROJECT,load,sha256
from freecus.run import write_json
from .protocol import BASE,validate_config,verify_upstream

PROFILE=os.environ.get('UNO512_PROFILE','second')
SUFFIX='' if PROFILE=='second' else '-'+PROFILE
APP='uno-512-seed42-after-freecus-20261112'+SUFFIX
RUNS='uno-512-seed42-20261112-runs'+SUFFIX
CACHE='hf-model-cache'
STATE=BASE/('outputs/modal512-20261112'+SUFFIX)
FREECUS_RUNS='freecus-512-paper-seed42-20261112-runs'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def prerequisite_ready(value):
    if value.get('status')!='complete': return False
    for bench,count in [('dreambench',750),('dreambench_plus',1170)]:
        coverage=value.get('evaluations',{}).get(bench,{}).get('coverage',{})
        export=value.get('exports',{}).get(bench,{})
        if (coverage.get('status')!='complete' or coverage.get('expected')!=count
                or coverage.get('generated')!=count or coverage.get('missing')
                or not export.get('verified') or export.get('missing_case_ids')):
            return False
    return True


def prepare():
    cfg=read(BASE/'config.512.json');validate_config(cfg);verify_upstream()
    paths=[*BASE.glob('*.py'),*BASE.glob('requirements*.txt'),BASE/'constraints.txt',
           BASE/'config.json',BASE/'config.512.json',BASE/'models.lock.json',BASE/'UPSTREAM.json']
    paths += [p for p in (BASE/'upstream').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    shared=['__init__.py','manifest.py','run.py','evaluate.py','modal_client.py','modal_cost.py',
            'operations/artifact_bundle.py','project_metrics/benchmark_metrics.py','project_metrics/SOURCE.json']
    paths += [PROJECT/'freecus'/p for p in shared]
    code={p.relative_to(PROJECT).as_posix():sha256(p) for p in sorted(paths)}
    source_hash=hashlib.sha256(json.dumps(code,sort_keys=True).encode()).hexdigest()
    source=STATE/'sources'/source_hash
    for p in paths:
        dst=source/p.relative_to(PROJECT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
    lock=read(BASE/'models.lock.json');plans=[]
    for bench,count in [('dreambench',750),('dreambench_plus',1350)]:
        manifest=PROJECT/f'freecus/manifests/{bench}_project_seed42.jsonl'
        rows=load(manifest,PROJECT)
        if len(rows)!=count or {r['seed'] for r in rows}!={42}: raise ValueError('Canonical cases differ')
        identity=dict(config=cfg,models=lock,code=code,manifest_sha256=sha256(manifest),
                      expected_generated=count,excluded_case_ids=[],style_included=True,modal_profile=PROFILE,
                      prerequisite='FreeCus 512 750+1170 generation/evaluation/verified ZIP export complete')
        run_id=f"uno512-{bench}-s42-"+hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
        local=BASE/'results_512'/bench/run_id;local.mkdir(parents=True,exist_ok=True)
        write_json(local/'provenance.json',identity)
        for p in [BASE/'config.512.json',BASE/'models.lock.json',manifest]:
            shutil.copy2(p,local/('manifest.jsonl' if p==manifest else p.name))
        with zipfile.ZipFile(local/'generation_source.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
            for p in source.rglob('*'):
                if p.is_file(): z.write(p,p.relative_to(source))
        refs=sorted({r['reference_id'] for r in rows});owner={r:i%2 for i,r in enumerate(refs)}
        lanes=[[r['case_id'] for r in rows if owner[r['reference_id']]==i] for i in range(2)]
        archive=STATE/'inputs'/f'{run_id}.zip';archive.parent.mkdir(parents=True,exist_ok=True)
        files={'manifest.jsonl':manifest.read_bytes(),'config.512.json':json.dumps(cfg).encode(),
               'models.lock.json':json.dumps(lock).encode(),'identity.json':json.dumps(identity).encode(),
               'source.zip':(local/'generation_source.zip').read_bytes()}
        for name in sorted({r['reference'] for r in rows}): files['input/'+name]=(PROJECT/name).read_bytes()
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as z:
            for name,data in files.items(): z.writestr(name,data)
            z.writestr('bundle.json',json.dumps(dict(files={n:hashlib.sha256(d).hexdigest() for n,d in files.items()})))
        plan=dict(run_id=run_id,benchmark=bench,lanes=lanes,source=str(source.resolve()),
                  archive=str(archive.resolve()),local=str(local.resolve()),identity=identity)
        write_json(local/'plan.json',plan);plans.append(plan)
    write_json(STATE/'plans.json',plans)
    print(json.dumps([dict(run_id=p['run_id'],cases=sum(map(len,p['lanes']))) for p in plans],indent=2))


def build(source):
    import modal
    from .modal_app import build_app
    app=build_app(source,app_name=APP,cache_name=CACHE,runs_name=RUNS,secret_name='huggingface-secret',
                  config_filename='config.512.json')
    runs=modal.Volume.from_name(RUNS);cache=modal.Volume.from_name(CACHE)
    freecus=modal.Volume.from_name(FREECUS_RUNS) if PROFILE=='second' else None
    cpu=(modal.Image.debian_slim(python_version='3.11').add_local_dir(source,'/root'))
    metric=(modal.Image.debian_slim(python_version='3.11').apt_install('git')
            .pip_install('torch==2.7.1','torchvision==0.22.1','numpy==1.26.4','Pillow==11.3.0',
                         'ftfy==6.3.1','regex','tqdm','packaging')
            .pip_install('git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6')
            .env(dict(TORCH_HOME='/cache/eval-torch',CLIP_DOWNLOAD_ROOT='/cache/eval-clip',PYTHONUNBUFFERED='1'))
            .add_local_dir(source,'/root'))

    @app.function(name='evaluate_export_512',serialized=True,image=metric,gpu='T4',
                  volumes={'/work':runs,'/cache':cache},cpu=(2,2),memory=(8192,8192),
                  timeout=7200,retries=0,max_containers=1,scaledown_window=2)
    def evaluate_export_512(run_id,benchmark):
        import csv
        import time
        from freecus.evaluate import prepare,validate,project_evaluator
        from freecus.run import artifact_paths,complete,write_json
        from freecus.manifest import load
        from freecus.operations.artifact_bundle import pack,verify
        start=time.monotonic();runs.reload();cache.reload()
        bundle=Path('/work/bundles')/run_id
        rows=load(bundle/'manifest.jsonl',bundle/'input')
        source=Path('/work/results')/run_id/'A100-80GB'
        lanes=[p for p in sorted(source.glob('lane-*')) if (p/'run.json').is_file()]
        folder=Path('/work/metrics')/run_id
        available=[];missing=[]
        destination=Path('/work/evaluation_inputs')/run_id
        if lanes:
            infos=[read(p/'run.json') for p in lanes]
            if any(i['fingerprint']!=infos[0]['fingerprint'] for i in infos): raise ValueError('Mixed run identities')
            destination.mkdir(parents=True,exist_ok=True)
            shutil.copy2(lanes[0]/'run.json',destination/'run.json')
        for row in rows:
            found=[artifact_paths(p,row) for p in lanes if complete(artifact_paths(p,row))]
            if len(found)>1: raise ValueError('Duplicate case '+row['case_id'])
            if not found:
                missing.append(row['case_id']);continue
            for kind,src in found[0].items():
                dst=artifact_paths(destination,row)[kind];dst.parent.mkdir(parents=True,exist_ok=True)
                if dst.exists():
                    if sha256(src)!=sha256(dst): raise ValueError('Existing assembled bytes differ')
                else:
                    try: dst.hardlink_to(src)
                    except OSError: shutil.copy2(src,dst)
            available.append(row['case_id'])
        coverage=dict(expected=len(rows),generated=len(available),missing_case_ids=missing,excluded_case_ids=[],
                      style_included=True,status='complete' if not missing else 'incomplete')
        write_json(folder/'coverage.json',coverage)
        if not available:
            runs.commit();return dict(coverage=coverage,verified=False,summary=None)
        evaluator=project_evaluator()
        plan=prepare(bundle/'manifest.jsonl',bundle/'input',destination,evaluator,'all',available,condition='uno_512')
        write_json(folder/'plan.json',plan)
        scores=folder/'metrics.json'
        if scores.exists(): result=read(scores)
        else:
            result=evaluator.evaluate(plan,scores,'cuda',32,validator=validate,
                    checkpoint=folder/'checkpoint.json',commit=lambda _:runs.commit())
            result.update(evaluation_selection=plan['selection'],resolution=[512,512],complete=not missing)
            write_json(scores,result)
        with (folder/'per_image.csv').open('w',newline='',encoding='utf-8-sig') as stream:
            fields=['case_id','reference_id','benchmark','category','condition','CLIP-I','CLIP-T','DINO-I']
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(result['scores'])
        extras={p.name:p for p in [bundle/'models.lock.json',bundle/'config.512.json',bundle/'source.zip',bundle/'identity.json',folder/'coverage.json']}
        for lane in lanes:
            for p in (lane/'conditioning').glob('*.png'): extras[f'conditioning/{lane.name}/{p.name}']=p
        archives=Path('/work/exports')/run_id
        index=pack(lanes,bundle/'manifest.jsonl',bundle/'input',archives,'uno_512',benchmark,run_id,
                   len(available),resolution=(512,512),provenance_files=extras)
        verify(archives,index)
        report=dict(coverage=coverage,summary=result['summary'],verified=True,parts=index['parts'],
                    elapsed_seconds=time.monotonic()-start)
        write_json(folder/'summary.json',report);write_json(archives/'verification.json',report)
        cache.commit();runs.commit();return report

    @app.function(name='queue_after_freecus',serialized=True,image=cpu,
                  volumes={'/work':runs,**({'/freecus':freecus} if freecus is not None else {})},cpu=(.125,.125),memory=(512,512),
                  timeout=86400,retries=0,max_containers=1,scaledown_window=2)
    def queue_after_freecus(queue_id,dependency,plans):
        import time
        import uuid
        import modal
        from uno.experiment_512 import prerequisite_ready
        report=Path('/work/workflows')/f'{queue_id}.json'
        runs.reload()
        if report.exists(): raise ValueError('Duplicate UNO queue')
        state=dict(queue_id=queue_id,status='waiting_for_freecus',dependency=dependency,plans=plans,
                   phases=[],evaluations={},started_at=time.time(),budget_policy='user-managed Modal workspace limit')
        def save():
            state['updated_at']=time.time();write_json(report,state);runs.commit()
        def phase(plan,kind,lanes):
            item=dict(benchmark=plan['benchmark'],run_id=plan['run_id'],kind=kind,calls=[])
            state['phases'].append(item);state['status']=kind+'_'+plan['benchmark'];save()
            calls=[];invocation=uuid.uuid4().hex
            for lane,cases in enumerate(lanes):
                call=modal.Function.from_name(APP,'gpu_lane').spawn(plan['run_id'],plan['benchmark'],invocation,lane,cases,21000)
                calls.append(call);item['calls'].append(dict(call_id=call.object_id,lane=lane,cases=len(cases)));save()
            for i,call in enumerate(calls):
                try: item['calls'][i]['result']=call.get()
                except Exception as exc: item['calls'][i]['result']=dict(status='failed',error=repr(exc))
                save()
            return item
        save()
        try:
            while True:
                if dependency.get('proof_sha256'):
                    runs.reload()
                    path=Path('/work/prerequisites/freecus-completion.json')
                    if sha256(path)!=dependency['proof_sha256']: raise ValueError('Prerequisite proof checksum mismatch')
                else:
                    freecus.reload()
                    path=Path('/freecus/workflows')/f"{dependency['workflow_id']}.json"
                if not path.exists(): raise ValueError('FreeCus dependency report missing')
                previous=read(path);state['freecus_status']=previous['status'];save()
                if prerequisite_ready(previous): break
                if previous['status'] in {'failed','pilot_failed','finished_with_missing_cases','complete'}:
                    state.update(status='blocked_by_freecus',reason=previous.get('error',previous['status']));save();return state
                try:
                    modal.FunctionCall.from_id(dependency['call_id']).get(timeout=0)
                except TimeoutError: pass
                except Exception as exc:
                    state.update(status='blocked_by_freecus',reason=repr(exc));save();return state
                else:
                    state.update(status='blocked_by_freecus',reason='Dependency exited without verified completion');save();return state
                time.sleep(60)
            state['prerequisite_verified_at']=time.time();save()
            for plan in plans:
                state['status']='staging_'+plan['benchmark'];save()
                call=modal.Function.from_name(APP,'stage').spawn(plan['run_id'])
                state.setdefault('stage_calls',[]).append(dict(run_id=plan['run_id'],call_id=call.object_id));save()
                call.get();runs.reload()
                bundle=Path('/work/bundles')/plan['run_id']
                rows=[json.loads(s) for s in (bundle/'manifest.jsonl').read_text().splitlines()]
                pilot=[]
                for ids in plan['lanes']:
                    selected=[r for r in rows if r['case_id'] in set(ids)]
                    refs={next(r['reference_id'] for r in selected if r['category']==c) for c in sorted({r['category'] for r in selected})}
                    pilot.append([r['case_id'] for r in selected if r['reference_id'] in refs and r['prompt_index'] in {0,4}])
                result=phase(plan,'pilot',pilot)
                if any(c['result']['status']!='complete' for c in result['calls']):
                    state['status']='pilot_failed';save();return state
                phase(plan,'generation',plan['lanes'])
                state['status']='evaluating_exporting_'+plan['benchmark'];save()
                call=modal.Function.from_name(APP,'evaluate_export_512').spawn(plan['run_id'],plan['benchmark'])
                state.setdefault('evaluation_calls',[]).append(dict(run_id=plan['run_id'],call_id=call.object_id));save()
                state['evaluations'][plan['benchmark']]=call.get();save()
                if state['evaluations'][plan['benchmark']]['coverage']['status']!='complete':
                    state['status']='finished_with_missing_cases';save();return state
            state['status']='complete';save();return state
        except BaseException as exc:
            state.update(status='failed',error=repr(exc));save();raise

    @app.function(name='inspect_512',serialized=True,image=cpu,volumes={'/work':runs},
                  cpu=(.125,.125),memory=(512,512),timeout=300,retries=0,scaledown_window=2)
    def inspect_512(queue_id):
        runs.reload();path=Path('/work/workflows')/f'{queue_id}.json'
        if not path.exists(): return dict(status='queued')
        state=read(path);state['progress']={}
        for plan in state['plans']:
            source=Path('/work/results')/plan['run_id']/'A100-80GB'
            state['progress'][plan['benchmark']]=dict(expected=sum(map(len,plan['lanes'])),
                                                    generated=sum(1 for _ in source.glob('lane-*/metadata/**/*.json')))
        state.pop('plans',None)
        for phase in state['phases']:
            for call in phase['calls']:
                if 'result' in call: call['result'].pop('timings',None)
        return state
    return app


def collect(modal,plans):
    from freecus.operations.artifact_bundle import digest,verify
    volume=modal.Volume.from_name(RUNS);reports=[]
    for plan in plans:
        run_id=plan['run_id'];local=Path(plan['local'])
        try: entries=list(volume.iterdir(f'metrics/{run_id}',recursive=True))
        except FileNotFoundError:
            reports.append(dict(run_id=run_id,status='not_ready'));continue
        for entry in entries:
            if entry.type==1:
                dest=local/Path(entry.path).name
                with dest.open('wb') as stream:
                    for chunk in volume.read_file(entry.path): stream.write(chunk)
        try: index=json.loads(b''.join(volume.read_file(f'exports/{run_id}/bundle_index.json')))
        except FileNotFoundError: continue
        folder=PROJECT.parent/'AI_PROJECT_images/uno_512'/plan['benchmark']/run_id
        folder.mkdir(parents=True,exist_ok=True)
        needed=sum(p['size_bytes'] for p in index['parts'] if not (folder/p['file']).exists())
        if shutil.disk_usage(folder).free<needed+200*1024**2: raise OSError('Insufficient disk space')
        for part in index['parts']:
            dest=folder/part['file']
            if dest.exists() and digest(dest)==part['sha256']: continue
            tmp=dest.with_suffix('.zip.partial')
            print('Downloading '+run_id+'/'+part['file'],flush=True)
            with tmp.open('wb') as stream:
                volume._read_file_into_fileobj(f"exports/{run_id}/{part['file']}",stream,concurrency=2)
            if tmp.stat().st_size!=part['size_bytes'] or digest(tmp)!=part['sha256']: raise ValueError('ZIP checksum mismatch')
            tmp.replace(dest)
        write_json(folder/'bundle_index.json',index);verify(folder,index)
        verified=dict(verified=True,parts=index['parts'],generated=index['generated_images'],
                      missing_case_ids=index['missing_case_ids'],archive_folder=str(folder))
        write_json(local/'bundle_index.json',index);write_json(local/'verification.json',verified)
        write_json(folder/'verification.json',verified);reports.append(verified)
    print(json.dumps(reports,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','deploy','queue','status','collect'])
    args=parser.parse_args();STATE.mkdir(parents=True,exist_ok=True)
    if args.command=='prepare': prepare();return
    from freecus.modal_client import connect
    modal=connect(PROFILE);plans=read(STATE/'plans.json')
    if args.command in {'deploy','queue'}:
        for p in plans:
            if any(sha256(PROJECT/n)!=v for n,v in p['identity']['code'].items()): raise ValueError('Source changed; prepare before submission')
    if args.command=='deploy':
        modal.Volume.from_name(RUNS,create_if_missing=True,version=2).hydrate()
        with modal.enable_output(): build(plans[0]['source']).deploy()
    elif args.command=='queue':
        receipt=STATE/'receipt.json'
        if receipt.exists(): raise ValueError('Queue already submitted; use status')
        volume=modal.Volume.from_name(RUNS)
        with volume.batch_upload(force=False) as upload:
            for p in plans: upload.put_file(p['archive'],f"/uploads/{p['run_id']}.zip")
            if PROFILE!='second': upload.put_file(STATE/'freecus-completion.json','/prerequisites/freecus-completion.json')
        dependency=read(PROJECT/'freecus/outputs/modal512-20261112/receipt.json')
        if PROFILE!='second':
            proof=read(STATE/'freecus-completion.json')
            if not prerequisite_ready(proof): raise ValueError('FreeCus completion proof is incomplete')
            dependency=dict(workflow_id=proof['workflow_id'],proof_sha256=sha256(STATE/'freecus-completion.json'),source_profile='second')
        queue_id='uno512-after-freecus-'+hashlib.sha256(''.join(p['run_id'] for p in plans).encode()).hexdigest()[:16]
        record=dict(queue_id=queue_id,profile=PROFILE,app=APP,dependency=dependency,status='submitting')
        write_json(receipt,record)
        call=modal.Function.from_name(APP,'queue_after_freecus').spawn(queue_id,
            dependency,
            [{k:p[k] for k in ['run_id','benchmark','lanes']} for p in plans])
        record.update(call_id=call.object_id,status='submitted',dashboard=f'https://modal.com/id/{call.object_id}')
        write_json(receipt,record)
        for p in plans: write_json(Path(p['local'])/'receipt.json',record)
        print(json.dumps(dict(queue_id=queue_id,call_id=call.object_id,dashboard=record['dashboard']),indent=2))
    elif args.command=='status':
        receipt=read(STATE/'receipt.json')
        result=modal.Function.from_name(APP,'inspect_512').remote(receipt['queue_id'])
        write_json(STATE/'status.json',result)
        for p in plans: write_json(Path(p['local'])/'workflow-status.json',result)
        print(json.dumps(result,indent=2))
    else: collect(modal,plans)


if __name__=='__main__': main()
