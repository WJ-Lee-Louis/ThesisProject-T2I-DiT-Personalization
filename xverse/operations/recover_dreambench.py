"""Resume exactly the 17 missing DreamBench cases on louislee-85468."""
import json
import uuid
from pathlib import Path
from freecus.manifest import PROJECT,sha256
from freecus.operations.artifact_bundle import write

STATE=PROJECT/'xverse/outputs/dreambench_recovery'
APP='xverse-dreambench-recovery'


def main():
    STATE.mkdir(parents=True,exist_ok=True)
    original=json.loads((PROJECT/'xverse/outputs/plan.json').read_text())
    source=Path(original['source'])
    plan=next(p for p in original['plans'] if p['benchmark']=='dreambench')
    bid=plan['bundle_id']
    index=json.loads((PROJECT/'xverse/results/dreambench'/bid/'bundle_index.json').read_text())
    cases=index['missing_case_ids']
    if len(cases)!=17:raise ValueError('Expected exactly 17 missing cases')
    run=json.loads(next((PROJECT/'xverse/outputs/collected/louislee').glob('*/run.json')).read_text())
    for name,value in run['code'].items():
        if sha256(source/name)!=value:raise ValueError('Original generation source changed: '+name)
    from freecus.modal_client import connect
    modal=connect('method-comparison')
    runs=modal.Volume.from_name('xverse-runs')
    cache=modal.Volume.from_name('freecus-method-comparison-models')
    receipt=STATE/'receipt.json'
    if receipt.exists():
        record=json.loads(receipt.read_text())
        try:result=modal.FunctionCall.from_id(record['call_id']).get(timeout=0)
        except TimeoutError:
            try:result=json.loads(b''.join(runs.read_file('dreambench_recovery/'+bid+'/state.json')))
            except FileNotFoundError:result=dict(status='starting')
        write(STATE/'status.json',result)
        print(json.dumps(result,indent=2));return
    from comparison_candidates.modal_app import build
    app=build(source,'xverse',app_name=APP,cache_name='freecus-method-comparison-models',
        secret_name='dreamo-huggingface',lane_count=1,host_memory_mib=32768)
    image=modal.Image.debian_slim(python_version='3.11').pip_install('Pillow==11.1.0').add_local_dir(source,'/root')

    @app.function(name='resume',serialized=True,image=image,volumes={'/work':runs,'/cache':cache},cpu=(.25,.25),
        memory=(512,512),timeout=3600,max_containers=1,retries=0,scaledown_window=2)
    def resume(bid,cases,invocation,model_lock):
        import time,zipfile,shutil
        from freecus.run import write_json
        from freecus.manifest import sha256
        runs.reload();cache.reload()
        state_path=Path('/work/dreambench_recovery')/bid/'state.json'
        state=dict(status='preparing',workspace='louislee-85468',bundle_id=bid,case_ids=cases,
            invocation=invocation,started_at=time.time(),gpu_seconds_cap=1800)
        def save():write_json(state_path,state);runs.commit()
        save()
        try:
            bundle=Path('/work/bundles')/bid
            bundle.mkdir(parents=True,exist_ok=True)
            with zipfile.ZipFile(Path('/work/uploads')/(bid+'.zip')) as z:
                for name in z.namelist():
                    if not (bundle/name).resolve().is_relative_to(bundle.resolve()):raise ValueError('Unsafe member')
                z.extractall(bundle)
            hashes=json.loads((bundle/'bundle.json').read_text())['files']
            for name,value in hashes.items():
                if sha256(bundle/name)!=value:raise ValueError('Input bytes differ')
            existing=Path('/work/bundles/f811b3fa80a9700e30dd8115/resolved_models.json')
            if not existing.exists():raise FileNotFoundError('Expected existing XVerse model cache paths')
            if json.loads(Path('/root/xverse/models.lock.json').read_text())!=model_lock:raise ValueError('Model lock differs')
            resolved=json.loads(existing.read_text())
            for key,item in model_lock.items():
                if not Path(resolved[key]).exists() or Path(resolved[key]).name!=item['revision']:
                    raise ValueError('Cached model revision differs')
            shutil.copyfile(existing,bundle/'resolved_models.json')
            state['status']='generating';save()
            call=modal.Function.from_name('xverse-dreambench-recovery','lane').spawn(bid,0,cases,1800,invocation)
            state['generation_call_id']=call.object_id;save()
            result=call.get();state['generation']=result
            state['status']='complete' if result['status']=='complete' and result['generated_total']==len(cases) else 'incomplete'
            save();return state
        except BaseException as exc:
            state['status']='failed';state['error']=repr(exc);save();raise

    # The supervisor needs the same account's cache only to inspect resolved paths.
    # Mount it at the original absolute path used in resolved_models.json.
    # The generation function independently mounts and reloads the cache.
    with modal.enable_output():app.deploy()
    with runs.batch_upload() as upload:upload.put_file(plan['archive'],'/uploads/'+bid+'.zip')
    invocation=uuid.uuid4().hex
    call=resume.spawn(bid,cases,invocation,json.loads((source/'xverse/models.lock.json').read_text()))
    write(receipt,dict(call_id=call.object_id,workspace='louislee-85468',profile='method-comparison',
        app=APP,bundle_id=bid,case_ids=cases,gpu='A100-80GB',gpu_seconds_cap=1800,
        generation_source_sha256=original['code_sha256'],generation_fingerprint=run['fingerprint']))
    print(receipt.read_text())


if __name__=='__main__':main()
