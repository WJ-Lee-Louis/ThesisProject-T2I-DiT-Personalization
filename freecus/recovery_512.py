"""Resume the accepted immutable 512 runs without regenerating DreamBench images."""
import argparse
import json
from pathlib import Path

from .experiment_512 import BASE, STATE, read
from .modal_client import connect
from .remote_512 import APP, CACHE, RUNS
from .run import write_json

RECOVERY_APP='freecus-512-cache-recovery-20261112'


def verified_copy(source,destination):
    import hashlib
    import shutil
    def digest(path):
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
        return h.hexdigest()
    original=digest(source)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        if digest(destination)!=original: raise ValueError('Existing DINO cache differs; refusing overwrite')
    else:
        temporary=destination.with_suffix('.copying')
        shutil.copyfile(source,temporary)
        if digest(temporary)!=original: raise ValueError('DINO copy checksum mismatch')
        temporary.replace(destination)
    return dict(source=str(source),destination=str(destination),sha256=original,bytes=source.stat().st_size,
                verified=True,tls_verification_disabled=False)


def build(source):
    import modal
    runs=modal.Volume.from_name(RUNS)
    cache=modal.Volume.from_name(CACHE)
    image=(modal.Image.debian_slim(python_version='3.11')
           .add_local_dir(source,'/root/freecus')
           .add_local_file(Path(__file__),'/root/freecus/recovery_512.py'))
    app=modal.App(RECOVERY_APP)

    @app.function(name='resume',serialized=True,image=image,volumes={'/work':runs,'/cache':cache},
                  cpu=(.125,.125),memory=(512,512),timeout=86400,retries=0,max_containers=1,scaledown_window=2)
    def resume(previous_id,recovery_id):
        import copy
        import time
        import modal
        from freecus.run import write_json
        from freecus.recovery_512 import verified_copy
        runs.reload();cache.reload()
        report=Path('/work/workflows')/f'{recovery_id}.json'
        if report.exists(): raise ValueError('Duplicate recovery')
        previous=json.loads((Path('/work/workflows')/f'{previous_id}.json').read_text())
        if previous['status']!='failed': raise ValueError('Original workflow is not failed')
        if 'CERTIFICATE_VERIFY_FAILED' not in previous.get('error',''):
            raise ValueError('Unexpected original failure')
        state=copy.deepcopy(previous)
        state.update(workflow_id=recovery_id,parent_workflow_id=previous_id,status='repairing_dino_cache',
                     previous_error=state.pop('error',None),recovery_started_at=time.time())
        def save():
            state['updated_at']=time.time();write_json(report,state);runs.commit()
        save()
        try:
            model='dino_deitsmall16_pretrain.pth'
            record=verified_copy(Path('/cache/torch/hub/checkpoints')/model,
                                 Path('/cache/eval-torch/hub/checkpoints')/model)
            record.update(model='DINOv1 vits16',
                          original_url='https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/'+model)
            state['cache_repair']=record;cache.commit();save()
            db=next(p for p in state['plans'] if p['benchmark']=='dreambench')
            state['status']='evaluating_dreambench';save()
            call=modal.Function.from_name(APP,'evaluate_512').spawn(db['run_id'])
            state.setdefault('evaluation_calls',[]).append(dict(run_id=db['run_id'],call_id=call.object_id,recovery=True));save()
            result=call.get()
            state.setdefault('evaluations',{})['dreambench']=result
            state['estimated_compute_usd']+=result['estimated_compute_usd'];save()
            state['status']='exporting_dreambench';save()
            call=modal.Function.from_name(APP,'export_512').spawn(db['run_id'],'dreambench')
            state.setdefault('export_calls',[]).append(dict(run_id=db['run_id'],call_id=call.object_id));save()
            state.setdefault('exports',{})['dreambench']=call.get();save()
            plus=next(p for p in state['plans'] if p['benchmark']=='dreambench_plus')
            child_id=recovery_id+'-dbpp'
            call=modal.Function.from_name(APP,'workflow_512').spawn(child_id,[plus])
            state['child_workflow']=dict(workflow_id=child_id,call_id=call.object_id)
            state['status']='starting_dreambench_plus';save()
            base_phases=copy.deepcopy(state['phases'])
            while True:
                try:
                    child=call.get(timeout=45)
                    break
                except TimeoutError:
                    runs.reload()
                    path=Path('/work/workflows')/f'{child_id}.json'
                    if path.exists():
                        child=json.loads(path.read_text())
                        state['status']=child['status']
                        state['phases']=base_phases+child.get('phases',[])
                        state.setdefault('pilot_measurements',{}).update(child.get('pilot_measurements',{}))
                        save()
            state['phases']=base_phases+child.get('phases',[])
            for key in ['evaluations','exports','pilot_measurements']:
                state.setdefault(key,{}).update(child.get(key,{}))
            state['estimated_compute_usd']+=child.get('estimated_compute_usd',0)
            state['status']=child['status']
            if child.get('error'): state['error']=child['error']
            save();return state
        except BaseException as exc:
            state['status']='failed';state['error']=repr(exc);save();raise
    return app,resume


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['submit','status'])
    args=parser.parse_args()
    modal=connect('second')
    target=STATE/'recovery-receipt.json'
    if args.command=='status':
        receipt=read(target)
        result=modal.Function.from_name(APP,'inspect_512').remote(receipt['workflow_id'])
        write_json(STATE/'status.json',result)
        for p in read(STATE/'plans.json'):
            write_json(Path(p['local'])/'workflow-status.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in {'plans','phases'}},indent=2))
        return
    if target.exists(): raise ValueError('Recovery already submitted; use status')
    receipt=read(STATE/'receipt.json')
    try: modal.FunctionCall.from_id(receipt['call_id']).get(timeout=0)
    except TimeoutError: raise ValueError('Original workflow still running; refusing duplicate')
    except Exception: pass
    else: raise ValueError('Original workflow succeeded; refusing duplicate')
    plans=read(STATE/'plans.json')
    app,function=build(plans[0]['source'])
    with modal.enable_output(): app.deploy()
    recovery_id=receipt['workflow_id']+'-cache-recovery-v1'
    record=dict(workflow_id=recovery_id,profile='second',app=APP,recovery_app=RECOVERY_APP,parent=receipt,
                status='submitting')
    write_json(target,record)
    call=modal.Function.from_name(RECOVERY_APP,'resume').spawn(receipt['workflow_id'],recovery_id)
    record.update(call_id=call.object_id,status='submitted',dashboard=f'https://modal.com/id/{call.object_id}')
    write_json(target,record)
    write_json(STATE/'receipt-before-recovery.json',receipt)
    write_json(STATE/'receipt.json',record)
    for p in plans:
        write_json(Path(p['local'])/'recovery-receipt.json',record)
    print(json.dumps(dict(workflow_id=recovery_id,call_id=call.object_id,dashboard=record['dashboard']),indent=2))


if __name__=='__main__': main()
