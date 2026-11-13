"""Verify completed child artifacts before reconciling status and unblocking UNO."""
import json
from pathlib import Path

from .experiment_512 import STATE,read
from .manifest import PROJECT
from .modal_client import connect
from .run import write_json

UNO_STATE=PROJECT/'uno/outputs/modal512-20261112'
UNO_APP='uno-512-seed42-after-freecus-20261112'
FREECUS_RUNS='freecus-512-paper-seed42-20261112-runs'
UNO_RUNS='uno-512-seed42-20261112-runs'


def main():
    modal=connect('second')
    current=read(STATE/'receipt.json');uno_receipt=read(UNO_STATE/'receipt.json')
    records=read(STATE/'plans.json');uno_plans=read(UNO_STATE/'plans.json')
    volume=modal.Volume.from_name(FREECUS_RUNS)
    uno_volume=modal.Volume.from_name(UNO_RUNS)
    # An active old queue must not be duplicated, even if its last written state is stale.
    try: old=modal.FunctionCall.from_id(uno_receipt['call_id']).get(timeout=0)
    except TimeoutError: raise ValueError('UNO queue is still running; refusing duplicate')
    if old.get('status')!='blocked_by_freecus': raise ValueError('Unexpected UNO queue status')
    image=(modal.Image.debian_slim(python_version='3.11')
           .add_local_dir(records[0]['source'],'/root/freecus'))
    app=modal.App('experiment-512-reconcile-20261112')

    @app.function(name='reconcile',serialized=True,image=image,
                  volumes={'/work':volume,'/uno':uno_volume},cpu=(.125,.125),memory=(512,512),
                  timeout=1800,retries=0,max_containers=1,scaledown_window=2)
    def reconcile(parent_id,old_queue_id,plans):
        import copy
        import hashlib
        import time
        import modal
        from pathlib import Path
        from freecus.run import write_json
        def read_json(path): return json.loads(path.read_text())
        def digest(path):
            h=hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda:stream.read(8*1024**2),b''): h.update(chunk)
            return h.hexdigest()
        volume.reload();uno_volume.reload()
        completion_id=parent_id+'-verified-completion'
        maintenance=Path('/work/workflows')/(completion_id+'-receipt.json')
        if maintenance.exists(): return read_json(maintenance)
        parent=read_json(Path('/work/workflows')/f'{parent_id}.json')
        child=read_json(Path('/work/workflows')/(parent['child_workflow']['workflow_id']+'.json'))
        if child['status']!='complete': raise ValueError('Child workflow not complete')
        combined=copy.deepcopy(parent)
        for key in ['evaluations','exports','pilot_measurements']:
            combined.setdefault(key,{}).update(child.get(key,{}))
        checks={}
        for bench,count in [('dreambench',750),('dreambench_plus',1170)]:
            coverage=combined['evaluations'][bench]['coverage']
            export=combined['exports'][bench]
            if (coverage['status']!='complete' or coverage['expected']!=count or coverage['generated']!=count
                    or coverage['missing'] or not export['verified'] or export['missing_case_ids']):
                raise ValueError('Incomplete or unverified outputs: '+bench)
            plan=next(p for p in parent['plans'] if p['benchmark']==bench)
            folder=Path('/work/exports')/plan['run_id']
            index=read_json(folder/'bundle_index.json')
            if index['generated_images']!=count or index['generation_resolution']!=[512,512]:
                raise ValueError('Archive identity mismatch')
            for part in index['parts']:
                path=folder/part['file']
                if path.stat().st_size!=part['size_bytes'] or digest(path)!=part['sha256']:
                    raise ValueError('Archive bytes changed: '+bench)
            checks[bench]=dict(generated=count,metric_coverage_complete=True,archive_sha256_verified=True,parts=index['parts'])
        for plan in plans:
            source=Path('/uno/results')/plan['run_id']
            if any(source.rglob('metadata/*.json')):
                raise ValueError('UNO already has images; inspect before queue replacement')
        combined.update(workflow_id=completion_id,status='complete',
                        reconciliation=dict(parent_workflow_id=parent_id,child_workflow_id=child['workflow_id'],
                                            parent_error=parent.get('error'),checks=checks,time=time.time()),
                        updated_at=time.time())
        combined.pop('error',None)
        combined['phases']=parent['phases']+child['phases']
        write_json(Path('/work/workflows')/f'{completion_id}.json',combined);volume.commit()
        new_queue=old_queue_id+'-verified-v1'
        dependency=dict(workflow_id=completion_id,call_id=parent['child_workflow']['call_id'])
        call=modal.Function.from_name('uno-512-seed42-after-freecus-20261112','queue_after_freecus').spawn(
            new_queue,dependency,plans)
        result=dict(freecus_workflow_id=completion_id,freecus_status='complete',checks=checks,
                    uno_queue_id=new_queue,uno_call_id=call.object_id,dependency=dependency)
        write_json(maintenance,result);volume.commit();return result

    with modal.enable_output(): app.deploy()
    call=modal.Function.from_name(app.name,'reconcile').spawn(current['workflow_id'],uno_receipt['queue_id'],
        [{k:p[k] for k in ['run_id','benchmark','lanes']} for p in uno_plans])
    write_json(STATE/'reconciliation-submission.json',dict(call_id=call.object_id,parent_workflow_id=current['workflow_id']))
    result=call.get()
    write_json(STATE/'reconciliation.json',result)
    write_json(STATE/'receipt-before-reconciliation.json',current)
    new_freecus=dict(current,workflow_id=result['freecus_workflow_id'],call_id=call.object_id,
                     status='complete',reconciled_from=current['workflow_id'])
    write_json(STATE/'receipt.json',new_freecus)
    write_json(UNO_STATE/'receipt-before-reconciliation.json',uno_receipt)
    new_uno=dict(uno_receipt,queue_id=result['uno_queue_id'],call_id=result['uno_call_id'],status='submitted',
                 dependency=result['dependency'],replaced_queue_id=uno_receipt['queue_id'],
                 dashboard='https://modal.com/id/'+result['uno_call_id'])
    write_json(UNO_STATE/'receipt.json',new_uno)
    for p in records: write_json(Path(p['local'])/'reconciliation.json',result)
    for p in uno_plans: write_json(Path(p['local'])/'receipt.json',new_uno)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
