"""Take over existing DreamBench workers without restarting healthy GPU jobs."""
import json
from pathlib import Path
from freecus.manifest import PROJECT
from freecus.run import write_json


def main():
    from freecus.modal_client import connect
    modal=connect('method-comparison-dbpp')
    base=PROJECT/'xverse'
    receipt=json.loads((base/'outputs/workflow.json').read_text())
    prepared=json.loads((base/'outputs/plan.json').read_text())
    runs=modal.Volume.from_name('xverse-runs')
    original=json.loads(b''.join(runs.read_file('workflows/'+receipt['workflow_id']+'.json')))
    benchmark=original['benchmarks'][0]
    phase=next(p for p in benchmark['phases'] if p['label']=='full')
    plan=next(p for p in prepared['plans'] if p['benchmark']=='dreambench')
    out=base/'outputs/dreambench-30usd-guard.json'
    if out.exists():raise ValueError('Guard already submitted')
    image=(modal.Image.debian_slim(python_version='3.11').pip_install('Pillow==11.1.0')
        .add_local_dir(prepared['source'],'/root')
        .add_local_file(__file__,'/root/comparison_candidates/guard_existing.py'))
    app=modal.App('xverse-dreambench-30usd-guard')

    @app.function(name='guard',serialized=True,image=image,volumes={'/work':runs},
        cpu=(0.25,0.25),memory=(512,512),timeout=86400,max_containers=1,retries=0,scaledown_window=2)
    def guard(original,phase,plan,parent_call):
        import time
        import modal
        from pathlib import Path
        from freecus.run import write_json
        rate=.000694+2*.0000131+64*.00000222
        folder=Path('/work/budget_guards')/original['workflow_id']
        folder.mkdir(parents=True,exist_ok=True)
        path=folder/'state.json'
        runs.reload()
        if path.exists():state=json.loads(path.read_text())
        else:
            state=dict(status='monitoring',benchmark='dreambench',bundle_id=plan['bundle_id'],budget_usd=30,
                generation_cap_usd=26.5,reserve_usd=3.5,base_compute_usd=original['generation_compute_usd'],
                calls=[dict(**c,attempt=0) for c in phase['calls']],started_at=time.time())
            write_json(path,state);runs.commit()
        # Only the orchestration parent is cancelled. Existing GPU children stay.
        modal.FunctionCall.from_id(parent_call).cancel(terminate_containers=True)
        def save():
            state['updated_at']=time.time();write_json(path,state);runs.commit()
        while True:
            runs.reload()
            for call in list(state['calls']):
                if 'result' in call:continue
                try:result=modal.FunctionCall.from_id(call['id']).get(timeout=0)
                except TimeoutError:continue
                call['result']=result;save()
                if result['status']!='complete' and call['attempt']<1:
                    recovery=modal.Function.from_name('xverse-method-comparison','lane').spawn(
                        plan['bundle_id'],call['lane'],plan['lanes'][call['lane']],7200,phase['invocation'])
                    state['calls'].append(dict(id=recovery.object_id,lane=call['lane'],attempt=call['attempt']+1))
                    save()
            total=state['base_compute_usd']
            latest={}
            for call in state['calls']:
                latest[call['lane']]=call
                if 'result' in call:elapsed=call['result']['elapsed_seconds']
                else:
                    journal=Path('/work/invocations')/phase['invocation']/f"lane-{call['lane']}"/'state.json'
                    try:elapsed=json.loads(journal.read_text()).get('elapsed_seconds',0)
                    except (FileNotFoundError,json.JSONDecodeError):elapsed=0
                total+=(elapsed+30)*rate
            state['estimated_generation_usd']=total;save()
            if total>=state['generation_cap_usd']:
                for c in latest.values():
                    if 'result' not in c:modal.FunctionCall.from_id(c['id']).cancel(terminate_containers=True)
                state['status']='budget_paused';save();return state
            if all('result' in c for c in latest.values()):
                if any(c['result']['status']!='complete' for c in latest.values()):
                    state['status']='generation_failed';save();return state
                break
            time.sleep(10)
        state['status']='evaluating';save()
        if state.get('evaluation_call'):
            score=modal.FunctionCall.from_id(state['evaluation_call'])
        else:
            score=modal.Function.from_name('xverse-method-comparison','evaluate').spawn(plan)
            state['evaluation_call']=score.object_id;save()
        state['evaluation']=score.get();state['status']='complete';save()
        return state
    with modal.enable_output():app.deploy()
    call=modal.Function.from_name('xverse-dreambench-30usd-guard','guard').spawn(original,phase,plan,receipt['call_id'])
    record=dict(call_id=call.object_id,workflow_id=original['workflow_id'],budget_usd=30,
                generation_cap_usd=26.5,benchmark='dreambench',app='xverse-dreambench-30usd-guard')
    write_json(out,record)
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
