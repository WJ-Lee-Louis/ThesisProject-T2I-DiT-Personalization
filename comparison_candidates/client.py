"""Explicit-profile, immutable-source preparation and detached submission."""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from freecus.manifest import PROJECT,load,sha256
from freecus.run import write_json

PROFILES={'diptych':'method-comparison','xverse':'method-comparison-dbpp'}


def prepare(method):
    base=PROJECT/method
    files=list((PROJECT/'comparison_candidates').glob('*.py'))
    files+=list(base.glob('*.json'))+list(base.glob('requirements*.txt'))
    files += [p for p in (base/'upstream').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    files += [PROJECT/'freecus'/n for n in ['__init__.py','manifest.py','run.py','evaluate.py',
        'project_metrics/benchmark_metrics.py','project_metrics/SOURCE.json']]
    identity={p.relative_to(PROJECT).as_posix():sha256(p) for p in files}
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    source=base/'outputs/sources'/digest
    for file in files:
        target=source/file.relative_to(PROJECT)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,target)
    plans=[]
    for benchmark,count in [('dreambench',750),('dreambench_plus',1350)]:
        manifest=PROJECT/f'freecus/manifests/{benchmark}_project_seed42.jsonl'
        rows=load(manifest,PROJECT)
        if len(rows)!=count or {r['seed'] for r in rows}!={42}:raise ValueError('Invalid canonical manifest')
        local=base/'manifests'/f'{benchmark}_seed42.jsonl'
        local.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(manifest,local)
        bundle_id=hashlib.sha256((digest+sha256(manifest)).encode()).hexdigest()[:24]
        archive=base/'outputs/uploads'/(bundle_id+'.zip');archive.parent.mkdir(parents=True,exist_ok=True)
        data={'manifest.jsonl':manifest.read_bytes()}
        for reference in sorted({r['reference'] for r in rows}):data['input/'+reference]=(PROJECT/reference).read_bytes()
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as z:
            for name,content in data.items():z.writestr(name,content)
            z.writestr('bundle.json',json.dumps(dict(files={n:hashlib.sha256(v).hexdigest() for n,v in data.items()})))
        owners={ref:i%4 for i,ref in enumerate(sorted({r['reference_id'] for r in rows}))}
        lanes=[[r['case_id'] for r in rows if owners[r['reference_id']]==i] for i in range(4)]
        plans.append(dict(benchmark=benchmark,cases=count,bundle_id=bundle_id,archive=str(archive.resolve()),lanes=lanes))
    cfg=json.loads((base/'config.json').read_text())
    state=dict(method=method,profile=PROFILES[method],source=str(source.resolve()),code_sha256=digest,plans=plans,
               gpu=cfg.get('gpu','A100-80GB'),admission_usd=cfg.get('admission_usd',40))
    write_json(base/'outputs/plan.json',state)
    print(json.dumps(dict(method=method,profile=state['profile'],code=digest,
         plans=[dict(benchmark=p['benchmark'],bundle_id=p['bundle_id'],lanes=list(map(len,p['lanes']))) for p in plans]),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','deploy','submit','status','logs'])
    p.add_argument('--method',choices=list(PROFILES),required=True)
    a=p.parse_args()
    base=PROJECT/a.method
    if a.command=='prepare':prepare(a.method);return
    plan=json.loads((base/'outputs/plan.json').read_text())
    from freecus.modal_client import connect
    modal=connect(plan['profile'])
    app=a.method+'-method-comparison'
    if a.command=='deploy':
        from .modal_app import build
        with modal.enable_output():result=build(plan['source'],a.method,plan.get('gpu','A100-80GB')).deploy()
        write_json(base/'outputs/deployment.json',dict(code=plan['code_sha256'],profile=plan['profile']))
        print('Deployed',app);return
    runs=modal.Volume.from_name(a.method+'-runs')
    receipt_path=base/'outputs/workflow.json'
    if a.command=='submit':
        if receipt_path.exists():raise ValueError('Already submitted; inspect status')
        deployment=json.loads((base/'outputs/deployment.json').read_text())
        if deployment!=dict(code=plan['code_sha256'],profile=plan['profile']):raise ValueError('Deployment mismatch')
        for item in plan['plans']:
            with runs.batch_upload() as upload:upload.put_file(item['archive'],'/uploads/'+item['bundle_id']+'.zip')
        import uuid
        workflow_id=uuid.uuid4().hex
        call=modal.Function.from_name(app,'workflow').spawn(plan['plans'],workflow_id,plan.get('admission_usd',40))
        receipt=dict(workflow_id=workflow_id,call_id=call.object_id,profile=plan['profile'],app=app,
                     code=plan['code_sha256'],budget_usd=plan.get('admission_usd',40),gpu=plan.get('gpu','A100-80GB'),max_gpu_workers=4)
        write_json(receipt_path,receipt);print(json.dumps(receipt,indent=2));return
    receipt=json.loads(receipt_path.read_text())
    guard_receipt=base/'outputs/dreambench-30usd-guard.json'
    if a.command=='status' and guard_receipt.exists():
        guard=json.loads(guard_receipt.read_text())
        try:status=modal.FunctionCall.from_id(guard['call_id']).get(timeout=0)
        except TimeoutError:
            status=json.loads(b''.join(runs.read_file('budget_guards/'+guard['workflow_id']+'/state.json')))
        write_json(base/'outputs/budget-guard-status.json',status)
        if status.get('evaluation',{}).get('status')=='complete':
            data=b''.join(runs.read_file(status['evaluation']['scores']))
            out=base/'results/dreambench'/status['bundle_id']/'metrics.json'
            out.parent.mkdir(parents=True,exist_ok=True)
            if out.exists() and out.read_bytes()!=data:raise ValueError('Existing scores differ')
            out.write_bytes(data)
        print(json.dumps(status,indent=2));return
    if a.command=='logs':
        status=json.loads(b''.join(runs.read_file('workflows/'+receipt['workflow_id']+'.json')))
        for benchmark in status.get('benchmarks',[]):
            for phase in benchmark['phases']:
                for call in phase['calls']:
                    name=f"invocations/{phase['invocation']}/lane-{call['lane']}/worker.log"
                    try:
                        log=b''.join(runs.read_file(name)).decode(errors='replace')
                        print(benchmark['benchmark'],phase['label'],call['lane'],log[-4500:])
                    except FileNotFoundError:pass
        return
    try:status=modal.FunctionCall.from_id(receipt['call_id']).get(timeout=0)
    except TimeoutError:
        try:status=json.loads(b''.join(runs.read_file('workflows/'+receipt['workflow_id']+'.json')))
        except FileNotFoundError:status=dict(status='starting')
    write_json(base/'outputs/workflow-status.json',status)
    # Download finalized scores only; image data remains durable in Modal Volume.
    for benchmark in status.get('benchmarks',[]):
        if benchmark.get('evaluation',{}).get('status')=='complete':
            data=b''.join(runs.read_file(benchmark['evaluation']['scores']))
            out=base/'results'/benchmark['benchmark']/benchmark['bundle_id']/'metrics.json'
            out.parent.mkdir(parents=True,exist_ok=True)
            if out.exists() and out.read_bytes()!=data:raise ValueError('Existing score differs')
            out.write_bytes(data)
    print(json.dumps(status,indent=2))


if __name__=='__main__':main()
