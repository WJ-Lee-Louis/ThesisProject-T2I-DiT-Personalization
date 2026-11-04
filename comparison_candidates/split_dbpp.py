"""Budget-weighted XVerse DB++ generation across three explicit workspaces."""
import argparse
import hashlib
import json
import shutil
import zipfile
from collections import Counter,defaultdict
from pathlib import Path
from freecus.manifest import PROJECT,load,sha256
from freecus.run import write_json

SHARES={
    'wj-lee-louis':dict(profile='wj-lee-louis',budget=21,references=51,workers=2,cache='uno-models',secret='uno-huggingface'),
    'woojin716':dict(profile='woojin716',budget=13,references=24,workers=1,cache='uno-models',secret='uno-huggingface'),
    'louislee-85468':dict(profile='method-comparison',budget=28,references=75,workers=3,cache='freecus-method-comparison-models',secret='dreamo-huggingface'),
}
STATE=PROJECT/'xverse/outputs/dbpp_split'
APP='xverse-dbpp-split'


def assignments(rows):
    groups=defaultdict(list)
    for r in rows:groups[r['reference_id']].append(r)
    if len(groups)!=150 or any(len(v)!=9 for v in groups.values()):raise ValueError('Expected 150 x 9')
    # Deal each category round-robin by normalized quota; prevents allocating all
    # expensive human/style cases to one account and keeps references indivisible.
    owners={};used=Counter()
    for category in sorted({r['category'] for r in rows}):
        refs=sorted(k for k,v in groups.items() if v[0]['category']==category)
        for ref in refs:
            candidates=[k for k,v in SHARES.items() if used[k]<v['references']]
            selected=min(candidates,key=lambda k:used[k]/SHARES[k]['references'])
            owners[ref]=selected;used[selected]+=1
    return {k:[r['case_id'] for r in rows if owners[r['reference_id']]==k] for k in SHARES}


def prepare():
    STATE.mkdir(parents=True,exist_ok=True)
    manifest=PROJECT/'xverse/manifests/dreambench_plus_seed42.jsonl'
    rows=load(manifest,PROJECT);parts=assignments(rows)
    files=list((PROJECT/'comparison_candidates').glob('*.py'))
    files+=list((PROJECT/'xverse').glob('*.json'))+list((PROJECT/'xverse').glob('requirements*.txt'))
    files+=[p for p in (PROJECT/'xverse/upstream').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    files += [PROJECT/'freecus'/n for n in ['__init__.py','manifest.py','run.py','evaluate.py',
                                          'project_metrics/benchmark_metrics.py','project_metrics/SOURCE.json']]
    hashes={p.relative_to(PROJECT).as_posix():sha256(p) for p in files}
    code=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    source=STATE/'sources'/code
    for file in files:
        target=source/file.relative_to(PROJECT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,target)
    bid=hashlib.sha256((code+sha256(manifest)+'budget-split-v1').encode()).hexdigest()[:24]
    archive=STATE/(bid+'.zip')
    data={'manifest.jsonl':manifest.read_bytes()}
    for name in sorted({r['reference'] for r in rows}):data['input/'+name]=(PROJECT/name).read_bytes()
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as z:
        for name,content in data.items():z.writestr(name,content)
        z.writestr('bundle.json',json.dumps(dict(files={n:hashlib.sha256(v).hexdigest() for n,v in data.items()})))
    plans={}
    for workspace,share in SHARES.items():
        ids=set(parts[workspace]);selected=[r for r in rows if r['case_id'] in ids]
        owner={ref:i%share['workers'] for i,ref in enumerate(sorted({r['reference_id'] for r in selected}))}
        lanes=[[r['case_id'] for r in selected if owner[r['reference_id']]==i] for i in range(share['workers'])]
        plans[workspace]=dict(**share,workspace=workspace,benchmark='dreambench_plus',cases=1350,
            assigned_case_ids=sorted(ids),bundle_id=bid,lanes=lanes,
            categories=dict(Counter(r['category'] for r in selected)))
    record=dict(source=str(source.resolve()),code_sha256=code,bundle_id=bid,archive=str(archive.resolve()),plans=plans)
    write_json(STATE/'plan.json',record)
    write_json(PROJECT/'xverse/verification/dbpp_split_plan.json',dict(bundle_id=bid,source_sha256=code,
        manifest_sha256=sha256(manifest),global_cases=1350,
        workspaces={k:{n:v for n,v in p.items() if n not in ['lanes','cache','secret']} for k,p in plans.items()}))
    print(json.dumps({k:dict(cases=len(p['assigned_case_ids']),workers=p['workers'],budget=p['budget'],categories=p['categories']) for k,p in plans.items()},indent=2))


def build_supervisor(source,part):
    import modal
    runs=modal.Volume.from_name('xverse-runs')
    image=(modal.Image.debian_slim(python_version='3.11').pip_install('Pillow==11.1.0')
           .add_local_dir(source,'/root'))
    app=modal.App(APP+'-supervisor')

    @app.function(name='supervise',serialized=True,image=image,volumes={'/work':runs},cpu=(.25,.25),
        memory=(512,512),timeout=86400,max_containers=1,retries=0,scaledown_window=2)
    def supervise(plan):
        import modal,time,uuid
        from pathlib import Path
        from freecus.run import write_json
        runs.reload()
        path=Path('/work/split_workflows')/(plan['bundle_id']+'-'+plan['workspace']+'.json')
        if path.exists():state=json.loads(path.read_text())
        else:state=dict(status='staging',workspace=plan['workspace'],budget_usd=plan['budget'],
            generation_cap_usd=plan['budget']-1.2,calls=[],invocation=uuid.uuid4().hex,bundle_id=plan['bundle_id'],
            requested_cases=len(plan['assigned_case_ids']),started_at=time.time())
        def save():
            state['updated_at']=time.time();write_json(path,state);runs.commit()
        save()
        if state['status']=='failed' and state.get('error')=='KeyboardInterrupt()':
            state['recovered_supervisor_interruption']=state.pop('error')
            state['status']='generating';save()
        if state['status'] in {'complete','budget_paused','generation_failed','failed'}:return state
        try:
            if state.get('stage_call'):call=modal.FunctionCall.from_id(state['stage_call'])
            else:
                call=modal.Function.from_name(APP,'stage').spawn([plan]);state['stage_call']=call.object_id;save()
            call.get()
            state['status']='generating';save()
            # Use allocated maximum RAM in the runtime limit to preserve budget
            # even if physical RAM exceeds the lower 32 GiB request.
            conservative_rate=.000694+2*.0000131+64*.00000222
            seconds=min(21000,state['generation_cap_usd']/plan['workers']/conservative_rate-60)
            for i,cases in enumerate(plan['lanes']):
                if any(c['lane']==i for c in state['calls']):continue
                call=modal.Function.from_name(APP,'lane').spawn(plan['bundle_id'],i,cases,seconds,state['invocation'])
                state['calls'].append(dict(lane=i,id=call.object_id));save()
            while True:
                for entry in state['calls']:
                    if 'result' in entry:continue
                    try:entry['result']=modal.FunctionCall.from_id(entry['id']).get(timeout=0);save()
                    except TimeoutError:pass
                if all('result' in c for c in state['calls']):break
                time.sleep(15)
            state['estimated_generation_usd']=sum((c['result']['elapsed_seconds']+60)*conservative_rate for c in state['calls'])
            state['generated_cases']=sum(c['result']['generated_total'] for c in state['calls'])
            if any(c['result']['status']!='complete' for c in state['calls']):
                state['status']='budget_paused' if any(c['result']['status']=='budget_paused' for c in state['calls']) else 'generation_failed'
                save();return state
            state['status']='evaluating';save()
            if state.get('evaluation_call'):score=modal.FunctionCall.from_id(state['evaluation_call'])
            else:
                score=modal.Function.from_name(APP,'evaluate').spawn(plan);state['evaluation_call']=score.object_id;save()
            state['evaluation']=score.get();state['status']='complete';save();return state
        except Exception as exc:
            state['status']='failed';state['error']=repr(exc);save();raise
    return app


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','deploy','submit','status'])
    p.add_argument('--workspace',choices=list(SHARES))
    a=p.parse_args()
    if a.command=='prepare':prepare();return
    if not a.workspace:p.error('--workspace required')
    allplans=json.loads((STATE/'plan.json').read_text());part=allplans['plans'][a.workspace]
    from freecus.modal_client import connect
    modal=connect(part['profile'])
    folder=STATE/a.workspace;folder.mkdir(parents=True,exist_ok=True)
    if a.command=='deploy':
        # Create new result volume only; preexisting model cache and HF Secret
        # are reused in the same account without transferring credentials.
        modal.Volume.from_name('xverse-runs',create_if_missing=True,version=2).hydrate()
        from .modal_app import build
        with modal.enable_output():
            build(allplans['source'],'xverse',app_name=APP,cache_name=part['cache'],secret_name=part['secret'],
                  lane_count=part['workers'],host_memory_mib=32768).deploy()
            build_supervisor(allplans['source'],part).deploy()
        write_json(folder/'deployment.json',dict(code=allplans['code_sha256']));return
    runs=modal.Volume.from_name('xverse-runs')
    receipt=folder/'receipt.json'
    if a.command=='submit':
        if receipt.exists():raise ValueError('Already submitted')
        if json.loads((folder/'deployment.json').read_text())['code']!=allplans['code_sha256']:raise ValueError('Stale deployment')
        with runs.batch_upload() as upload:upload.put_file(allplans['archive'],'/uploads/'+part['bundle_id']+'.zip')
        call=modal.Function.from_name(APP+'-supervisor','supervise').spawn(part)
        write_json(receipt,dict(call_id=call.object_id,workspace=a.workspace,bundle_id=part['bundle_id'],
                               budget_usd=part['budget'],cases=len(part['assigned_case_ids'])))
        print(receipt.read_text());return
    record=json.loads(receipt.read_text())
    try:status=modal.FunctionCall.from_id(record['call_id']).get(timeout=0)
    except TimeoutError:
        try:status=json.loads(b''.join(runs.read_file('split_workflows/'+part['bundle_id']+'-'+a.workspace+'.json')))
        except FileNotFoundError:status=dict(status='starting')
    write_json(folder/'status.json',status)
    if status.get('evaluation',{}).get('status')=='complete':
        data=b''.join(runs.read_file(status['evaluation']['scores']))
        dest=PROJECT/'xverse/results/dreambench_plus'/part['bundle_id']/a.workspace/'metrics.json'
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists() and dest.read_bytes()!=data:raise ValueError('Existing score mismatch')
        dest.write_bytes(data)
    print(json.dumps(status,indent=2))


if __name__=='__main__':main()
