"""Independent baseline/source/target ablation, generation followed by pinned metrics."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from dataclasses import replace
import argparse,csv,hashlib,json,math,os,subprocess,sys,time,traceback
ROOT=Path('/workspace/Ours/ThesisProject-T2I-DiT-Personalization/Orthogonalization-InstantX-IP-Adapter')
sys.path.insert(0,str(ROOT/'scripts'))
import benchmark_runpod as base
import benchmark_metrics as metrics
METHOD=os.environ['EXPERIMENT_METHOD']
CONFIG={'baseline_0.8':(.8,None),'baseline_0.9':(.9,None),'source_only':(.7,'source'),'target_only':(.7,'target')}
SCALE,ORIGIN=CONFIG[METHOD]
OUTPUT=Path('/workspace/instantx-0807/results/ablation_20261026')/METHOD
GEN='/workspace/instantx-0807/venvs/generation/bin/python'
METRIC='/workspace/instantx-0807/venvs/metric/bin/python'

def jobs_for(limit=0):
    code=base.common.digest({'base':base.revision(),'ablation':base.source_sha(__file__)})
    hashes={};jobs=[]
    for row in base.rows_for(limit):
        path=ROOT/row['image_path']
        if str(path) not in hashes:hashes[str(path)]=metrics.sha(path)
        cfg=dict(condition=METHOD,ip_adapter_scale=SCALE,seed=42,steps=24,width=1024,height=1024,
                 guidance_scale=3.5,orthogonalization_mode='semantic_local' if ORIGIN else 'off',mask_single_text_rows=False,origin=ORIGIN,
                 implementation_sha256=code)
        data=dict(row=row,settings=cfg,image_sha256=hashes[str(path)])
        if ORIGIN:data['conditioning']=base.common.read(ROOT/row['conditioning_path'])
        jobs.append(dict(**data,signature=base.common.digest(data)))
    return jobs

def paths(job):return base.common.paths_for(OUTPUT,METHOD,job['row'])

def generation_kwargs(job,data):
    cfg=job['settings']
    kwargs=dict(prompt=job['row']['target_prompt'],reference_image_url='',reference_image_bytes=data,
        ip_adapter_scale=cfg['ip_adapter_scale'],guidance_scale=cfg['guidance_scale'],num_inference_steps=cfg['steps'],
        seed=cfg['seed'],width=cfg['width'],height=cfg['height'],orthogonalization_mode='off',
        orthogonalization_strength=0.,object_prompt='',object_behavior_prompt='',object_background_prompt='',
        object_behavior_background_prompt='',remove_behavior=False,remove_background=False,include_interaction=False,
        restore_visual_norm=bool(ORIGIN),mask_single_text_rows=False)
    if ORIGIN:
        from orthogonalization_instantx_ip_adapter.semantic_contrasts import SemanticContrastConditioning
        _,payload=base.validate_cache(job)
        conditioning=replace(SemanticContrastConditioning.from_dict(payload),origin=ORIGIN)
        kwargs.update(orthogonalization_mode='semantic_local',orthogonalization_strength=1.,remove_behavior=True,remove_background=True,contrast_analysis_payload=conditioning.to_dict())
    return kwargs

def validate_diagnostics(diag,cfg):
    if diag['mask_single_text_rows'] is not False:raise ValueError('SSG must be OFF')
    if diag['orthogonalization']['enabled'] != bool(ORIGIN):raise ValueError('Projection mismatch')
    if ORIGIN:
        if diag['semantic_conditioning']['selection']['origin']!=ORIGIN:raise ValueError('Origin mismatch')
    elif diag['semantic_conditioning'] is not None:raise ValueError('Baseline conditioning must be absent')
    wanted=dict(prompt=None,ip_adapter_scale=cfg['ip_adapter_scale'],width=cfg['width'],height=cfg['height'],
                seed=cfg['seed'],num_inference_steps=cfg['steps'],guidance_scale=cfg['guidance_scale'])
    for key,value in wanted.items():
        if key!='prompt' and diag['generation'][key]!=value:raise ValueError('Generation setting mismatch: '+key)

def finished(job):
    png,meta=paths(job)
    if not base.common.complete(png,meta,job['signature']):return False
    record=base.common.read(meta)
    if record['row']!=job['row'] or record['settings']!=job['settings']:raise ValueError('Provenance mismatch')
    validate_diagnostics(record['diagnostics'],job['settings'])
    from PIL import Image
    with Image.open(png) as image:
        if image.size!=(1024,1024):raise ValueError('Image dimensions mismatch')
    return True

def worker(lane):
    jobs=jobs_for()[lane::1];pending=[j for j in jobs if not finished(j)]
    if not pending:return
    for package,expected in {'torch':'2.7.1','diffusers':'0.35.2','transformers':'4.54.0','accelerate':'1.10.0','huggingface-hub':'0.34.3'}.items():
        if version(package).split('+')[0]!=expected:raise ValueError('Package version mismatch: '+package)
    import torch
    if torch.cuda.device_count()!=1 or torch.cuda.get_device_properties(0).total_memory<75*1024**3:
        raise ValueError('Each worker requires one visible >=75 GiB GPU')
    from orthogonalization_instantx_ip_adapter import model_loader,inference
    from orthogonalization_instantx_ip_adapter.config import BASE_MODEL_ID,BASE_MODEL_REVISION
    snapshot=Path(os.environ['HF_HOME'])/'hub'/('models--'+BASE_MODEL_ID.replace('/','--'))/'snapshots'/BASE_MODEL_REVISION
    if not snapshot.is_dir():raise FileNotFoundError(snapshot)
    model_loader.BASE_MODEL_ID=str(snapshot)
    model=inference.load_ip_adapter_model();inference.load_ip_adapter_model=lambda:model
    failures=[];consecutive=0;completed=len(jobs)-len(pending)
    for number,job in enumerate(pending,1):
        try:
            start=time.monotonic()
            data=(ROOT/job['row']['image_path']).read_bytes()
            if hashlib.sha256(data).hexdigest()!=job['image_sha256']:raise ValueError('Reference changed')
            png,diag=inference.generate_png_bytes(**generation_kwargs(job,data))
            validate_diagnostics(diag,job['settings'])
            target,meta=paths(job);target.parent.mkdir(parents=True,exist_ok=True)
            temp=target.with_suffix('.png.tmp');temp.write_bytes(png);temp.replace(target)
            metrics.write(meta,dict(row=job['row'],settings=job['settings'],job_signature=job['signature'],
                png_sha256=hashlib.sha256(png).hexdigest(),diagnostics=diag,elapsed_seconds=time.monotonic()-start,
                runtime=dict(provider='runpod',gpu=torch.cuda.get_device_name(0),pod_id=os.environ.get('RUNPOD_POD_ID'))))
            completed+=1;consecutive=0
            print(f"SAVED {completed}/{len(jobs)} {METHOD} {job['row']['case_id']}",flush=True)
        except Exception as exc:
            traceback.print_exc();failures.append(dict(case_id=job['row']['case_id'],error=str(exc)));consecutive+=1
        metrics.write(OUTPUT/f'state/lane-{lane}.json',dict(processed=number,assigned=len(jobs),complete=completed,
            failures=failures,updated_at=time.time()))
        if consecutive>=3:raise RuntimeError('Stopping after 3 consecutive failures')
    if failures:raise RuntimeError(f'{len(failures)} failed jobs; inspect before resuming')

def score(lane):
    jobs=jobs_for()[lane::1];records=[]
    for job in jobs:
        if not finished(job):raise ValueError('Generation incomplete')
        row=job['row']
        records.append(dict(condition=METHOD,category=metrics.category(row),row=row,
            reference=str((ROOT/row['image_path']).resolve()),reference_sha256=job['image_sha256'],generated=str(paths(job)[0])))
    plan=dict(protocol=metrics.PROTOCOL,records=records)
    dest=OUTPUT/'metrics';dest.mkdir(exist_ok=True)
    metrics.write(dest/f'pairs_lane_{lane}.json',plan)
    def validate(plan,account=None):
        if len(plan['records'])!=2100 or not all(finished(j) for j in jobs):raise ValueError('Invalid lane inputs')
    metrics.evaluate(plan,dest/f'scores_lane_{lane}.json','cuda',32,
        checkpoint=dest/f'checkpoint_lane_{lane}.json',validator=validate)

def merge():
    parts=[base.common.read(OUTPUT/f'metrics/scores_lane_{i}.json') for i in (0,)]
    env=lambda p:{k:p[k] for k in ('device','device_name','versions')}
    if any(p['protocol']!=metrics.PROTOCOL for p in parts) or env(parts[0])!=env(parts[1]):raise ValueError('Mixed metric environments')
    jobs=jobs_for();expected={(j['row']['benchmark'],j['row']['case_id']):j for j in jobs};seen=set();scores=[]
    for part in parts:
        for s in part['scores']:
            key=(s['benchmark'],s['case_id'])
            if key not in expected or key in seen or s['condition']!=METHOD:raise ValueError('Missing/duplicate/unexpected score')
            job=expected[key];meta=base.common.read(paths(job)[1])
            if s['generated_sha256']!=meta['png_sha256'] or s['reference_sha256']!=job['image_sha256']:
                raise ValueError('Score image hash mismatch')
            if s['target_prompt']!=job['row']['target_prompt'] or s['reference_id']!=job['row']['reference_id']:
                raise ValueError('Score input mismatch')
            if not all(math.isfinite(s[m]) for m in metrics.METRICS):raise ValueError('Nonfinite score')
            seen.add(key);scores.append(s)
    if seen!=set(expected):raise ValueError('Incomplete score coverage')
    result=dict(protocol=metrics.PROTOCOL,**env(parts[0]),summary=metrics.summarize(scores),scores=scores)
    metrics.write(OUTPUT/'metrics/scores.json',result);metrics.write(OUTPUT/'metrics/summary.json',result['summary'])
    with (OUTPUT/'metrics/summary.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['benchmark/condition','images',*metrics.METRICS])
        for k,v in result['summary'].items():w.writerow([k,v['cases'],*[v['subject_macro'][m] for m in metrics.METRICS]])

def launch(lane):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=os.environ['EXPERIMENT_GPU'])
    with (OUTPUT/f'lane-{lane}.log').open('a') as log:
        for command,python in (('worker',GEN),('score',METRIC)):
            metrics.write(OUTPUT/f'state/controller_lane_{lane}.json',dict(stage=command,status='running',updated_at=time.time()))
            subprocess.run([python,__file__,command,'--lane',str(lane)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    metrics.write(OUTPUT/f'state/controller_lane_{lane}.json',dict(stage='score',status='complete',updated_at=time.time()))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['run','worker','score','merge']);parser.add_argument('--lane',type=int,choices=(0,),default=0)
    args=parser.parse_args()
    if args.command=='worker':worker(args.lane);return
    if args.command=='score':score(args.lane);return
    if args.command=='merge':merge();return
    OUTPUT.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (OUTPUT/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        metrics.write(OUTPUT/'run.json',dict(status='running',condition=METHOD,scale=SCALE,projection=bool(ORIGIN),origin=ORIGIN,gate=False,
            images=2100,gpus=1,per_gpu=2100,started_at=time.time(),script_sha256=base.source_sha(__file__)))
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                futures=[pool.submit(launch,i) for i in (0,)]
                for f in futures:f.result()
            merge()
            metrics.write(OUTPUT/'status.json',dict(status='complete',images=2100,scored=2100,updated_at=time.time()))
        except Exception as exc:
            metrics.write(OUTPUT/'status.json',dict(status='failed',error=str(exc),updated_at=time.time()));raise
if __name__=='__main__':main()
