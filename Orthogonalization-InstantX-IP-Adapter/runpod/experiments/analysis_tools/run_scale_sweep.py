"""Two GPU scale sweep. Reuses pinned generation/evaluation without editing historical code."""
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import argparse, csv, hashlib, json, os, subprocess, sys, time
ROOT=Path('/workspace/Ours/ThesisProject-T2I-DiT-Personalization/Orthogonalization-InstantX-IP-Adapter')
sys.path.insert(0,str(ROOT/'scripts'))
import benchmark_runpod as base
import benchmark_metrics as metrics
SCALES=(0.6,0.9,1.0)
OUTPUT=Path('/workspace/instantx-0807/results/scale_sweep_20261025')
GEN='/workspace/instantx-0807/venvs/generation/bin/python'
METRIC='/workspace/instantx-0807/venvs/metric/bin/python'
original_paths=base.paths

def jobs_for(limit=0):
    rows=base.rows_for(limit)
    code=base.common.digest({'base':base.revision(),'sweep':base.source_sha(__file__)})
    jobs=[]
    references={}
    caches={}
    for scale in SCALES:
        for row in rows:
            ref=ROOT/row['image_path']
            if str(ref) not in references: references[str(ref)]=metrics.sha(ref)
            image_hash=references[str(ref)]
            if row['case_id'] not in caches: caches[row['case_id']]=base.common.read(ROOT/row['conditioning_path'])
            cache=caches[row['case_id']]
            cfg=dict(condition='source_target_ssg',ip_adapter_scale=scale,strength=1.,seed=42,
                     steps=24,width=1024,height=1024,guidance_scale=3.5,implementation_sha256=code)
            signature=base.common.digest(dict(row=row,settings=cfg,cache=cache,image_sha256=image_hash))
            jobs.append(dict(row=row,settings=cfg,signature=signature,image_sha256=image_hash))
    return jobs

def paths(output,job):
    return original_paths(output/f"scale_{job['settings']['ip_adapter_scale']:.1f}",job)
base.jobs_for=jobs_for
base.paths=paths

def lane_jobs(lane): return jobs_for()[lane::2]

def score(lane):
    jobs=lane_jobs(lane)
    records=[]
    for job in jobs:
        if not base.finished(OUTPUT,job): raise ValueError('Generation incomplete')
        row=job['row']
        records.append(dict(condition=f"method_scale_{job['settings']['ip_adapter_scale']:.1f}",category=metrics.category(row),
            row=row,reference=str((ROOT/row['image_path']).resolve()),reference_sha256=job['image_sha256'],
            generated=str(paths(OUTPUT,job)[0])))
    plan=dict(protocol=metrics.PROTOCOL,records=records)
    dest=OUTPUT/'metrics'; dest.mkdir(exist_ok=True)
    metrics.write(dest/f'pairs_lane_{lane}.json',plan)
    def validate(plan,account=None):
        if len(plan['records'])!=3150 or not all(base.finished(OUTPUT,j) for j in jobs):
            raise ValueError('Invalid lane images')
    metrics.evaluate(plan,dest/f'scores_lane_{lane}.json','cuda',32,
                     checkpoint=dest/f'checkpoint_lane_{lane}.json',validator=validate)

def merge():
    parts=[base.common.read(OUTPUT/f'metrics/scores_lane_{lane}.json') for lane in range(2)]
    if any(p['protocol']!=metrics.PROTOCOL for p in parts): raise ValueError('Metric protocol mismatch')
    env=lambda p:{k:p[k] for k in ('device','device_name','versions')}
    if env(parts[0])!=env(parts[1]): raise ValueError('Metric environment mismatch')
    scores=[s for p in parts for s in p['scores']]
    expected={(f'method_scale_{scale:.1f}',r['benchmark'],r['case_id']) for scale in SCALES for r in base.rows_for()}
    keys=[(s['condition'],s['benchmark'],s['case_id']) for s in scores]
    if len(keys)!=len(set(keys)) or set(keys)!=expected: raise ValueError('Missing/duplicate scores')
    result=dict(protocol=metrics.PROTOCOL,**env(parts[0]),summary=metrics.summarize(scores),scores=scores)
    metrics.write(OUTPUT/'metrics/scores.json',result)
    metrics.write(OUTPUT/'metrics/summary.json',result['summary'])
    with (OUTPUT/'metrics/summary.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['benchmark/condition','images',*metrics.METRICS])
        for k,v in result['summary'].items():w.writerow([k,v['cases'],*[v['subject_macro'][m] for m in metrics.METRICS]])

def launch(lane):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(lane))
    with (OUTPUT/f'lane-{lane}.log').open('a') as log:
        for command,python in (('worker',GEN),('score',METRIC)):
            metrics.write(OUTPUT/f'state/controller_lane_{lane}.json',dict(stage=command,status='running',updated_at=time.time()))
            subprocess.run([python,__file__,command,'--lane',str(lane)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    metrics.write(OUTPUT/f'state/controller_lane_{lane}.json',dict(stage='score',status='complete',updated_at=time.time()))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['run','worker','score','merge']);parser.add_argument('--lane',type=int,default=0)
    args=parser.parse_args()
    if args.command=='worker':
        # Resolve the exact cached revision locally; diffusers otherwise calls model_info even offline.
        from orthogonalization_instantx_ip_adapter import model_loader
        from orthogonalization_instantx_ip_adapter.config import BASE_MODEL_ID, BASE_MODEL_REVISION
        model_loader.BASE_MODEL_ID=str(Path(os.environ['HF_HOME'])/'hub'/('models--'+BASE_MODEL_ID.replace('/','--'))/'snapshots'/BASE_MODEL_REVISION)
        if not Path(model_loader.BASE_MODEL_ID).is_dir(): raise FileNotFoundError('Pinned FLUX snapshot missing')
        base.worker(SimpleNamespace(limit=0,lane=args.lane,total_workers=2,output=OUTPUT));return
    if args.command=='score':score(args.lane);return
    if args.command=='merge':merge();return
    OUTPUT.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (OUTPUT/'.sweep.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        metrics.write(OUTPUT/'run.json',dict(status='running',scales=SCALES,gate=True,projection=True,
            images=6300,gpus=2,per_gpu=3150,started_at=time.time(),script_sha256=base.source_sha(__file__)))
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(launch,i) for i in range(2)]
                for future in futures:future.result()
            merge()
            metrics.write(OUTPUT/'status.json',dict(status='complete',images=6300,scored=6300,updated_at=time.time()))
        except Exception as exc:
            metrics.write(OUTPUT/'status.json',dict(status='failed',error=str(exc),updated_at=time.time()))
            raise
if __name__=='__main__':main()
