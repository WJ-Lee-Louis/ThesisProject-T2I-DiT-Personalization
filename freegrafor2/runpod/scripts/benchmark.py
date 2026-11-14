"""Resumable native FreeGraftor generation and unchanged InstantX metric math.

The upstream subject-detection failures are recorded, never replaced with
text-only images, altered prompts, lower thresholds, or other methods.
"""
from pathlib import Path
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import argparse,csv,fcntl,gc,hashlib,importlib.metadata,json,math,os,shutil,subprocess,sys,time,traceback
import project_metrics as metrics

ROOT=Path('/workspace/freegraftor')
OUTPUT=ROOT/'results/seed42_512'
GEN=Path('/opt/freegraftor-runtime/generation-local/bin/python')
METRIC=Path('/workspace/instantx-0807/venvs/metric/bin/python')
SETTINGS=dict(condition='freegraftor',seed=42,width=512,height=512,num_steps=25,guidance=3.,sim_threshold=.2,cyc_threshold=1.5,inject_match_dropout=.5,start_inject_step=0,end_inject_step=25,inject_block_ids=list(range(57)),alignment='center',requires_offload=False,precision='bfloat16',upstream_commit='b8e0f93e91753265e46e9d0b364daa38e25f1afa')

def read(p):return json.loads(Path(p).read_text())
def write(p,x):metrics.write(p,x)
def sha(p):return metrics.sha(p)
def digest(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
def now():return time.time()

def source_digest():
    paths=sorted((ROOT/'src_snapshot/src').rglob('*.py'))+[Path(__file__),ROOT/'scripts/project_metrics.py']
    return digest({str(p.relative_to(ROOT)):sha(p) for p in paths})

def jobs_for(lane=None,pilot=False):
    rows=read(ROOT/'inputs/manifest.json');refs=read(ROOT/'inputs/references.json')
    assignment=read(ROOT/'inputs/assignment.json')
    selected=set(assignment[lane]) if lane is not None else {r['case_id'] for r in rows}
    code=source_digest();models=sha(ROOT/'cache/flux-native/conversion.json')
    jobs=[]
    for row in rows:
        if row['case_id'] not in selected:continue
        ref=refs[row['reference_id']]
        signature=digest(dict(row=row,settings=SETTINGS,image_sha256=ref['sha256'],implementation=code,models=models))
        jobs.append(dict(row=row,reference=ref['path'],reference_sha256=ref['sha256'],signature=signature))
    if pilot:
        seen=set();chosen=[]
        for j in jobs:
            k=j['row']['benchmark'],metrics.category(j['row'])
            if k not in seen:chosen.append(j);seen.add(k)
        return chosen
    return jobs

def paths(job):
    p=OUTPUT/'images'/job['row']['benchmark']/(job['row']['case_id']+'.png')
    return p,p.with_suffix('.json')

def finished(job):
    png,meta=paths(job)
    if not png.is_file() or not meta.is_file():return False
    r=read(meta)
    if r['job_signature']!=job['signature']:raise ValueError(f'Existing result settings/source changed: {png}')
    if r['row']!=job['row'] or r['settings']!=SETTINGS:raise ValueError('Sidecar mismatch')
    if sha(png)!=r['png_sha256']:raise ValueError('PNG checksum mismatch')
    from PIL import Image
    with Image.open(png) as im:
        assert im.size==(512,512)
        im.verify()
    return True

def failure_path(job):return OUTPUT/'errors'/job['row']['benchmark']/(job['row']['case_id']+'.json')

def evaluation_policy(pilot=False):
    jobs=jobs_for(0,True)+jobs_for(1,True) if pilot else jobs_for()
    style_jobs=[j for j in jobs if metrics.category(j['row'])=='style']
    failures=[read(failure_path(j)) for j in style_jobs if failure_path(j).exists()]
    policy=dict(exclude_dbpp_style=bool(failures),reason='native_style_generation_or_subject_mask_failure' if failures else 'all_attempted_style_cases_generated_without_fallback',style_attempted=len(style_jobs),style_failed=len(failures),fallback_images_used=0,primary_dbpp_scope='non-style' if failures else 'all-classes',updated_at=now())
    write(OUTPUT/('evaluation-policy-pilot.json' if pilot else 'evaluation-policy.json'),policy)
    return policy

def include_metric(job,policy):
    return not (policy['exclude_dbpp_style'] and metrics.category(job['row'])=='style')

def generate(lane,pilot=False,retry=False):
    import torch
    torch.set_num_threads(8)
    if torch.cuda.device_count()!=1 or torch.cuda.get_device_properties(0).total_memory<75*1024**3:raise RuntimeError('Each worker needs one visible A100 80GB')
    versions={p:importlib.metadata.version(p) for p in ['torch','torchvision','transformers','diffusers','numpy','Pillow','modelscope','segment-anything-py','kornia']}
    for p,v in {'torch':'2.6.0','torchvision':'0.21.0','transformers':'4.51.3','diffusers':'0.31.0'}.items():
        assert versions[p].split('+')[0]==v,(p,versions[p])
    jobs=jobs_for(lane,pilot);pending=[j for j in jobs if not finished(j) and (retry or not failure_path(j).exists())]
    if not pending:return
    from pipeline import FreeGraftorPipeline,GenerationConfig,ConceptConfig
    cfg=GenerationConfig(**{k:v for k,v in SETTINGS.items() if k in GenerationConfig.__dataclass_fields__})
    work=ROOT/'cache/work'/f'lane-{lane}'
    work.mkdir(parents=True,exist_ok=True)
    pipe=FreeGraftorPipeline(device='cuda',image_cache_dir=str(work/'images'),image_info_cache_dir=str(work/'info'),requires_offload=False)
    templates={};failures=[];success=0;consecutive_runtime=0
    for j in pending:
        row=j['row'];started=now();print('START',lane,row['benchmark'],row['case_id'],row['identity_term'],flush=True)
        try:
            if sha(j['reference'])!=j['reference_sha256']:raise ValueError('Reference changed')
            # Per-job caches prevent stale inversions; preserve just template RNG state.
            jobwork=work/'current'
            if jobwork.exists():shutil.rmtree(jobwork)
            pipe.image_cache_dir=pipe.collage_creator.image_cache_dir=str(jobwork/'images')
            pipe.image_info_cache_dir=pipe.collage_creator.image_info_cache_dir=str(jobwork/'info')
            key=digest(dict(prompt=row['target_prompt'],seed=42,settings=SETTINGS))
            t0=now()
            if key not in templates:
                torch.manual_seed(42)
                image,_=pipe.generate_template(row['target_prompt'],cfg)
                dest=work/'templates'/(key+'.png');dest.parent.mkdir(parents=True,exist_ok=True);image.save(dest)
                templates[key]=(str(dest),torch.get_rng_state().clone(),[s.clone() for s in torch.cuda.get_rng_state_all()])
            template,cpu,cuda=templates[key]
            template_seconds=now()-t0
            torch.set_rng_state(cpu);torch.cuda.set_rng_state_all(cuda)
            image=pipe(concept_configs=[ConceptConfig(class_name=row['identity_term'],image_path=j['reference'],alignment='center')],prompt=row['target_prompt'],template_prompt=row['target_prompt'],template_path=template,output_dir=str(jobwork/'scratch'),config=cfg,save_output=False)
            assert image.size==(512,512)
            png,meta=paths(j);png.parent.mkdir(parents=True,exist_ok=True)
            temp=png.with_suffix('.png.tmp');image.save(temp,format='PNG');temp.replace(png)
            write(meta,dict(row=row,settings=SETTINGS,job_signature=j['signature'],reference=j['reference'],reference_sha256=j['reference_sha256'],png_sha256=sha(png),elapsed_seconds=now()-started,template_seconds=template_seconds,runtime=dict(gpu=torch.cuda.get_device_name(0),physical_gpu=lane,versions=versions),completed_at=now()))
            failure_path(j).unlink(missing_ok=True);success+=1;consecutive_runtime=0
            print('SAVED',row['case_id'],round(now()-started,2),'s',flush=True)
        except Exception as e:
            traceback.print_exc()
            detection=isinstance(e,RuntimeError) and str(e).startswith('Grounding DINO could not detect')
            record=dict(row=row,job_signature=j['signature'],error_type=type(e).__name__,error=str(e),detection_failure=detection,category=metrics.category(row),elapsed_seconds=now()-started,updated_at=now())
            write(failure_path(j),record);failures.append(record)
            consecutive_runtime=0 if detection else consecutive_runtime+1
            if isinstance(e,torch.cuda.OutOfMemoryError) or consecutive_runtime>=3:raise
        finally:
            gc.collect();torch.cuda.empty_cache()
            write(OUTPUT/'state'/f'generation-lane-{lane}{"-pilot" if pilot else ""}.json',dict(status='running',assigned=len(jobs),new_successes=success,new_failures=len(failures),last_case_id=row['case_id'],updated_at=now()))
    write(OUTPUT/'state'/f'generation-lane-{lane}{"-pilot" if pilot else ""}.json',dict(status='complete_with_failures' if failures else 'complete',assigned=len(jobs),new_successes=success,new_failures=len(failures),updated_at=now()))

def score(lane,pilot=False):
    jobs=jobs_for(lane,pilot);records=[]
    policy=read(OUTPUT/('evaluation-policy-pilot.json' if pilot else 'evaluation-policy.json'))
    for j in jobs:
        if finished(j):
            if not include_metric(j,policy):continue
            records.append(dict(condition='freegraftor',category=metrics.category(j['row']),row=j['row'],reference=j['reference'],reference_sha256=j['reference_sha256'],generated=str(paths(j)[0])))
        elif not failure_path(j).exists():raise ValueError('Generation neither complete nor explicitly failed')
    if not records:raise RuntimeError('No successful images for evaluation')
    plan=dict(protocol=metrics.PROTOCOL,records=records)
    tag=f'lane-{lane}'+('-pilot' if pilot else '')
    write(OUTPUT/'metrics'/f'pairs-{tag}.json',plan)
    by_case={j['row']['case_id']:j for j in jobs}
    def validate(plan,account=None):
        for r in plan['records']:
            j=by_case[r['row']['case_id']]
            if not finished(j) or sha(j['reference'])!=j['reference_sha256']:raise ValueError('Invalid metric inputs')
    metrics.evaluate(plan,OUTPUT/'metrics'/f'scores-{tag}.json','cuda',32,checkpoint=OUTPUT/'metrics'/f'checkpoint-{tag}.json',validator=validate)

def merge(pilot=False):
    jobs=jobs_for();expected={(j['row']['benchmark'],j['row']['case_id']):j for j in jobs};scores=[];seen=set();env=None
    policy=read(OUTPUT/('evaluation-policy-pilot.json' if pilot else 'evaluation-policy.json'))
    for lane in range(2):
        tag=f'lane-{lane}'+('-pilot' if pilot else '')
        p=read(OUTPUT/'metrics'/f'scores-{tag}.json')
        assert p['protocol']==metrics.PROTOCOL
        e={k:p[k] for k in ['device','device_name','versions']}
        if env is not None:assert e==env
        env=e
        for s in p['scores']:
            key=s['benchmark'],s['case_id'];assert key not in seen and key in expected
            j=expected[key];m=read(paths(j)[1])
            assert include_metric(j,policy),'Excluded style case in primary metrics'
            assert s['condition']=='freegraftor' and s['generated_sha256']==m['png_sha256'] and s['reference_sha256']==j['reference_sha256']
            assert s['target_prompt']==j['row']['target_prompt'] and s['reference_id']==j['row']['reference_id']
            assert all(math.isfinite(s[k]) for k in metrics.METRICS)
            seen.add(key);scores.append(s)
    failures=[read(failure_path(j)) for j in jobs if failure_path(j).exists()]
    generated=[j for j in jobs if paths(j)[0].exists() and paths(j)[1].exists()]
    excluded=[j for j in generated if not include_metric(j,policy)]
    if not pilot:
        generated_keys={(j['row']['benchmark'],j['row']['case_id']) for j in generated}
        attempted=generated_keys|{(f['row']['benchmark'],f['row']['case_id']) for f in failures}
        assert attempted==set(expected) and len(generated_keys)+len(failures)==2100
        assert seen=={(j['row']['benchmark'],j['row']['case_id']) for j in generated if include_metric(j,policy)}
    scores.sort(key=lambda s:(s['benchmark'],s['case_id']))
    summary=metrics.summarize(scores)
    suffix='-pilot' if pilot else ''
    write(OUTPUT/'metrics'/f'scores{suffix}.json',dict(protocol=metrics.PROTOCOL,**env,summary=summary,scores=scores))
    write(OUTPUT/'metrics'/f'summary{suffix}.json',summary)
    write(OUTPUT/f'coverage{suffix}.json',dict(expected=2100 if not pilot else len(jobs_for(0,True))+len(jobs_for(1,True)),metric_expected=1920 if policy['exclude_dbpp_style'] and not pilot else 2100 if not pilot else None,generated=len(generated),scored=len(scores),excluded_generated_style=len(excluded),evaluation_policy=policy,failed=len(failures),failures_by_category=dict(Counter(f['category'] for f in failures)),failures=failures))
    with (OUTPUT/'metrics'/f'summary{suffix}.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['benchmark/condition','images',*metrics.METRICS])
        for k,v in summary.items():w.writerow([k,v['cases'],*[v['subject_macro'][m] for m in metrics.METRICS]])
    if not pilot:compare(scores)
    return len(generated),len(scores),len(failures)

def compare(scores):
    sources=[Path('/workspace/instantx-0807/results/full/metrics/scores.json'),Path('/workspace/instantx-0807/results/scale_sweep_20261025/metrics/scores.json')]
    current={(s['benchmark'],s['case_id']):s for s in scores}
    matched=[]
    for p in sources:
        data=read(p);assert data['protocol']==metrics.PROTOCOL
        for s in data['scores']:
            k=s['benchmark'],s['case_id']
            if k in current:
                other=current[k]
                assert all(s[x]==other[x] for x in ['reference_sha256','reference_id','target_prompt'])
                matched.append(s)
    summary=metrics.summarize(scores+matched)
    deltas={}
    for key,v in summary.items():
        b,c=key.split('/')
        if c=='freegraftor':continue
        fg=summary[b+'/freegraftor']
        deltas[key]={m:v['subject_macro'][m]-fg['subject_macro'][m] for m in metrics.METRICS}
    write(OUTPUT/'metrics/comparison.json',dict(note='Matched successful FreeGraftor cases only. Native method resolutions differ: FreeGraftor 512, InstantX 1024. Positive delta means InstantX higher. This is not an equal-resolution causal test.',summary=summary,instantx_minus_freegraftor=deltas))

def launch(lane,pilot=False,stage='generate'):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(lane))
    tag=f'lane-{lane}'+('-pilot' if pilot else '')
    with (OUTPUT/'logs'/f'{tag}.log').open('a') as log:
        for cmd,python in [(stage,GEN if stage=='generate' else METRIC)]:
            write(OUTPUT/'state'/f'controller-{tag}.json',dict(stage=cmd,status='running',updated_at=now()))
            args=[str(python),'-u',__file__,cmd,'--lane',str(lane)]+(['--pilot'] if pilot else [])
            subprocess.run(args,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        write(OUTPUT/'state'/f'controller-{tag}.json',dict(stage=stage,status='complete',updated_at=now()))

def phase(stage,pilot=False):
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(launch,lane,pilot,stage) for lane in range(2)]
        for future in futures:future.result()

def run():
    for name in ['logs','state','metrics','errors']: (OUTPUT/name).mkdir(parents=True,exist_ok=True)
    with (OUTPUT/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        runfile=OUTPUT/'run.json'
        provenance=dict(settings=SETTINGS,implementation=source_digest(),input_manifest_sha256=sha(ROOT/'inputs/manifest.json'),conversion_sha256=sha(ROOT/'cache/flux-native/conversion.json'),expected=2100,gpus=2,per_gpu=1050)
        if runfile.exists():assert read(runfile)['provenance']==provenance,'Existing run source/config changed'
        else:write(runfile,dict(provenance=provenance,started_at=now()))
        try:
            if not (OUTPUT/'pilot-passed.json').exists():
                write(OUTPUT/'status.json',dict(status='running',stage='pilot',updated_at=now()))
                phase('generate',True)
                evaluation_policy(True)
                phase('score',True)
                n,scored,failed=merge(True)
                pilot_scores=read(OUTPUT/'metrics/scores-pilot.json')['scores']
                required={('dreambench','animal'),('dreambench','object'),('dreambench_plus','animal'),('dreambench_plus','human'),('dreambench_plus','object')}
                have={(s['benchmark'],s['category']) for s in pilot_scores}
                assert required<=have,f'Pilot failed required categories: {required-have}'
                write(OUTPUT/'pilot-passed.json',dict(generated=n,scored=scored,failed=failed,categories=sorted(have),updated_at=now()))
            write(OUTPUT/'status.json',dict(status='running',stage='full_generation_and_metrics',updated_at=now()))
            phase('generate')
            policy=evaluation_policy()
            write(OUTPUT/'status.json',dict(status='running',stage='full_metrics',evaluation_policy=policy,updated_at=now()))
            phase('score')
            n,scored,failed=merge()
            write(OUTPUT/'status.json',dict(status='complete_with_failures' if failed else 'complete',stage='all',expected=2100,images=n,scored=scored,failed=failed,evaluation_policy=policy,updated_at=now()))
        except Exception as e:
            write(OUTPUT/'status.json',dict(status='failed',error=str(e),updated_at=now()));raise

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','generate','score','merge','plan']);p.add_argument('--lane',type=int,choices=[0,1]);p.add_argument('--pilot',action='store_true');p.add_argument('--retry-failures',action='store_true');a=p.parse_args()
    if a.command=='generate':generate(a.lane,a.pilot,a.retry_failures)
    elif a.command=='score':score(a.lane,a.pilot)
    elif a.command=='merge':merge(a.pilot)
    elif a.command=='plan':print(json.dumps(dict(settings=SETTINGS,counts=dict(Counter(j['row']['benchmark'] for j in jobs_for(a.lane,a.pilot)))),indent=2))
    else:run()

if __name__=='__main__':main()
