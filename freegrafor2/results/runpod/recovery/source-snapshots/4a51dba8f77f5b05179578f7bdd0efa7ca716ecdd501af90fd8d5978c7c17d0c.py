"""Audited subject-label retries; never change the diffusion prompt or reference.

Native source files and existing signatures stay frozen. Repaired images carry
an additional signed preprocessing record, preserving the original failed run.
"""
from pathlib import Path
import argparse,gc,json,os,signal,subprocess,sys,time,traceback
import benchmark as base

ROOT=base.ROOT
REC=ROOT/'recovery'
ALIASES=REC/'detector_aliases.json'

def active_failures(lane=None,reference=None):
    out=[]
    for j in base.jobs_for(lane):
        if base.metrics.category(j['row'])=='style':continue
        if reference and j['row']['reference_id']!=reference:continue
        if base.failure_path(j).exists() and not base.paths(j)[0].exists():out.append(j)
    return out

def validate_repair(j):
    if not base.finished(j):return False
    m=base.read(base.paths(j)[1]);audit=m.get('mask_recovery')
    if audit:
        signature=m.get('mask_recovery_signature')
        assert signature==base.digest(audit),'Mask recovery metadata changed'
        assert audit['original_job_signature']==j['signature']
        assert audit['generation_prompt_unchanged'] and audit['reference_bytes_unchanged']
        for item in audit['mask_artifacts']:
            assert base.sha(item['mask_path'])==item['mask_sha256'],'Mask changed'
    return True

def repair(lane,reference=None,shared=False):
    import numpy as np
    import torch
    from PIL import Image
    from pipeline import FreeGraftorPipeline,GenerationConfig,ConceptConfig
    torch.set_num_threads(8)
    if shared:torch.cuda.set_per_process_memory_fraction(.49)
    jobs=active_failures(lane,reference)
    if not jobs:return
    queries=base.read(ALIASES)
    source=base.sha(__file__)
    work=ROOT/'cache/recovery-work'/f'lane-{lane}'
    cfg=GenerationConfig(**{k:v for k,v in base.SETTINGS.items() if k in GenerationConfig.__dataclass_fields__})
    pipe=FreeGraftorPipeline(device='cuda',image_cache_dir=str(work/'images'),image_info_cache_dir=str(work/'info'),requires_offload=False)
    original_grounding=pipe.collage_creator.grounding
    original_segment=pipe.collage_creator.segment
    context={}

    def grounding(image_path=None,pil_image=None,prompt='',box_threshold=.35,text_threshold=.25):
        # Same thresholds as the native run. Only an equivalent subject noun
        # phrase is retried after the original label fails to find any boxes.
        j=context['job'];row=j['row']
        options=[prompt]+queries.get(row['reference_id'],{}).get('queries',[])
        if prompt and not prompt.startswith(('a ','an ')):options.append('a '+prompt)
        options=list(dict.fromkeys(options))
        stage='reference' if image_path else 'template'
        attempts=[]
        for query in options:
            result=original_grounding(image_path=image_path,pil_image=pil_image,prompt=query,box_threshold=box_threshold,text_threshold=text_threshold)
            attempts.append(dict(query=query,box_threshold=box_threshold,text_threshold=text_threshold,scores=result['scores'].detach().float().cpu().tolist(),boxes=result['boxes'].detach().float().cpu().tolist()))
            if len(result['boxes']):
                context['detector_calls'].append(dict(stage=stage,original_query=prompt,selected_query=query,attempts=attempts))
                return result
        context['detector_calls'].append(dict(stage=stage,original_query=prompt,selected_query=None,attempts=attempts))
        return result

    def segment(image_path=None,pil_image=None,prompt=None):
        masks,boxes=original_segment(image_path=image_path,pil_image=pil_image,prompt=prompt)
        if masks:
            mask=masks[0].squeeze().detach().cpu().numpy().astype(bool)
            fraction=float(mask.mean())
            if not mask.any() or fraction>=.98:raise RuntimeError('Recovery SAM mask is empty or almost full-frame; manual subject-mask review required')
            row=context['job']['row'];stage='reference' if image_path else 'template'
            dest=REC/'masks'/row['benchmark']/row['case_id'];dest.mkdir(parents=True,exist_ok=True)
            mask_path=dest/(stage+'-mask.png');Image.fromarray(mask.astype('uint8')*255).save(mask_path)
            image=np.array(Image.open(image_path).convert('RGB') if image_path else pil_image.convert('RGB'))
            image[mask]=(image[mask]*.55+np.array([255,60,0])*.45).astype('uint8')
            Image.fromarray(image).save(dest/(stage+'-overlay.png'))
            context['mask_artifacts'].append(dict(stage=stage,mask_path=str(mask_path),mask_sha256=base.sha(mask_path),foreground_fraction=fraction,box=boxes[0].detach().float().cpu().tolist()))
        return masks,boxes

    pipe.collage_creator.grounding=grounding
    pipe.collage_creator.segment=segment
    versions={p:base.importlib.metadata.version(p) for p in ['torch','torchvision','transformers','diffusers','numpy','Pillow','modelscope','segment-anything-py','kornia']}
    success=0;failures=[]
    for j in jobs:
        row=j['row'];started=time.time();original_error=base.read(base.failure_path(j))
        archive=REC/'original_failures'/row['benchmark']/(row['case_id']+'.json')
        if not archive.exists():base.write(archive,original_error)
        context.clear();context.update(job=j,detector_calls=[],mask_artifacts=[])
        print('REPAIR START',row['benchmark'],row['case_id'],flush=True)
        try:
            assert base.sha(j['reference'])==j['reference_sha256']
            current=work/row['case_id']
            pipe.image_cache_dir=pipe.collage_creator.image_cache_dir=str(current/'images')
            pipe.image_info_cache_dir=pipe.collage_creator.image_info_cache_dir=str(current/'info')
            torch.manual_seed(42)
            template,_=pipe.generate_template(row['target_prompt'],cfg)
            template_path=current/'template.png';template_path.parent.mkdir(parents=True,exist_ok=True);template.save(template_path)
            cpu=torch.get_rng_state().clone();cuda=[s.clone() for s in torch.cuda.get_rng_state_all()]
            torch.set_rng_state(cpu);torch.cuda.set_rng_state_all(cuda)
            image=pipe(concept_configs=[ConceptConfig(class_name=row['identity_term'],image_path=j['reference'],alignment='center')],prompt=row['target_prompt'],template_prompt=row['target_prompt'],template_path=str(template_path),output_dir=str(current/'scratch'),config=cfg,save_output=False)
            assert image.size==(512,512)
            assert {a['stage'] for a in context['mask_artifacts']}=={'reference','template'}
            audit=dict(original_job_signature=j['signature'],recovery_source_sha256=source,alias_evidence=queries.get(row['reference_id']),original_error_record=str(archive),generation_prompt_unchanged=True,reference_bytes_unchanged=True,detector_thresholds_unchanged=True,detector_calls=context['detector_calls'],mask_artifacts=context['mask_artifacts'],completed_at=time.time())
            png,meta=base.paths(j);png.parent.mkdir(parents=True,exist_ok=True)
            temp=png.with_suffix('.png.tmp');image.save(temp,format='PNG');temp.replace(png)
            base.write(meta,dict(row=row,settings=base.SETTINGS,job_signature=j['signature'],reference=j['reference'],reference_sha256=j['reference_sha256'],png_sha256=base.sha(png),elapsed_seconds=time.time()-started,template_seconds=None,mask_recovery=audit,mask_recovery_signature=base.digest(audit),runtime=dict(gpu=torch.cuda.get_device_name(0),physical_gpu=lane,versions=versions,recovery=True,shared_memory_budget=shared),completed_at=time.time()))
            assert validate_repair(j)
            base.failure_path(j).unlink();success+=1
            print('REPAIR SAVED',row['case_id'],round(time.time()-started,2),flush=True)
        except Exception as e:
            traceback.print_exc();failures.append(dict(case_id=row['case_id'],error=str(e)))
            base.write(REC/'unresolved'/row['benchmark']/(row['case_id']+'.json'),dict(row=row,error=str(e),detector_calls=context['detector_calls'],updated_at=time.time()))
            if isinstance(e,torch.cuda.OutOfMemoryError):raise
        finally:
            gc.collect();torch.cuda.empty_cache()
            base.write(REC/f'repair-lane-{lane}.json',dict(status='running',assigned=len(jobs),success=success,failures=failures,updated_at=time.time()))
    base.write(REC/f'repair-lane-{lane}.json',dict(status='complete' if not failures else 'needs_mask_review',assigned=len(jobs),success=success,failures=failures,updated_at=time.time()))

def immediate(worker_pid,lane,reference):
    cmd=Path(f'/proc/{worker_pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert 'benchmark.py generate' in cmd and f'--lane {lane}' in cmd,'Unexpected worker'
    assert not (REC/'immediate.lock').exists(),'Immediate recovery already started; inspect lock/status before another run'
    (REC/'immediate.lock').write_text(str(os.getpid()))
    stopped=False
    try:
        os.kill(worker_pid,signal.SIGSTOP);stopped=True
        base.write(REC/'immediate-status.json',dict(status='running',paused_worker=worker_pid,lane=lane,reference=reference,started_at=time.time()))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(lane))
        with (REC/'immediate-worker.log').open('a') as log:
            subprocess.run([str(base.GEN),'-u',__file__,'repair','--lane',str(lane),'--reference',reference,'--shared-gpu'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
        base.write(REC/'immediate-status.json',dict(status='complete',paused_worker=worker_pid,lane=lane,reference=reference,remaining_non_style_failures=len(active_failures()),updated_at=time.time()))
    except Exception as e:
        base.write(REC/'immediate-status.json',dict(status='failed',error=str(e),updated_at=time.time()));raise
    finally:
        if stopped:
            try:os.kill(worker_pid,signal.SIGCONT)
            except ProcessLookupError:pass
        (REC/'immediate.lock').unlink(missing_ok=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['repair','immediate']);p.add_argument('--lane',type=int,choices=[0,1],required=True);p.add_argument('--reference');p.add_argument('--worker-pid',type=int);p.add_argument('--shared-gpu',action='store_true');a=p.parse_args()
    if a.command=='repair':repair(a.lane,a.reference,a.shared_gpu)
    else:immediate(a.worker_pid,a.lane,a.reference)

if __name__=='__main__':main()
