"""Explicit GPT-5.6 Sol paired A/B pilot. No image generation or original-file edits."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import json,sys,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from orthogonalization_instantx_ip_adapter.llm_provider import ResponsesJSONProvider
from orthogonalization_instantx_ip_adapter.semantic_contrasts import SemanticLLM,CaptionRecord,ReferenceFingerprint,SemanticContrastConditioning,prepare_semantic_conditioning
from orthogonalization_instantx_ip_adapter.contrast_prompt import CAPTION_PROMPT_SHA256,CONTRAST_PROMPT_SHA256
from run_joint_pilot import read,save
OUT=ROOT/'outputs/joint_pilot_gpt56_sol_concept_v6'
CAPTION_CACHE=OUT/'captions'
MODEL='gpt-5.6-sol'
def llm():
    p=ResponsesJSONProvider.from_env(MODEL);p.timeout=240
    return SemanticLLM(p)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');args=parser.parse_args()
    baseline=read(ROOT/'outputs/joint_pilot_v2/run.json')
    refs=read(ROOT/'datasets/caption_monitoring/module_a_v2/references.json')
    marker=dict(baseline,caption_prompt_sha256=CAPTION_PROMPT_SHA256,contrast_prompt_sha256=CONTRAST_PROMPT_SHA256,model=MODEL,reasoning_effort='provider default (medium)',caption_source='fresh compact Sol A with preserve list')
    if (OUT/'run.json').exists() and read(OUT/'run.json')!=marker: raise ValueError('Run metadata changed')
    save(OUT/'run.json',marker)
    from orthogonalization_instantx_ip_adapter.contrast_prompt import CAPTION_SYSTEM_PROMPT,CONTRAST_SYSTEM_PROMPT
    save(OUT/'prompts.json',dict(caption=CAPTION_SYSTEM_PROMPT,contrast=CONTRAST_SYSTEM_PROMPT))
    if not args.run: print('Prepared up to 19 A (compatible captions reused) + 40 joint B calls (excluding validation retries)');return
    new_caption_attempts=[]
    new_contrast_attempts=[]
    def caption(i):
        path=OUT/'captions'/f'{i:03d}.json';c=refs[i];data=(ROOT/c['image_path']).read_bytes()
        if path.exists():
            r=CaptionRecord.from_dict(read(path));r.validate_reuse(identity_term=c['identity_term'],reference=ReferenceFingerprint.from_image(data),model=MODEL);return i,r
        cached=CAPTION_CACHE/f'{i:03d}.json'
        if cached.exists():
            r=CaptionRecord.from_dict(read(cached));r.validate_reuse(identity_term=c['identity_term'],reference=ReferenceFingerprint.from_image(data),model=MODEL);save(path,r.to_dict());print('A',i,'REUSED',flush=True);return i,r
        r=llm().caption(data,c['identity_term']);new_caption_attempts.append(len(r.trace.attempts));save(path,r.to_dict());print('A',i,'PASS',flush=True);return i,r
    # First request verifies model/vision/schema availability before fanout.
    indices=sorted({r['reference_index'] for r in baseline['selection']})
    records=dict([caption(indices[0])])
    with ThreadPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(caption,i) for i in indices[1:]]):
            i,r=f.result();records[i]=r
    def contrast(row):
        path=OUT/'cases'/(row['case_id']+'.json');i=row['reference_index'];c=refs[i]
        data=(ROOT/c['image_path']).read_bytes()
        if path.exists():
            v=read(path)
            if v['status']=='PASS':
                SemanticContrastConditioning.from_dict(v['conditioning']).validate_reuse(identity_term=c['identity_term'],target_prompt=row['target_prompt'],reference=ReferenceFingerprint.from_image(data),source_caption=records[i].text,model=MODEL,preserve=records[i].preserve)
            return v
        v=dict(row,source_caption=records[i].text)
        try:
            r=prepare_semantic_conditioning(llm=llm(),identity_term=c['identity_term'],target_prompt=row['target_prompt'],reference_image_bytes=data,caption_record=records[i])
            v.update(status='PASS',conditioning=r.to_dict())
        except Exception as e:
            v.update(status='ERROR',error=f'{type(e).__name__}: {e}')
            if getattr(e,'trace',None):v['trace']=e.trace.to_dict()
        trace=v['conditioning']['llm']['contrast'] if v['status']=='PASS' else v.get('trace',{})
        new_contrast_attempts.append(len(trace.get('attempts',[])))
        save(path,v);print('B',row['case_id'],v['status'],flush=True);return v
    results=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(contrast,r) for r in baseline['selection']]):results.append(f.result())
    save(OUT/'summary.json',dict(model=MODEL,captions=len(records),total=len(results),passed=sum(r['status']=='PASS' for r in results),errors=[dict(case_id=r['case_id'],error=r['error']) for r in results if r['status']!='PASS'],caption_trace_attempts=sum(len(r.trace.attempts) for r in records.values()),new_caption_attempts=sum(new_caption_attempts),new_contrast_attempts=sum(new_contrast_attempts),contrast_attempts=sum(len((r['conditioning']['llm']['contrast'] if r['status']=='PASS' else r.get('trace',{})).get('attempts',[])) for r in results)))
if __name__=='__main__': main()
