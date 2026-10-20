"""Explicit paid B pilot using the new 180-caption observation run; never generates images."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from orthogonalization_instantx_ip_adapter.contrast_prompt import CAPTION_PROMPT_VERSION, CAPTION_PROMPT_SHA256, CONTRAST_PROMPT_SHA256
from orthogonalization_instantx_ip_adapter.semantic_contrasts import CaptionRecord, ReferenceFingerprint, StageTrace, AttemptRecord, SemanticLLM, prepare_semantic_conditioning
from orthogonalization_instantx_ip_adapter.llm_provider import ResponsesJSONProvider

# Fixed benchmark indices, zero-based. Known development cases and new probes are labelled.
SELECTION = {0:[0,9,21,24],3:[2,14,21,24],10:[10,13,21],30:[6,7],51:[7],75:[6,7,8],89:[7],115:[4,6,7],135:[1,7],161:[0,7,8],162:[1,8],169:[3,5],178:[0,5,7,8],98:[0],131:[0],166:[6],173:[7],176:[7],179:[8]}
KNOWN = {(30,6):'T13',(51,7):'T16',(75,8):'T17',(89,7):'T20',(115,6):'T22',(135,7):'T23',(161,8):'T01',(162,8):'T03',(161,7):'cloud development'}
OUT=ROOT/'outputs/joint_pilot_v2'
CAP=ROOT/'datasets/caption_monitoring/module_a_v2'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def caption_record(case,record):
    if record['status']!='PASS' or record['prompt_sha256']!=CAPTION_PROMPT_SHA256:
        raise ValueError('caption not valid for current A')
    ref=ReferenceFingerprint.from_image((ROOT/case['image_path']).read_bytes())
    if ref.bytes_sha256 != record['image_sha256']: raise ValueError('reference changed')
    seed=CaptionRecord.manual(record['source_caption'],case['identity_term'],ref,preserve=record['preserve'])
    trace=StageTrace('caption',record['requested_model'],seed.input_sha256,CAPTION_PROMPT_VERSION,CAPTION_PROMPT_SHA256,(AttemptRecord(record['raw_response'],None),))
    return CaptionRecord(seed.text,seed.identity_term,ref,'generated_v2',trace,preserve=seed.preserve)
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    refs=read(CAP/'references.json'); records={r['id']:r for r in read(CAP/'captions.json')['records']}
    if len(records)!=180: raise ValueError('Complete A run first')
    selection=[]
    for i,js in SELECTION.items():
        for j in js:
            c=refs[i]; selection.append(dict(case_id=f'{i:03d}-{j:02d}',reference_index=i,target_index=j,identity_term=c['identity_term'],target_prompt=c['target_prompts'][j],reference_id=c['id'],known_case=KNOWN.get((i,j)),selection_role='known development' if (i,j) in KNOWN else 'additional benchmark probe'))
    marker=dict(caption_prompt_sha256=CAPTION_PROMPT_SHA256,contrast_prompt_sha256=CONTRAST_PROMPT_SHA256,selection=selection)
    if (OUT/'run.json').exists() and read(OUT/'run.json')!=marker: raise ValueError('Different run exists')
    save(OUT/'run.json',marker)
    for c in refs:
        r=records[c['id']]
        if r['status']=='PASS': save(CAP/'caption_cache'/(c['id'].replace('/','__')+'.json'),caption_record(c,r).to_dict())
    if not args.run:
        print(f'Prepared {len(selection)} B pairs; A cache traces reconstructed from preserved first API responses, without new A calls.'); return
    def run(row):
        path=OUT/'cases'/(row['case_id']+'.json')
        c=refs[row['reference_index']]
        if path.exists(): return read(path)
        result=dict(row)
        try:
            llm=SemanticLLM(ResponsesJSONProvider.from_env())
            llm.provider.timeout=180
            conditioning=prepare_semantic_conditioning(llm=llm,identity_term=c['identity_term'],target_prompt=row['target_prompt'],reference_image_bytes=(ROOT/c['image_path']).read_bytes(),caption_record=caption_record(c,records[c['id']]))
            result.update(status='PASS',conditioning=conditioning.to_dict())
        except Exception as error:
            result.update(status='ERROR',error=f'{type(error).__name__}: {error}')
            if getattr(error,'trace',None): result['trace']=error.trace.to_dict()
        save(path,result); return result
    with ThreadPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(run,r) for r in selection]):
            r=f.result(); print(r['case_id'],r['status'],flush=True)
    values=[read(OUT/'cases'/(r['case_id']+'.json')) for r in selection]
    save(OUT/'summary.json',dict(total=len(values),passed=sum(v['status']=='PASS' for v in values),errors=[dict(case_id=v['case_id'],error=v['error']) for v in values if v['status']!='PASS']))
if __name__=='__main__': main()
