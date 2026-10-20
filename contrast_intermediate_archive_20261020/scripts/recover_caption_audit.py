"""Explicit one-retry recovery retaining the original first-response audit."""
import json
from dataclasses import replace
from run_joint_pilot import ROOT,CAP,read,save,caption_record
from orthogonalization_instantx_ip_adapter.llm_provider import ResponsesJSONProvider
from orthogonalization_instantx_ip_adapter.contrast_prompt import CAPTION_SYSTEM_PROMPT,CAPTION_SCHEMA,CAPTION_PROMPT_SHA256
from orthogonalization_instantx_ip_adapter.semantic_contrasts import parse_json_object,identity_occurrences,AttemptRecord
refs={c['id']:c for c in read(CAP/'references.json')}
for old in read(CAP/'captions.json')['records']:
    if old['status']=='PASS': continue
    if old['prompt_sha256'] != CAPTION_PROMPT_SHA256: raise ValueError('Historical recovery requires its original prompt; use a fresh run')
    dest=CAP/'recovery'/(old['id'].replace('/','__')+'.json')
    if dest.exists(): raise ValueError('Recovery already exists; inspect instead of overwriting')
    c=refs[old['id']]; provider=ResponsesJSONProvider.from_env(old['requested_model'])
    raw=provider.complete(system_prompt=CAPTION_SYSTEM_PROMPT,payload={'identity_term':c['identity_term']},schema=CAPTION_SCHEMA,image_bytes=(ROOT/c['image_path']).read_bytes(),feedback=old['error'])
    recovered=dict(old,raw_response=raw,status='INVALID',attempt=2)
    try:
        value=parse_json_object(raw)
        if set(value)!={'source_caption','preserve'}: raise ValueError('unexpected response keys')
        identity_occurrences(value['source_caption'],c['identity_term'])
        recovered.update(status='PASS',source_caption=value['source_caption'],preserve=value['preserve'])
        recovered.pop('error',None)
        record=caption_record(c,recovered)
        trace=replace(record.trace,attempts=(AttemptRecord(old['raw_response'],old['error']),AttemptRecord(raw,None)))
        record=replace(record,trace=trace)
        save(CAP/'caption_cache'/(old['id'].replace('/','__')+'.json'),record.to_dict())
    except ValueError as e: recovered['error']=str(e)
    save(dest,dict(first_record=old,retry_record=recovered))
    print(c['id'],recovered['status'],recovered.get('source_caption'),flush=True)
