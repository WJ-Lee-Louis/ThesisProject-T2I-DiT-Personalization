"""Correct stage labels from the first recovery worker, preserving its audit.

Native segment() forwards a PIL image to grounding() even for reference files.
The actual ordered mask artifacts retain the correct stages and SHA256 values.
This changes logging only; generation images and native job inputs are untouched.
"""
from pathlib import Path
import copy,json
import benchmark as base

ROOT=base.ROOT
def normalize():
    corrected=0
    for p in (base.OUTPUT/'images').glob('*/*.json'):
        record=base.read(p);audit=record.get('mask_recovery')
        if not audit:continue
        calls=audit['detector_calls'];masks=audit['mask_artifacts']
        if len(calls)!=len(masks):raise ValueError('Detector call and mask provenance need review')
        if all(c['stage']==m['stage'] for c,m in zip(calls,masks)):continue
        assert record['mask_recovery_signature']==base.digest(audit)
        archive=ROOT/'recovery/original_audits'/record['row']['benchmark']/(record['row']['case_id']+'.json')
        if not archive.exists():base.write(archive,copy.deepcopy(audit))
        for c,m in zip(calls,masks):c['stage']=m['stage']
        audit['logging_correction']=dict(reason='Native segment passes PIL images for both reference and template; use ordered signed mask artifacts for their stage labels',original_audit_path=str(archive),original_audit_sha256=base.sha(archive),correction_source_sha256=base.sha(__file__))
        record['mask_recovery_signature']=base.digest(audit);base.write(p,record);corrected+=1
    print('Recovery stage logging corrected:',corrected,'PNG bytes unchanged',flush=True)
if __name__=='__main__':normalize()
