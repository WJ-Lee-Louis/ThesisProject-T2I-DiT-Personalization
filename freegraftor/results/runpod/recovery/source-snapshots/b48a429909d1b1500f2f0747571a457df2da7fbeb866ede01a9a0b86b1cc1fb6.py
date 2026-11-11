"""Check every non-style selected reference before late benchmark failures."""
from pathlib import Path
import json,os,time
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
from segment_anything import sam_model_registry,SamPredictor
from core.collage_creation import CollageCreator
import benchmark as base

torch.set_num_threads(8)
root=base.ROOT;dest=root/'recovery/reference-preflight';dest.mkdir(parents=True,exist_ok=True)
obj=CollageCreator.__new__(CollageCreator);obj.device='cuda'
obj.grounding_dino_processor=AutoProcessor.from_pretrained(os.environ['GROUNDING_DINO'])
obj.grounding_dino=AutoModelForZeroShotObjectDetection.from_pretrained(os.environ['GROUNDING_DINO']).to('cuda',dtype=torch.bfloat16).eval()
obj.sam=SamPredictor(sam_model_registry['vit_h'](checkpoint=os.environ['SAM']))
obj.sam.model=obj.sam.model.to('cuda',dtype=torch.bfloat16).eval()
refs={j['row']['reference_id']:j for j in base.jobs_for() if base.metrics.category(j['row'])!='style'}
aliases=base.read(root/'recovery/detector_aliases.json');results=[]
for ref,j in sorted(refs.items()):
    row=j['row'];original=row['identity_term'];attempts=[];success=None
    queries=list(dict.fromkeys([original]+aliases.get(ref,{}).get('queries',[])+['a '+original]))
    for q in queries:
        started=time.time();d=obj.grounding(image_path=j['reference'],prompt=q)
        attempt=dict(query=q,boxes=d['boxes'].detach().float().cpu().tolist(),scores=d['scores'].detach().float().cpu().tolist(),box_threshold=.35,text_threshold=.25)
        if len(d['boxes']):
            masks,boxes=obj.segment(image_path=j['reference'],prompt=q)
            mask=masks[0].squeeze().detach().cpu().numpy().astype(bool)
            attempt['mask_fraction']=float(mask.mean());attempt['nonempty']=bool(mask.any())
            if mask.any() and mask.mean()<.98:
                success=q
                if q!=original:
                    folder=dest/ref;folder.mkdir(parents=True,exist_ok=True)
                    Image.fromarray(mask.astype('uint8')*255).save(folder/'mask.png')
                    image=np.array(Image.open(j['reference']).convert('RGB'));image[mask]=(image[mask]*.55+np.array([255,60,0])*.45).astype('uint8');Image.fromarray(image).save(folder/'overlay.png')
        attempt['seconds']=time.time()-started;attempts.append(attempt)
        if success:break
    item=dict(reference_id=ref,reference=j['reference'],reference_sha256=j['reference_sha256'],original_query=original,selected_query=success,status='native_ok' if success==original else 'noun_phrase_retry_ok' if success else 'needs_mask_review',attempts=attempts)
    results.append(item)
    base.write(dest/'results.json',dict(expected=160,processed=len(results),results=results,updated_at=time.time()))
    print(len(results),ref,item['status'],success,flush=True)
summary=dict(expected=160,processed=len(results),native_ok=sum(r['status']=='native_ok' for r in results),noun_phrase_retry_ok=sum(r['status']=='noun_phrase_retry_ok' for r in results),needs_mask_review=[r for r in results if r['status']=='needs_mask_review'])
base.write(dest/'summary.json',summary);print(json.dumps(summary),flush=True)
