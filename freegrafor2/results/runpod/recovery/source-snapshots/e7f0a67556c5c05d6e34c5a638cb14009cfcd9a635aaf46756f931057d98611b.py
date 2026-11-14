"""Compare the original detector label with the official can example label."""
from pathlib import Path
import json,os,time
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
from segment_anything import sam_model_registry,SamPredictor
from core.collage_creation import CollageCreator

torch.set_num_threads(8)
root=Path('/workspace/freegraftor')
dest=root/'recovery/probes/can';dest.mkdir(parents=True,exist_ok=True)
reference=root/'inputs/references/dreambench/can/01.jpg'
obj=CollageCreator.__new__(CollageCreator);obj.device='cuda'
obj.grounding_dino_processor=AutoProcessor.from_pretrained(os.environ['GROUNDING_DINO'])
obj.grounding_dino=AutoModelForZeroShotObjectDetection.from_pretrained(os.environ['GROUNDING_DINO']).to('cuda',dtype=torch.bfloat16).eval()
obj.sam=SamPredictor(sam_model_registry['vit_h'](checkpoint=os.environ['SAM']))
obj.sam.model=obj.sam.model.to('cuda',dtype=torch.bfloat16).eval()
results=[]
for q in ['can','beer can','beverage can','tin can']:
    started=time.time()
    detections=obj.grounding(image_path=str(reference),prompt=q)
    masks,boxes=obj.segment(image_path=str(reference),prompt=q)
    item=dict(query=q,box_threshold=.35,text_threshold=.25,count=len(masks),scores=detections['scores'].cpu().float().tolist(),boxes=[b.cpu().float().tolist() for b in boxes],elapsed_seconds=time.time()-started)
    if masks:
        mask=masks[0].squeeze().cpu().numpy().astype(bool)
        item['mask_fraction']=float(mask.mean())
        Image.fromarray(mask.astype('uint8')*255).save(dest/(q.replace(' ','_')+'-mask.png'))
        image=np.array(Image.open(reference).convert('RGB'));overlay=image.copy()
        overlay[mask]=(image[mask]*.55+np.array([255,60,0])*.45).astype('uint8')
        Image.fromarray(overlay).save(dest/(q.replace(' ','_')+'-overlay.png'))
    results.append(item);print(json.dumps(item),flush=True)
(dest/'results.json').write_text(json.dumps(results,indent=2))
