"""Probe equivalent film noun phrases on the preserved failed template.

No diffusion prompts, seeds, detector thresholds or benchmark outputs change.
"""
from pathlib import Path
import json
import os
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from segment_anything import sam_model_registry, SamPredictor
from core.collage_creation import CollageCreator


def main():
    torch.set_num_threads(8)
    root = Path('/workspace/freegraftor')
    dest = root / 'recovery/probes/film-20261114'
    dest.mkdir(parents=True, exist_ok=True)
    template = root / 'cache/recovery-work/lane-0/156-04/template.png'
    obj = CollageCreator.__new__(CollageCreator)
    obj.device = 'cuda'
    obj.grounding_dino_processor = AutoProcessor.from_pretrained(os.environ['GROUNDING_DINO'])
    obj.grounding_dino = AutoModelForZeroShotObjectDetection.from_pretrained(
        os.environ['GROUNDING_DINO']).to('cuda', dtype=torch.bfloat16).eval()
    obj.sam = SamPredictor(sam_model_registry['vit_h'](checkpoint=os.environ['SAM']))
    obj.sam.model = obj.sam.model.to('cuda', dtype=torch.bfloat16).eval()
    # Fix the semantic alternatives before inspecting detector results.
    queries = ['roll of film', 'a roll of film', 'film strip', 'photographic film',
               'unrolled photographic film', 'strip of photographic film']
    results = []
    for query in queries:
        started = time.time()
        detections = obj.grounding(image_path=str(template), prompt=query)
        masks, boxes = obj.segment(image_path=str(template), prompt=query)
        item = dict(query=query, box_threshold=.35, text_threshold=.25,
                    scores=detections['scores'].float().cpu().tolist(),
                    boxes=[b.float().cpu().tolist() for b in boxes],
                    elapsed_seconds=time.time()-started)
        if masks:
            mask = masks[0].squeeze().cpu().numpy().astype(bool)
            item['mask_fraction'] = float(mask.mean())
            name = query.replace(' ', '_')
            Image.fromarray(mask.astype('uint8')*255).save(dest/(name+'-mask.png'))
            overlay = np.array(Image.open(template).convert('RGB'))
            overlay[mask] = (overlay[mask]*.55+np.array([255, 60, 0])*.45).astype('uint8')
            Image.fromarray(overlay).save(dest/(name+'-overlay.png'))
        results.append(item)
        (dest/'results.json').write_text(json.dumps(results, indent=2))
        print(json.dumps(item), flush=True)


if __name__ == '__main__':
    main()
