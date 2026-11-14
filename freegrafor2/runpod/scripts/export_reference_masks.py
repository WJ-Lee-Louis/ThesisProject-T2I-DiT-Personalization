"""Re-extract reviewable masks for every selected reference; never alter a run.

These are newly computed masks, not a claim to recover deleted historical masks.
"""
from pathlib import Path
import hashlib
import html
import importlib.metadata
import json
import os
import shutil
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from segment_anything import sam_model_registry, SamPredictor
from core.collage_creation import CollageCreator

ROOT = Path('/workspace/freegraftor')
DEST = ROOT/'recovery/all-reference-masks-20261114'


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def main():
    torch.set_num_threads(8)
    DEST.mkdir(parents=True, exist_ok=True)
    references = json.loads((ROOT/'inputs/references.json').read_text())
    aliases = json.loads((ROOT/'recovery/detector_aliases.json').read_text())
    obj = CollageCreator.__new__(CollageCreator)
    obj.device = 'cuda'
    obj.grounding_dino_processor = AutoProcessor.from_pretrained(os.environ['GROUNDING_DINO'])
    obj.grounding_dino = AutoModelForZeroShotObjectDetection.from_pretrained(
        os.environ['GROUNDING_DINO']).to('cuda', dtype=torch.bfloat16).eval()
    obj.sam = SamPredictor(sam_model_registry['vit_h'](checkpoint=os.environ['SAM']))
    obj.sam.model = obj.sam.model.to('cuda', dtype=torch.bfloat16).eval()
    provenance = dict(kind='new_reference_mask_reextraction', historical_masks=False,
                      generated_at=time.time(), box_threshold=.35, text_threshold=.25,
                      source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      versions={p:importlib.metadata.version(p) for p in
                                ['torch', 'transformers', 'segment-anything-py']},
                      gpu=torch.cuda.get_device_name(0))
    write(DEST/'provenance.json', provenance)
    results = []
    cards = []
    for ref, record in sorted(references.items()):
        folder = DEST/ref
        folder.mkdir(parents=True, exist_ok=True)
        source = Path(record['path'])
        assert hashlib.sha256(source.read_bytes()).hexdigest() == record['sha256']
        reference_file = folder/('reference'+source.suffix)
        shutil.copy2(source, reference_file)
        original = record['identity_term']
        queries = list(dict.fromkeys([original]+aliases.get(ref, {}).get('queries', [])+['a '+original]))
        item = dict(reference_id=ref, reference_sha256=record['sha256'],
                    original_query=original, selected_query=None, attempts=[],
                    style_excluded_from_primary_metrics='/style/' in ref,
                    historical_mask=False, status='not_detected')
        for query in queries:
            detections = obj.grounding(image_path=str(source), prompt=query)
            item['attempts'].append(dict(query=query,
                scores=detections['scores'].float().cpu().tolist(),
                boxes=detections['boxes'].float().cpu().tolist()))
            if not len(detections['boxes']):
                continue
            masks, boxes = obj.segment(image_path=str(source), prompt=query)
            if not masks:
                continue
            mask = masks[0].squeeze().cpu().numpy().astype(bool)
            Image.fromarray(mask.astype('uint8')*255).save(folder/'mask.png')
            overlay = np.array(Image.open(source).convert('RGB'))
            overlay[mask] = (overlay[mask]*.55+np.array([255, 60, 0])*.45).astype('uint8')
            Image.fromarray(overlay).save(folder/'overlay.png')
            fraction = float(mask.mean())
            item.update(selected_query=query, mask_fraction=fraction,
                        selected_box=boxes[0].float().cpu().tolist(),
                        status=('needs_review' if fraction == 0 or fraction >= .98 else
                                'native_detected' if query == original else 'alias_detected'),
                        mask_sha256=hashlib.sha256((folder/'mask.png').read_bytes()).hexdigest())
            break
        write(folder/'record.json', item)
        results.append(item)
        prefix=ref+'/'
        caption=html.escape(ref+' | '+item['status']+' | '+str(item['selected_query']))
        images=f'<figure><img loading="lazy" src="{prefix+reference_file.name}"><figcaption>Reference</figcaption></figure>'
        if item['selected_query']:
            for filename, label in [('overlay.png', 'Mask overlay'), ('mask.png', 'Binary mask')]:
                images+=f'<figure><img loading="lazy" src="{prefix+filename}"><figcaption>{label}</figcaption></figure>'
        cards.append(f'<section data-ref="{html.escape(ref)}"><h2>{caption}</h2>{images}<p><a href="{prefix}record.json">Detection record</a></p></section>')
        write(DEST/'results.json', dict(expected=len(references), processed=len(results),
                                       results=results, updated_at=time.time()))
        print(len(results), ref, item['status'], flush=True)
    summary=dict(expected=len(references), processed=len(results),
                 detected=sum(r['selected_query'] is not None for r in results),
                 not_detected=[r['reference_id'] for r in results if r['selected_query'] is None],
                 needs_review=[r['reference_id'] for r in results if r['status']=='needs_review'],
                 historical_masks=False)
    write(DEST/'summary.json', summary)
    page='''<!doctype html><meta charset="utf-8"><title>FreeGraftor reference mask review</title>
<style>body{font-family:system-ui;margin:24px;background:#fafafa}section{border-top:1px solid #ccc;padding:16px 0}h2{font-size:16px}figure{display:inline-block;margin:8px;vertical-align:top}img{width:260px;height:260px;object-fit:contain;background:#eee}input{padding:10px;width:360px}</style>
<h1>FreeGraftor: all selected reference masks</h1>
<p>New extraction on 2026-11-14 with original thresholds. These are not historical mask artifacts. Red overlay marks the SAM foreground. Detection does not establish semantic correctness. Style references are excluded from primary metrics.</p>
<input id="filter" placeholder="Filter reference ID or subject">
'''+''.join(cards)+'''<script>document.querySelector('#filter').oninput=e=>{for(const s of document.querySelectorAll('section'))s.hidden=!s.textContent.toLowerCase().includes(e.target.value.toLowerCase())}</script>'''
    (DEST/'index.html').write_text(page)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
