"""Canonical 1024 single-reference USO run with shared artifact contracts."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
from freecus.manifest import load, sha256
from freecus.run import write_json, artifact_paths, complete


def execute(args):
    import torch
    from PIL import Image
    from uso.runtime import Engine
    bundle = args.bundle
    cfg = json.loads((bundle / 'config.json').read_text())
    identity = json.loads((bundle / 'identity.json').read_text())
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    rows = load(bundle / 'manifest.jsonl', bundle / 'input')
    if len(rows) not in (750, 1350) or {r['seed'] for r in rows} != {42}:
        raise ValueError('Noncanonical cases')
    selected = [r for r in rows if r['case_id'] in set(args.case_ids)]
    if len(selected) != len(set(args.case_ids)):
        raise ValueError('Unknown cases')
    versions = {n: importlib.metadata.version(n) for n in ['torch', 'torchvision', 'transformers', 'Pillow', 'numpy']}
    marker = args.output / 'run.json'
    record = dict(config=dict(cfg, seeds=[42]), fingerprint=fingerprint,
                  manifest_sha256=sha256(bundle / 'manifest.jsonl'), code=identity['code'],
                  models=identity['models'], versions=versions, gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda)
    if marker.exists():
        old = json.loads(marker.read_text())
        if old['fingerprint'] != fingerprint or old['versions'] != versions:
            raise ValueError('Existing run identity differs')
    else:
        write_json(marker, record)
    pending = [r for r in selected if not complete(artifact_paths(args.output, r))]
    if not pending:
        return
    paths = json.loads((bundle / 'resolved_models.json').read_text())
    if any(Path(paths[k]).name != v['revision'] for k, v in identity['models'].items()):
        raise ValueError('Model revisions differ')
    torch.cuda.reset_peak_memory_stats()
    began = time.monotonic()
    engine = Engine(cfg, paths)
    write_json(args.output / 'initialization.json', dict(seconds=time.monotonic()-began,
               peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
    for row in pending:
        began = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        with Image.open(bundle / 'input' / row['reference']) as source:
            reference = source.convert('RGB')
        image = engine.generate(row, reference)
        torch.cuda.synchronize()
        if image.size != (1024, 1024):
            raise ValueError('Generated resolution differs')
        paths = artifact_paths(args.output, row)
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        image.save(paths['tgt_image'])
        reference.save(paths['src_image'])
        paths['text'].write_text(row['prompt'], encoding='utf-8')
        timing = dict(seconds=time.monotonic()-began, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        write_json(paths['metadata'], dict(**row, fingerprint=fingerprint, generation_prompt=row['prompt'],
                   uso_task='style' if row['category']=='style' else 'subject', timing=timing,
                   artifact_sha256={k: sha256(p) for k,p in paths.items() if k!='metadata'}))
        print('saved', row['case_id'], json.dumps(timing), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case-ids', nargs='+', required=True)
    execute(parser.parse_args())
