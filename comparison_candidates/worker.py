"""Resume-safe artifacts matching the established selected-reference evaluator."""
import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path
from freecus.manifest import load, sha256
from freecus.run import artifact_paths, complete, write_json


def run(a):
    import torch
    from PIL import Image
    from transformers import set_seed
    from .runtime import Diptych, XVerse
    project = Path(__file__).resolve().parents[1]
    base = project/a.method
    upstream = json.loads((base/'UPSTREAM.json').read_text())
    for name,digest in upstream['files'].items():
        if sha256(base/'upstream'/name) != digest:
            raise ValueError('Upstream hash mismatch: '+name)
    rows = load(a.bundle/'manifest.jsonl',a.bundle/'input')
    wanted = set(a.cases)
    selected = [r for r in rows if r['case_id'] in wanted]
    if len(selected) != len(wanted) or {r['seed'] for r in rows} != {42}:
        raise ValueError('Noncanonical case selection')
    cfg = json.loads((getattr(a, 'config', None) or base/'config.json').read_text())
    files = (list(base.glob('*.json'))+list(base.glob('requirements*.txt'))+
             [p for p in (base/'upstream').rglob('*') if p.is_file() and '__pycache__' not in p.parts]+
             list(base.glob('*.py'))+
             list((project/'comparison_candidates').glob('*.py'))+
             [project/'freecus'/n for n in ['manifest.py','run.py','evaluate.py','project_metrics/benchmark_metrics.py']])
    code = {p.relative_to(project).as_posix():sha256(p) for p in files}
    identity = dict(config=cfg,manifest_sha256=sha256(a.bundle/'manifest.jsonl'),code=code,
                    models=json.loads((base/'models.lock.json').read_text()))
    fingerprint = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    versions = {n:importlib.metadata.version(n) for n in ['torch','torchvision','diffusers','transformers','numpy','Pillow']}
    marker = a.output/'run.json'
    record = dict(**identity,fingerprint=fingerprint,versions=versions,gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda)
    if marker.exists() and json.loads(marker.read_text()) != record:
        previous=json.loads(marker.read_text())
        raise ValueError('Run identity changed: '+str([k for k in record if previous.get(k)!=record[k]]))
    write_json(marker,record)
    pending = [r for r in selected if not complete(artifact_paths(a.output,r))]
    if not pending:
        return
    set_seed(42)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    engine = (Diptych if a.method=='diptych' else XVerse)(json.loads((a.bundle/'resolved_models.json').read_text()),cfg)
    write_json(a.output/'initialization.json',dict(seconds=time.perf_counter()-start,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
    for row in pending:
        set_seed(row['seed'])
        start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            image,prompt = engine.generate(row,a.bundle/'input'/row['reference'])
        torch.cuda.synchronize()
        if image.size != (cfg['width'],cfg['height']):
            raise ValueError('Output resolution changed')
        paths = artifact_paths(a.output,row)
        for p in paths.values():
            p.parent.mkdir(parents=True,exist_ok=True)
        image.save(paths['tgt_image'])
        with Image.open(a.bundle/'input'/row['reference']) as ref:
            ref.convert('RGB').save(paths['src_image'])
        paths['text'].write_text(row['prompt'],encoding='utf-8')
        timing = dict(seconds=time.perf_counter()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        write_json(paths['metadata'],dict(**row,fingerprint=fingerprint,generation_prompt=prompt,timing=timing,
            artifact_sha256={k:sha256(v) for k,v in paths.items() if k!='metadata'}))
        print(json.dumps(dict(saved=row['case_id'],**timing)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--method',choices=['diptych','xverse'],required=True)
    p.add_argument('--config',type=Path)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cases',nargs='+',required=True)
    run(p.parse_args())
