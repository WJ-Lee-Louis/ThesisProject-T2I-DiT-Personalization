"""512-only worker: preserve canonical cases and record failures without substitution."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
import time
import traceback
from collections import defaultdict
from pathlib import Path

from .manifest import load, sha256
from .run import artifact_paths, complete, validate_config, write_json


def execute(job, emit):
    import torch
    from PIL import Image
    from .runtime import Engine
    config = json.loads(Path(job['config']).read_text())
    validate_config(config)
    if (config['width'], config['height'], config['caption_seed'], config['seeds']) != (512, 512, 42, [42]):
        raise ValueError('Dedicated 512/seed42 worker only')
    root, out = Path(job['data_root']), Path(job['output'])
    manifest = Path(job['manifest'])
    rows = load(manifest, root)
    wanted = set(job['case_ids'])
    selected = [r for r in rows if r['case_id'] in wanted]
    if {r['case_id'] for r in selected} != wanted:
        raise ValueError('Unknown cases')
    source = Path(__file__).parent
    files = [*source.glob('*.py'), *source.glob('runtime_src/*.py'), *source.glob('upstream/*.py')]
    identity = dict(config=config, manifest_sha256=sha256(manifest),
                    code={f.relative_to(source).as_posix(): sha256(f) for f in sorted(files)})
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    versions = {n: importlib.metadata.version(n) for n in
                ['torch','torchvision','diffusers','transformers','accelerate','numpy','Pillow']}
    marker = out / 'run.json'
    if marker.exists():
        old = json.loads(marker.read_text())
        if old['fingerprint'] != fingerprint or old['versions'] != versions:
            raise ValueError('Refusing to mix runs')
    else:
        write_json(marker, dict(fingerprint=fingerprint, **identity, versions=versions,
                               python=platform.python_version(), gpu=torch.cuda.get_device_name(),
                               cuda=torch.version.cuda,
                               model_lock=json.loads(manifest.with_name('models.lock.json').read_text())))
    pending = [r for r in selected if not complete(artifact_paths(out, r))
               and not (out/'failures'/f"{r['case_id']}.json").exists()]
    groups = defaultdict(list)
    for row in pending:
        groups[row['subject']].append(row)
    engine = Engine(config)
    captions = {}

    def failed(row, phase, exc):
        record = dict(case_id=row['case_id'], reference_id=row['reference_id'], category=row['category'],
                      prompt=row['prompt'], phase=phase, error=repr(exc), traceback=traceback.format_exc(),
                      fingerprint=fingerprint, resolution=[512,512], automatic_retry=False)
        write_json(out/'failures'/f"{row['case_id']}.json", record)
        emit(dict(phase='case_failed', **{k:v for k,v in record.items() if k not in {'phase','traceback'}}))

    for key, group in groups.items():
        cache = out/'captions'/f'{key}.json'
        if cache.exists():
            captions[key] = json.loads(cache.read_text())
            continue
        start = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        try:
            raw, filtered = engine.describe(group[0], root)
            captions[key] = dict(raw=raw, filtered=filtered, reference=group[0]['reference'], seed=42)
            write_json(cache, captions[key])
            emit(dict(phase='caption', subject=key, seconds=time.monotonic()-start,
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
        except Exception as exc:
            for row in group:
                failed(row, 'captioning', exc)
            torch.cuda.empty_cache()
    engine.release_caption_models()
    emit(dict(phase='caption_models_released', allocated_gib=torch.cuda.memory_allocated()/2**30))
    for row in pending:
        if row['subject'] not in captions:
            continue
        start = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        try:
            image, augmented = engine.generate(row, root, captions[row['subject']]['filtered'])
            torch.cuda.synchronize()
            if image.size != (512,512):
                raise ValueError(f'Actual dimensions: {image.size}')
            timing = dict(seconds=time.monotonic()-start,
                          peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                          peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
            paths = artifact_paths(out, row)
            for p in paths.values():
                p.parent.mkdir(parents=True, exist_ok=True)
            image.save(paths['tgt_image'])
            with Image.open(root/row['reference']) as ref:
                ref.convert('RGB').save(paths['src_image'])
            paths['text'].write_text(row['prompt'], encoding='utf-8')
            write_json(paths['metadata'], dict(**row, augmented_prompt=augmented, caption=captions[row['subject']],
                       fingerprint=fingerprint, timing=timing,
                       artifact_sha256={k:sha256(p) for k,p in paths.items() if k!='metadata'}))
            emit(dict(phase='generated', case_id=row['case_id'], id=row['id'], actual_resolution=[512,512], **timing))
        except Exception as exc:
            failed(row, 'generation', exc)
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('job', type=Path)
    args = parser.parse_args()
    start = time.monotonic()
    def emit(value):
        value.update(worker_elapsed_seconds=time.monotonic()-start,
                     host_peak_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
        with args.job.with_name('events.jsonl').open('a') as stream:
            stream.write(json.dumps(value)+'\n')
            stream.flush()
        print(json.dumps(value), flush=True)
    execute(json.loads(args.job.read_text()), emit)


if __name__ == '__main__':
    main()
