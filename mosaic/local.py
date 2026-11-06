"""GPU-0 local generation, complete-only scoring and portable benchmark export."""
import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import urllib.request
import zipfile

from freecus.manifest import PROJECT, load, sha256
from freecus.run import artifact_paths, complete, write_json

BASE = PROJECT / 'mosaic'
WORK = BASE / 'outputs/local'
BENCHMARKS = {'dreambench': 750, 'dreambench_plus': 1350}
RUN_ID = 'local-gpu0-seed42-bf16-v1'
DATA_REVISION = '459b06499fad3dac842c4b1f7a053fb3328d91c7'


def manifest(benchmark):
    return BASE / 'manifests' / (benchmark + '_seed42.jsonl')


def lane(benchmark):
    return WORK / benchmark / RUN_ID


def results(benchmark):
    return BASE / 'results' / benchmark / RUN_ID


def inputs():
    """Restore only canonical missing references, checking original byte hashes."""
    rows = [json.loads(s) for b in BENCHMARKS for s in manifest(b).read_text().splitlines() if s.strip()]
    refs = {r['reference']: r['reference_sha256'] for r in rows}
    missing = [name for name in refs if not (PROJECT / name).exists()]
    if missing:
        if any('/dreambench_plus_raw/' not in name for name in missing):
            raise ValueError('Missing DreamBench input; restore the canonical dataset first')
        archive = BASE / 'outputs/dreambench_plus_data.zip'
        if not archive.exists():
            archive.parent.mkdir(parents=True, exist_ok=True)
            temp = archive.with_suffix('.download')
            urllib.request.urlretrieve(
                f'https://huggingface.co/datasets/yuangpeng/dreambench_plus/resolve/{DATA_REVISION}/data.zip', temp)
            temp.replace(archive)
        with zipfile.ZipFile(archive) as z:
            for name in missing:
                suffix = name.split('/extracted/', 1)[1]
                candidates = [p for p in z.namelist() if p == suffix or p.endswith('/' + suffix)]
                if len(candidates) != 1:
                    raise ValueError('Ambiguous/missing reference in dataset ZIP: ' + suffix)
                data = z.read(candidates[0])
                import hashlib
                if hashlib.sha256(data).hexdigest() != refs[name]:
                    raise ValueError('Downloaded reference does not match canonical manifest: ' + name)
                target = PROJECT / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
    report = {}
    for b, count in BENCHMARKS.items():
        selected = load(manifest(b), PROJECT)
        if len(selected) != count or len({r['case_id'] for r in selected}) != count:
            raise ValueError('Incomplete canonical manifest')
        if any(r['seed'] != 42 or r['reference_policy'] != 'selected_one' for r in selected):
            raise ValueError('Noncanonical generation settings')
        report[b] = dict(cases=count, references=len({r['reference_id'] for r in selected}),
                         manifest_sha256=sha256(manifest(b)))
    write_json(WORK / 'inputs.json', dict(status='verified', benchmarks=report,
        restored_dataset=dict(repo='yuangpeng/dreambench_plus', revision=DATA_REVISION)))
    print(json.dumps(report), flush=True)


def prepare():
    inputs()
    import torch
    from huggingface_hub import hf_hub_download, snapshot_download
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or torch.cuda.device_count() != 1:
        raise ValueError('Run with CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0')
    if not torch.cuda.is_bf16_supported() or torch.cuda.get_device_capability() not in [(12, 0)]:
        raise ValueError('Expected the local RTX PRO 6000 Blackwell on physical GPU 0')
    # Exercise CUDA before committing to model downloads.
    a = torch.ones((32, 32), device='cuda', dtype=torch.bfloat16)
    assert (a @ a).float().mean().item() == 32
    locks = json.loads((BASE / 'models.lock.json').read_text())
    # Fail promptly on missing gated-model authorization, before large downloads.
    hf_hub_download(locks['flux']['repo'], 'model_index.json', revision=locks['flux']['revision'])
    paths = {}
    for key, item in locks.items():
        print('Preparing model: ' + key, flush=True)
        allow = ['model_index.json', 'scheduler/*', 'tokenizer/*', 'tokenizer_2/*',
                 'text_encoder/*', 'text_encoder_2/*', 'transformer/*', 'vae/*'] if key == 'flux' else None
        paths[key] = snapshot_download(item['repo'], revision=item['revision'], allow_patterns=allow if key=='flux' else [item['weight_name']],
            ignore_patterns=['*.bin', '*.onnx', '*.h5', '*.msgpack', '*.tflite', '*.ot'])
    write_json(WORK / 'resolved_models.json', paths)
    cfg = json.loads((BASE / 'config.json').read_text())
    cfg.update(gpu=torch.cuda.get_device_name(), execution_backend='server', physical_gpu_index=0)
    config = WORK / 'config.json'
    if config.exists() and json.loads(config.read_text()) != cfg:
        raise ValueError('Local configuration changed; preserve the old run and choose a new RUN_ID')
    write_json(config, cfg)
    for b in BENCHMARKS:
        bundle = WORK / 'bundles' / b
        bundle.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(manifest(b), bundle / 'manifest.jsonl')
        if not (bundle / 'input').exists():
            (bundle / 'input').symlink_to(PROJECT, target_is_directory=True)
        write_json(bundle / 'resolved_models.json', paths)
    snapshot()


def snapshot():
    files = [p for folder in ['mosaic', 'freecus']
             for p in (PROJECT / folder).rglob('*') if p.is_file()
             and not any(part.startswith('.') or part in {'outputs', 'results', 'artifacts', '__pycache__'}
                         for part in p.relative_to(PROJECT / folder).parts)
             and p.suffix in {'.py', '.json', '.txt', '.md', '.jsonl', '.sh'}]
    target = WORK / 'generation_source.zip'
    if target.exists():
        with zipfile.ZipFile(target) as z:
            if set(z.namelist()) != {p.relative_to(PROJECT).as_posix() for p in files}:
                raise ValueError('Generation source file list changed')
            for p in files:
                if z.read(p.relative_to(PROJECT).as_posix()) != p.read_bytes():
                    raise ValueError('Generation source changed: ' + str(p))
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):
            z.write(p, p.relative_to(PROJECT).as_posix())


def generate(benchmark, pilot=False):
    rows = load(manifest(benchmark), PROJECT)
    if pilot:
        seen = set()
        selected = []
        for r in rows:
            if r['category'] not in seen:
                selected.append(r)
                seen.add(r['category'])
    else:
        selected = rows
    subprocess.run([sys.executable, '-m', 'mosaic.worker', '--method', 'mosaic',
        '--config', str(WORK / 'config.json'), '--bundle', str(WORK / 'bundles' / benchmark),
        '--output', str(lane(benchmark)), '--cases', *[r['case_id'] for r in selected]],
        cwd=PROJECT, check=True)


def export(benchmark):
    from freecus.operations.artifact_bundle import pack, verify, digest
    folder = results(benchmark)
    archives = BASE / 'artifacts' / benchmark / RUN_ID
    index = pack([lane(benchmark)], manifest(benchmark), PROJECT, archives, 'mosaic',
        benchmark, RUN_ID, BENCHMARKS[benchmark], resolution=(512, 512))
    verify(archives, index)
    if index['missing_case_ids']:
        raise ValueError('Refusing partial benchmark export')
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / 'bundle_index.json', index)
    shutil.copyfile(manifest(benchmark), folder / 'manifest.jsonl')
    shutil.copyfile(WORK / 'generation_source.zip', folder / 'generation_source.zip')
    from importlib.metadata import distributions
    (folder / 'environment.freeze.txt').write_text('\n'.join(sorted(
        d.metadata['Name'] + '==' + d.version for d in distributions())) + '\n', encoding='utf-8')
    write_json(folder / 'provenance.json', dict(method='mosaic', benchmark=benchmark, bundle_id=RUN_ID,
        backend='local', physical_gpu_index=0, generation_resolution=[512, 512],
        source_archive_sha256=digest(folder / 'generation_source.zip'),
        manifest_sha256=sha256(manifest(benchmark)), generated_images=index['generated_images'],
        generation_fingerprint=index['generation_fingerprint'], missing_case_ids=[]))
    write_json(folder / 'verification.json', dict(status='verified', generated_images=index['generated_images'],
        zip_files=len(index['parts']), total_archive_bytes=sum(p['size_bytes'] for p in index['parts']),
        all_member_sha256_verified=True, source_reference_paths_recoverable=True))


def evaluate(benchmark):
    from freecus.evaluate import prepare as scoring_plan, project_evaluator
    from .evaluation import validate
    evaluator = project_evaluator()
    folder = results(benchmark)
    plan = scoring_plan(manifest(benchmark), PROJECT, lane(benchmark), evaluator, 'all')
    for rec in plan['records']:
        rec['condition'] = 'mosaic'
    write_json(folder / 'evaluation_plan.json', plan)
    result = evaluator.evaluate(plan, folder / 'metrics.json', 'cuda', 32, validator=validate,
        checkpoint=WORK / benchmark / 'metric_checkpoint.json')
    index = json.loads((folder / 'bundle_index.json').read_text())
    canonical = {r['case_id']: r for r in load(manifest(benchmark), PROJECT)}
    if {s['case_id'] for s in result['scores']} != set(canonical) or len(result['scores']) != len(canonical):
        raise ValueError('Incomplete metric coverage')
    for score in result['scores']:
        row = canonical[score['case_id']]
        if score['generated_sha256'] != index['members']['tgt_image/' + row['id'] + '.png']['sha256']:
            raise ValueError('Metric image hash differs from archived image')
        if not all(math.isfinite(score[k]) for k in evaluator.METRICS):
            raise ValueError('Non-finite metric')
    result.update(generation_resolution=[512, 512], bundle_id=RUN_ID,
        generation_fingerprint=index['generation_fingerprint'],
        coverage=dict(status='complete', generated_cases=len(canonical), canonical_cases=len(canonical), missing_case_ids=[]))
    write_json(folder / 'metrics.json', result)
    report = BASE / 'results/reports'
    report.mkdir(parents=True, exist_ok=True)
    write_json(report / (benchmark + '_scores.json'), result)
    with (report / (benchmark + '_per_image.csv')).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['case_id', 'reference_id', 'category', 'target_prompt',
            'generated_sha256', 'CLIP-I', 'CLIP-T', 'DINO-I'], extrasaction='ignore')
        writer.writeheader()
        writer.writerows(result['scores'])
    from freecus.operations.artifact_bundle import digest
    provenance = json.loads((folder / 'provenance.json').read_text())
    provenance.update(metrics_sha256=digest(folder / 'metrics.json'), metric_cases=len(canonical),
        metric_versions=result['versions'], metric_device=result['device_name'], metric_protocol=result['protocol'])
    write_json(folder / 'provenance.json', provenance)
    verification = json.loads((folder / 'verification.json').read_text())
    verification.update(metric_cases=len(canonical), score_image_sha256_verified=True, coverage='complete', missing_case_ids=[])
    write_json(folder / 'verification.json', verification)


def reports():
    summaries, inventory = [], []
    for b in BENCHMARKS:
        folder = results(b)
        result = json.loads((folder / 'metrics.json').read_text())
        index = json.loads((folder / 'bundle_index.json').read_text())
        summary = result['summary'][b + '/mosaic']
        summaries.append(dict(method='mosaic', benchmark=b, cases=len(result['scores']),
            references=summary['subjects'], canonical_cases=BENCHMARKS[b], missing_cases=0,
            selection='all', **summary['subject_macro']))
        inventory.append(dict(method='mosaic', benchmark=b, bundle_id=RUN_ID,
            **json.loads((folder / 'verification.json').read_text()),
            archives=[dict(**p, path=(Path('mosaic/artifacts') / b / RUN_ID / p['file']).as_posix())
                      for p in index['parts']]))
    folder = BASE / 'results/reports'
    with (folder / 'metrics_summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    write_json(folder / 'metrics_audit.json', dict(status='verified', evaluations=summaries,
        generation_resolution=[512, 512], physical_gpu_index=0,
        scoring_source_sha256=sha256(PROJECT / 'freecus/project_metrics/benchmark_metrics.py')))
    write_json(BASE / 'results/transfer_inventory.json', dict(bundles=inventory))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['inputs', 'prepare', 'run', 'status', 'evaluate'])
    p.add_argument('--benchmark', choices=list(BENCHMARKS))
    a = p.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    if a.command == 'status':
        state = WORK / 'status.json'
        print(state.read_text() if state.exists() else 'No local run submitted')
        for b, count in BENCHMARKS.items():
            n = len(list((lane(b) / 'metadata').rglob('*.json')))
            print(f'{b}: {n}/{count} saved metadata files')
        return
    if a.command == 'inputs':
        inputs()
        return
    if a.command == 'evaluate':
        if not a.benchmark:
            p.error('--benchmark is required for evaluate')
        evaluate(a.benchmark)
        return
    with (WORK / 'workflow.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            write_json(WORK / 'status.json', dict(status='preparing', pid=os.getpid(), run_id=RUN_ID))
            prepare()
            if a.command == 'prepare':
                write_json(WORK / 'status.json', dict(status='ready', run_id=RUN_ID))
                return
            for b in BENCHMARKS:
                write_json(WORK / 'status.json', dict(status='piloting', benchmark=b, pid=os.getpid(), run_id=RUN_ID))
                generate(b, pilot=True)
            for b in BENCHMARKS:
                write_json(WORK / 'status.json', dict(status='generating', benchmark=b, pid=os.getpid(), run_id=RUN_ID))
                generate(b)
                if not all(complete(artifact_paths(lane(b), r)) for r in load(manifest(b), PROJECT)):
                    raise ValueError('Incomplete generation: ' + b)
                export(b)
                write_json(WORK / 'status.json', dict(status='evaluating', benchmark=b, pid=os.getpid(), run_id=RUN_ID))
                subprocess.run([sys.executable, '-m', 'mosaic.local', 'evaluate', '--benchmark', b], cwd=PROJECT, check=True)
            reports()
            write_json(WORK / 'status.json', dict(status='complete', generated_images=2100, evaluated_images=2100, run_id=RUN_ID))
        except Exception as e:
            write_json(WORK / 'status.json', dict(status='failed', error=type(e).__name__ + ': ' + str(e), run_id=RUN_ID))
            traceback.print_exc()
            raise SystemExit(1)


if __name__ == '__main__':
    main()
