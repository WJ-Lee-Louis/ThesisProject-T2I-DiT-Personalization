"""Standalone RunPod generation and existing CLIP/DINO evaluation; no Modal calls."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import csv
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent)]
import benchmark_modal as common
import benchmark_metrics as metrics

EXPERIMENTS = {'source_target_ssg': .8, 'source_target': .7}
OUTPUT = ROOT / 'outputs/runpod_0807'
MAX_CONSECUTIVE_FAILURES = 3


def source_sha(path):
    """Ignore platform newlines, but retain every other source byte."""
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def revision():
    paths = list((ROOT / 'orthogonalization_instantx_ip_adapter').glob('*.py'))
    paths += list((ROOT.parent / 'shared').glob('*.py'))
    paths += [Path(__file__), ROOT / 'scripts/benchmark_modal.py']
    return common.digest([(p.relative_to(ROOT.parent).as_posix(), source_sha(p))
                          for p in sorted(paths, key=lambda p: p.relative_to(ROOT.parent).as_posix())])


def rows_for(limit=0):
    if not common.MANIFEST.exists():
        raise FileNotFoundError('Inputs missing. Run: python runpod/input_bundle.py restore /path/to/runpod_inputs.tar.gz')
    rows = common.read(common.MANIFEST)
    if Counter(r['benchmark'] for r in rows) != {'dreambench': 750, 'dreambench_plus': 1350}:
        raise ValueError('Expected 750 + 1350 cases')
    if len({r['case_id'] for r in rows}) != 2100 or any(r['status'] != 'READY' for r in rows):
        raise ValueError('Invalid manifest')
    if limit:
        rows = [r for b in ('dreambench', 'dreambench_plus')
                for r in [r for r in rows if r['benchmark'] == b][:limit]]
    return rows


def jobs_for(limit=0):
    code = revision()
    jobs = []
    for row in rows_for(limit):
        image_hash = metrics.sha(ROOT / row['image_path'])
        cache = common.read(ROOT / row['conditioning_path'])
        for condition, scale in EXPERIMENTS.items():
            cfg = dict(condition=condition, ip_adapter_scale=scale, strength=1., seed=42,
                       steps=24, width=1024, height=1024, guidance_scale=3.5,
                       implementation_sha256=code)
            signature = common.digest(dict(row=row, settings=cfg, cache=cache, image_sha256=image_hash))
            jobs.append(dict(row=row, settings=cfg, signature=signature, image_sha256=image_hash))
    return jobs


def validate_cache(job):
    from orthogonalization_instantx_ip_adapter.semantic_contrasts import SemanticContrastConditioning, ReferenceFingerprint
    row = job['row']
    data = (ROOT / row['image_path']).read_bytes()
    if hashlib.sha256(data).hexdigest() != job['image_sha256']:
        raise ValueError('Reference changed')
    cached = SemanticContrastConditioning.from_dict(common.read(ROOT / row['conditioning_path']))
    cached.validate_reuse(identity_term=row['identity_term'], target_prompt=row['target_prompt'],
                          reference=ReferenceFingerprint.from_image(data), model='gpt-5.6-sol')
    return data, replace(cached, origin='source_target').to_dict()


def paths(output, job):
    return common.paths_for(output, job['settings']['condition'], job['row'])


def finished(output, job):
    png, sidecar = paths(output, job)
    if not common.complete(png, sidecar, job['signature']):
        return False
    record = common.read(sidecar)
    if record['row'] != job['row'] or record['settings'] != job['settings']:
        raise ValueError(f'Provenance mismatch: {sidecar}')
    validate_diagnostics(record['diagnostics'], job['settings'])
    from PIL import Image
    with Image.open(png) as im:
        if im.size != (1024, 1024):
            raise ValueError(f'Wrong size: {png}')
    return True


def validate_diagnostics(diag, cfg):
    if diag['mask_single_text_rows'] != (cfg['condition'] == 'source_target_ssg'):
        raise ValueError('SSG does not match experiment')
    if not diag['orthogonalization']['enabled']:
        raise ValueError('Projection must be enabled')
    if diag['generation']['ip_adapter_scale'] != cfg['ip_adapter_scale']:
        raise ValueError('Scale does not match experiment')


def preflight(args):
    jobs = jobs_for(args.limit)
    if args.command in ('generate', 'plan'):
        jobs = assigned(jobs, args.total_workers, args.worker_offset, args.gpus)
    seen = set()
    complete = 0
    for job in jobs:
        if job['row']['case_id'] not in seen:
            validate_cache(job)
            seen.add(job['row']['case_id'])
        complete += finished(args.output, job)
    counts = Counter(j['settings']['condition'] + '/' + j['row']['benchmark'] for j in jobs)
    print(json.dumps(dict(expected=len(jobs), complete=complete, pending=len(jobs)-complete,
                         counts=counts, local_gpus=args.gpus, total_workers=args.total_workers,
                         worker_offset=args.worker_offset), indent=2), flush=True)
    return jobs


def assigned(jobs, total, offset, count):
    return [job for i, job in enumerate(jobs) if offset <= i % total < offset + count]


def worker(args):
    # Parent assigns exactly one physical GPU before importing torch.
    jobs = jobs_for(args.limit)[args.lane::args.total_workers]
    pending = [j for j in jobs if not finished(args.output, j)]
    if not pending:
        return
    from importlib.metadata import version
    for package, expected in {'torch': '2.7.1', 'diffusers': '0.35.2', 'transformers': '4.54.0',
                              'accelerate': '1.10.0', 'huggingface-hub': '0.34.3'}.items():
        if version(package).split('+')[0] != expected:
            raise ValueError(f'Unexpected {package} version; run runpod/setup.sh')
    import torch
    from orthogonalization_instantx_ip_adapter import inference
    model = inference.load_ip_adapter_model()
    inference.load_ip_adapter_model = lambda: model
    failures = []
    consecutive_failures = 0
    for number, job in enumerate(pending, 1):
        try:
            started = time.monotonic()
            data, payload = validate_cache(job)
            cfg = job['settings']
            png, diag = inference.generate_png_bytes(
                prompt=job['row']['target_prompt'], reference_image_url='', reference_image_bytes=data,
                ip_adapter_scale=cfg['ip_adapter_scale'], guidance_scale=3.5, num_inference_steps=24,
                seed=42, width=1024, height=1024, orthogonalization_mode='semantic_local',
                orthogonalization_strength=1., object_prompt='', object_behavior_prompt='',
                object_background_prompt='', object_behavior_background_prompt='', remove_behavior=True,
                remove_background=True, include_interaction=False, restore_visual_norm=True,
                mask_single_text_rows=cfg['condition'] == 'source_target_ssg', contrast_analysis_payload=payload)
            validate_diagnostics(diag, cfg)
            target, meta = paths(args.output, job)
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_suffix('.png.tmp')
            temp.write_bytes(png)
            temp.replace(target)
            metrics.write(meta, dict(row=job['row'], settings=cfg, job_signature=job['signature'],
                                    png_sha256=hashlib.sha256(png).hexdigest(), diagnostics=diag,
                                    elapsed_seconds=time.monotonic()-started,
                                    runtime=dict(provider='runpod', gpu=torch.cuda.get_device_name(0),
                                                 pod_id=os.environ.get('RUNPOD_POD_ID'))))
            print(f"SAVED {number}/{len(pending)} {cfg['condition']} {job['row']['case_id']}", flush=True)
            consecutive_failures = 0
        except Exception as exc:
            import traceback
            traceback.print_exc()
            failures.append(dict(case_id=job['row']['case_id'], condition=job['settings']['condition'], error=str(exc)))
            consecutive_failures += 1
        metrics.write(args.output / 'state' / f'lane-{args.lane}.json',
                      dict(processed=number, assigned=len(pending), failures=failures, updated_at=time.time()))
        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            raise RuntimeError('Stopping after 3 consecutive failures; fix the error before resuming')
    if failures:
        raise RuntimeError(f'{len(failures)} jobs failed; rerun same command to resume')


def generate(args):
    import torch
    if torch.cuda.device_count() < args.gpus:
        raise ValueError(f'Requested {args.gpus} GPUs, visible {torch.cuda.device_count()}')
    for i in range(args.gpus):
        if torch.cuda.get_device_properties(i).total_memory < 75 * 1024**3:
            raise ValueError('This run requires >= 75 GiB VRAM per GPU')
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', ','.join(map(str, range(args.gpus)))).split(',')
    children, logs = [], []
    try:
        for lane in range(args.gpus):
            global_lane = args.worker_offset + lane
            log = (args.output / f'worker-{global_lane}.log').open('a', encoding='utf-8')
            logs.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices[lane], PYTHONUNBUFFERED='1')
            children.append(subprocess.Popen([sys.executable, str(Path(__file__)), 'worker',
                '--output', str(args.output), '--gpus', str(args.gpus), '--lane', str(global_lane),
                '--total-workers', str(args.total_workers), '--worker-offset', str(args.worker_offset),
                '--limit', str(args.limit)], env=env, stdout=log, stderr=subprocess.STDOUT))
        wait_workers(children)
    finally:
        for p in children:
            if p.poll() is None:
                p.terminate()
        for p in children:
            p.wait()
        for log in logs:
            log.close()


def wait_workers(children):
    """Notice a failed lane without waiting for an earlier lane to finish."""
    while True:
        codes = [p.poll() for p in children]
        if any(code not in (None, 0) for code in codes):
            raise RuntimeError(f'Workers failed {codes}; see worker-*.log, rerun to resume')
        if all(code is not None for code in codes):
            return
        time.sleep(1)


def evaluate(args):
    jobs = jobs_for(args.limit)
    records = []
    for job in jobs:
        row = job['row']
        records.append(dict(condition=job['settings']['condition'], category=metrics.category(row), row=row,
                            reference=str((ROOT / row['image_path']).resolve()), reference_sha256=job['image_sha256'],
                            generated=str(paths(args.output, job)[0])))
    plan = dict(protocol=metrics.PROTOCOL, records=records)
    def validate(plan, account=None):
        if len(plan['records']) != len(jobs) or not all(finished(args.output, j) for j in jobs):
            raise ValueError('Missing/invalid images; evaluation blocked')
    destination = args.output / 'metrics'
    destination.mkdir(exist_ok=True)
    metrics.write(destination / 'pairs.json', plan)
    result = metrics.evaluate(plan, destination / 'scores.json', 'cuda', 32,
                              checkpoint=destination / 'checkpoint.json', validator=validate)
    with (destination / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['benchmark/condition', 'images', *metrics.METRICS])
        for key, value in result['summary'].items():
            writer.writerow([key, value['cases'], *[value['subject_macro'][m] for m in metrics.METRICS]])
    metrics.write(destination / 'summary.json', result['summary'])


def stop_pod():
    import urllib.request
    pod = os.environ['RUNPOD_POD_ID']
    if not pod.isalnum():
        raise ValueError('Invalid Pod ID')
    request = urllib.request.Request(f'https://rest.runpod.io/v1/pods/{pod}/stop', data=b'', method='POST',
                                    headers={'Authorization': 'Bearer ' + os.environ['RUNPOD_API_KEY']})
    with urllib.request.urlopen(request, timeout=30) as response:
        print('Stop requested:', response.status, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'generate', 'evaluate', 'all', 'worker', '_score'])
    parser.add_argument('--metric-python', default=str(Path(os.environ.get(
        'RUNPOD_WORKDIR', '/workspace/instantx-0807')) / 'venvs/metric/bin/python'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--total-workers', type=int, help='global GPU count across Pods, default: --gpus')
    parser.add_argument('--worker-offset', type=int, default=0, help='first global worker owned by this Pod')
    parser.add_argument('--lane', type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument('--limit', type=int, default=0, help='pilot: cases per benchmark, both conditions; 0=full')
    parser.add_argument('--stop-pod', action='store_true', help='stop this Pod after success OR failure; network volume required')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.total_workers = args.total_workers or args.gpus
    if (not 1 <= args.gpus <= args.total_workers <= 8 or not 0 <= args.lane < args.total_workers
            or not 0 <= args.worker_offset <= args.total_workers - args.gpus or args.limit < 0):
        parser.error('invalid gpus/lane/limit')
    if args.command == 'all' and args.total_workers != args.gpus:
        parser.error('Multi-Pod: generate on each Pod, then evaluate once after all finish')
    if args.command == 'worker':
        worker(args)
        return
    if args.command == '_score':
        evaluate(args)
        return
    if args.command == 'plan':
        preflight(args)
        return
    if args.stop_pod and not all(os.environ.get(k) for k in ('RUNPOD_POD_ID', 'RUNPOD_API_KEY')):
        parser.error('--stop-pod requires RUNPOD_POD_ID and RUNPOD_API_KEY')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command in ('all', 'evaluate') and not Path(args.metric_python).is_file():
        parser.error('Metric Python missing; run runpod/setup.sh or set --metric-python')
    # OS lock is released on crash; prevents duplicate writers but permits resume.
    import fcntl
    with ExitStack() as stack:
        # Shared lock for nonoverlapping generation Pods; exclusive for evaluation.
        lock = stack.enter_context((args.output / '.run.lock').open('a'))
        mode = fcntl.LOCK_SH if args.command == 'generate' else fcntl.LOCK_EX
        fcntl.flock(lock, mode | fcntl.LOCK_NB)
        lanes = range(args.worker_offset, args.worker_offset + args.gpus)
        for lane in lanes:
            lane_lock = stack.enter_context((args.output / f'.lane-{lane}.lock').open('a'))
            fcntl.flock(lane_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        topology = args.output / 'topology.json'
        topology_lock = stack.enter_context((args.output / '.topology.lock').open('a'))
        fcntl.flock(topology_lock, fcntl.LOCK_EX)
        if args.command == 'generate':
            wanted = dict(total_workers=args.total_workers, limit=args.limit)
            if topology.exists() and common.read(topology) != wanted:
                raise ValueError('Topology changed. Stop all writers and remove topology.json before repartitioning.')
            metrics.write(topology, wanted)
        fcntl.flock(topology_lock, fcntl.LOCK_UN)
        status = args.output / f'status-{args.command}-{args.worker_offset}.json'
        try:
            jobs = preflight(args)
            if args.command in ('generate', 'all'):
                generate(args)
            if not all(finished(args.output, j) for j in jobs):
                raise ValueError('Generation incomplete')
            if args.command in ('evaluate', 'all'):
                subprocess.run([args.metric_python, str(Path(__file__)), '_score', '--output',
                                str(args.output), '--limit', str(args.limit)], check=True)
            metrics.write(status, dict(status='complete', stage=args.command,
                          images=len(jobs), pilot=bool(args.limit), updated_at=time.time()))
        except BaseException as exc:
            metrics.write(status, dict(status='failed', error=str(exc), updated_at=time.time()))
            raise
        finally:
            if args.stop_pod:
                stop_pod()


if __name__ == '__main__':
    main()
