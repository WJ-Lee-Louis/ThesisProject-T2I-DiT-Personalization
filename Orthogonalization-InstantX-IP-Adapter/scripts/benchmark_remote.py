"""Plan/submit/status/download/resume durable DreamBench jobs (no local GPU)."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_modal as original
import benchmark_remote_worker as remote


def transport_digest():
    return original.digest([hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (Path(__file__), ROOT / "benchmark_remote_worker.py")])


def build_bundle(args, directory):
    """Validate caches once per case and preserve every compatible local result."""
    if not args.conditions or not set(args.conditions).issubset(remote.DEFAULT_CONDITIONS):
        raise ValueError('Only baseline/source_target_ssg are enabled; source_target is on hold')
    from orthogonalization_instantx_ip_adapter.semantic_contrasts import (
        SemanticContrastConditioning, ReferenceFingerprint,
    )
    rows = original.read(original.MANIFEST)
    if (len(rows) != 2100 or len({r['case_id'] for r in rows}) != 2100
            or any(r['status'] != 'READY' for r in rows)
            or sum(r['benchmark'] == 'dreambench' for r in rows) != 750
            or sum(r['benchmark'] == 'dreambench_plus' for r in rows) != 1350):
        raise ValueError("Unexpected manifest")
    rows = original.select_rows(rows, args.benchmark, args.shard_index, args.num_shards)
    revision = original.implementation_digest()
    settings = dict(strength=args.strength, seed=args.seed, steps=args.steps, width=args.width,
                    height=args.height, ip_adapter_scale=0.7, guidance_scale=3.5,
                    gpu=args.gpu, implementation_sha256=revision)
    directory.mkdir(parents=True, exist_ok=False)
    run_id = directory.name
    summary = {c: dict(selected=len(rows), preserved=0, pending=0) for c in args.conditions}
    plan = dict(schema_version=1, run_id=run_id, profile=os.environ.get('MODAL_PROFILE'),
                shard_index=args.shard_index, num_shards=args.num_shards,
                workers=args.workers, conditions=args.conditions, transport_sha256=transport_digest(), jobs=[])
    bundle = directory / 'bundle.zip'
    seen_images = set()
    # Keep conditions sequential within each remote lane.
    jobs_by_condition = {c: [] for c in args.conditions}
    with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for index, row in enumerate(rows):
            image_bytes = (ROOT / row['image_path']).read_bytes()
            raw = original.read(ROOT / row['conditioning_path'])
            cached = SemanticContrastConditioning.from_dict(raw)
            cached.validate_reuse(identity_term=row['identity_term'], target_prompt=row['target_prompt'],
                                  reference=ReferenceFingerprint.from_image(image_bytes), model='gpt-5.6-sol')
            image_hash = hashlib.sha256(image_bytes).hexdigest()
            member = f'inputs/{image_hash}.bin'
            if member not in seen_images:
                archive.writestr(member, image_bytes)
                seen_images.add(member)
            for condition in args.conditions:
                cfg = dict(condition=condition, **settings)
                payload = None
                if condition in ('source_target', 'source_target_ssg'):
                    payload = replace(cached, origin='source_target').to_dict()
                signature = original.digest(dict(row=row, settings=cfg, cache=raw, image_sha256=image_hash))
                job = dict(row=row, settings=cfg, signature=signature, payload=payload,
                           image_member=member, image_sha256=image_hash)
                jobs_by_condition[condition].append(job)
                png, sidecar = original.paths_for(args.output, condition, row)
                if original.complete(png, sidecar, signature):
                    record = original.read(sidecar)
                    if record['row'] != row or record['settings'] != cfg:
                        raise ValueError(f'Local provenance mismatch: {sidecar}')
                    result_member = f'results/{condition}/{row["benchmark"]}/{row["case_id"]}'
                    archive.write(png, result_member + '.png')
                    archive.write(sidecar, result_member + '.json')
                    summary[condition]['preserved'] += 1
                else:
                    summary[condition]['pending'] += 1
            if index % 100 == 0 or index + 1 == len(rows):
                print(f'Validated {index + 1}/{len(rows)} cases', flush=True)
        plan['jobs'] = [job for condition in args.conditions for job in jobs_by_condition[condition]]
        archive.writestr('plan.json', json.dumps(plan, ensure_ascii=False))
    remote.atomic_json(directory / 'plan.json', plan)
    print(json.dumps(summary, indent=2), flush=True)
    return plan, bundle, summary


def require_profile(profile=None):
    current = os.environ.get('MODAL_PROFILE')
    if not current or (profile is not None and profile != current):
        raise ValueError(f'Set MODAL_PROFILE to {profile or "the intended workspace profile"}')
    if os.environ.get('MODAL_TOKEN_ID') or os.environ.get('MODAL_TOKEN_SECRET'):
        raise ValueError('Remove MODAL_TOKEN_ID/SECRET overrides')


def require_legacy_stopped():
    if os.name == 'nt':
        command = ("Get-CimInstance Win32_Process -Filter \"name = 'python.exe'\" | "
                   "Where-Object { $_.CommandLine -match 'benchmark_modal[.]py\\s' } | "
                   "Select-Object -ExpandProperty ProcessId")
        result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command],
                                capture_output=True, text=True, check=True,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        if result.stdout.strip():
            raise ValueError('The old benchmark_modal.py is still running. Stop both old terminals before submitting.')


def read_remote(volume, path):
    return b''.join(volume.read_file(path))


def load_receipt(path, allow_held=False):
    receipt = original.read(path)
    require_profile(receipt['profile'])
    plan = original.read(receipt['plan_path'])
    if receipt['run_id'] != plan['run_id']:
        raise ValueError('Receipt/plan mismatch')
    if not allow_held and (not plan['conditions'] or not set(plan['conditions']).issubset(remote.DEFAULT_CONDITIONS)):
        raise ValueError('This is an old three-condition plan. Run repair before resuming.')
    return receipt, plan


def spawn_lanes(args, receipt, plan, receipt_path, initialize=False):
    import modal
    require_legacy_stopped()
    if sys.version_info[:2] != (3, 11):
        raise ValueError('Use the prepared Python 3.11 environment')
    if transport_digest() != plan['transport_sha256']:
        raise ValueError('Remote transport code changed; review before resuming this plan')
    if any(j['settings']['implementation_sha256'] != original.implementation_digest() for j in plan['jobs'][:1]):
        raise ValueError('Inference implementation changed; cannot resume this plan')
    app, initializer, worker = remote.build_app(plan['workers'], plan['jobs'][0]['settings']['gpu'])
    receipt['phase'] = 'initializing' if initialize else 'resubmitting'
    remote.atomic_json(receipt_path, receipt)
    with modal.enable_output(), app.run(detach=True):
        if initialize:
            counts = initializer.remote(plan['run_id'])
            print('Uploaded results validated:', json.dumps(counts), flush=True)
        receipt['app_id'] = app.app_id
        receipt.setdefault('attempts', []).append(dict(app_id=app.app_id, calls=[]))
        remote.atomic_json(receipt_path, receipt)
        for lane in range(plan['workers']):
            receipt['phase'] = f'submitting_lane_{lane}'
            remote.atomic_json(receipt_path, receipt)
            call = worker().generate_lane.spawn(plan['run_id'], lane, receipt['profile'])
            receipt['attempts'][-1]['calls'].append(dict(lane=lane, call_id=call.object_id, submitted_at=time.time()))
            remote.atomic_json(receipt_path, receipt)
            print(f'ACCEPTED lane={lane} call={call.object_id}', flush=True)
        receipt['phase'] = 'submitted'
        remote.atomic_json(receipt_path, receipt)
    print(f'SUBMITTED: {receipt_path}', flush=True)
    print('All lanes accepted. The laptop may now be shut down.', flush=True)


def submit(args):
    import modal
    require_profile()
    require_legacy_stopped()
    if sys.version_info[:2] != (3, 11):
        raise ValueError('Use the prepared Python 3.11 environment')
    if args.receipt.exists():
        raise ValueError('Receipt already exists. Use status/resume; do not submit duplicate work.')
    directory = ROOT / 'outputs/remote_benchmark_runs' / uuid.uuid4().hex
    plan, bundle, summary = build_bundle(args, directory)
    receipt = dict(schema_version=1, run_id=plan['run_id'], profile=plan['profile'],
                   plan_path=str((directory / 'plan.json').resolve()), output=str(args.output.resolve()),
                   volume=remote.VOLUME_NAME, summary=summary, phase='uploading', attempts=[])
    remote.atomic_json(args.receipt, receipt)
    volume = modal.Volume.from_name(remote.VOLUME_NAME, create_if_missing=True)
    print(f'Uploading inputs and preserved images ({bundle.stat().st_size / 1024**2:.1f} MiB)...', flush=True)
    with volume.batch_upload() as batch:
        batch.put_file(str(bundle), f'/uploads/{plan["run_id"]}.zip')
    spawn_lanes(args, receipt, plan, args.receipt, initialize=True)


def status(args):
    import modal
    receipt, plan = load_receipt(args.receipt)
    volume = modal.Volume.from_name(receipt['volume'])
    prefix = f'runs/{plan["run_id"]}/'
    try:
        entries = volume.listdir('/' + prefix, recursive=True)
    except FileNotFoundError:
        print(json.dumps(dict(phase=receipt['phase'], message='Remote initialization not yet complete')))
        return
    names = {e.path.lstrip('/') for e in entries}
    counts = {c: dict(expected=0, committed_pairs=0) for c in plan['conditions']}
    for job in plan['jobs']:
        condition = job['settings']['condition']
        stem = f'{prefix}results/{condition}/{job["row"]["benchmark"]}/{job["row"]["case_id"]}'
        counts[condition]['expected'] += 1
        counts[condition]['committed_pairs'] += int(stem + '.png' in names and stem + '.json' in names)
    states = []
    for lane in range(plan['workers']):
        path = f'{prefix}state/lane-{lane}.json'
        if path in names:
            state = json.loads(read_remote(volume, '/' + path))
            failures = state.pop('failed', [])
            state.update(failure_count=len(failures), first_failures=failures[:2])
            states.append(state)
    calls = []
    if receipt['attempts']:
        for item in receipt['attempts'][-1]['calls']:
            entry = dict(item)
            try:
                result = modal.FunctionCall.from_id(item['call_id']).get(timeout=0)
                entry['status'] = result.get('status', 'finished') if isinstance(result, dict) else 'finished'
            except TimeoutError:
                entry['status'] = 'active'
            except modal.exception.OutputExpiredError:
                entry['status'] = 'output_expired'
            except Exception as exc:
                entry.update(status='error', detail=repr(exc))
            calls.append(entry)
    print(json.dumps(dict(run_id=plan['run_id'], profile=receipt['profile'],
                          phase=receipt['phase'], counts=counts, lanes=states, calls=calls), indent=2))
    print('Pair counts are progress indicators. Download validates signatures and PNG hashes.')


def require_finished_call(call_id):
    """A user exception also ends a call; network errors must fail closed."""
    import modal
    call = modal.FunctionCall.from_id(call_id)
    try:
        call.get(timeout=0)
    except TimeoutError:
        raise ValueError(f'Remote call is still active: {call_id}')
    except modal.exception.OutputExpiredError:
        return
    except Exception:
        from modal.call_graph import InputStatus
        graph = call.get_call_graph()
        terminal = {InputStatus.SUCCESS, InputStatus.FAILURE, InputStatus.INIT_FAILURE,
                    InputStatus.TERMINATED, InputStatus.TIMEOUT}
        if not graph or any(node.status not in terminal for node in graph):
            raise


def repaired_plan(plan, inference_digest, new_transport_digest):
    if not plan['jobs'] or any(j['settings']['implementation_sha256'] != inference_digest for j in plan['jobs']):
        raise ValueError('Inference implementation changed; repair cannot re-label old results')
    if not set(remote.DEFAULT_CONDITIONS).issubset(plan['conditions']):
        raise ValueError('Original plan does not contain both requested conditions')
    jobs = [j for c in remote.DEFAULT_CONDITIONS for j in plan['jobs'] if j['settings']['condition'] == c]
    return dict(plan, conditions=list(remote.DEFAULT_CONDITIONS), jobs=jobs, transport_sha256=new_transport_digest)


def repair(args):
    """Migrate a stopped run's metadata; retain inputs and every result byte."""
    import modal
    receipt, old_plan = load_receipt(args.receipt, allow_held=True)
    require_legacy_stopped()
    for attempt in receipt['attempts']:
        for item in attempt['calls']:
            require_finished_call(item['call_id'])
    locks = modal.Dict.from_name(remote.LOCK_NAME, create_if_missing=True)
    for lane in range(old_plan['workers']):
        key = f'{old_plan["run_id"]}:{lane}'
        lock = locks.get(key)
        if lock:
            require_finished_call(lock['call_id'])
            locks.pop(key, None)
    current_transport = transport_digest()
    plan = repaired_plan(old_plan, original.implementation_digest(), current_transport)
    if plan == old_plan:
        receipt['summary'] = {c: receipt.get('summary', {}).get(c, {}) for c in plan['conditions']}
        remote.atomic_json(args.receipt, receipt)
        print('ALREADY REPAIRED: no metadata changes')
        return
    volume = modal.Volume.from_name(receipt['volume'])
    prefix = f'/runs/{plan["run_id"]}'
    stored = json.loads(read_remote(volume, prefix + '/plan.json'))
    if original.digest(stored) != original.digest(old_plan):
        raise ValueError('Local/remote plans differ; refusing to overwrite remote metadata')
    old_path = Path(receipt['plan_path'])
    new_path = old_path.parent / f'plan.repaired-{current_transport[:12]}.json'
    remote.atomic_json(new_path, plan)
    archived_name = f'plan-before-{old_plan["transport_sha256"][:12]}.json'
    with volume.batch_upload(force=True) as batch:
        batch.put_file(io.BytesIO(json.dumps(old_plan, ensure_ascii=False).encode()), prefix + '/history/' + archived_name)
        batch.put_file(str(new_path), prefix + '/plan.json')
        for lane in range(plan['workers']):
            state = dict(lane=lane, status='ready', saved=0, skipped=0, failed=[], updated_at=time.time())
            batch.put_file(io.BytesIO(json.dumps(state).encode()), prefix + f'/state/lane-{lane}.json')
    receipt.setdefault('repair_history', []).append(dict(previous_plan=str(old_path),
        old_conditions=old_plan['conditions'], previous_transport=old_plan['transport_sha256'],
        new_transport=current_transport, repaired_at=time.time()))
    receipt.update(plan_path=str(new_path.resolve()), phase='repaired',
                   summary={c: receipt.get('summary', {}).get(c, {}) for c in plan['conditions']})
    remote.atomic_json(args.receipt, receipt)
    print(f'REPAIRED: {args.receipt}; {len(plan["jobs"])} planned images; GPU jobs not submitted')


def resume(args):
    import modal
    receipt, plan = load_receipt(args.receipt)
    # A lost receipt write during spawn is deliberately fail-closed.
    if receipt['phase'].startswith('submitting_lane_'):
        raise ValueError('Submission was interrupted during spawn. Inspect/stop its app in Modal before recovery.')
    calls = [item for attempt in receipt['attempts'] for item in attempt['calls']]
    for item in calls:
        require_finished_call(item['call_id'])
    locks = modal.Dict.from_name(remote.LOCK_NAME, create_if_missing=True)
    for lane in range(plan['workers']):
        key = f'{plan["run_id"]}:{lane}'
        lock = locks.get(key)
        if lock:
            require_finished_call(lock['call_id'])
            locks.pop(key, None)
    if not receipt['attempts']:
        volume = modal.Volume.from_name(receipt['volume'], create_if_missing=True)
        bundle = Path(receipt['plan_path']).parent / 'bundle.zip'
        with volume.batch_upload(force=True) as batch:
            batch.put_file(str(bundle), f'/uploads/{plan["run_id"]}.zip')
    spawn_lanes(args, receipt, plan, args.receipt, initialize=not receipt['attempts'])


def download_job(volume, prefix, output, job):
    png, sidecar = original.paths_for(output, job['settings']['condition'], job['row'])
    if original.complete(png, sidecar, job['signature']):
        return 'already_local'
    stem = f'{prefix}/results/{job["settings"]["condition"]}/{job["row"]["benchmark"]}/{job["row"]["case_id"]}'
    try:
        metadata = read_remote(volume, stem + '.json')
    except FileNotFoundError:
        return 'pending'
    record = json.loads(metadata)
    if (record['job_signature'] != job['signature'] or record['row'] != job['row']
            or record['settings'] != job['settings']):
        raise ValueError(f'Remote provenance mismatch: {stem}')
    data = read_remote(volume, stem + '.png')
    if hashlib.sha256(data).hexdigest() != record['png_sha256']:
        raise ValueError(f'Remote image hash mismatch: {stem}')
    from PIL import Image
    with Image.open(io.BytesIO(data)) as image:
        image.verify()
    png.parent.mkdir(parents=True, exist_ok=True)
    temporary = png.with_suffix('.png.tmp')
    temporary.write_bytes(data)
    temporary.replace(png)
    remote.atomic_json(sidecar, record)
    return 'downloaded'


def download(args):
    import modal
    from collections import Counter
    receipt, plan = load_receipt(args.receipt)
    volume = modal.Volume.from_name(receipt['volume'])
    prefix = f'/runs/{plan["run_id"]}'
    output = args.output or Path(receipt['output'])
    counts = Counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = pool.map(lambda job: download_job(volume, prefix, output, job), plan['jobs'])
        for index, result in enumerate(results, 1):
            counts[result] += 1
            if index % 100 == 0:
                print(f'{index}/{len(plan["jobs"])} {dict(counts)}', flush=True)
    print(json.dumps(dict(counts), indent=2))
    if counts['pending']:
        raise RuntimeError('Some images are still pending. Run status; download can be repeated safely.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('plan', 'submit'):
        command = sub.add_parser(name)
        command.add_argument('--conditions', nargs='+', choices=remote.DEFAULT_CONDITIONS, default=list(remote.DEFAULT_CONDITIONS))
        command.add_argument('--benchmark', choices=('all', 'dreambench', 'dreambench_plus'), default='all')
        command.add_argument('--shard-index', type=int, required=True)
        command.add_argument('--num-shards', type=int, default=2)
        command.add_argument('--workers', type=int, default=3)
        command.add_argument('--gpu', default='A100-80GB')
        command.add_argument('--strength', type=float, default=1.0)
        command.add_argument('--seed', type=int, default=42)
        command.add_argument('--steps', type=int, default=24)
        command.add_argument('--width', type=int, default=1024)
        command.add_argument('--height', type=int, default=1024)
        command.add_argument('--output', type=Path, default=original.OUTPUT)
        command.add_argument('--receipt', type=Path, default=ROOT / 'outputs/remote_benchmark_receipt.json')
    for name in ('status', 'resume', 'download', 'repair'):
        command = sub.add_parser(name)
        command.add_argument('--receipt', type=Path, required=True)
        if name == 'download':
            command.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.command in ('plan', 'submit'):
        if (args.workers < 1 or args.steps < 1 or min(args.width, args.height) < 16
                or not 0 <= args.strength <= 1 or len(set(args.conditions)) != len(args.conditions)):
            parser.error('Invalid workers/steps/dimensions/strength/duplicate conditions')
    if args.command == 'plan':
        directory = ROOT / 'outputs/remote_benchmark_runs' / ('preview-' + uuid.uuid4().hex)
        build_bundle(args, directory)
        print(f'Offline preview bundle: {directory}')
    else:
        globals()[args.command](args)


if __name__ == '__main__':
    main()
