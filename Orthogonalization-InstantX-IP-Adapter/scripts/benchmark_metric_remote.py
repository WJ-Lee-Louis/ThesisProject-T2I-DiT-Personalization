"""CPU prepare, then detached Modal CLIP/DINO pilot + evaluation, status and download."""
from __future__ import annotations

import argparse
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
sys.path.insert(0, str(ROOT / 'scripts'))
import benchmark_metrics as metrics

PROFILES = {'A': 'wj-lee-louis', 'B': 'modal-research-2'}
OUTPUT = ROOT / 'outputs/benchmark_metrics_v1'


def worker():
    import benchmark_metric_worker
    return benchmark_metric_worker


def receipt_path(account):
    return ROOT / 'outputs' / ('metric_receipt_' + account + '.json')


def require_profile(account):
    if os.environ.get('MODAL_PROFILE') != PROFILES[account]:
        raise ValueError('Set MODAL_PROFILE=' + PROFILES[account])
    if os.environ.get('MODAL_TOKEN_ID') or os.environ.get('MODAL_TOKEN_SECRET'):
        raise ValueError('Remove MODAL_TOKEN_ID/MODAL_TOKEN_SECRET overrides')


def load_receipt(account):
    require_profile(account)
    receipt = metrics.read(receipt_path(account))
    if receipt['account'] != account or receipt['profile'] != PROFILES[account]:
        raise ValueError('Metric receipt account/profile mismatch')
    return receipt


def read_remote(volume, path):
    return b''.join(volume.read_file(path))


def build_bundle(plan, account, destination):
    """Keep full metadata/references, but only the assigned generated PNG/JSON pairs."""
    files = {ROOT / 'outputs/semantic_benchmark_gpt56_sol/generation_manifest.json'}
    for record in plan['records']:
        files.add(Path(record['reference']))
        files.add(ROOT / record['row']['conditioning_path'])
    for record in metrics.select_records(plan, account):
        image = Path(record['generated'])
        files.update((image, image.with_suffix('.json')))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(ROOT).as_posix())
    temporary.replace(destination)


def prepare():
    if sys.version_info[:2] != (3, 11):
        raise ValueError('Use the prepared Python 3.11 environment')
    code = worker().code_digest()
    marker = OUTPUT / 'preparation.json'
    if marker.exists():
        preparation = metrics.read(marker)
        if preparation['implementation_sha256'] != code:
            raise ValueError('Metric implementation changed since preparation; review existing metric runs before replacing them')
        run_id = preparation['run_id']
        if all(receipt_path(a).exists() and Path(metrics.read(receipt_path(a))['bundle']).exists() for a in PROFILES):
            print('PREPARED: existing input bundles are ready. Continue with start --account A/B.', flush=True)
            return
    else:
        run_id = uuid.uuid4().hex
        metrics.write(marker, dict(run_id=run_id, implementation_sha256=code, phase='fetching'))
    # A separate interpreter per profile prevents Modal client credential caching across workspaces.
    for account, profile in PROFILES.items():
        source_receipt = ROOT / 'outputs' / ('remote_receipt_' + account + '.json')
        if metrics.read(source_receipt)['profile'] != profile:
            raise ValueError('Source generation profile differs from the configured account')
        environment = dict(os.environ, MODAL_PROFILE=profile, PYTHONUTF8='1')
        environment.pop('MODAL_TOKEN_ID', None)
        environment.pop('MODAL_TOKEN_SECRET', None)
        print(f'CPU FETCH {account}: preserving valid local images; downloading missing committed pairs...', flush=True)
        subprocess.run([sys.executable, str(ROOT / 'scripts/benchmark_remote.py'), 'download',
                        '--receipt', str(source_receipt)], env=environment, cwd=ROOT, check=True)
    plan = metrics.prepare(ROOT / 'outputs/semantic_benchmark_gpt56_sol/generation_manifest.json',
                           ROOT / 'outputs/benchmark_images_v1')
    metrics.validate_pairs(plan)
    metrics.write(OUTPUT / 'pairs.json', plan)
    for account, profile in PROFILES.items():
        bundle = OUTPUT / 'runs' / run_id / ('inputs_' + account + '.zip')
        print(f'CPU PACKAGE {account}: {len(metrics.select_records(plan, account))} generated images', flush=True)
        build_bundle(plan, account, bundle)
        receipt = dict(run_id=run_id, account=account, profile=profile, bundle=str(bundle),
                       archive_sha256=worker().file_sha(bundle), implementation_sha256=code,
                       phase='prepared', volume=worker().VOLUME, attempts=[],
                       expected=len(metrics.select_records(plan, account)))
        metrics.write(receipt_path(account), receipt)
    metrics.write(marker, dict(run_id=run_id, implementation_sha256=code, phase='prepared'))
    print('PREPARED: validated 4200 PNG/JSON pairs; A=2700, B=1500. No GPU evaluation submitted.', flush=True)


def call_state(call_id):
    import modal
    call = modal.FunctionCall.from_id(call_id)
    try:
        return 'completed', call.get(timeout=0)
    except TimeoutError:
        return 'active', None
    except modal.exception.OutputExpiredError:
        return 'expired', None
    except Exception as exc:
        from modal.call_graph import InputStatus
        terminal = {InputStatus.SUCCESS, InputStatus.FAILURE, InputStatus.INIT_FAILURE,
                    InputStatus.TERMINATED, InputStatus.TIMEOUT}
        graph = call.get_call_graph()
        if not graph or any(node.status not in terminal for node in graph):
            raise  # A network error is not evidence that resubmission is safe.
        return 'failed', repr(exc)


def repair(account):
    """One audited cache-mount repair; never reuse numerical scores across changed code."""
    import modal
    receipt = load_receipt(account)
    implementation = worker()
    current = implementation.code_digest()
    previous = receipt['implementation_sha256']
    if previous == current:
        print(f'REPAIRED {account}: already using current code; inputs preserved.', flush=True)
        return
    if previous not in ('20ce247c84e933ac9568b52e4e06d701cae3c43197657e205e5818e3525a68ce',
                        '5260fe7116c56d72d9764f7d7d692cc5b4c30d9afa6e45f39849aff6903791be'):
        raise ValueError('Repair only supports the known pre-evaluation cache-mount failure')
    for attempt in receipt['attempts']:
        if call_state(attempt['call_id'])[0] == 'active':
            raise ValueError('Stop the old metric app before repair')
    target = modal.Volume.from_name(receipt['volume'])
    root = '/runs/' + receipt['run_id']
    for name in ('checkpoint.json', 'scores.json'):
        try:
            read_remote(target, root + '/' + name)
        except FileNotFoundError:
            pass
        else:
            raise ValueError('Numerical scores already exist; do not silently migrate their implementation')
    ready = json.loads(read_remote(target, root + '/ready.json'))
    expected = dict(run_id=receipt['run_id'], account=account, archive_sha256=receipt['archive_sha256'],
                    implementation_sha256=previous, protocol=metrics.PROTOCOL)
    if ready['configuration'] != expected:
        raise ValueError('Remote input provenance differs; repair refused')
    history = dict(previous=previous, current=current, reason='Separate model cache mount and stage unchanged archive on worker disk', at=time.time())
    ready['configuration']['implementation_sha256'] = current
    ready.setdefault('repair_history', []).append(history)
    with target.batch_upload(force=True) as upload:
        upload.put_file(io.BytesIO(json.dumps(ready, ensure_ascii=False, indent=2).encode()), root + '/ready.json')
    receipt['implementation_sha256'] = current
    receipt['phase'] = 'prepared'
    receipt.setdefault('repair_history', []).append(history)
    metrics.write(receipt_path(account), receipt)
    if all(metrics.read(receipt_path(a))['implementation_sha256'] == current for a in PROFILES):
        preparation = metrics.read(OUTPUT / 'preparation.json')
        preparation['implementation_sha256'] = current
        metrics.write(OUTPUT / 'preparation.json', preparation)
    print(f'REPAIRED {account}: validated remote inputs preserved; no archive reupload needed.', flush=True)


def start(account, pilot_only=False):
    import modal
    receipt = load_receipt(account)
    implementation = worker()
    if sys.version_info[:2] != (3, 11) or receipt['implementation_sha256'] != implementation.code_digest():
        raise ValueError('Use Python 3.11 and the unchanged prepared metric implementation')
    if receipt['phase'].startswith('spawning_'):
        raise ValueError(f'Uncertain interrupted submission. Inspect app {receipt.get("app_id")} before retrying; do not duplicate GPU work.')
    for attempt in receipt['attempts']:
        state, _ = call_state(attempt['call_id'])
        if state == 'active':
            print(f'ALREADY_ACTIVE {attempt["mode"]} {attempt["call_id"]}; use status. If this is a pilot, rerun start after it completes.', flush=True)
            return
    target = modal.Volume.from_name(receipt['volume'], create_if_missing=True)
    root = '/runs/' + receipt['run_id']
    try:
        ready = json.loads(read_remote(target, root + '/ready.json'))
    except FileNotFoundError:
        ready = None
    if ready is None:
        bundle = Path(receipt['bundle'])
        if implementation.file_sha(bundle) != receipt['archive_sha256']:
            raise ValueError('Prepared input archive changed')
        receipt['phase'] = 'uploading'
        metrics.write(receipt_path(account), receipt)
        print(f'CPU UPLOAD {account}: {bundle.stat().st_size / 1024**2:.1f} MiB; keep this terminal open.', flush=True)
        with target.batch_upload(force=True) as upload:
            upload.put_file(str(bundle), '/uploads/' + receipt['run_id'] + '.zip')
    try:
        state = json.loads(read_remote(target, root + '/state.json'))
    except FileNotFoundError:
        state = {}
    if state.get('status') == 'completed':
        print('EVALUATION_COMPLETE: all assigned case scores are already saved. Use download.', flush=True)
        return
    locks = modal.Dict.from_name(implementation.LOCKS, create_if_missing=True)
    key = receipt['run_id'] + ':' + account
    previous = locks.get(key)
    if previous:
        if call_state(previous['call_id'])[0] == 'active':
            raise ValueError('A metric worker still owns this run')
        locks.pop(key, None)
    with modal.enable_output(), implementation.app.run(detach=True):
        receipt['app_id'] = implementation.app.app_id
        receipt['phase'] = 'validating'
        metrics.write(receipt_path(account), receipt)
        result = implementation.initialize.remote(receipt['run_id'], account, receipt['archive_sha256'], receipt['implementation_sha256'])
        print(f'INPUTS_VALIDATED {result["expected"]}; GPU pilot/evaluation can now run.', flush=True)
        pilot_done = state.get('status') == 'pilot_completed' or any(
            a.get('pilot_passed') for a in receipt['attempts'])
        if pilot_only and pilot_done:
            print('PILOT_COMPLETE: existing successful pilot will be reused by start.', flush=True)
            return
        modes = ['pilot'] if pilot_only else (['full'] if pilot_done else ['pilot', 'full'])
        for mode in modes:
            receipt['phase'] = 'spawning_' + mode
            metrics.write(receipt_path(account), receipt)
            call = implementation.run_metrics.spawn(receipt['run_id'], account, receipt['implementation_sha256'],
                                                     pilot_only=mode == 'pilot', batch_size=32)
            attempt = dict(call_id=call.object_id, mode=mode, app_id=implementation.app.app_id, submitted_at=time.time())
            receipt['attempts'].append(attempt)
            receipt['phase'] = mode + '_submitted'
            metrics.write(receipt_path(account), receipt)
            if mode == 'pilot':
                print('PILOT_SUBMITTED: waiting for class-covering pilot; keep this terminal open.', flush=True)
                result = call.get()
                if result['status'] != 'pilot_completed':
                    raise RuntimeError('Pilot did not pass; full evaluation was not submitted')
                attempt['pilot_passed'] = True
                receipt['phase'] = 'pilot_completed'
                metrics.write(receipt_path(account), receipt)
                print('PILOT_COMPLETE: checkpoint preserved and reused by full evaluation.', flush=True)
            else:
                print(f'SUBMITTED {account}: {call.object_id}', flush=True)
    if pilot_only:
        print('PILOT_COMPLETE: full evaluation was not submitted. Use start when ready.', flush=True)
    else:
        print('DETACHED: this account no longer needs the local terminal/laptop. Check both accounts before shutdown.', flush=True)


def status(account):
    import modal
    receipt = load_receipt(account)
    volume = modal.Volume.from_name(receipt['volume'], create_if_missing=True)
    try:
        state = json.loads(read_remote(volume, '/runs/' + receipt['run_id'] + '/state.json'))
    except FileNotFoundError:
        state = dict(status='not_started', scored=0)
    call = None
    if receipt['attempts']:
        latest = receipt['attempts'][-1]
        current, detail = call_state(latest['call_id'])
        call = dict(latest, status=current, detail=detail)
    print(json.dumps(dict(account=account, expected=receipt['expected'], phase=receipt['phase'],
                          state=state, latest_call=call), ensure_ascii=False, indent=2))


def download(account):
    import modal
    receipt = load_receipt(account)
    volume = modal.Volume.from_name(receipt['volume'])
    root = '/runs/' + receipt['run_id']
    state = json.loads(read_remote(volume, root + '/state.json'))
    if state['status'] != 'completed' or state['scored'] != receipt['expected']:
        raise ValueError('Evaluation is not complete; use status/start, not partial download')
    result = json.loads(read_remote(volume, root + '/scores.json'))
    plan = metrics.read(OUTPUT / 'pairs.json')
    environment = {k: result[k] for k in ('device', 'device_name', 'versions')}
    output = OUTPUT / ('scores_' + account + '.json')
    # Reuse checkpoint guards to validate every returned case against local staged image hashes.
    temporary = OUTPUT / ('download_' + account + '.json')
    metrics.write(temporary, result)
    scores = metrics.cached_scores(temporary, metrics.select_records(plan, account), environment)
    if len(scores) != receipt['expected']:
        raise ValueError('Downloaded result has missing cases')
    if output.exists() and metrics.read(output) != result:
        raise ValueError('Different local score file already exists; preserve it before replacing')
    metrics.write(output, result)
    print(f'DOWNLOADED {account}: {len(scores)} case scores -> {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'repair', 'pilot', 'start', 'status', 'download'))
    parser.add_argument('--account', choices=('A', 'B'))
    args = parser.parse_args()
    if args.command != 'prepare' and args.account is None:
        parser.error('--account A/B is required')
    if args.command == 'prepare':
        prepare()
    elif args.command == 'pilot':
        start(args.account, pilot_only=True)
    else:
        globals()[args.command](args.account)


if __name__ == '__main__':
    main()
