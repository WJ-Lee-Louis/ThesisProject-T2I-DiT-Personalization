"""Source-defined Modal metric functions; no serialized SDK wrapper closures."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import time
import traceback
import zipfile

import modal

ROOT = Path(__file__).resolve().parent
VOLUME = 'orthogonalization-dreambench-metrics-v1'
LOCKS = VOLUME + '-locks'
MANIFEST = 'outputs/semantic_benchmark_gpt56_sol/generation_manifest.json'
CODE_FILES = ('benchmark_metric_worker.py', 'scripts/benchmark_metric_remote.py',
              'scripts/benchmark_metrics.py', 'requirements-benchmark-metrics.txt')
MODEL_CACHE = '/metric_models'


def code_digest():
    return hashlib.sha256(b''.join(hashlib.sha256((ROOT / p).read_bytes()).digest() for p in CODE_FILES)).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_extract(archive_path, target):
    target = Path(target).resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            p = PurePosixPath(member.filename)
            if (p.is_absolute() or '..' in p.parts or '\\' in member.filename or ':' in member.filename
                    or member.is_dir() or ((member.external_attr >> 16) & 0o170000) == 0o120000
                    or not target.joinpath(*p.parts).resolve().is_relative_to(target)):
                raise ValueError('Unsafe archive member: ' + member.filename)
        archive.extractall(target)


def load_metrics(input_root):
    script_root = str(ROOT / 'scripts')
    if script_root not in sys.path:
        sys.path.insert(0, script_root)
    import benchmark_metrics as metrics
    metrics.ROOT = Path(input_root)
    return metrics


def stage_local_inputs(run_id, archive_sha256, remote_base='/evaluation/uploads', local_base='/tmp/dreambench-metric-inputs'):
    """Stream one archive off the Volume; avoid thousands of remote small-file reads."""
    target = Path(local_base) / run_id
    marker = target / 'staged.sha256'
    if marker.exists() and marker.read_text(encoding='utf-8') == archive_sha256:
        return target / 'inputs'
    target.mkdir(parents=True, exist_ok=True)
    archive = target / 'inputs.zip'
    shutil.copyfile(Path(remote_base) / (run_id + '.zip'), archive)
    if file_sha(archive) != archive_sha256:
        raise ValueError('Local staging archive hash mismatch')
    safe_extract(archive, target / 'inputs')
    marker.write_text(archive_sha256, encoding='utf-8')
    return target / 'inputs'


def run_root(run_id):
    if not re.fullmatch('[a-f0-9]{32}', run_id):
        raise ValueError('Invalid metric run ID')
    return Path('/evaluation/runs') / run_id


CODE_DIGEST = os.environ.get('DREAMBENCH_METRIC_CODE_DIGEST') or code_digest()
volume = modal.Volume.from_name(VOLUME, create_if_missing=True)
cache = modal.Volume.from_name(VOLUME + '-models', create_if_missing=True)
base = (modal.Image.debian_slim(python_version='3.11').pip_install('Pillow==11.3.0')
        .env({'DREAMBENCH_METRIC_CODE_DIGEST': CODE_DIGEST, 'PYTHONUNBUFFERED': '1'})
        .add_local_file(ROOT / 'scripts/benchmark_metrics.py', '/root/scripts/benchmark_metrics.py', copy=True)
        .add_local_file(ROOT / 'requirements-benchmark-metrics.txt', '/root/requirements-benchmark-metrics.txt', copy=True))
gpu_image = (base.apt_install('git')
             .pip_install_from_requirements(ROOT / 'requirements-benchmark-metrics.txt')
             .env({'TORCH_HOME': MODEL_CACHE + '/torch', 'CLIP_DOWNLOAD_ROOT': MODEL_CACHE + '/clip'})
             .run_commands('test ! -e /metric_models'))
app = modal.App('orthogonalization-dreambench-metrics-v1')


@app.function(image=base, volumes={'/evaluation': volume}, timeout=3600, memory=4096)
def initialize(run_id, account, archive_sha256, implementation_sha256):
    if account not in ('A', 'B') or implementation_sha256 != CODE_DIGEST:
        raise ValueError('Account/code mismatch')
    volume.reload()
    root = run_root(run_id)
    metrics = load_metrics(root / 'inputs')
    configuration = dict(run_id=run_id, account=account, archive_sha256=archive_sha256,
                         implementation_sha256=implementation_sha256, protocol=metrics.PROTOCOL)
    ready = root / 'ready.json'
    if ready.exists():
        if metrics.read(ready)['configuration'] != configuration:
            raise ValueError('Existing metric input configuration differs')
        return metrics.read(ready)
    archive = Path('/evaluation/uploads') / (run_id + '.zip')
    if file_sha(archive) != archive_sha256:
        raise ValueError('Uploaded archive hash mismatch')
    safe_extract(archive, root / 'inputs')
    plan = metrics.prepare(root / 'inputs' / MANIFEST, root / 'inputs/outputs/benchmark_images_v1')
    metrics.validate_pairs(plan, account)
    records = metrics.select_records(plan, account)
    expected = 2700 if account == 'A' else 1500
    if len(records) != expected:
        raise ValueError('Wrong evaluation case count')
    result = dict(configuration=configuration, expected=expected, validated_at=time.time())
    metrics.write(ready, result)
    volume.commit()
    return result


@app.function(image=gpu_image, gpu='T4', max_containers=1, min_containers=0,
              scaledown_window=2, timeout=7200, startup_timeout=1800, retries=0,
              memory=8192, volumes={'/evaluation': volume, MODEL_CACHE: cache})
def run_metrics(run_id, account, implementation_sha256, pilot_only=False, batch_size=32):
    if implementation_sha256 != CODE_DIGEST or batch_size != 32:
        raise ValueError('Metric code/batch configuration differs')
    volume.reload()
    root = run_root(run_id)
    metrics = load_metrics(root / 'inputs')
    ready = metrics.read(root / 'ready.json')
    if ready['configuration']['account'] != account or ready['configuration']['implementation_sha256'] != CODE_DIGEST:
        raise ValueError('Unvalidated metric input configuration')
    locks = modal.Dict.from_name(LOCKS, create_if_missing=True)
    key = run_id + ':' + account
    owner = modal.current_function_call_id()
    if not locks.put(key, dict(call_id=owner), skip_if_exists=True):
        raise RuntimeError('Metric run already locked; inspect status before retrying')
    state = dict(status='pilot_running' if pilot_only else 'running', account=account,
                 expected=ready['expected'], updated_at=time.time(), call_id=owner, failure=None)

    def commit(result=None):
        if result is not None:
            state['scored'] = len(result['scores'])
        state['updated_at'] = time.time()
        metrics.write(root / 'state.json', state)
        volume.commit()

    try:
        commit()
        print('STAGING_INPUTS: streaming input archive to worker temporary disk', flush=True)
        input_root = stage_local_inputs(run_id, ready['configuration']['archive_sha256'])
        metrics = load_metrics(input_root)
        print('INPUTS_STAGED: validation and scoring now use local worker files', flush=True)
        plan = metrics.prepare(input_root / MANIFEST, input_root / 'outputs/benchmark_images_v1')
        if not pilot_only:
            prior = metrics.read(root / 'checkpoint.json')
            pilot_keys = {(r['condition'], r['row']['case_id']) for r in metrics.select_records(plan, account, True)}
            if not pilot_keys.issubset({(r['condition'], r['case_id']) for r in prior['scores']}):
                raise ValueError('A successful class-covering pilot is required before full evaluation')
        result = metrics.evaluate(plan, root / 'scores.json', 'cuda', batch_size, account,
                                  checkpoint=root / 'checkpoint.json', commit=commit, pilot_only=pilot_only)
        state['status'] = 'pilot_completed' if pilot_only else 'completed'
        cache.commit()
        commit(result)
        print(('PILOT_COMPLETE' if pilot_only else 'EVALUATION_COMPLETE') + f" {len(result['scores'])}/{ready['expected']}", flush=True)
        return state
    except Exception:
        state.update(status='failed', failure=traceback.format_exc())
        commit()
        raise
    finally:
        locks.pop(key, None)
