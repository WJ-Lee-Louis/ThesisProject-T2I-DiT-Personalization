"""Metric transport safety, class-covering pilots and interrupted score checkpoints."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

import benchmark_metric_worker as worker

SPEC = importlib.util.spec_from_file_location('metric_remote_client', Path(__file__).resolve().parents[1] / 'scripts/benchmark_metric_remote.py')
client = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(client)
metrics = client.metrics


def pair(tmp_path, ref, condition, case='case'):
    image = tmp_path / (condition + '-' + case + '.png')
    image.write_bytes(b'unchanged-generated-image')
    row = dict(reference_id=ref, benchmark=ref.split('/')[0], case_id=case, target_prompt='original target')
    return dict(row=row, condition=condition, category=metrics.category(row), reference_sha256='reference', generated=str(image))


def score(record):
    return dict(record['row'], condition=record['condition'], category=record['category'],
                reference_sha256=record['reference_sha256'], generated_sha256=metrics.sha(record['generated']),
                **{m: .5 for m in metrics.METRICS})


def test_pilot_covers_classes_and_paired_methods(tmp_path):
    records = [pair(tmp_path, ref, condition, f'{i}-{prompt}')
               for i, ref in enumerate(('dreambench_plus/live_subject/animal/00',
                                         'dreambench_plus/live_subject/human/00',
                                         'dreambench_plus/object/00', 'dreambench_plus/style/00'))
               for prompt in range(2) for condition in metrics.CONDITIONS]
    selected = metrics.select_records(dict(records=records), 'A', True)
    assert len(selected) == 8
    assert {r['category'] for r in selected} == {'animal', 'human', 'object', 'style'}
    assert all(r['row']['case_id'].endswith('-0') for r in selected)


def test_checkpoint_reuses_only_completed_valid_cases(tmp_path):
    records = [pair(tmp_path, 'dreambench/cat', c) for c in metrics.CONDITIONS]
    environment = dict(device='cuda', device_name='Tesla T4', versions={'torch': '2.7.1'})
    checkpoint = tmp_path / 'checkpoint.json'
    metrics.write(checkpoint, dict(protocol=metrics.PROTOCOL, **environment, scores=[score(records[0])]))
    restored = metrics.cached_scores(checkpoint, records, environment)
    assert len(restored) == 1
    assert restored[0]['condition'] == 'baseline'
    # An interruption before the next batch commit leaves the previous batch recoverable.
    checkpoint.with_suffix('.json.tmp').write_text('incomplete', encoding='utf-8')
    assert metrics.cached_scores(checkpoint, records, environment) == restored
    Path(records[0]['generated']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='Checkpoint image'):
        metrics.cached_scores(checkpoint, records, environment)


def test_checkpoint_rejects_mixed_environment_and_duplicates(tmp_path):
    record = pair(tmp_path, 'dreambench/cat', 'baseline')
    env = dict(device='cuda', device_name='Tesla T4', versions={})
    checkpoint = tmp_path / 'checkpoint.json'
    metrics.write(checkpoint, dict(protocol=metrics.PROTOCOL, **env, scores=[score(record)] * 2))
    with pytest.raises(ValueError, match='duplicate'):
        metrics.cached_scores(checkpoint, [record], env)
    with pytest.raises(ValueError, match='device'):
        metrics.cached_scores(checkpoint, [record], dict(env, device_name='different'))


@pytest.mark.parametrize('member', ['../escape', '/absolute', 'C:/escape', 'dir\\..\\escape'])
def test_input_archive_rejects_unsafe_paths(tmp_path, member):
    archive = tmp_path / 'bad.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr(member, b'bad')
    with pytest.raises(ValueError, match='Unsafe'):
        worker.safe_extract(archive, tmp_path / 'inputs')


def test_failure_is_committed_and_lock_released(tmp_path, monkeypatch):
    metrics.write(tmp_path / 'ready.json', dict(expected=2, configuration=dict(account='B', implementation_sha256=worker.CODE_DIGEST, archive_sha256='archive')))
    commits = []
    locks = {}

    class Lock:
        def put(self, key, value, skip_if_exists):
            if key in locks:
                return False
            locks[key] = value
            return True

        def pop(self, key, default=None):
            return locks.pop(key, default)

    def fail(*args, **kwargs):
        raise RuntimeError('test model failure')

    fake_metrics = SimpleNamespace(read=metrics.read, write=metrics.write, prepare=lambda *a: {}, evaluate=fail)
    monkeypatch.setattr(worker, 'load_metrics', lambda path: fake_metrics)
    monkeypatch.setattr(worker, 'run_root', lambda run: tmp_path)
    monkeypatch.setattr(worker, 'stage_local_inputs', lambda *a: tmp_path / 'inputs')
    monkeypatch.setattr(worker, 'volume', SimpleNamespace(reload=lambda: None, commit=lambda: commits.append(True)))
    monkeypatch.setattr(worker.modal.Dict, 'from_name', lambda *a, **kw: Lock())
    monkeypatch.setattr(worker.modal, 'current_function_call_id', lambda: 'test-call')
    with pytest.raises(RuntimeError, match='test model failure'):
        worker.run_metrics.local('a' * 32, 'B', worker.CODE_DIGEST, pilot_only=True)
    state = metrics.read(tmp_path / 'state.json')
    assert state['status'] == 'failed'
    assert 'test model failure' in state['failure']
    assert len(commits) == 2
    assert not locks


def test_cli_profile_guard_and_source_defined_functions(monkeypatch):
    monkeypatch.setenv('MODAL_PROFILE', 'modal-research-2')
    with pytest.raises(ValueError, match='wj-lee-louis'):
        client.require_profile('A')
    # Production functions are top-level source functions, not serialized SDK closures.
    assert worker.run_metrics.get_raw_f().__module__ == 'benchmark_metric_worker'
    assert worker.initialize.get_raw_f().__module__ == 'benchmark_metric_worker'


@pytest.mark.parametrize('has_scores', [False, True])
def test_mount_repair_preserves_input_and_refuses_existing_scores(tmp_path, monkeypatch, has_scores):
    old = '20ce247c84e933ac9568b52e4e06d701cae3c43197657e205e5818e3525a68ce'
    configuration = dict(run_id='a' * 32, account='A', archive_sha256='unchanged-archive',
                         implementation_sha256=old, protocol=metrics.PROTOCOL)
    ready = dict(configuration=configuration, expected=2700)
    remote_files = {'/runs/' + 'a' * 32 + '/ready.json': __import__('json').dumps(ready).encode()}
    if has_scores:
        remote_files['/runs/' + 'a' * 32 + '/checkpoint.json'] = b'{}'

    class Volume:
        def read_file(self, path):
            if path not in remote_files:
                raise FileNotFoundError(path)
            return [remote_files[path]]

        def batch_upload(self, **kwargs):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def put_file(self, source, path):
            remote_files[path] = source.read()

    receipt = dict(configuration, profile='wj-lee-louis', volume='test', phase='pilot_submitted',
                   attempts=[dict(call_id='ended-call')])
    for a in ('A', 'B'):
        metrics.write(tmp_path / (a + '.json'), receipt if a == 'A' else dict(implementation_sha256='new-code'))
    metrics.write(tmp_path / 'preparation.json', dict(implementation_sha256=old))
    monkeypatch.setattr(client, 'OUTPUT', tmp_path)
    monkeypatch.setattr(client, 'receipt_path', lambda a: tmp_path / (a + '.json'))
    monkeypatch.setattr(client, 'load_receipt', lambda a: metrics.read(tmp_path / (a + '.json')))
    monkeypatch.setattr(client, 'worker', lambda: SimpleNamespace(code_digest=lambda: 'new-code'))
    monkeypatch.setattr(client, 'call_state', lambda call: ('failed', None))
    monkeypatch.setattr(worker.modal.Volume, 'from_name', lambda *a, **k: Volume())
    if has_scores:
        with pytest.raises(ValueError, match='scores already exist'):
            client.repair('A')
        assert metrics.read(tmp_path / 'A.json')['implementation_sha256'] == old
    else:
        client.repair('A')
        updated = __import__('json').loads(remote_files['/runs/' + 'a' * 32 + '/ready.json'])
        assert updated['configuration']['archive_sha256'] == 'unchanged-archive'
        assert updated['configuration']['implementation_sha256'] == 'new-code'
        assert metrics.read(tmp_path / 'A.json')['attempts'] == receipt['attempts']


def test_archive_is_staged_once_then_reused_locally(tmp_path):
    remote = tmp_path / 'remote'
    remote.mkdir()
    run = 'b' * 32
    archive = remote / (run + '.zip')
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('reference.png', b'original-image')
    digest = worker.file_sha(archive)
    local = tmp_path / 'local'
    root = worker.stage_local_inputs(run, digest, remote, local)
    assert (root / 'reference.png').read_bytes() == b'original-image'
    archive.rename(remote / 'temporarily-unavailable.zip')
    assert worker.stage_local_inputs(run, digest, remote, local) == root
