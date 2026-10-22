"""Durable checkpoints, migration compatibility, and interrupted downloads."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile
import sys

import pytest
from PIL import Image

import benchmark_remote_worker as worker

SPEC = importlib.util.spec_from_file_location('durable_client', Path(__file__).resolve().parents[1] / 'scripts/benchmark_remote.py')
client = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(client)


def fixture_plan(tmp_path, n=2):
    image = b'reference-test'
    (tmp_path / 'inputs').mkdir()
    (tmp_path / 'inputs/ref.bin').write_bytes(image)
    jobs = [dict(row=dict(benchmark='dreambench', case_id=f'000-{i:02}'),
                 settings=dict(condition='source_target_ssg'), signature=f'sig-{i}',
                 image_member='inputs/ref.bin', image_sha256=hashlib.sha256(image).hexdigest(), payload=None)
            for i in range(n)]
    return dict(run_id='test-run', workers=1, conditions=['source_target_ssg'], jobs=jobs)


def result(job):
    buffer = io.BytesIO()
    Image.new('RGB', (16, 16), 'red').save(buffer, format='PNG')
    data = buffer.getvalue()
    return dict(png=data, row=job['row'], settings=job['settings'],
                job_signature=job['signature'], png_sha256=hashlib.sha256(data).hexdigest(), diagnostics={})


def test_failed_job_resumes_without_regenerating_committed_image(tmp_path):
    plan = fixture_plan(tmp_path)
    calls, commits = [], []

    def first_pass(job):
        calls.append(job['signature'])
        if job['signature'] == 'sig-1':
            raise RuntimeError('GPU failure')
        return result(job)

    state = worker.run_lane(tmp_path, plan, 0, first_pass, lambda: commits.append(True), 'profile-A')
    assert state['status'] == 'partial'
    assert state['saved'] == 1
    png, sidecar = worker.result_paths(tmp_path, plan['jobs'][0])
    before = (png.read_bytes(), sidecar.read_bytes())
    calls.clear()
    state = worker.run_lane(tmp_path, plan, 0, lambda j: calls.append(j['signature']) or result(j),
                            lambda: None, 'profile-A')
    assert calls == ['sig-1']
    assert state['skipped'] == 1 and state['saved'] == 1 and state['status'] == 'completed'
    assert (png.read_bytes(), sidecar.read_bytes()) == before
    assert len(commits) >= 3


def test_lane_lock_blocks_duplicate_work(tmp_path):
    plan = fixture_plan(tmp_path)
    with pytest.raises(RuntimeError, match='already claimed'):
        worker.run_lane(tmp_path, plan, 0, lambda j: pytest.fail('duplicate generation'),
                        lambda: None, 'A', claim=lambda: False)


def test_interrupted_commit_stops_and_releases_lock(tmp_path):
    plan = fixture_plan(tmp_path)
    released = []
    with pytest.raises(OSError, match='commit failure'):
        worker.run_lane(tmp_path, plan, 0, result,
                        lambda: (_ for _ in ()).throw(OSError('commit failure')),
                        'A', claim=lambda: True, release=lambda: released.append(True))
    assert released == [True]


def test_corrupt_existing_result_is_not_silently_regenerated(tmp_path):
    plan = fixture_plan(tmp_path, n=1)
    job = plan['jobs'][0]
    worker.save_result(tmp_path, job, result(job), 'A', 'test-run')
    png, _ = worker.result_paths(tmp_path, job)
    png.write_bytes(b'corruption')
    with pytest.raises(ValueError, match='hash mismatch'):
        worker.run_lane(tmp_path, plan, 0, lambda j: pytest.fail('must stop'), lambda: None, 'A')


def test_bundle_rejects_path_traversal(tmp_path):
    bundle = tmp_path / 'bad.zip'
    with zipfile.ZipFile(bundle, 'w') as archive:
        archive.writestr('../escape.txt', 'bad')
    with pytest.raises(ValueError, match='Unsafe'):
        worker.extract_bundle(bundle, tmp_path / 'out')
    assert not (tmp_path / 'escape.txt').exists()


def test_download_preserves_existing_files_and_checks_remote_hash(tmp_path):
    plan = fixture_plan(tmp_path, n=1)
    job = plan['jobs'][0]
    worker.save_result(tmp_path, job, result(job), 'A', 'test-run')
    png, meta = worker.result_paths(tmp_path, job)
    data = {'/run/results/source_target_ssg/dreambench/000-00.png': png.read_bytes(),
            '/run/results/source_target_ssg/dreambench/000-00.json': meta.read_bytes()}
    volume = SimpleNamespace(read_file=lambda path: iter([data[path]]))
    output = tmp_path / 'download'
    assert client.download_job(volume, '/run', output, job) == 'downloaded'
    before = (output / 'source_target_ssg/dreambench/000-00.json').read_bytes()
    assert client.download_job(None, '/run', output, job) == 'already_local'
    assert (output / 'source_target_ssg/dreambench/000-00.json').read_bytes() == before
    data['/run/results/source_target_ssg/dreambench/000-00.png'] = b'bad'
    with pytest.raises(ValueError, match='hash mismatch'):
        client.download_job(volume, '/run', tmp_path / 'other', job)
    assert not (tmp_path / 'other/source_target_ssg/dreambench/000-00.png').exists()


def test_migration_bundle_preserves_legacy_signature_and_original_files(tmp_path, monkeypatch):
    from dataclasses import replace
    from orthogonalization_instantx_ip_adapter.semantic_contrasts import SemanticContrastConditioning
    original = client.original
    real_rows = original.read(original.MANIFEST)
    # Retain the real manifest validation but select two real rows for a quick fixture.
    monkeypatch.setattr(original, 'select_rows', lambda *args: real_rows[:2])
    args = SimpleNamespace(benchmark='all', shard_index=0, num_shards=2, workers=2,
                           conditions=['baseline','source_target_ssg'], strength=1.0, seed=42,
                           steps=24, width=1024, height=1024, gpu='A100-80GB', output=tmp_path/'local')
    row = real_rows[0]
    raw = original.read(original.ROOT / row['conditioning_path'])
    image = (original.ROOT / row['image_path']).read_bytes()
    cfg = dict(condition='baseline', strength=1.0, seed=42, steps=24, width=1024, height=1024,
               ip_adapter_scale=0.7, guidance_scale=3.5, gpu='A100-80GB', implementation_sha256=original.implementation_digest())
    sig = original.digest(dict(row=row, settings=cfg, cache=raw, image_sha256=hashlib.sha256(image).hexdigest()))
    png, sidecar = original.paths_for(args.output, 'baseline', row)
    png.parent.mkdir(parents=True)
    Image.new('RGB', (16,16)).save(png)
    original_record = dict(row=row, settings=cfg, job_signature=sig,
                           png_sha256=hashlib.sha256(png.read_bytes()).hexdigest(), modal_profile='A')
    sidecar.write_text(json.dumps(original_record))
    before = (png.read_bytes(), sidecar.read_bytes())
    plan, bundle, counts = client.build_bundle(args, tmp_path/'bundle-run')
    assert counts['baseline']['preserved'] == 1
    assert counts['source_target_ssg']['pending'] == 2
    assert plan['jobs'][0]['signature'] == sig
    projection = plan['jobs'][2]
    assert projection['payload'] == replace(SemanticContrastConditioning.from_dict(raw), origin='source_target').to_dict()
    assert projection['settings']['condition'] == 'source_target_ssg'
    worker.extract_bundle(bundle, tmp_path/'remote')
    assert worker.valid_result(tmp_path/'remote', plan['jobs'][0])
    assert (png.read_bytes(), sidecar.read_bytes()) == before


def test_modal_definitions_build_without_cloud_or_gpu():
    app, initializer, generator = worker.build_app(2, 'A100-80GB')
    assert generator._get_user_cls().__dict__['generate_lane'] is not None


def test_cli_defaults_to_only_the_requested_comparison(monkeypatch):
    observed = []
    monkeypatch.setattr(sys, 'argv', ['benchmark_remote.py', 'plan', '--shard-index', '0'])
    monkeypatch.setattr(client, 'build_bundle', lambda args, directory: observed.append((args.conditions, args.workers)))
    client.main()
    assert observed == [(['baseline', 'source_target_ssg'], 3)]


@pytest.mark.parametrize('condition', ['source', 'target', 'ssg', 'source_target'])
def test_remote_cli_rejects_unrequested_ablation_conditions(monkeypatch, condition):
    monkeypatch.setattr(sys, 'argv', ['benchmark_remote.py', 'plan', '--shard-index', '0', '--conditions', condition])
    with pytest.raises(SystemExit) as error:
        client.main()
    assert error.value.code == 2


@pytest.mark.parametrize('condition,gate,mode', [
    ('baseline', False, 'off'),
    ('source_target_ssg', True, 'semantic_local'),
])
def test_actual_generator_flags_match_requested_baseline_and_full_method(monkeypatch, condition, gate, mode):
    import orthogonalization_instantx_ip_adapter as package
    original = client.original
    seen = []
    model = object()
    fake = SimpleNamespace(load_ip_adapter_model=lambda: model,
                           generate_png_bytes=lambda **kwargs: (seen.append(kwargs) or b'fake', {}))
    monkeypatch.setitem(sys.modules, 'orthogonalization_instantx_ip_adapter.inference', fake)
    monkeypatch.setattr(package, 'inference', fake, raising=False)
    from modal._serialization import serialize, deserialize
    # Exercise the production helper AFTER serialization, which the old test missed.
    generate = deserialize(serialize(worker.generation_function(2, 'A100-80GB')), None)
    instance = SimpleNamespace(model=None)
    row = original.read(original.MANIFEST)[0]
    payload = original.read(original.ROOT / row['conditioning_path']) if mode == 'semantic_local' else None
    job = dict(row=row, settings=dict(condition=condition, strength=1.0, steps=24, seed=42,
                    width=1024, height=1024, ip_adapter_scale=0.7, guidance_scale=3.5),
               signature='test', image_bytes=(original.ROOT/row['image_path']).read_bytes(), payload=payload)
    generate(instance, job)
    assert seen[0]['mask_single_text_rows'] is gate
    assert seen[0]['orthogonalization_mode'] == mode
    assert seen[0]['contrast_analysis_payload'] == payload
    if payload is not None:
        assert payload['selection'] == dict(origin='source_target', strategy='local')


def test_two_conditions_partition_into_2100_jobs_per_account_without_overlap():
    original = client.original
    rows = original.read(original.MANIFEST)
    accounts = []
    for shard in range(2):
        selected = original.select_rows(rows, 'all', shard, 2)
        jobs = {(condition, row['case_id']) for condition in worker.DEFAULT_CONDITIONS for row in selected}
        assert len(jobs) == 2100
        for condition in worker.DEFAULT_CONDITIONS:
            assert sum(c == condition for c, _ in jobs) == 1050
        assert sum(r['benchmark'] == 'dreambench' for r in selected) == 375
        assert sum(r['benchmark'] == 'dreambench_plus' for r in selected) == 675
        ordered_jobs = [(condition, row['case_id']) for condition in worker.DEFAULT_CONDITIONS for row in selected]
        lanes = [set(ordered_jobs[lane::3]) for lane in range(3)]
        assert all(len(lane) == 700 for lane in lanes)
        assert set.union(*lanes) == jobs
        assert sum(len(lane) for lane in lanes) == len(jobs)
        for lane in lanes:
            for condition in worker.DEFAULT_CONDITIONS:
                assert sum(c == condition for c, _ in lane) == 350
        accounts.append(jobs)
    assert accounts[0].isdisjoint(accounts[1])
    assert len(accounts[0] | accounts[1]) == 4200


@pytest.mark.parametrize('synchronizer_failure,expected_attempts', [(False, 3), (True, 1)])
def test_repeated_errors_stop_lane_before_wasting_more_gpu(tmp_path, synchronizer_failure, expected_attempts):
    plan = fixture_plan(tmp_path, n=5)
    attempts = []
    def fail(job):
        attempts.append(job['signature'])
        if synchronizer_failure:
            raise AttributeError("'Synchronizer' object has no attribute '_original_attr'")
        raise RuntimeError('systemic inference error')
    with pytest.raises(RuntimeError, match='Stopping lane'):
        worker.run_lane(tmp_path, plan, 0, fail, lambda: None, 'A')
    assert len(attempts) == expected_attempts
    state = json.loads((tmp_path/'state/lane-0.json').read_text())
    assert state['status'] == 'interrupted'
    assert len(state['failed']) == expected_attempts
    assert 'Traceback' in state['failed'][0]['traceback']


def test_repair_removes_held_condition_without_changing_inputs_or_signatures():
    old_jobs = [dict(settings=dict(condition=c, implementation_sha256='inference'),
                    signature=c, payload=dict(origin='source_target'), row=dict(case_id='000-00'),
                    image_member='inputs/ref', image_sha256='image')
                for c in ('baseline','source_target','source_target_ssg')]
    old = dict(conditions=['baseline','source_target','source_target_ssg'], jobs=old_jobs,
               transport_sha256='old', workers=3, run_id='run')
    before = json.dumps(old)
    fixed = client.repaired_plan(old, 'inference', 'fixed')
    assert fixed['conditions'] == ['baseline','source_target_ssg']
    assert fixed['jobs'] == [old_jobs[0], old_jobs[2]]
    assert json.dumps(old) == before
    assert fixed['transport_sha256'] == 'fixed' and fixed['workers'] == 3
    with pytest.raises(ValueError, match='Inference implementation changed'):
        client.repaired_plan(old, 'different-inference', 'fixed')
