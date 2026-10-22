"""Partition coverage, resume integrity, and independent gate/scale validation."""
import importlib.util
import sys
from pathlib import Path

import pytest
from PIL import Image

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import benchmark_runpod as runner


@pytest.mark.parametrize('sizes', [(8,), (4, 4), (2, 2, 2, 2), (1,) * 8, (3, 2, 3)])
def test_disjoint_complete_multi_pod_partition(sizes):
    jobs = list(range(4200))
    groups, offset = [], 0
    for size in sizes:
        groups.append(runner.assigned(jobs, 8, offset, size))
        offset += size
    assert sorted(j for group in groups for j in group) == jobs
    assert all(len(group) == 525 * size for group, size in zip(groups, sizes))


def test_projection_without_gate_and_scale():
    cfg = dict(condition='source_target', ip_adapter_scale=.7)
    diag = dict(mask_single_text_rows=False, orthogonalization=dict(enabled=True),
                generation=dict(ip_adapter_scale=.7))
    runner.validate_diagnostics(diag, cfg)
    for wrong in (
        dict(diag, mask_single_text_rows=True),
        dict(diag, orthogonalization=dict(enabled=False)),
        dict(diag, generation=dict(ip_adapter_scale=.8)),
    ):
        with pytest.raises(ValueError):
            runner.validate_diagnostics(wrong, cfg)


def test_resume_checks_signature_pixels_and_diagnostics(tmp_path):
    job = dict(row=dict(benchmark='dreambench', case_id='000-00'),
               settings=dict(condition='source_target', ip_adapter_scale=.7), signature='expected')
    png, sidecar = runner.paths(tmp_path, job)
    assert not runner.finished(tmp_path, job)
    png.parent.mkdir(parents=True)
    Image.new('RGB', (1024, 1024)).save(png)
    meta = dict(row=job['row'], settings=job['settings'], job_signature='expected',
                png_sha256=runner.metrics.sha(png), diagnostics=dict(mask_single_text_rows=False,
                orthogonalization=dict(enabled=True), generation=dict(ip_adapter_scale=.7)))
    runner.metrics.write(sidecar, meta)
    assert runner.finished(tmp_path, job)
    with pytest.raises(ValueError, match='mismatch'):
        runner.finished(tmp_path, dict(job, signature='different'))
    Image.new('RGB', (1024, 1024), 'red').save(png)
    with pytest.raises(ValueError, match='hash mismatch'):
        runner.finished(tmp_path, job)


def test_jobs_preserve_seed_and_independent_scales(monkeypatch, tmp_path):
    (tmp_path / 'image').write_bytes(b'reference')
    runner.metrics.write(tmp_path / 'cache.json', {})
    row = dict(case_id='one', image_path='image', conditioning_path='cache.json')
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'revision', lambda: 'code')
    monkeypatch.setattr(runner, 'rows_for', lambda limit: [row])
    jobs = runner.jobs_for()
    assert [j['settings']['ip_adapter_scale'] for j in jobs] == [.8, .7]
    assert all(j['settings']['seed'] == 42 and j['settings']['strength'] == 1 for j in jobs)
    assert jobs[0]['signature'] != jobs[1]['signature']


@pytest.mark.parametrize('job_count', [2, 6])
def test_worker_failure_resume_preserves_completed_case(monkeypatch, tmp_path, job_count):
    import io
    import importlib.metadata
    from types import SimpleNamespace, ModuleType
    import orthogonalization_instantx_ip_adapter as package
    jobs = [dict(row=dict(benchmark='dreambench', case_id=str(i), target_prompt=str(i)),
                 settings=dict(condition='source_target', ip_adapter_scale=.7), signature=str(i))
            for i in range(job_count)]
    monkeypatch.setattr(runner, 'jobs_for', lambda limit: jobs)
    monkeypatch.setattr(runner, 'validate_cache', lambda job: (b'image', {}))
    versions = {'torch': '2.7.1', 'diffusers': '0.35.2', 'transformers': '4.54.0',
                'accelerate': '1.10.0', 'huggingface-hub': '0.34.3'}
    monkeypatch.setattr(importlib.metadata, 'version', versions.__getitem__)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(get_device_name=lambda i: 'test')))
    fake = ModuleType('orthogonalization_instantx_ip_adapter.inference')
    fake.load_ip_adapter_model = lambda: object()
    calls = []
    fail = [True]
    def generate(**kwargs):
        calls.append(kwargs['prompt'])
        if kwargs['prompt'] != '0' and fail[0]:
            raise RuntimeError('temporary GPU failure')
        output = io.BytesIO()
        Image.new('RGB', (1024, 1024)).save(output, format='PNG')
        return output.getvalue(), dict(mask_single_text_rows=False, orthogonalization=dict(enabled=True),
                                      generation=dict(ip_adapter_scale=.7))
    fake.generate_png_bytes = generate
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    monkeypatch.setattr(package, 'inference', fake, raising=False)
    args = SimpleNamespace(limit=0, lane=0, total_workers=1, output=tmp_path)
    with pytest.raises(RuntimeError, match='jobs failed|3 consecutive failures'):
        runner.worker(args)
    attempted = ['0', '1'] if job_count == 2 else ['0', '1', '2', '3']
    assert calls == attempted
    fail[0] = False
    runner.worker(args)
    assert calls == attempted + [str(i) for i in range(1, job_count)]
    assert all(runner.finished(tmp_path, job) for job in jobs)


def test_source_hash_normalizes_only_crlf(tmp_path):
    path = tmp_path / 'source.py'
    path.write_bytes(b'a = 1\r\nb = 2\r\n')
    expected = runner.source_sha(path)
    path.write_bytes(b'a = 1\nb = 2\n')
    assert runner.source_sha(path) == expected
    path.write_bytes(b'a = 3\nb = 2\n')
    assert runner.source_sha(path) != expected


def test_failed_later_lane_is_detected_before_waiting(monkeypatch):
    from types import SimpleNamespace
    children = [SimpleNamespace(poll=lambda: None), SimpleNamespace(poll=lambda: 1)]
    monkeypatch.setattr(runner.time, 'sleep', lambda _: pytest.fail('Must notice failed lane immediately'))
    with pytest.raises(RuntimeError, match='Workers failed'):
        runner.wait_workers(children)
