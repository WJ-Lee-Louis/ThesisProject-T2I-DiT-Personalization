import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('input_bundle', Path(__file__).resolve().parents[1] / 'runpod/input_bundle.py')
bundle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bundle)


@pytest.fixture
def source(tmp_path):
    root = tmp_path / 'source'
    reference = root / 'datasets/example/ref.png'
    reference.parent.mkdir(parents=True)
    reference.write_bytes(b'example reference')
    cache = root / 'outputs/semantic_benchmark_gpt56_sol/conditioning/shared.json'
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b'{"cached": true}')
    rows = [dict(case_id=str(i), benchmark='dreambench' if i < 750 else 'dreambench_plus',
                 status='READY', image_path=reference.relative_to(root).as_posix(),
                 conditioning_path=cache.relative_to(root).as_posix()) for i in range(2100)]
    (root / bundle.MANIFEST).write_text(json.dumps(rows), encoding='utf-8')
    lock = tmp_path / 'input_bundle.json'
    archive = bundle.build(root, lock)
    return root, lock, archive


def test_deterministic_build_and_idempotent_restore(source, tmp_path):
    root, lock, archive = source
    first = archive.read_bytes()
    bundle.build(root, lock)
    assert archive.read_bytes() == first
    target = tmp_path / 'clone'
    bundle.restore(archive, target, lock)
    bundle.restore(archive, target, lock)
    assert (target / bundle.MANIFEST).read_bytes() == (root / bundle.MANIFEST).read_bytes()
    assert (target / 'datasets/example/ref.png').read_bytes() == b'example reference'
    assert not (target / 'scripts').exists()


def test_wrong_bundle_and_existing_input_are_not_overwritten(source, tmp_path):
    root, lock, archive = source
    target = tmp_path / 'clone'
    conflict = target / 'datasets/example/ref.png'
    conflict.parent.mkdir(parents=True)
    conflict.write_bytes(b'keep this')
    with pytest.raises(ValueError, match='Existing input differs'):
        bundle.restore(archive, target, lock)
    assert conflict.read_bytes() == b'keep this'
    assert not (target / bundle.MANIFEST).exists()
    archive.write_bytes(archive.read_bytes() + b'corrupted')
    with pytest.raises(ValueError, match='SHA256/size mismatch'):
        bundle.restore(archive, target, lock)


def test_manifest_cannot_restore_outside_input_directories(source):
    root, lock, archive = source
    rows = json.loads((root / bundle.MANIFEST).read_text(encoding='utf-8'))
    rows[0]['image_path'] = 'datasets/../../secrets'
    with pytest.raises(ValueError, match='Invalid input path'):
        bundle.names_for(rows)
