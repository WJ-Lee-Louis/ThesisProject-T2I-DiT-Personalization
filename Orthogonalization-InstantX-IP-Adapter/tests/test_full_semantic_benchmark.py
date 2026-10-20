"""Batch orchestration safety, without paid requests."""
from pathlib import Path

import pytest

from scripts import run_full_semantic_benchmark as batch


def test_atomic_save_retries_windows_reader_lock(tmp_path, monkeypatch):
    path = tmp_path / "progress.json"
    path.write_text('{"old": true}', encoding="utf-8")
    original_replace = Path.replace
    attempts = []

    def replace(source, target):
        attempts.append(target)
        if len(attempts) < 3:
            raise PermissionError("destination temporarily open")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(batch.time, "sleep", lambda _: None)
    batch.save(path, {"done": 10})
    assert batch.read(path) == {"done": 10}
    assert len(attempts) == 3
    assert not path.with_suffix(".json.tmp").exists()


def test_transport_retry_is_bounded(monkeypatch):
    attempts = []
    monkeypatch.setattr(batch.time, "sleep", lambda _: None)

    def fail():
        attempts.append(1)
        raise batch.LLMTransportError("LLM HTTP 503")

    with pytest.raises(batch.LLMTransportError):
        batch.request(fail)
    assert len(attempts) == 3


def test_auth_error_does_not_fan_out_retries():
    attempts = []

    def fail():
        attempts.append(1)
        raise batch.LLMTransportError("LLM HTTP 401")

    with pytest.raises(batch.LLMTransportError):
        batch.request(fail)
    assert len(attempts) == 1


def test_retry_preserves_failed_paid_trace(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "OUT", tmp_path)
    path = tmp_path / "errors" / "A-001.json"
    failure = {"status": "ERROR", "trace": {"attempts": ["original response"]}}
    batch.save(path, failure)
    batch.retain_previous_failure(path)
    assert not path.exists()
    assert batch.read(tmp_path / "recovered_failures" / path.name) == [failure]
