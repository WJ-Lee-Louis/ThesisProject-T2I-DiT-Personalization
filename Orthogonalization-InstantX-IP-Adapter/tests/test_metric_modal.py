"""Tests for safe retries of failed Modal metric runs."""

import json
from pathlib import Path

import pytest

import metric_modal


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _config(digest: str = "old") -> dict:
    return {
        "schema_version": 1,
        "run_id": "animal-6cond-v1",
        "metric": "dpg",
        "generation_batches": {"subset": {}, "custom": {}},
        "implementation_digest": digest,
        "xverse_revision": "revision",
        "requested_gpu": "A10G",
    }


def test_failed_metric_accepts_digest_only_upgrade(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    status_path = tmp_path / "status.json"
    _write(config_path, _config("old"))
    _write(status_path, {"state": "failed"})

    metric_modal._validate_existing_config(config_path, status_path, _config("new"))


@pytest.mark.parametrize("state", ["running", "completed"])
def test_nonfailed_metric_rejects_digest_change(tmp_path: Path, state: str) -> None:
    config_path = tmp_path / "config.json"
    status_path = tmp_path / "status.json"
    _write(config_path, _config("old"))
    _write(status_path, {"state": state})

    with pytest.raises(RuntimeError):
        metric_modal._validate_existing_config(config_path, status_path, _config("new"))


def test_failed_metric_still_rejects_semantic_change(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    status_path = tmp_path / "status.json"
    _write(config_path, _config("old"))
    _write(status_path, {"state": "failed"})
    changed = _config("new")
    changed["generation_batches"] = {"subset": {"baseline": "other"}}

    with pytest.raises(RuntimeError):
        metric_modal._validate_existing_config(config_path, status_path, changed)
