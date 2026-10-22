"""Offline checks for cross-account coverage and provenance-preserving resume/export."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

from PIL import Image
import pytest

SPEC = importlib.util.spec_from_file_location("benchmark_modal", Path(__file__).resolve().parents[1] / "scripts/benchmark_modal.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_real_manifest_shards_cover_benchmarks_without_overlap():
    rows = runner.read(runner.MANIFEST)
    shards = [runner.select_rows(rows, "all", index, 2) for index in range(2)]
    assert all(len(shard) == 1050 for shard in shards)
    assert {r["case_id"] for r in shards[0]}.isdisjoint({r["case_id"] for r in shards[1]})
    assert {r["case_id"] for shard in shards for r in shard} == {r["case_id"] for r in rows}
    for shard in shards:
        assert sum(r["benchmark"] == "dreambench" for r in shard) == 375
        assert sum(r["benchmark"] == "dreambench_plus" for r in shard) == 675
    assert {r["reference_id"] for r in shards[0]}.isdisjoint({r["reference_id"] for r in shards[1]})


def test_invalid_shard_fails():
    with pytest.raises(ValueError):
        runner.select_rows([], "all", 2, 2)


def test_resume_rejects_changed_settings_or_corrupted_image(tmp_path):
    png, sidecar = tmp_path / "a.png", tmp_path / "a.json"
    assert not runner.complete(png, sidecar, "sig")
    Image.new("RGB", (16, 16)).save(png)
    sidecar.write_text(json.dumps(dict(job_signature="sig", png_sha256=runner.hashlib.sha256(png.read_bytes()).hexdigest())))
    assert runner.complete(png, sidecar, "sig")
    with pytest.raises(ValueError, match="mismatch"):
        runner.complete(png, sidecar, "different")
    png.write_bytes(b"broken")
    with pytest.raises(ValueError, match="hash mismatch"):
        runner.complete(png, sidecar, "sig")


def test_dreambench_plus_export_matches_upstream_collection_names(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    manifest = root / "manifest.json"
    row = dict(case_id="030-00", reference_id="dreambench_plus/live_subject/animal/00", benchmark="dreambench_plus",
               target_prompt="A kitten playing", image_path="reference.jpg")
    manifest.write_text(json.dumps([row]))
    ledger = root / "datasets/semantic_benchmark/references.json"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(json.dumps([dict(id=row["reference_id"], subject_id="live_subject/animal/00", identity_term="kitten")]))
    Image.new("RGB", (16, 16)).save(root / "reference.jpg")
    output = root / "outputs"
    png, sidecar = runner.paths_for(output, "source_target", row)
    png.parent.mkdir(parents=True)
    Image.new("RGB", (16, 16)).save(png)
    sidecar.write_text(json.dumps(dict(row=row, settings=dict(condition="source_target"), job_signature="sig",
                                      png_sha256=runner.hashlib.sha256(png.read_bytes()).hexdigest())))
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "MANIFEST", manifest)
    destination = root / "export"
    args = SimpleNamespace(benchmark="dreambench_plus", output=output, condition="source_target", export_dir=destination)
    runner.export(args)
    assert (destination / "tgt_image/live_subject_animal_00_kitten/0_0.jpg").exists()
    assert (destination / "text/live_subject_animal_00_kitten/0_0.txt").read_text() == row["target_prompt"]
    with pytest.raises(ValueError, match="must be new"):
        runner.export(args)


def test_worker_reuses_model_preserves_prompt_and_validates_before_loading(monkeypatch):
    import orthogonalization_instantx_ip_adapter as package
    calls, loads = [], []
    model = object()

    def load():
        loads.append(True)
        return model

    def generate(**kwargs):
        assert fake.load_ip_adapter_model() is model
        calls.append(kwargs)
        return b"test-png", {"mock": True}

    fake = SimpleNamespace(load_ip_adapter_model=load, generate_png_bytes=generate)
    monkeypatch.setitem(sys.modules, "orthogonalization_instantx_ip_adapter.inference", fake)
    monkeypatch.setattr(package, "inference", fake, raising=False)
    _, cls = runner.build_generator(SimpleNamespace(gpu="A100-80GB", workers=2))
    user_cls = cls._get_user_cls()
    instance = user_cls()
    user_cls.__dict__["enter"]._get_raw_f()(instance)
    execute = user_cls.__dict__["generate"]._get_raw_f()
    row = runner.read(runner.MANIFEST)[0]
    payload = runner.read(runner.ROOT / row["conditioning_path"])
    job = dict(row=row, settings=dict(condition="source_target", strength=1.0, steps=24, seed=42,
                                     width=1024, height=1024, ip_adapter_scale=0.7, guidance_scale=3.5),
               signature="test", image_bytes=(runner.ROOT / row["image_path"]).read_bytes(), payload=payload)
    execute(instance, job)
    execute(instance, job)
    assert len(loads) == 1
    assert fake.load_ip_adapter_model is load
    assert calls[0]["prompt"] == row["target_prompt"]
    assert calls[0]["contrast_analysis_payload"] == payload
    assert calls[0]["orthogonalization_mode"] == "semantic_local"
    assert calls[0]["mask_single_text_rows"] is False
    bad_job = dict(job, row=dict(row, target_prompt="a completely different target"))
    fresh = user_cls()
    user_cls.__dict__["enter"]._get_raw_f()(fresh)
    with pytest.raises(ValueError):
        execute(fresh, bad_job)
    assert len(loads) == 1
    assert fresh.model is None
