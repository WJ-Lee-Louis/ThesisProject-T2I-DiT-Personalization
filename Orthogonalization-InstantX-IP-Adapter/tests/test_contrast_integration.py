from dataclasses import replace
from io import BytesIO
import json
import sys
from types import SimpleNamespace

from PIL import Image
import pytest
import torch

from test_semantic_contrasts import prepared, image_bytes, Provider, example
from orthogonalization_instantx_ip_adapter.adapter import OrthogonalizationRequest, OrthogonalizedIPAdapterMixin
from orthogonalization_instantx_ip_adapter.contrast_directions import (
    contrast_plan, contrast_diagnostics, build_contrast_directions,
)
from orthogonalization_instantx_ip_adapter.semantic_contrasts import SemanticLLM, prepare_semantic_conditioning
from orthogonalization_instantx_ip_adapter.semantic_cli import prepare_cli_conditioning, resolve_contrast_options
from orthogonalization_instantx_ip_adapter.orthogonalization import normalize_fp32, estimate_orthonormal_basis


@pytest.mark.parametrize("origin,count", [("source", 3), ("target", 1), ("source_target", 4)])
def test_local_plan_encodes_exact_llm_negatives_and_originals(origin, count, monkeypatch):
    from orthogonalization_instantx_ip_adapter.semantic_components import ComponentAnalysis
    monkeypatch.setattr(ComponentAnalysis, "prompt_for_subset", lambda *a: pytest.fail("legacy deletion called"))
    conditioning, _ = prepared(1, origin)
    plan = contrast_plan(conditioning)
    seen = []
    lookup = {text: torch.tensor([float(i + 1), 2., 3., 4.])
              for i, text in enumerate(dict.fromkeys(t for row in plan for t in (row.positive, row.negative)))}
    def encode(texts):
        seen.extend(texts)
        return torch.stack([lookup[t] for t in texts])
    directions = build_contrast_directions(conditioning, encode)
    assert len(plan) == count == directions.values.shape[0]
    assert len(seen) == len(set(seen))
    for index, row in enumerate(plan):
        positive = conditioning.analysis.source_caption if row.origin == "source" else conditioning.analysis.target_prompt
        assert row.positive == positive
        raw = normalize_fp32(lookup[positive]) - normalize_fp32(lookup[row.negative])
        torch.testing.assert_close(directions.values[index], normalize_fp32(raw))
        torch.testing.assert_close(directions.raw_norms[index], raw.norm())
    assert [r["negative"] for r in contrast_diagnostics(conditioning)["contrasts"]] == [r.negative for r in plan]


def test_cross_side_identical_text_reuses_encoding_but_retains_rows():
    out = dict(identity_term="dog", source_preserved=["dog"],
               source=[dict(unit="sitting", negative="a dog")],
               target=[dict(unit="sitting", negative="a dog")], issues=[])
    c = prepare_semantic_conditioning(llm=SemanticLLM(Provider(out)), identity_term="dog",
                                      source_caption="a dog sitting", target_prompt="a dog sitting", preserve=[])
    calls = []
    def encode(texts):
        calls.append(texts)
        return torch.tensor([[1., 1., 0.], [1., 0., 0.]])
    d = build_contrast_directions(c, encode)
    assert calls == [["a dog sitting", "a dog"]]
    assert d.names == ("source:removal_0", "target:removal_0")
    assert estimate_orthonormal_basis(d.values).diagnostics.retained_rank == 1


@pytest.mark.parametrize("delta,zero", [(0., True), (1e-13, True), (1e-8, False)])
def test_degenerate_differences_keep_count_and_numeric_policy(delta, zero):
    c, _ = prepared(origin="target")
    d = build_contrast_directions(c, lambda texts: torch.tensor([[1., 0.], [1., delta]]))
    assert d.values.shape == (1, 2)
    assert bool((d.values == 0).all()) == zero
    assert estimate_orthonormal_basis(d.values).diagnostics.retained_rank == (0 if zero else 1)
    torch.testing.assert_close(d.raw_norms, torch.tensor([delta]))


def test_empty_removals_use_only_dimension_probe():
    c, _ = prepared(5)
    calls = []
    def encode(texts):
        calls.append(texts)
        return torch.ones(1, 4)
    d = build_contrast_directions(c, encode)
    assert calls == [["sunset"]]
    assert d.values.shape == (0, 4) and not d.names
    assert contrast_diagnostics(c)["selected_direction_count"] == 0


@pytest.mark.parametrize("bad", [torch.ones(4), torch.ones(1, 3), torch.ones(2, 0),
                                 torch.ones(2, 3, dtype=torch.int32), torch.full((2, 3), float("nan"))])
def test_bad_encoder_output_rejected(bad):
    c, _ = prepared(origin="target")
    with pytest.raises(ValueError, match="encoder"):
        build_contrast_directions(c, lambda texts: bad)


def model_mock(monkeypatch):
    from orthogonalization_instantx_ip_adapter import inference
    adapter = object.__new__(OrthogonalizedIPAdapterMixin)
    adapter.device = "cpu"
    adapter.checkpoint_load_diagnostics = {}
    adapter.last_single_stream_text_seq_len = 4
    encoded, generated, projected = [], [], []
    def encode(texts):
        encoded.extend(texts)
        return torch.tensor([[float(len(t)), float(t.count('a')), float(t.count('o')), 2.] for t in texts])
    adapter._encode_component_text_pool = encode
    adapter.image_proj_model = lambda value: projected.append(value) or value
    def generate(**kwargs):
        generated.append(kwargs)
        adapter._active_orthogonalization = kwargs["orthogonalization"]
        adapter.get_image_embeds(clip_image_embeds=torch.tensor([[1., 2., 3., 4.]]))
        return [Image.new("RGB", (3, 2))]
    adapter.generate = generate
    monkeypatch.setattr(inference, "load_ip_adapter_model", lambda: adapter)
    monkeypatch.setitem(sys.modules, "infer_flux_ipa_siglip", SimpleNamespace(resize_img=lambda x: x))
    return adapter, encoded, generated, projected


@pytest.mark.parametrize("origin", ["source", "target", "source_target"])
def test_python_a_joint_b_to_real_adapter_geometry_and_original_generation(monkeypatch, origin):
    from orthogonalization_instantx_ip_adapter.inference import generate_semantic_png_bytes
    adapter, encoded, generated, projected = model_mock(monkeypatch)
    inp, out = example(1)
    provider = Provider({"source_caption": inp["source_caption"], "preserve": inp["preserve"]}, out)
    png, diagnostics = generate_semantic_png_bytes(llm=SemanticLLM(provider), identity_term=inp["identity_term"],
        target_prompt=inp["target_prompt"], reference_image_bytes=image_bytes(), origin=origin)
    assert png.startswith(b"\x89PNG") and len(projected) == 1
    assert len(provider.calls) == 2
    assert generated[0]["prompt"] == "A purple backpack"
    assert generated[0]["orthogonalization"].mode == "semantic_local"
    orth = diagnostics["orthogonalization"]
    expected = {"source": 3, "target": 1, "source_target": 4}[origin]
    assert orth["input_direction_count"] == orth["selected_direction_count"] == expected
    assert all(r["negative"] in encoded for r in orth["contrasts"])
    assert orth["basis_rank"] <= expected
    torch.testing.assert_close(projected[0].float().norm(), torch.tensor(30.).sqrt(), rtol=.01, atol=.01)


def test_cached_python_generation_no_llm_and_origin_override(monkeypatch):
    from orthogonalization_instantx_ip_adapter.inference import generate_semantic_png_bytes
    _, _, generated, _ = model_mock(monkeypatch)
    c, _ = prepared()
    generate_semantic_png_bytes(identity_term="backpack", target_prompt=c.analysis.target_prompt,
        reference_image_bytes=image_bytes(), contrast_analysis_payload=c.to_dict(), origin="target")
    assert generated[0]["orthogonalization"].semantic_conditioning.origin == "target"


@pytest.mark.parametrize("change", ["target", "reference", "version", "unbound"])
def test_remote_rejects_invalid_cache_before_model_load(monkeypatch, change):
    from orthogonalization_instantx_ip_adapter import inference
    c, _ = prepared()
    payload, target, raw = c.to_dict(), c.analysis.target_prompt, image_bytes()
    if change == "target": target = "A purple backpack"
    if change == "reference": raw = image_bytes("blue")
    if change == "version": payload["schema_version"] = 2
    if change == "unbound": payload["reference"] = None
    monkeypatch.setattr(inference, "load_ip_adapter_model", lambda: pytest.fail("GPU loaded"))
    with pytest.raises(ValueError):
        inference.generate_png_bytes(prompt=target, reference_image_url="", reference_image_bytes=raw,
            ip_adapter_scale=.7, guidance_scale=3.5, num_inference_steps=1, seed=1, width=32, height=32,
            orthogonalization_mode="semantic_local", orthogonalization_strength=1., object_prompt="",
            object_behavior_prompt="", object_background_prompt="", object_behavior_background_prompt="",
            remove_behavior=True, remove_background=True, include_interaction=False, restore_visual_norm=True,
            mask_single_text_rows=True, contrast_analysis_payload=payload)


@pytest.mark.parametrize("kwargs", [{"component_strategy": "factorial"}, {"component_max_contexts": 256},
    {"contrast_origin": "source", "component_origin": "target"}, {"contrast_origin": "invalid"}])
def test_cli_invalid_options_rejected(kwargs):
    with pytest.raises(ValueError):
        resolve_contrast_options(**kwargs)


def test_cli_alias_and_cached_selection_with_no_provider(tmp_path):
    c, _ = prepared(origin="source")
    path = tmp_path / "saved.contrasts.json"
    path.write_text(json.dumps(c.to_dict()), encoding="utf-8")
    with pytest.warns(FutureWarning):
        origin, selected = resolve_contrast_options(component_analysis_path=str(path), component_strategy="local")
    def forbidden(model): pytest.fail("provider created for cached run")
    restored = prepare_cli_conditioning(image_bytes=image_bytes(), target_prompt=c.analysis.target_prompt,
        analysis_path=selected, origin=origin, provider_factory=forbidden)
    assert restored.origin == "source"
    assert prepare_cli_conditioning(image_bytes=image_bytes(), target_prompt=c.analysis.target_prompt,
        analysis_path=str(path), origin="target", provider_factory=forbidden).origin == "target"


def test_cli_manual_target_still_uses_one_joint_b():
    inp, out = example()
    provider = Provider(out)
    c = prepare_cli_conditioning(image_bytes=image_bytes(), **{k:v for k,v in inp.items() if k != "preserve"},
                                 preserve_json=json.dumps(inp["preserve"]), origin="target",
                                 provider_factory=lambda model: provider)
    assert len(provider.calls) == 1 and provider.calls[0]["payload"] == inp
    assert c.direction_count == 1


def test_new_request_rejects_legacy_type_and_mixed_geometry():
    c, _ = prepared()
    with pytest.raises(ValueError):
        OrthogonalizationRequest(mode="components", semantic_conditioning=c)
    with pytest.raises(ValueError, match="cannot mix"):
        OrthogonalizationRequest(mode="semantic_local", semantic_conditioning=c, include_interaction=True)
    with pytest.raises(ValueError, match="cannot mix"):
        OrthogonalizationRequest(mode="semantic_local", semantic_conditioning=c, include_behavior=False)


def test_modal_entrypoint_url_is_fetched_once_and_cache_roundtrips(monkeypatch, tmp_path):
    import modal_inference as modal_module
    from orthogonalization_instantx_ip_adapter.llm_provider import ResponsesJSONProvider
    _, _, generated, _ = model_mock(monkeypatch)
    inp, out = example(1)
    provider = Provider({"source_caption": inp["source_caption"], "preserve": inp["preserve"]}, out)
    monkeypatch.setattr(ResponsesJSONProvider, "from_env", lambda model=None: provider)
    downloads = []
    def download(url, timeout):
        downloads.append(url)
        return BytesIO(image_bytes())
    monkeypatch.setattr("urllib.request.urlopen", download)
    remote = modal_module.generate_experiment.get_raw_f()
    monkeypatch.setattr(modal_module, "generate_experiment", SimpleNamespace(remote=remote))
    modal_module.main(prompt=inp["target_prompt"], reference_image_path="",
        reference_image_url="https://example.invalid/reference.png", identity_term="backpack",
        contrast_origin="source", output_path=str(tmp_path / "image.png"))
    assert len(downloads) == 1 and len(provider.calls) == 2
    paths = list(tmp_path.glob("*.contrasts.json"))
    assert len(paths) == 1 and not list(tmp_path.glob("*.components.json"))
    stored = json.loads(paths[0].read_text(encoding="utf-8"))
    assert stored["inputs"]["target_prompt"] == inp["target_prompt"]
    local_reference = tmp_path / "reference.png"
    local_reference.write_bytes(image_bytes())
    modal_module.main(prompt=inp["target_prompt"], reference_image_path=str(local_reference),
        contrast_analysis_path=str(paths[0]), output_path=str(tmp_path / "image.png"))
    assert len(provider.calls) == 2 and len(downloads) == 1
    assert generated[-1]["orthogonalization"].semantic_conditioning.origin == "source"
    assert len(list(tmp_path.glob("*.contrasts.json"))) == 2


def test_cli_caption_cache_and_legacy_import(tmp_path):
    from orthogonalization_instantx_ip_adapter.semantic_contrasts import ReferenceFingerprint
    c, _ = prepared()
    path = tmp_path / "caption.json"
    path.write_text(json.dumps(c.caption.to_dict()), encoding="utf-8")
    inp, out = example()
    provider = Provider(out, out)
    common = dict(image_bytes=image_bytes(), target_prompt=inp["target_prompt"], identity_term="backpack",
                  provider_factory=lambda model: provider)
    cached = prepare_cli_conditioning(**common, caption_cache_path=str(path))
    assert cached.caption.origin == "generated_v2" and len(provider.calls) == 1
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps(dict(identity_term="backpack", source_caption=inp["source_caption"], status="PASS",
        image_sha256=ReferenceFingerprint.from_image(image_bytes()).bytes_sha256,
        requested_model="old", prompt_sha256="a" * 64)), encoding="utf-8")
    imported = prepare_cli_conditioning(**common, import_legacy_caption=str(legacy), preserve_json=json.dumps(inp["preserve"]))
    assert imported.caption.origin == "imported_legacy" and len(provider.calls) == 2
    assert imported.caption.imported_from.model == "old"


def test_monitor_audit_and_invalid_cache_stop_before_gpu(monkeypatch, tmp_path):
    import scripts.monitor_semantic_pipeline as monitor
    import scripts.semantic_monitor_modal as remote
    from orthogonalization_instantx_ip_adapter import inference
    c, _ = prepared()
    assert monitor.audit(c)["contrasts"][0]["negative"] == c.analysis.source[0].negative
    legacy_dir = tmp_path / "old"
    legacy_dir.mkdir()
    (legacy_dir / "conditioning_local.json").write_text("{}")
    with pytest.raises(ValueError, match="legacy"):
        monitor.check_run(legacy_dir)
    monkeypatch.setattr(inference, "load_ip_adapter_model", lambda: pytest.fail("GPU loaded"))
    with pytest.raises(ValueError, match="reference"):
        list(remote.run_cases.get_raw_f()([dict(case_id="x", image_bytes=image_bytes("blue"),
                                               contrast_analysis_payload=c.to_dict())]))


def test_monitor_analyze_and_render_use_new_files_without_real_calls(monkeypatch, tmp_path):
    import scripts.monitor_semantic_pipeline as monitor
    import scripts.render_semantic_monitor as renderer
    inp, out = example()
    raw_path = tmp_path / "reference.png"
    raw_path.write_bytes(image_bytes())
    case = dict(case_id="01-01", identity_term=inp["identity_term"], target_prompt=inp["target_prompt"],
                reference_image=str(raw_path), selection_reason="offline fixture")
    monkeypatch.setattr(monitor, "cases", lambda: [case])
    provider = Provider({"source_caption": inp["source_caption"], "preserve": inp["preserve"]}, out)
    monkeypatch.setattr(monitor.ResponsesJSONProvider, "from_env", lambda model: provider)
    directory = tmp_path / "run"
    assert monitor.analyze(directory, "test-model")[0]["status"] == "analysis_passed"
    assert monitor.analyze(directory, "test-model")[0]["status"] == "cached"
    assert len(provider.calls) == 2
    renderer.main(directory)
    page = (directory / "index.html").read_text(encoding="utf-8")
    assert out["source"][0]["negative"] in page
    assert "No generated image" in page
    assert not list(directory.rglob("*factorial*"))


@pytest.mark.parametrize("kwargs", [{"strategy": "factorial"}, {"origin": "bad"}])
def test_python_bad_semantic_options_never_call_llm_or_generation(monkeypatch, kwargs):
    from orthogonalization_instantx_ip_adapter import inference
    provider = Provider()
    monkeypatch.setattr(inference, "generate_png_bytes", lambda **kw: pytest.fail("generation called"))
    with pytest.raises(ValueError):
        inference.generate_semantic_png_bytes(identity_term="dog", target_prompt="a dog",
            reference_image_bytes=image_bytes(), llm=SemanticLLM(provider), **kwargs)
    assert provider.calls == []


@pytest.mark.parametrize("preserve_json", ["", "{}", '["backpack"]', '["not observed"]', '"magenta"'])
def test_cli_invalid_or_missing_manual_preserve_stops_before_provider(preserve_json):
    def forbidden(model):
        pytest.fail("provider must not be constructed for invalid manual preservation")
    with pytest.raises(ValueError):
        prepare_cli_conditioning(image_bytes=image_bytes(), identity_term="backpack",
            source_caption="A magenta backpack", target_prompt="A backpack", preserve_json=preserve_json,
            provider_factory=forbidden)


def test_monitor_rejects_pre_preserve_schema_marker(tmp_path):
    from scripts.monitor_semantic_pipeline import check_run
    (tmp_path / "joint_local_run.json").write_text(json.dumps({"schema_version": 3}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        check_run(tmp_path)
    assert json.loads((tmp_path / "joint_local_run.json").read_text())["schema_version"] == 3
