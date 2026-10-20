from copy import deepcopy
from dataclasses import replace

import pytest

from test_semantic_contrasts import Provider, image_bytes
from test_contrast_integration import model_mock
from orthogonalization_instantx_ip_adapter.semantic_contrasts import (
    SemanticLLM, IdentityConflictError, SemanticContrastConditioning,
    prepare_source_only_fallback,
)
from orthogonalization_instantx_ip_adapter.contrast_directions import contrast_plan, contrast_diagnostics


def fixture():
    target = "A figurine transforming into an animal under a starry sky"
    conflict = dict(identity_term="figurine", source_preserved=[], source=[], target=[],
                    issues=[dict(kind="identity_conflict", side="target",
                                 evidence="transforming into an animal", reason="identity transformation")])
    source = dict(identity_term="figurine", source_preserved=["figurine", "white"],
                  source=[dict(unit="table location", negative="a white figurine")], target=[], issues=[])
    provider = Provider(dict(source_caption="a white figurine on a table", preserve=["white"]), conflict, source)
    llm = SemanticLLM(provider)
    caption = llm.caption(image_bytes(), "figurine")
    with pytest.raises(IdentityConflictError) as error:
        llm.contrast(caption.text, target, "figurine", caption.preserve)
    return provider, llm, caption, target, error.value.trace


def fallback():
    provider, llm, caption, target, trace = fixture()
    result = prepare_source_only_fallback(llm=llm, caption_record=caption, target_prompt=target,
                                         reference_image_bytes=image_bytes(), failed_trace=trace)
    return result, provider


def test_fallback_preserves_trace_and_only_encodes_source():
    result, provider = fallback()
    assert len(provider.calls) == 3
    assert provider.calls[-1]["payload"]["target_prompt"] == result.analysis.target_prompt
    assert result.analysis.trace.stage == "contrast_source_fallback"
    assert result.fallback_trace.stage == "contrast"
    assert result.to_dict()["kind"] == "semantic_source_fallback"
    assert SemanticContrastConditioning.from_dict(result.to_dict()) == result
    assert [(p.origin, p.positive, p.negative) for p in contrast_plan(result)] == [
        ("source", "a white figurine on a table", "a white figurine")]
    assert contrast_diagnostics(result)["target_projection_skipped"]
    assert replace(result, origin="source_target").direction_count == 1
    with pytest.raises(ValueError):
        replace(result, origin="target")


@pytest.mark.parametrize("change", ["target", "reference", "model", "source_issue"])
def test_invalid_fallback_is_rejected_before_paid_call(change):
    provider, llm, caption, target, trace = fixture()
    data = image_bytes()
    if change == "target":
        target += " at night"
    elif change == "reference":
        data = image_bytes("blue")
    elif change == "model":
        trace = replace(trace, model="different-model")
    else:
        import json
        response = json.loads(trace.attempts[-1].raw_response)
        response["issues"][0].update(side="source", evidence="figurine")
        trace = replace(trace, attempts=(replace(trace.attempts[-1], raw_response=json.dumps(response)),))
    with pytest.raises(ValueError):
        prepare_source_only_fallback(llm=llm, caption_record=caption, target_prompt=target,
                                     reference_image_bytes=data, failed_trace=trace)
    assert len(provider.calls) == 2


@pytest.mark.parametrize("change", ["remove_provenance", "normal_kind", "wrong_policy"])
def test_fallback_cannot_be_disguised_as_normal_success(change):
    result, _ = fallback()
    value = deepcopy(result.to_dict())
    if change == "remove_provenance":
        value.pop("fallback")
    elif change == "normal_kind":
        value["kind"] = "semantic_local_contrast"
        value.pop("fallback")
    else:
        value["fallback"]["policy_version"] = "unknown"
    with pytest.raises(ValueError):
        SemanticContrastConditioning.from_dict(value)


def test_cached_fallback_reaches_adapter_and_keeps_original_target(monkeypatch):
    from orthogonalization_instantx_ip_adapter.inference import generate_semantic_png_bytes
    _, encoded, generated, projected = model_mock(monkeypatch)
    result, _ = fallback()
    png, diagnostics = generate_semantic_png_bytes(
        identity_term="figurine", target_prompt=result.analysis.target_prompt,
        reference_image_bytes=image_bytes(), contrast_analysis_payload=result.to_dict())
    assert png.startswith(b"\x89PNG") and len(projected) == 1
    assert generated[0]["prompt"] == result.analysis.target_prompt
    assert "a white figurine" in encoded
    assert result.analysis.target_prompt not in encoded
    assert diagnostics["orthogonalization"]["target_projection_skipped"]
