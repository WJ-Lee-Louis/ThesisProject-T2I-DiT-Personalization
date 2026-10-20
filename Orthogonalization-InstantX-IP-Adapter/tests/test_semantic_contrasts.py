"""Offline contract tests. Authored responses do not measure LLM semantics."""
from copy import deepcopy
from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path

from PIL import Image
import pytest

from orthogonalization_instantx_ip_adapter.contrast_prompt import (
    CAPTION_EXAMPLES, CAPTION_SCHEMA, CAPTION_SYSTEM_PROMPT,
    CONTRAST_EXAMPLES, CONTRAST_SCHEMA, CONTRAST_SYSTEM_PROMPT,
)
from orthogonalization_instantx_ip_adapter.llm_provider import LLMTransportError, ResponsesJSONProvider
from orthogonalization_instantx_ip_adapter.semantic_contrasts import (
    CaptionRecord, IdentityConflictError, InvalidLLMOutputError, JointContrastAnalysis,
    LegacyCaptionProvenance, MAX_REMOVALS, ReferenceFingerprint, Removal,
    SemanticContrastConditioning, SemanticLLM,
    identity_occurrences, parse_json_object, prepare_semantic_conditioning,
)


class Provider:
    model = "test-model"

    def __init__(self, *outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        value = next(self.outputs)
        if isinstance(value, Exception):
            raise value
        return value if isinstance(value, str) else json.dumps(value)


@pytest.mark.parametrize("identity", ["cherry blossoms", "headphones", "sunglasses"])
def test_target_initial_capital_full_pipeline_and_roundtrip(identity):
    target = identity[0].upper() + identity[1:] + " on a table"
    output = dict(identity_term=identity, source_preserved=[identity], source=[],
                  target=[dict(unit="table placement", negative=target.removesuffix(" on a table"))], issues=[])
    provider = Provider(dict(source_caption=identity, preserve=[]), output)
    result = prepare_semantic_conditioning(llm=SemanticLLM(provider), identity_term=identity,
                                          target_prompt=target, reference_image_bytes=image_bytes())
    restored = SemanticContrastConditioning.from_dict(result.to_dict())
    assert restored.analysis.target_prompt == target
    assert restored.analysis.target[0].negative == target.removesuffix(" on a table")
    assert len(provider.calls) == 2


@pytest.mark.parametrize("text", ["a Headphones", "HEADPHONES", "HeadPhones", "Headphoness"])
def test_initial_capital_does_not_relax_other_spelling(text):
    with pytest.raises(ValueError):
        identity_occurrences(text, "headphones", allow_initial_capital=True)


def test_caption_identity_stays_case_sensitive():
    with pytest.raises(ValueError):
        CaptionRecord.manual("Headphones", "headphones", preserve=[])


def image_bytes(color="red", format="PNG"):
    buffer = BytesIO()
    Image.new("RGB", (3, 2), color).save(buffer, format=format)
    return buffer.getvalue()


def example(index=0):
    return deepcopy(CONTRAST_EXAMPLES[index])


def prepared(index=0, origin="source_target"):
    inp, out = example(index)
    provider = Provider({"source_caption": inp["source_caption"], "preserve": inp["preserve"]}, out)
    result = prepare_semantic_conditioning(llm=SemanticLLM(provider), identity_term=inp["identity_term"],
                                          target_prompt=inp["target_prompt"], reference_image_bytes=image_bytes(),
                                          origin=origin)
    return result, provider


def assert_schema(value, schema, root=None):
    """Exercise the actual emitted schema without an optional jsonschema dependency."""
    root = schema if root is None else root
    if "$ref" in schema:
        schema = root["$defs"][schema["$ref"].rsplit("/", 1)[-1]]
    if schema["type"] == "object":
        assert isinstance(value, dict)
        assert set(value) == set(schema["required"]) == set(schema["properties"])
        assert schema["additionalProperties"] is False
        for key, item in value.items():
            assert_schema(item, schema["properties"][key], root)
    elif schema["type"] == "array":
        assert isinstance(value, list)
        for item in value:
            assert_schema(item, schema["items"], root)
    elif schema["type"] == "string":
        assert isinstance(value, str)
        if "enum" in schema:
            assert value in schema["enum"]
    else:
        pytest.fail("unexpected schema type")


@pytest.mark.parametrize("inp,out", [pair for pair in CONTRAST_EXAMPLES if not pair[1]["issues"]])
def test_executable_joint_few_shots(inp, out):
    assert_schema(out, CONTRAST_SCHEMA)
    provider = Provider(out)
    result = SemanticLLM(provider).contrast(**inp)
    assert len(provider.calls) == 1
    assert provider.calls[0]["payload"] == inp
    assert provider.calls[0]["schema"] == CONTRAST_SCHEMA
    assert provider.calls[0]["image_bytes"] is None
    assert result.response_dict() == out
    assert result.inputs == inp
    assert JointContrastAnalysis.from_dict(result.to_dict()) == result


@pytest.mark.parametrize("inp,out", CAPTION_EXAMPLES)
def test_caption_wording_fixtures_and_image_only_payload(inp, out):
    assert_schema(out, CAPTION_SCHEMA)
    provider = Provider(out)
    raw = image_bytes()
    result = SemanticLLM(provider).caption(raw, inp["identity_term"])
    assert result.text == out["source_caption"]
    assert result.origin == "generated_v2"
    assert provider.calls[0]["payload"] == {"identity_term": inp["identity_term"]}
    assert provider.calls[0]["image_bytes"] == raw
    assert CaptionRecord.from_dict(result.to_dict()) == result


@pytest.mark.parametrize("origin,count", [("source", 2), ("target", 1), ("source_target", 3)])
def test_all_origins_use_a_then_one_joint_b(origin, count):
    result, provider = prepared(origin=origin)
    inp, _ = example()
    assert len(provider.calls) == 2
    assert provider.calls[0]["payload"] == {"identity_term": inp["identity_term"]}
    assert provider.calls[1]["payload"] == inp
    assert result.direction_count == count
    assert len(result.analysis.source) == 2 and len(result.analysis.target) == 1
    assert SemanticContrastConditioning.from_dict(result.to_dict()) == result


@pytest.mark.parametrize("origin", ["source", "target", "source_target"])
def test_manual_caption_skips_only_a_and_preserves_sentence(origin):
    inp, out = example(3)
    provider = Provider(out)
    result = prepare_semantic_conditioning(llm=SemanticLLM(provider), **inp, origin=origin)
    assert result.caption.text == "A brown dog with its mouth open rests on grass."
    assert result.caption.trace is None
    assert result.caption.origin == "manual"
    assert result.caption.reference is None
    assert provider.calls[0]["payload"] == inp
    assert len(provider.calls) == 1
    with pytest.raises(ValueError, match="reference binding"):
        result.validate_reuse(identity_term=inp["identity_term"], target_prompt=inp["target_prompt"],
                              reference=ReferenceFingerprint.from_image(image_bytes()))


def test_caption_cache_is_target_independent_b_reanalyzes_target():
    first, _ = prepared()
    inp, out = example(1)
    provider = Provider(out)
    second = prepare_semantic_conditioning(llm=SemanticLLM(provider), identity_term=inp["identity_term"],
                                          target_prompt=inp["target_prompt"], caption_record=first.caption,
                                          reference_image_bytes=image_bytes(), origin="target")
    assert len(provider.calls) == 1
    assert second.caption == first.caption
    assert second.analysis.target_prompt != first.analysis.target_prompt
    assert second.direction_count == 1


def test_full_cache_roundtrip_and_selection_override_require_no_llm():
    result, provider = prepared()
    cached = SemanticContrastConditioning.from_dict(parse_json_object(json.dumps(result.to_dict())))
    cached.validate_reuse(identity_term="backpack", target_prompt=result.analysis.target_prompt,
                          source_caption=result.caption.text, reference=ReferenceFingerprint.from_image(image_bytes()),
                          model="test-model")
    assert len(provider.calls) == 2
    assert replace(cached, origin="target").direction_count == 1
    assert replace(cached, origin="target").analysis == result.analysis


@pytest.mark.parametrize("override", [
    {"origin": "bad"}, {"strategy": "factorial"}, {"identity_term": "Backpack"},
    {"source_caption": "a dog"}, {"source_caption": " backpack"},
    {"reference_image_bytes": b"invalid image"}, {"reference_image_bytes": b""},
    {"target_prompt": "backpackhouse"}, {"target_prompt": " A backpack in a forest"},
])
def test_invalid_inputs_fail_before_any_provider_call(override):
    provider = Provider()
    kwargs = dict(llm=SemanticLLM(provider), identity_term="backpack", target_prompt="A backpack in a forest",
                  reference_image_bytes=image_bytes())
    kwargs.update(override)
    with pytest.raises(ValueError):
        prepare_semantic_conditioning(**kwargs)
    assert provider.calls == []


def test_target_only_now_requires_source_or_image():
    provider = Provider()
    with pytest.raises(ValueError, match="including target"):
        prepare_semantic_conditioning(llm=SemanticLLM(provider), identity_term="dog",
                                      target_prompt="a dog", origin="target")
    assert provider.calls == []


@pytest.mark.parametrize("value", [-1, 6, True, 1.5, "2"])
def test_invalid_retry_settings(value):
    with pytest.raises(ValueError):
        SemanticLLM(Provider(), max_retries=value)


@pytest.mark.parametrize("bad", [
    "{}", "[]", '{"identity_term":"dog","identity_term":"dog"}',
    '{"invalid":NaN}', '```json\n{}\n```', "not JSON",
])
def test_strict_json_and_retry_feedback(bad):
    inp, out = example()
    provider = Provider(bad, out)
    result = SemanticLLM(provider, max_retries=1).contrast(**inp)
    assert len(provider.calls) == 2
    assert provider.calls[0]["feedback"] is None
    assert provider.calls[1]["feedback"]
    assert provider.calls[1]["payload"] == inp
    assert result.trace.attempts[0].raw_response == bad
    assert result.trace.attempts[0].validation_error


def mutate_response(value, mutation):
    if mutation == "extra": value["explanation"] = "no"
    elif mutation == "missing": del value["target"]
    elif mutation == "identity": value["identity_term"] = "Backpack"
    elif mutation == "list_type": value["source"] = {}
    elif mutation == "none": value["source"] = None
    elif mutation == "row_type": value["source"][0] = "standing"
    elif mutation == "row_extra": value["source"][0]["spans"] = []
    elif mutation == "unit_type": value["source"][0]["unit"] = 3
    elif mutation == "blank": value["source"][0]["negative"] = " "
    elif mutation == "outer": value["source"][0]["negative"] += " "
    elif mutation == "no_identity": value["source"][0]["negative"] = "A bag"
    elif mutation == "subword": value["source"][0]["negative"] = "A backpackhouse"
    elif mutation == "lost_preserved": value["source"][0]["negative"] = "A backpack"
    elif mutation == "bad_preserved": value["source_preserved"].append("not observed")
    elif mutation == "preserved_type": value["source_preserved"].append({})
    elif mutation == "no_preserved_identity": value["source_preserved"].remove("backpack")
    elif mutation == "duplicate": value["source"].append(deepcopy(value["source"][0]))
    elif mutation == "duplicate_unit": value["source"][1]["unit"] = value["source"][0]["unit"]
    elif mutation == "duplicate_negative": value["source"][1]["negative"] = value["source"][0]["negative"]
    elif mutation == "unchanged": value["source"][0]["negative"] = example()[0]["source_caption"]
    elif mutation == "too_many": value["source"] *= MAX_REMOVALS
    else: raise AssertionError(mutation)


@pytest.mark.parametrize("mutation", [
    "extra", "missing", "identity", "list_type", "none", "row_type", "row_extra", "unit_type", "blank",
    "outer", "no_identity", "subword", "lost_preserved", "bad_preserved", "preserved_type",
    "no_preserved_identity", "duplicate", "duplicate_unit", "duplicate_negative", "unchanged", "too_many",
])
def test_malformed_joint_response_rejected(mutation):
    inp, out = example()
    mutate_response(out, mutation)
    provider = Provider(out)
    if mutation == "lost_preserved":
        # Semantic loss is not structurally decidable now that grammatical rewriting is allowed.
        result = SemanticLLM(provider, max_retries=0).contrast(**inp)
        assert result.source[0].negative == "A backpack"
        return
    with pytest.raises(InvalidLLMOutputError) as caught:
        SemanticLLM(provider, max_retries=0).contrast(**inp)
    assert len(provider.calls) == 1
    assert caught.value.trace.attempts[-1].validation_error


def test_b_retry_does_not_repeat_a_and_exhaustion_is_auditable():
    inp, out = example()
    provider = Provider({"source_caption": inp["source_caption"], "preserve": inp["preserve"]}, {}, out)
    result = prepare_semantic_conditioning(llm=SemanticLLM(provider), identity_term=inp["identity_term"],
                                          target_prompt=inp["target_prompt"], reference_image_bytes=image_bytes())
    assert len(provider.calls) == 3
    assert len(result.caption.trace.attempts) == 1
    assert len(result.analysis.trace.attempts) == 2
    bad = Provider({}, {}, {})
    with pytest.raises(InvalidLLMOutputError) as caught:
        SemanticLLM(bad).contrast(**inp)
    assert len(caught.value.trace.attempts) == 3


def test_caption_validation_retry_keeps_case_without_rewriting():
    provider = Provider({"source_caption": "Cherry blossoms on a branch", "preserve": []},
                        {"source_caption": "cherry blossoms on a branch", "preserve": []})
    result = SemanticLLM(provider).caption(image_bytes(), "cherry blossoms")
    assert len(provider.calls) == 2
    assert result.text.startswith("cherry blossoms")


def issue_response(identity, kind, evidence):
    return dict(identity_term=identity, source_preserved=[], source=[], target=[],
                issues=[dict(kind=kind, side="target", evidence=evidence, reason="Authored test issue.")])


@pytest.mark.parametrize("kind,error,identity,source,target,evidence", [
    ("identity_conflict", IdentityConflictError, "brown dog", "a brown dog sitting on grass",
     "a brown dog whose fur is entirely purple instead of brown", "whose fur is entirely purple instead of brown"),
])
def test_valid_issues_stop_without_retry_or_partial_result(kind, error, identity, source, target, evidence):
    out = issue_response(identity, kind, evidence)
    assert_schema(out, CONTRAST_SCHEMA)
    provider = Provider(out)
    with pytest.raises(error) as caught:
        SemanticLLM(provider).contrast(source, target, identity, preserve=[])
    assert len(provider.calls) == 1
    assert caught.value.issues[0].evidence == evidence
    assert caught.value.trace.attempts[0].validation_error is None
    assert json.loads(caught.value.trace.attempts[0].raw_response) == out


@pytest.mark.parametrize("mutation", ["evidence", "partial", "kind", "side", "blank_reason", "extra"])
def test_malformed_issue_is_output_error_not_domain_error(mutation):
    inp, success = example()
    out = issue_response("backpack", "identity_conflict", "in a forest")
    if mutation == "evidence": out["issues"][0]["evidence"] = "not in target"
    elif mutation == "partial": out["source_preserved"] = ["backpack"]
    elif mutation == "kind": out["issues"][0]["kind"] = "unknown"
    elif mutation == "side": out["issues"][0]["side"] = "source"
    elif mutation == "blank_reason": out["issues"][0]["reason"] = ""
    elif mutation == "extra": out["issues"][0]["extra"] = 1
    provider = Provider(out, success)
    result = SemanticLLM(provider).contrast(**inp)
    assert len(result.trace.attempts) == 2


def test_transport_error_is_not_an_output_retry():
    provider = Provider(LLMTransportError("offline test"))
    with pytest.raises(LLMTransportError):
        SemanticLLM(provider).contrast(**example()[0])
    assert len(provider.calls) == 1


def test_empty_provider_validation_error_is_recorded_and_retried():
    inp, out = example()
    provider = Provider(ValueError(), out)
    result = SemanticLLM(provider).contrast(**inp)
    assert len(provider.calls) == 2
    assert result.trace.attempts[0].raw_response is None
    assert result.trace.attempts[0].validation_error == "ValueError"


def test_features_released_only_in_their_own_local_negative():
    forest = JointContrastAnalysis.from_response(example()[1], **example()[0])
    purple = JointContrastAnalysis.from_response(example(1)[1], **example(1)[0])
    assert "magenta" in forest.source_preserved
    assert "magenta" not in purple.source_preserved
    assert "magenta" not in purple.source[0].negative
    assert all("magenta" in row.negative for row in purple.source[1:])
    assert all("gold zippers" in row.negative and "colorful pins" in row.negative for row in purple.source)
    assert purple.target[0].negative == "A backpack"
    assert purple.target_prompt == "A purple backpack"


def test_contents_grammar_style_and_empty_fixtures():
    bowl, _ = prepared(2)
    assert bowl.analysis.source[0].negative == "A white bowl filled with blueberries"
    grammar, _ = prepared(3)
    assert grammar.analysis.source[0].negative == "A brown dog rests on grass."
    assert grammar.analysis.source[1].negative == "A brown dog with its mouth open is on grass."
    assert grammar.analysis.source[2].negative == "A brown dog with its mouth open rests."
    assert grammar.analysis.target[0].negative.startswith("a dog")
    assert "and" not in grammar.analysis.target[1].negative
    assert "brown" in grammar.analysis.source_preserved  # Hat is not a fur recolor.
    style, _ = prepared(4)
    assert style.analysis.source[0].negative == "soft pastel tones"
    empty, provider = prepared(5)
    assert empty.direction_count == 0
    assert len(provider.calls) == 2


def test_same_text_on_both_sides_is_not_semantically_filtered():
    response = dict(identity_term="dog", source_preserved=["dog"],
                    source=[dict(unit="sitting", negative="a dog")],
                    target=[dict(unit="sitting", negative="a dog")], issues=[])
    inp = dict(identity_term="dog", source_caption="a dog sitting", target_prompt="a dog sitting", preserve=[])
    result = prepare_semantic_conditioning(llm=SemanticLLM(Provider(response)), **inp)
    assert result.direction_count == 2


def test_repeated_identity_count_and_exact_article():
    response = dict(identity_term="dog", source_preserved=["dog"], source=[],
                    target=[dict(unit="brown", negative="a dog")], issues=[])
    with pytest.raises(ValueError, match="all identity occurrences"):
        JointContrastAnalysis.from_response(response, preserve=[], source_caption="dog", target_prompt="a brown dog beside a dog",
                                            identity_term="dog")
    with pytest.raises(ValueError):
        identity_occurrences("a brown dog", "a dog")
    assert identity_occurrences("a dog with brown fur", "a dog") == ((0, 5),)


@pytest.mark.parametrize("override", [
    {"identity_term": "bag"}, {"target_prompt": "A purple backpack"}, {"source_caption": "A backpack"},
    {"model": "different-model"}, {"reference": ReferenceFingerprint.from_image(image_bytes("blue"))},
])
def test_cache_reuse_binding_mismatch(override):
    result, _ = prepared()
    kwargs = dict(identity_term="backpack", target_prompt=result.analysis.target_prompt,
                  reference=ReferenceFingerprint.from_image(image_bytes()), model="test-model")
    kwargs.update(override)
    with pytest.raises(ValueError):
        result.validate_reuse(**kwargs)


def test_rgb_binding_survives_reencoding_but_not_pixel_change():
    png = ReferenceFingerprint.from_image(image_bytes())
    bmp = ReferenceFingerprint.from_image(image_bytes(format="BMP"))
    assert png.bytes_sha256 != bmp.bytes_sha256
    assert png.binding == bmp.binding
    result, _ = prepared()
    result.validate_reuse(identity_term="backpack", target_prompt=result.analysis.target_prompt, reference=bmp)


@pytest.mark.parametrize("mutation", [
    "old_version", "bool_version", "old_policy", "extra", "source", "target", "reference", "caption",
    "hash", "prompt_hash", "prompt_version", "trace_input", "raw_response", "no_trace", "limit", "strategy", "created_at",
])
def test_strict_storage_rejects_stale_or_inconsistent_data(mutation):
    result, _ = prepared()
    value = result.to_dict()
    if mutation == "old_version": value["schema_version"] = 2
    elif mutation == "bool_version": value["schema_version"] = True
    elif mutation == "old_policy": value["policy_version"] = "target_aware_source_v1"
    elif mutation == "extra": value["components"] = []
    elif mutation == "source": value["inputs"]["source_caption"] = "A backpack"
    elif mutation == "target": value["inputs"]["target_prompt"] = "A purple backpack"
    elif mutation == "reference": value["reference"]["rgb_sha256"] = "0" * 64
    elif mutation == "caption": value["caption"]["text"] = "A backpack"
    elif mutation == "hash": value["input_sha256"] = "0" * 64
    elif mutation == "prompt_hash": value["llm"]["contrast"]["prompt_sha256"] = "0" * 64
    elif mutation == "prompt_version": value["llm"]["contrast"]["prompt_version"] = "old"
    elif mutation == "trace_input": value["llm"]["contrast"]["input_sha256"] = "0" * 64
    elif mutation == "raw_response": value["llm"]["contrast"]["attempts"][-1]["raw_response"] = "{}"
    elif mutation == "no_trace": value["llm"]["contrast"] = None
    elif mutation == "limit": value["limits"]["max_removals_per_side"] = 8
    elif mutation == "strategy": value["selection"]["strategy"] = "factorial"
    elif mutation == "created_at": value["created_at"] = "2026-10-19T00:00:00"
    with pytest.raises(ValueError):
        SemanticContrastConditioning.from_dict(value)


def test_old_component_payload_and_caption_cannot_be_silently_promoted():
    with pytest.raises(ValueError):
        SemanticContrastConditioning.from_dict(dict(schema_version=2, target={}, source=None,
                                                    origin="target", strategy="local", max_contexts=256))
    with pytest.raises(ValueError):
        CaptionRecord.from_dict({"source_caption": "a dog"})
    with pytest.raises(ValueError):
        JointContrastAnalysis.from_dict({"identity_term": "dog", "components": []})


def test_explicit_legacy_caption_import_keeps_provenance_and_skips_a():
    inp, out = example()
    provenance = LegacyCaptionProvenance("old/captions.json", "a" * 64, "old-model", "b" * 64)
    reference = ReferenceFingerprint.from_image(image_bytes())
    record = CaptionRecord.import_legacy(inp["source_caption"], inp["identity_term"],
                                         provenance=provenance, reference=reference, status="PASS", preserve=inp["preserve"])
    provider = Provider(out)
    result = prepare_semantic_conditioning(llm=SemanticLLM(provider), identity_term=inp["identity_term"],
                                          target_prompt=inp["target_prompt"], caption_record=record,
                                          reference_image_bytes=image_bytes())
    assert len(provider.calls) == 1
    assert result.caption.origin == "imported_legacy"
    assert result.caption.trace is None
    assert result.caption.imported_from == provenance
    assert SemanticContrastConditioning.from_dict(result.to_dict()) == result
    with pytest.raises(ValueError, match="failed"):
        CaptionRecord.import_legacy("dog", "dog", provenance=provenance, status="FAIL")


def test_caption_cache_mismatch_and_mixed_inputs_stop_before_b():
    result, _ = prepared()
    provider = Provider()
    kwargs = dict(llm=SemanticLLM(provider), identity_term="backpack", target_prompt="A backpack",
                  caption_record=result.caption, reference_image_bytes=image_bytes("blue"))
    with pytest.raises(ValueError, match="fingerprint"):
        prepare_semantic_conditioning(**kwargs)
    kwargs["source_caption"] = "A backpack"
    with pytest.raises(ValueError, match="mutually exclusive"):
        prepare_semantic_conditioning(**kwargs)
    assert provider.calls == []


@pytest.mark.parametrize("identity,target,negatives", [
    ("kitten", "A retro-style digital art of a kitten wearing a tiny bow tie and exploring old books", [
        "a kitten wearing a tiny bow tie and exploring old books",
        "A retro-style digital art of a kitten exploring old books",
        "A retro-style digital art of a kitten wearing a tiny bow tie"]),
    ("Spider-Man", "Spider-Man weaving a giant web between the moon and Earth, silhouetted against a cosmic backdrop", [
        "Spider-Man silhouetted against a cosmic backdrop", "Spider-Man weaving a giant web between the moon and Earth"]),
    ("Magic Cube", "An anime-style drawing of a character intensely focusing on solving a glowing Magic Cube", [
        "a character intensely focusing on solving a glowing Magic Cube",
        "An anime-style drawing of a glowing Magic Cube",
        "An anime-style drawing of a character intensely focusing on solving a Magic Cube"]),
    ("Siamese cat", "A Siamese cat walking gracefully along the rim of a giant teacup floating in the sky", [
        "A Siamese cat along the rim of a giant teacup floating in the sky", "A Siamese cat walking gracefully"]),
    ("space ranger", "A depiction of a space ranger riding a comet, steering it through a field of asteroids", [
        "a space ranger riding a comet, steering it through a field of asteroids", "A depiction of a space ranger"]),
    ("a dog", "a retro-style digital drawing of a dog playing the guitar on a sofa", [
        "a dog playing the guitar on a sofa", "a retro-style digital drawing of a dog on a sofa",
        "a retro-style digital drawing of a dog playing the guitar"]),
    ("pastel", "a cat rendered in soft pastel tones", ["soft pastel tones", "a cat rendered in pastel tones"]),
    ("a fox", "a fox on a platform that is floating in a lake", ["a fox"]),
], ids=["T13", "T20", "T22", "T16", "T17", "F02", "F04", "F08"])
def test_authored_dependency_regressions_use_direct_negative_text(identity, target, negatives):
    out = dict(identity_term=identity, source_preserved=[identity], source=[],
               target=[dict(unit=f"removal {i}", negative=n) for i, n in enumerate(negatives)], issues=[])
    result = SemanticLLM(Provider(out)).contrast(identity, target, identity, preserve=[])
    assert [r.negative for r in result.target] == negatives
    assert result.target_prompt == target
    assert not hasattr(result, "prompt_for_subset")


def test_shared_prompt_documents_match_runtime_exactly():
    root = Path(__file__).resolve().parents[1]
    shared = (root / "docs/prompts/llm_modules_share.md").read_text(encoding="utf-8")
    caption = (root / "docs/prompts/module_a_caption_system_prompt.md").read_text(encoding="utf-8")
    assert CAPTION_SYSTEM_PROMPT in shared and CONTRAST_SYSTEM_PROMPT in shared
    assert json.loads(shared.split("```json\n", 1)[1].split("\n```", 1)[0]) == CAPTION_SCHEMA
    assert caption.split("```text\n", 1)[1].split("\n```", 1)[0] == CAPTION_SYSTEM_PROMPT
    contrast = (root / "docs/prompts/module_b_contrast_system_prompt.md").read_text(encoding="utf-8")
    assert contrast.split("```text\n", 1)[1].split("\n```", 1)[0] == CONTRAST_SYSTEM_PROMPT


def test_joint_schema_passes_through_existing_structured_http_provider(monkeypatch):
    inp, out = example()
    requests = []

    class Response(BytesIO):
        pass

    def urlopen(request, timeout):
        requests.append(json.loads(request.data))
        return Response(json.dumps({"status": "completed", "output": [{"type": "message", "content": [
            {"type": "output_text", "text": json.dumps(out)}]}]}).encode())

    monkeypatch.setattr("orthogonalization_instantx_ip_adapter.llm_provider.urlopen", urlopen)
    result = SemanticLLM(ResponsesJSONProvider("test-model", "test-secret")).contrast(**inp)
    assert result.response_dict() == out
    assert len(requests) == 1
    assert requests[0]["text"]["format"]["schema"] == CONTRAST_SCHEMA
    assert requests[0]["text"]["format"]["strict"] is True
    assert requests[0]["store"] is False
    assert len(requests[0]["input"][1]["content"]) == 1


@pytest.mark.parametrize("raw, expected", [
    ('{"source_caption":"a dog","preserve":[]}', "PASS"),
    ('{"source_caption":" a dog"}', "INVALID"),
    ('{"source_caption":"a dog","source_caption":"a dog"}', "INVALID"),
    ('{"source_caption":"a dog","extra":1}', "INVALID"),
])
def test_caption_monitor_uses_current_strict_contract(tmp_path, monkeypatch, raw, expected):
    from scripts import monitor_reference_captions as monitor
    data = image_bytes()
    (tmp_path / "reference.png").write_bytes(data)
    monkeypatch.setattr(monitor, "ROOT", tmp_path)
    provider = Provider(raw)
    result = monitor.run_case(dict(image_path="reference.png", image_sha256=monitor.sha(data),
                                   identity_term="dog"), provider, CAPTION_SYSTEM_PROMPT,
                              dict(prompt_sha256=monitor.sha(CAPTION_SYSTEM_PROMPT.encode())))
    assert result["status"] == expected
    assert result["raw_response"] == raw
    assert len(provider.calls) == 1  # First-response audit never silently repairs/retries.
    assert provider.calls[0]["payload"] == {"identity_term": "dog"}


def test_structure_validation_is_not_a_semantic_judge():
    # Deliberately bad meaning: both unrelated context units disappear at once.
    # The parser cannot prove single-unit deletion or completeness of preservation.
    out = dict(identity_term="dog", source_preserved=["dog", "brown"],
               source=[dict(unit="sitting pose", negative="A brown dog")], target=[], issues=[])
    result = JointContrastAnalysis.from_response(out, preserve=["brown"], source_caption="A brown dog sitting on grass",
                                                target_prompt="A dog", identity_term="dog")
    assert result.source[0].negative == "A brown dog"
    # Acceptance here documents the limit; this is NOT a desired contrast fixture.


@pytest.mark.parametrize("source,target,identity,negatives", [
    ("pixel art", "A magical portal opening in a pixel art enchanted forest", "pixel art",
     ["a pixel art enchanted forest", "A magical portal opening in a pixel art forest"]),
    ("pixel art", "A tiny dragon perched on a pixel art cloud", "pixel art", ["a pixel art cloud"]),
    ("sunset", "A giant phoenix soaring across the sky, its feathers echoing the colors of the sunset", "sunset", ["the sunset"]),
    ("Rolls-Royce hood ornament", "A Rolls-Royce hood ornament transformed into a sleek spaceship leading a caravan of vintage cars across a galaxy",
     "Rolls-Royce hood ornament", ["A Rolls-Royce hood ornament"]),
    ("dog", "A dog with red ears and paws", "dog", ["A dog"]),
])
def test_authored_conservative_grouping(source, target, identity, negatives):
    # Human policy fixtures, not observations of model judgment.
    out = dict(identity_term=identity, source_preserved=[identity], source=[],
               target=[dict(unit=f"group {i}", negative=n) for i, n in enumerate(negatives)], issues=[])
    result = SemanticLLM(Provider(out)).contrast(source, target, identity, preserve=[])
    assert [r.negative for r in result.target] == negatives
    assert result.target_prompt == target


def test_old_scope_fallback_is_rejected_then_retried_under_new_policy():
    inp, good = example(6)
    old = issue_response("pixel art", "unrepresentable_contrast", "pixel art lake")
    provider = Provider(old, good)
    result = SemanticLLM(provider).contrast(**inp)
    assert len(provider.calls) == 2
    assert result.target[0].negative == "a pixel art lake"
    assert result.trace.attempts[0].validation_error


def test_previous_joint_policy_cache_is_not_reused():
    result, _ = prepared()
    value = result.to_dict()
    value["policy_version"] = "joint_local_v1"
    with pytest.raises(ValueError, match="policy version"):
        SemanticContrastConditioning.from_dict(value)


def test_antecedent_expansion_can_repeat_protected_identity():
    response = dict(identity_term="dog", source_preserved=["dog"], source=[],
                    target=[dict(unit="grass placement", negative="A dog watching the dog's reflection")], issues=[])
    result = JointContrastAnalysis.from_response(response, preserve=[], source_caption="dog",
        target_prompt="A dog on grass watching its reflection", identity_term="dog")
    assert result.target[0].negative == "A dog watching the dog's reflection"
    # This verifies lexical compatibility only; antecedent correctness is semantic.


@pytest.mark.parametrize("preserve", [None, "magenta", [""], [" magenta"], [1],
                                       ["magenta", "magenta"], ["not visible"],
                                       ["backpack"], ["magenta backpack"], ["magenta"] * 33])
def test_invalid_preserve_rejected_before_b(preserve):
    inp, out = example()
    inp["preserve"] = preserve
    provider = Provider(out)
    with pytest.raises(ValueError):
        SemanticLLM(provider).contrast(**inp)
    assert provider.calls == []


def test_a_preserve_is_forwarded_to_b_and_bound_in_storage():
    c, provider = prepared()
    assert c.caption.preserve == ("magenta", "with gold zippers and colorful pins")
    assert provider.calls[0]["payload"] == {"identity_term": "backpack"}
    assert provider.calls[1]["payload"]["preserve"] == list(c.caption.preserve)
    assert c.analysis.preserve == c.caption.preserve
    assert c.to_dict()["inputs"]["preserve"] == list(c.caption.preserve)
    assert SemanticContrastConditioning.from_dict(c.to_dict()) == c
    with pytest.raises(ValueError, match="preserve mismatch"):
        c.validate_reuse(identity_term="backpack", target_prompt=c.analysis.target_prompt,
                         reference=ReferenceFingerprint.from_image(image_bytes()), preserve=["magenta"])
    tampered = c.to_dict()
    tampered["caption"]["preserve"] = ["magenta"]
    with pytest.raises(ValueError):
        SemanticContrastConditioning.from_dict(tampered)


def test_manual_preserve_is_required_and_empty_is_explicit_identity_only():
    provider = Provider(dict(identity_term="dog", source_preserved=["dog"], source=[], target=[], issues=[]))
    with pytest.raises(ValueError, match="explicit list"):
        prepare_semantic_conditioning(llm=SemanticLLM(provider), source_caption="dog",
                                      target_prompt="dog", identity_term="dog")
    assert not provider.calls
    c = prepare_semantic_conditioning(llm=SemanticLLM(provider), source_caption="dog",
                                      target_prompt="dog", identity_term="dog", preserve=[])
    assert c.caption.preserve == () and len(provider.calls) == 1


def test_preserved_relation_allows_grammatical_glue_repair():
    # The original preposition is not an immutable token; semantics are authored here.
    inp = dict(source_caption="A dog with brown fur and with white paws sitting", target_prompt="A dog",
               identity_term="dog", preserve=["and with white paws"])
    out = dict(identity_term="dog", source_preserved=["dog", "and with white paws"],
               source=[dict(unit="brown fur", negative="A dog with white paws sitting")], target=[], issues=[])
    c = SemanticLLM(Provider(out)).contrast(**inp)
    assert c.source[0].negative == "A dog with white paws sitting"
    # The protected white paws survive while the conjunction changes.


def test_caption_retries_missing_or_identity_repeating_preserve():
    good = {"source_caption": "A magenta backpack", "preserve": ["magenta"]}
    provider = Provider({"source_caption": "A magenta backpack"},
                        {"source_caption": "A magenta backpack", "preserve": ["magenta backpack"]}, good)
    c = SemanticLLM(provider).caption(image_bytes(), "backpack")
    assert c.preserve == ("magenta",) and len(provider.calls) == 3
    assert len(c.trace.attempts) == 3


def test_old_caption_schema_does_not_acquire_silent_empty_preserve():
    c, _ = prepared()
    old = c.caption.to_dict()
    old["caption_schema_version"] = 1
    old.pop("preserve")
    with pytest.raises(ValueError):
        CaptionRecord.from_dict(old)



def test_authored_partial_conflict_keeps_rest_of_preserve_phrase():
    # Authored semantics: only white conflicts; pattern/contents survive in both negatives.
    inp = dict(source_caption="A white patterned bowl filled with blueberries on fabric",
               target_prompt="A purple bowl", identity_term="bowl",
               preserve=["white patterned", "filled with blueberries"])
    out = dict(identity_term="bowl", source_preserved=["bowl", "patterned", "filled with blueberries"],
               source=[dict(unit="white body color", negative="A patterned bowl filled with blueberries on fabric"),
                       dict(unit="fabric placement", negative="A white patterned bowl filled with blueberries")],
               target=[dict(unit="purple body color", negative="A bowl")], issues=[])
    result = SemanticLLM(Provider(out)).contrast(**inp)
    assert result.preserve == ("white patterned", "filled with blueberries")
    assert "white" not in result.source[0].negative
    assert "white" in result.source[1].negative
    assert all("patterned" in r.negative and "filled with blueberries" in r.negative for r in result.source)
    assert result.target_prompt == inp["target_prompt"]



def test_concept_caption_passes_empty_baseline_to_joint_b():
    inp = dict(identity_term="low-poly geometric",
               source_caption="an island with angular trees in low-poly geometric style",
               target_prompt="a fox in low-poly geometric style")
    output = dict(identity_term=inp["identity_term"],
                  source_preserved=["low-poly geometric", "style"],
                  source=[dict(unit="depicted island and trees", negative="low-poly geometric style")],
                  target=[dict(unit="depicted fox", negative="low-poly geometric style")], issues=[])
    provider = Provider(dict(source_caption=inp["source_caption"], preserve=[]), output)
    c = prepare_semantic_conditioning(llm=SemanticLLM(provider), identity_term=inp["identity_term"],
        target_prompt=inp["target_prompt"], reference_image_bytes=image_bytes())
    assert len(provider.calls) == 2
    assert c.caption.preserve == c.analysis.preserve == ()
    assert provider.calls[1]["payload"]["preserve"] == []
    assert c.analysis.source[0].negative == "low-poly geometric style"
    assert c.analysis.target_prompt == inp["target_prompt"]
    # Authored policy example; no assertion about actual model classification accuracy.


def test_pre_concept_caption_policy_requires_regeneration():
    c, _ = prepared()
    old = c.caption.to_dict()
    old["caption_policy_version"] = "identity_np_preserve_v3"
    with pytest.raises(ValueError, match="caption policy"):
        CaptionRecord.from_dict(old)
