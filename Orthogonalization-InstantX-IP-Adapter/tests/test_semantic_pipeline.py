from io import BytesIO
import json
from types import SimpleNamespace
from urllib.error import HTTPError

from PIL import Image
import pytest
import torch

from orthogonalization_instantx_ip_adapter.adapter import OrthogonalizationRequest, OrthogonalizedIPAdapterMixin
from orthogonalization_instantx_ip_adapter.component_directions import build_component_directions, component_contrast_plan
from orthogonalization_instantx_ip_adapter.llm_provider import ResponsesJSONProvider, LLMTransportError
from orthogonalization_instantx_ip_adapter.semantic_components import (
    ComponentAnalysis, SemanticConditioning, SemanticLLM, prepare_semantic_conditioning,
)
from orthogonalization_instantx_ip_adapter.orthogonalization import build_factorial_directions, build_local_directions


class FakeProvider:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        output = next(self.outputs)
        return output if isinstance(output, str) else json.dumps(output)


IDENTITY = "a dog"
SOURCE = "a dog sleeping on the snow"
TARGET = "a retro-style digital drawing of a dog playing the guitar on a sofa"
SOURCE_COMPONENTS = ["sleeping", "on the snow"]
TARGET_COMPONENTS = ["a retro-style digital drawing of", "playing the guitar", "on a sofa"]


@pytest.mark.parametrize("explicit, environment, expected", [
    (None, None, "gpt-6-luna"),
    (None, "", "gpt-6-luna"),
    (None, "environment-model", "environment-model"),
    ("explicit-model", "environment-model", "explicit-model"),
])
def test_provider_model_configuration_precedence(monkeypatch, explicit, environment, expected):
    monkeypatch.setenv("SEMANTIC_LLM_API_KEY", "test-secret")
    monkeypatch.delenv("SEMANTIC_LLM_ENDPOINT", raising=False)
    if environment is None:
        monkeypatch.delenv("SEMANTIC_LLM_MODEL", raising=False)
    else:
        monkeypatch.setenv("SEMANTIC_LLM_MODEL", environment)
    assert ResponsesJSONProvider.from_env(explicit).model == expected


def test_default_model_still_requires_api_key(monkeypatch):
    monkeypatch.delenv("SEMANTIC_LLM_API_KEY", raising=False)
    monkeypatch.delenv("SEMANTIC_LLM_MODEL", raising=False)
    with pytest.raises(ValueError, match="API key"):
        ResponsesJSONProvider.from_env()


def output(identity, components):
    return {"identity_term": identity, "components": components}


def conditioning(strategy="local", origin="source_target"):
    return SemanticConditioning(ComponentAnalysis(TARGET, IDENTITY, tuple(TARGET_COMPONENTS)),
                                ComponentAnalysis(SOURCE, IDENTITY, tuple(SOURCE_COMPONENTS)), origin, strategy)


def source_output():
    return dict(identity_term=IDENTITY, preserved_spans=[IDENTITY], retained_spans=[], neutral_spans=[],
                components=[{"spans": [c]} for c in SOURCE_COMPONENTS], conflicts=[])


def source_conditioning():
    from orthogonalization_instantx_ip_adapter.semantic_components import ComponentGroup
    return SemanticConditioning(ComponentAnalysis(TARGET, IDENTITY, tuple(TARGET_COMPONENTS)),
        ComponentAnalysis(SOURCE, IDENTITY, tuple(ComponentGroup((c,)) for c in SOURCE_COMPONENTS),
                          (IDENTITY,), (), (), TARGET))


def test_case1_zero_shot_and_manual_caption():
    provider = FakeProvider([{"source_caption": SOURCE}, source_output(), output(IDENTITY, TARGET_COMPONENTS)])
    result = prepare_semantic_conditioning(llm=SemanticLLM(provider), identity_term=IDENTITY,
                                          target_prompt=TARGET, reference_image_bytes=b"fake image")
    assert result == source_conditioning()
    assert provider.calls[0]["image_bytes"] == b"fake image"
    assert [call["payload"] for call in provider.calls[1:]] == [
        {"source_caption": SOURCE, "target_prompt": TARGET, "identity_term": IDENTITY}, {"text": TARGET, "identity_term": IDENTITY}]
    manual = FakeProvider([source_output(), output(IDENTITY, TARGET_COMPONENTS)])
    assert prepare_semantic_conditioning(llm=SemanticLLM(manual), identity_term=IDENTITY,
                                        target_prompt=TARGET, source_caption=SOURCE) == result
    assert len(manual.calls) == 2
    assert SemanticConditioning.from_dict(result.to_dict()) == result


def test_case2_long_free_form_cat():
    text = "In a softly lit room, a cat wearing a blue scarf is sitting beside a gramophone, with tall windows overlooking a garden, rendered as an expressive oil painting with visible brush strokes."
    phrases = ["In a softly lit room,", "wearing a blue scarf", "is sitting beside a gramophone,",
               "with tall windows overlooking a garden,", "rendered as an expressive oil painting with visible brush strokes."]
    provider = FakeProvider([output("a cat", phrases)])
    result = prepare_semantic_conditioning(llm=SemanticLLM(provider), identity_term="a cat", target_prompt=text, origin="target")
    assert result.target.components == tuple(phrases)
    assert result.source is None


@pytest.mark.parametrize("text", ["a cat on the snow", "a doghouse", "A dog sleeping"])
def test_case3_identity_absent_stops_before_llm(text):
    provider = FakeProvider([])
    with pytest.raises(ValueError, match="verbatim"):
        prepare_semantic_conditioning(llm=SemanticLLM(provider), identity_term=IDENTITY, target_prompt=text)
    assert provider.calls == []


@pytest.mark.parametrize("components", [("sleeping",), ("sleeping", "on snow", "at night", "in oil paint"), ()])
@pytest.mark.parametrize("strategy", ["local", "factorial"])
def test_case4_case5_arbitrary_count(components, strategy):
    text = " ".join((IDENTITY, *components))
    result = SemanticConditioning(ComponentAnalysis(text, IDENTITY, components), origin="target", strategy=strategy)
    lookup = {}
    def encode(prompts):
        for prompt in prompts:
            if prompt not in lookup:
                generator = torch.Generator().manual_seed(len(lookup))
                lookup[prompt] = torch.randn(8, generator=generator)
        return torch.stack([lookup[prompt] for prompt in prompts])
    directions = build_component_directions(result, encode)
    assert directions.values.shape == (len(components), 8)


def test_case6_style_identity_keeps_subject_as_removable_component():
    text = "watercolor style depiction of a cat sleeping on a sofa"
    phrases = ["depiction of a cat", "sleeping", "on a sofa"]
    provider = FakeProvider([output("watercolor style", phrases)])
    result = prepare_semantic_conditioning(llm=SemanticLLM(provider), identity_term="watercolor style", target_prompt=text, origin="target")
    assert all("watercolor style" in p for _, pairs in component_contrast_plan(result) for pair in pairs for p in pair)


def grouped_output(identity, preserved, groups):
    return {"identity_term": identity, "preserved_spans": preserved,
            "components": [{"spans": spans} for spans in groups]}


@pytest.mark.parametrize("strategy", ["local", "factorial"])
def test_grouped_style_location_is_one_direction_and_posture_is_separate(strategy):
    text = "A tiny dragon perched on a pixel art cloud"
    provider = FakeProvider([grouped_output("pixel art", ["pixel art"],
        [["A tiny dragon"], ["perched"], ["on a", "cloud"]])])
    analysis = SemanticLLM(provider).analyze(text, "pixel art")
    assert analysis.prompt_for_subset([0, 1, 2]) == text
    assert analysis.prompt_for_subset([0, 1]) == "A tiny dragon perched pixel art"
    assert analysis.prompt_for_subset([0, 2]) == "A tiny dragon on a pixel art cloud"
    assert analysis.prompt_for_subset([]) == "pixel art"
    conditioning = SemanticConditioning(analysis, origin="target", strategy=strategy)
    plan = component_contrast_plan(conditioning)
    assert len(plan) == 3  # Three semantic groups, not four span fragments.
    for positive, negative in plan[2][1]:
        assert "on a" in positive and "cloud" in positive
        assert "on a" not in negative and "cloud" not in negative
        assert "pixel art" in positive and "pixel art" in negative
    assert SemanticConditioning.from_dict(conditioning.to_dict()) == conditioning


@pytest.mark.parametrize("text,identity,preserved,groups", [
    ("a cat sleeping in watercolor style", "watercolor style", ["in watercolor style"], [["a cat"], ["sleeping"]]),
    ("A cup of tea on a pastel tablecloth", "pastel", ["pastel"], [["A cup of tea"], ["on a", "tablecloth"]]),
    ("A beach during sunset", "sunset", ["during sunset"], [["A beach"]]),
    ("a cat rendered in soft pastel tones", "pastel", ["pastel"], [["a cat"], ["rendered in soft", "tones"]]),
])
def test_preserved_carrier_and_grouped_components_cover_source(text, identity, preserved, groups):
    analysis = SemanticLLM(FakeProvider([grouped_output(identity, preserved, groups)])).analyze(text, identity)
    assert analysis.prompt_for_subset(range(len(groups))) == text
    assert analysis.prompt_for_subset([]) == " ".join(preserved)


@pytest.mark.parametrize("bad", [
    grouped_output("pixel art", ["pixel art"], [["A tiny dragon"], ["perched"], ["cloud"]]),
    grouped_output("pixel art", ["pixel art"], [["A tiny dragon"], ["perched"], ["on a cloud"]]),
    grouped_output("pixel art", ["pixel art"], [["A tiny dragon"], ["perched"], ["cloud", "on a"]]),
    grouped_output("pixel art", ["pixel art"], [["A tiny dragon"], ["perched"], ["on a", "art cloud"]]),
    grouped_output("pixel art", ["pixel art"], [[]]),
    grouped_output("pixel art", ["pixel art"], [[""]]),
    grouped_output("pixel art", [], [["A tiny dragon"]]),
    grouped_output("pixel art", ["cloud"], [["A tiny dragon"]]),
])
def test_bad_group_partition_retries_with_validation_feedback(bad):
    good = grouped_output("pixel art", ["pixel art"], [["A tiny dragon"], ["perched"], ["on a", "cloud"]])
    provider = FakeProvider([bad, good])
    analysis = SemanticLLM(provider, max_retries=1).analyze("A tiny dragon perched on a pixel art cloud", "pixel art")
    assert analysis.prompt_for_subset([]) == "pixel art"
    assert provider.calls[1]["feedback"]


def test_all_grouped_identity_occurrences_and_exact_whitespace_reconstruction():
    text = "pixel art beside  pixel art"
    analysis = SemanticLLM(FakeProvider([grouped_output("pixel art", ["pixel art", "pixel art"], [["beside"]])])).analyze(text, "pixel art")
    assert analysis.prompt_for_subset([0]) == text
    assert analysis.prompt_for_subset([]) == "pixel art pixel art"
    missing = FakeProvider([grouped_output("pixel art", ["pixel art"], [["beside"]])])
    with pytest.raises(ValueError, match="every identity occurrence"):
        SemanticLLM(missing, max_retries=0).analyze(text, "pixel art")


def neutral_output(identity, preserved, neutral, groups):
    return {**grouped_output(identity, preserved, groups), "neutral_spans": neutral}


@pytest.mark.parametrize("max_retries, status, calls", [(0, "FAIL", 1), (2, "PASS", 2)])
def test_monitor_can_evaluate_first_response_without_retrying(max_retries, status, calls):
    from scripts.monitor_dreambench_target_components import run_case

    bad = neutral_output("lamp", ["lamp"], [], [["on a desk"]])
    good = neutral_output("lamp", ["a lamp"], [], [["on a desk"]])
    provider = FakeProvider([bad, good])
    result = run_case({"text": "a lamp on a desk", "identity_term": "lamp"}, provider, max_retries)
    assert result["status"] == status
    assert len(result["attempts"]) == len(provider.calls) == calls
    if max_retries == 0:
        assert "unassigned lexical content" in result["error"]


@pytest.mark.parametrize("strategy", ["local", "factorial"])
def test_qualified_identity_article_survives_without_false_subword_ambiguity(strategy):
    text = "a cube shaped <backpack> backpack"
    response = neutral_output("<backpack> backpack", ["<backpack> backpack"], ["a"], [["cube shaped"]])
    provider = FakeProvider([response])
    analysis = SemanticLLM(provider, max_retries=0).analyze(text, "<backpack> backpack")
    assert len(provider.calls) == 1
    assert analysis.prompt_for_subset([]) == "a <backpack> backpack"
    assert component_contrast_plan(SemanticConditioning(analysis, origin="target", strategy=strategy)) == (
        ("target:component_0", ((text, "a <backpack> backpack"),)),)


def test_neutral_article_excludes_positions_owned_by_a_complete_component():
    text = "a red dog wearing a hat"
    response = neutral_output("dog", ["dog"], ["a"], [["red"], ["wearing a hat"]])
    analysis = SemanticLLM(FakeProvider([response]), max_retries=0).analyze(text, "dog")
    assert analysis.prompt_for_subset([]) == "a dog"
    assert analysis.prompt_for_subset([1]) == "a dog wearing a hat"


def test_genuinely_ambiguous_repeated_neutral_span_still_rejected():
    response = neutral_output("dog", ["dog"], ["a", "a"], [["red"], ["blue"]])
    with pytest.raises(ValueError, match="ambiguous span"):
        SemanticLLM(FakeProvider([response]), max_retries=0).analyze("a red dog a blue", "dog")


def test_components_cannot_partition_a_word_into_fragments():
    response = neutral_output("a dog", ["a dog"], [], [["runn"], ["ing"]])
    with pytest.raises(ValueError, match="span is absent"):
        SemanticLLM(FakeProvider([response]), max_retries=0).analyze("a dog running", "a dog")


@pytest.mark.parametrize("strategy", ["local", "factorial"])
def test_generic_wrapper_and_temporal_relation_are_not_projection_directions(strategy):
    text = "A photo of a serene beach during sunset, capturing the golden horizon"
    response = neutral_output("sunset", ["during sunset"], ["A photo of"],
                              [["a serene beach"], ["capturing the golden horizon"]])
    analysis = SemanticLLM(FakeProvider([response])).analyze(text, "sunset")
    conditioning = SemanticConditioning(analysis, origin="target", strategy=strategy)
    assert analysis.prompt_for_subset([0, 1]) == text
    assert len(component_contrast_plan(conditioning)) == 2
    for _, pairs in component_contrast_plan(conditioning):
        for pair in pairs:
            for prompt in pair:
                assert "A photo of" in prompt and "during sunset" in prompt
    assert SemanticConditioning.from_dict(conditioning.to_dict()) == conditioning


@pytest.mark.parametrize("wrapper", ["a photo of", "A photograph of", "an image of", "a picture of"])
def test_generic_wrapper_identity_only_has_no_removal_component(wrapper):
    text = wrapper + " a dog"
    analysis = SemanticLLM(FakeProvider([neutral_output("a dog", ["a dog"], [wrapper], [])])).analyze(text, "a dog")
    assert analysis.prompt_for_subset([]) == text
    assert component_contrast_plan(SemanticConditioning(analysis, origin="target")) == ()


@pytest.mark.parametrize("bad", [
    neutral_output("sunset", ["sunset"], ["A photo of"], [["a serene beach during"]]),
    neutral_output("sunset", ["during sunset"], [], [["A photo of"], ["a serene beach"]]),
    neutral_output("sunset", ["during sunset"], [], [["A photo of a serene beach"]]),
    neutral_output("sunset", ["during sunset"], ["A photo of"], [["photo of a serene beach"]]),
    neutral_output("sunset", ["during sunset"], ["sunset"], [["A photo of a serene beach"]]),
    neutral_output("sunset", ["during sunset"], ["invented words"], [["a serene beach"]]),
    neutral_output("sunset", ["during sunset"], [""], [["a serene beach"]]),
    {**neutral_output("sunset", ["during sunset"], [], [["a serene beach"]]), "neutral_spans": None},
])
def test_incorrect_neutral_and_temporal_partition_retries(bad):
    good = neutral_output("sunset", ["during sunset"], ["A photo of"], [["a serene beach"]])
    provider = FakeProvider([bad, good])
    result = SemanticLLM(provider, max_retries=1).analyze("A photo of a serene beach during sunset", "sunset")
    assert result.neutral_spans == ("A photo of",)
    assert provider.calls[1]["feedback"]


def test_temporal_connector_can_be_neutral_when_qualifier_interrupts():
    text = "A beach during a vibrant sunset"
    result = SemanticLLM(FakeProvider([neutral_output("sunset", ["sunset"], ["during a"],
                            [["A beach"], ["vibrant"]])])).analyze(text, "sunset")
    assert result.prompt_for_subset([]) == "during a sunset"


@pytest.mark.parametrize("text,groups", [
    ("a retro-style digital drawing of a dog", [["a retro-style digital drawing of"]]),
    ("a dog holding a photo of a cat", [["holding a photo of a cat"]]),
])
def test_informative_medium_and_physical_photo_stay_removable(text, groups):
    result = SemanticLLM(FakeProvider([neutral_output("a dog", ["a dog"], [], groups)])).analyze(text, "a dog")
    assert result.prompt_for_subset([]) == "a dog"
    assert len(component_contrast_plan(SemanticConditioning(result, origin="target"))) == 1


@pytest.mark.parametrize("bad", ["not JSON", "```json\n{}\n```", "[]", "null",
    '{"identity_term":"a dog","components":[],"components":[]}',
    {"components": ["sleeping"]}, output("dog", ["sleeping"]), output(IDENTITY, "sleeping"),
    output(IDENTITY, [""]), output(IDENTITY, [" "]), output(IDENTITY, [None]),
    output(IDENTITY, ["a dog"]), output(IDENTITY, ["dancing"]),
    output(IDENTITY, ["sleeping", "sleeping"]), output(IDENTITY, ["on the snow", "sleeping"]),
    output(IDENTITY, ["sleeping on", "on the snow"]), output(IDENTITY, ["dog sleeping"]),
    {"identity_term": IDENTITY, "components": [], "extra": True}])
def test_malformed_components_retry_then_recover(bad):
    provider = FakeProvider([bad, output(IDENTITY, SOURCE_COMPONENTS)])
    result = SemanticLLM(provider, max_retries=1).analyze(SOURCE, IDENTITY)
    assert result.components == tuple(SOURCE_COMPONENTS)
    assert provider.calls[1]["feedback"]


def test_exhausted_retry_and_bad_caption():
    provider = FakeProvider([{"source_caption": "a cat"}] * 3)
    with pytest.raises(ValueError, match="3 attempts"):
        SemanticLLM(provider).caption(b"image", IDENTITY)
    assert len(provider.calls) == 3


def test_all_identity_occurrences_protected():
    text = "a dog beside a dog sleeping"
    result = ComponentAnalysis(text, IDENTITY, ("beside", "sleeping"))
    assert result.prompt_for_subset(()) == "a dog a dog"
    with pytest.raises(ValueError):
        ComponentAnalysis(text, IDENTITY, ("dog sleeping",))


@pytest.mark.parametrize("strategy", ["local", "factorial"])
def test_two_components_reproduce_legacy_math(strategy):
    analysis = ComponentAnalysis("a dog sleeping on the snow", IDENTITY, tuple(SOURCE_COMPONENTS))
    result = SemanticConditioning(analysis, origin="target", strategy=strategy)
    generator = torch.Generator().manual_seed(3)
    e00, e10, e01, e11 = torch.randn(4, 8, generator=generator)
    lookup = {IDENTITY: e00, "a dog sleeping": e10, "a dog on the snow": e01, SOURCE: e11}
    new = build_component_directions(result, lambda prompts: torch.stack([lookup[prompt] for prompt in prompts]))
    kwargs = dict(object_behavior=e10, object_background=e01, object_behavior_background=e11)
    old = build_local_directions(e00, **kwargs) if strategy == "local" else build_factorial_directions(e00, **kwargs)
    torch.testing.assert_close(new.values, old.values)
    torch.testing.assert_close(new.raw_norms, old.raw_norms)


def test_factorial_budget_and_three_component_contexts():
    result = conditioning("factorial")
    assert len(component_contrast_plan(result)[2][1]) == 4
    with pytest.raises(ValueError, match="budget"):
        SemanticConditioning(result.target, origin="target", strategy="factorial", max_contexts=4)


@pytest.mark.parametrize("origin,count", [("source", 2), ("target", 3), ("source_target", 5)])
def test_origin_selects_correct_rows(origin, count):
    assert len(component_contrast_plan(conditioning(origin=origin))) == count


def test_projection_integration_on_cpu():
    adapter = object.__new__(OrthogonalizedIPAdapterMixin)
    adapter.device = "cpu"
    adapter._active_orthogonalization = OrthogonalizationRequest(mode="components", strength=1,
                                                               semantic_conditioning=conditioning())
    adapter._encode_component_text_pool = lambda prompts: torch.stack([
        torch.tensor([len(p), p.count("a"), p.count("o"), p.count("s"), 2., 3., 5., 7.]) for p in prompts])
    captured = []
    adapter.image_proj_model = lambda pool: captured.append(pool) or pool
    visual = torch.tensor([[1., 2., 3., 4., 5., 6., 7., 8.]])
    result = adapter.get_image_embeds(clip_image_embeds=visual)
    assert len(captured) == 1
    assert result.shape == visual.shape
    assert adapter.last_orthogonalization_diagnostics["input_direction_count"] == 5
    assert adapter.last_orthogonalization_diagnostics["direction_set_count"] == 2
    assert adapter.last_orthogonalization_diagnostics["max_abs_basis_dot"] < 1e-5
    torch.testing.assert_close(result.float().norm(), visual.norm(), rtol=.01, atol=.01)


def test_long_prompt_guard_rejects_instead_of_truncating():
    adapter = object.__new__(OrthogonalizedIPAdapterMixin)
    adapter.clip_image_processor = SimpleNamespace(tokenizer=lambda *args, **kwargs: {"input_ids": [[1] * 65]})
    adapter._get_siglip_text_encoder = lambda: SimpleNamespace(config=SimpleNamespace(max_position_embeddings=64))
    with pytest.raises(ValueError, match="64-token"):
        adapter._encode_component_text_pool(["a cat " * 70])


def test_http_transport_request_mock(monkeypatch):
    buffer = BytesIO()
    Image.new("RGB", (2, 2)).save(buffer, format="PNG")
    calls = []
    class Response(BytesIO):
        pass
    def urlopen(request, timeout):
        calls.append(json.loads(request.data))
        return Response(json.dumps({"status": "completed", "output": [{"type": "message", "content": [
            {"type": "output_text", "text": json.dumps({"source_caption": SOURCE})}]}]}).encode())
    monkeypatch.setattr("orthogonalization_instantx_ip_adapter.llm_provider.urlopen", urlopen)
    provider = ResponsesJSONProvider("test-model", "test-secret")
    assert SemanticLLM(provider).caption(buffer.getvalue(), IDENTITY) == SOURCE
    assert "test-secret" not in repr(provider)
    assert calls[0]["text"]["format"]["strict"] is True
    assert calls[0]["input"][1]["content"][1]["image_url"].startswith("data:image/png;base64,")


def test_transport_errors_do_not_trigger_output_retries():
    class Provider:
        def complete(self, **kwargs):
            raise LLMTransportError("authentication failed")
    with pytest.raises(LLMTransportError):
        SemanticLLM(Provider()).analyze(SOURCE, IDENTITY)


def test_http_error_exposes_parameter_but_redacts_keys(monkeypatch):
    secret = "test-private-key"
    error_body = {"error": {"type": "invalid_request_error", "code": "unsupported_parameter",
        "param": "text.format", "message": "Unsupported parameter; " + secret + " sk-other-secret Bearer another-secret"},
        "private_debug": "DO NOT PRINT"}
    calls = []
    def urlopen(*args, **kwargs):
        calls.append(True)
        raise HTTPError("https://api.openai.com/v1/responses", 400, "Bad Request", {},
                        BytesIO(json.dumps(error_body).encode()))
    monkeypatch.setattr("orthogonalization_instantx_ip_adapter.llm_provider.urlopen", urlopen)
    with pytest.raises(LLMTransportError) as captured:
        SemanticLLM(ResponsesJSONProvider("test-model", secret)).analyze(SOURCE, IDENTITY)
    message = str(captured.value)
    assert "HTTP 400" in message and "param=text.format" in message
    assert "Unsupported parameter" in message
    assert all(value not in message for value in (secret, "sk-other-secret", "another-secret", "DO NOT PRINT"))
    assert len(calls) == 1


@pytest.mark.parametrize("body", [b"<html>proxy error</html>", b"[]", b'{"error":"invalid"}', b"{" ])
def test_unstructured_http_errors_use_safe_fallback(monkeypatch, body):
    def urlopen(*args, **kwargs):
        raise HTTPError("https://api.openai.com/v1/responses", 400, "Bad Request", {}, BytesIO(body))
    monkeypatch.setattr("orthogonalization_instantx_ip_adapter.llm_provider.urlopen", urlopen)
    with pytest.raises(LLMTransportError, match="HTTP 400; response_bytes="):
        SemanticLLM(ResponsesJSONProvider("test-model", "test-secret")).analyze(SOURCE, IDENTITY)


def test_plaintext_gateway_error_is_bounded_and_redacted(monkeypatch):
    def urlopen(*args, **kwargs):
        body = b"<html>Bad request test-secret Bearer hidden-key sk-other-key</html>" + b"x" * 1000
        raise HTTPError("https://api.openai.com/v1/responses", 400, "Bad Request",
                        {"content-type": "text/html", "server": "gateway"}, BytesIO(body))
    monkeypatch.setattr("orthogonalization_instantx_ip_adapter.llm_provider.urlopen", urlopen)
    with pytest.raises(LLMTransportError) as captured:
        SemanticLLM(ResponsesJSONProvider("test-model", "test-secret")).analyze(SOURCE, IDENTITY)
    message = str(captured.value)
    assert "content-type=text/html" in message and "server=gateway" in message
    assert "body_preview=Bad request" in message
    assert all(value not in message for value in ("test-secret", "hidden-key", "sk-other-key", "<html>"))
    assert len(message) < 600


def test_empty_http_error_body_is_identified(monkeypatch):
    def urlopen(*args, **kwargs):
        raise HTTPError("https://api.openai.com/v1/responses", 400, "Bad Request", {}, BytesIO(b""))
    monkeypatch.setattr("orthogonalization_instantx_ip_adapter.llm_provider.urlopen", urlopen)
    with pytest.raises(LLMTransportError, match="response_bytes=0; empty_error_body"):
        SemanticLLM(ResponsesJSONProvider("test-model", "test-secret")).analyze(SOURCE, IDENTITY)


def test_modes_do_not_mix_legacy_settings():
    with pytest.raises(ValueError, match="only valid"):
        OrthogonalizationRequest(semantic_conditioning=conditioning())
    with pytest.raises(ValueError, match="cannot mix"):
        OrthogonalizationRequest(mode="components", semantic_conditioning=conditioning(), include_interaction=True)


def test_python_inference_connects_analysis_to_generation(monkeypatch):
    from orthogonalization_instantx_ip_adapter import inference
    calls = []
    def generate(**kwargs):
        calls.append(kwargs)
        return b"mock PNG", {"semantic_conditioning": kwargs["component_analysis_payload"]}
    monkeypatch.setattr(inference, "generate_png_bytes", generate)
    provider = FakeProvider([{"source_caption": SOURCE}, source_output(), output(IDENTITY, TARGET_COMPONENTS)])
    png, diagnostics = inference.generate_semantic_png_bytes(
        identity_term=IDENTITY, target_prompt=TARGET, reference_image_bytes=b"image", llm=SemanticLLM(provider))
    assert png == b"mock PNG"
    assert SemanticConditioning.from_dict(diagnostics["semantic_conditioning"]) == source_conditioning()
    assert calls[0]["orthogonalization_mode"] == "components"
    assert calls[0]["prompt"] == TARGET
    assert calls[0]["reference_image_bytes"] == b"image"
    assert calls[0]["mask_single_text_rows"] is True


def test_python_inference_invalid_identity_stops_before_generation(monkeypatch):
    from orthogonalization_instantx_ip_adapter import inference
    calls = []
    monkeypatch.setattr(inference, "generate_png_bytes", lambda **kwargs: calls.append(kwargs))
    provider = FakeProvider([])
    with pytest.raises(ValueError, match="verbatim"):
        inference.generate_semantic_png_bytes(identity_term=IDENTITY, target_prompt="a cat sleeping",
                                             reference_image_bytes=b"image", llm=SemanticLLM(provider))
    assert not calls and not provider.calls


def test_invalid_remote_payload_does_not_load_gpu(monkeypatch):
    from orthogonalization_instantx_ip_adapter import inference
    calls = []
    monkeypatch.setattr(inference, "load_ip_adapter_model", lambda: calls.append(True))
    with pytest.raises(ValueError, match="does not match"):
        inference.generate_png_bytes(
            prompt="a cat", reference_image_url="", reference_image_bytes=b"image",
            ip_adapter_scale=.7, guidance_scale=3.5, num_inference_steps=24, seed=42, width=1024, height=1024,
            orthogonalization_mode="components", orthogonalization_strength=1., object_prompt="",
            object_behavior_prompt="", object_background_prompt="", object_behavior_background_prompt="",
            remove_behavior=True, remove_background=True, include_interaction=False, restore_visual_norm=True,
            mask_single_text_rows=True, component_analysis_payload=conditioning().to_dict())
    assert not calls
