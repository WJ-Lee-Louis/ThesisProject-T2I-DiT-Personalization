from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from orthogonalization_instantx_ip_adapter.adapter import (
    ClassDirectionSet,
    OrthogonalizationRequest,
    OrthogonalizedIPAdapterMixin,
    PromptContrastPair,
    TextDirectionSet,
)
from orthogonalization_instantx_ip_adapter.orthogonalization import (
    PromptSet,
    build_factorial_directions,
    build_local_directions,
)


class _DummyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            num_attention_heads=1,
            attention_head_dim=128,
            joint_attention_dim=128,
        )
        self._processors = {
            "transformer_blocks.0.attn.processor": object(),
            "single_transformer_blocks.0.attn.processor": object(),
        }

    @property
    def attn_processors(self):
        return self._processors

    def set_attn_processor(self, processors):
        self._processors = processors

    def forward(self, hidden_states, encoder_hidden_states=None):
        return hidden_states


class _DummyUpstreamAdapter:
    def __init__(self):
        self.device = "cpu"
        self.num_tokens = 128
        self.pipe = SimpleNamespace(transformer=_DummyTransformer())
        self.set_ip_adapter()
        self.image_encoder = SimpleNamespace(dtype=torch.bfloat16)
        self.image_encoder_path = "unused"

    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        return torch.tensor([123.0])


class _ComposedAdapter(OrthogonalizedIPAdapterMixin, _DummyUpstreamAdapter):
    pass


class _CheckpointTransformer(_DummyTransformer):
    def __init__(self):
        super().__init__()
        self._processors = {
            **{
                f"transformer_blocks.{index}.attn.processor": object()
                for index in range(19)
            },
            **{
                f"single_transformer_blocks.{index}.attn.processor": object()
                for index in range(38)
            },
        }


class _CheckpointUpstreamAdapter:
    def __init__(self, checkpoint_path=None):
        self.device = "cpu"
        self.num_tokens = 128
        self.pipe = SimpleNamespace(transformer=_CheckpointTransformer())
        self.set_ip_adapter()
        self.image_proj_model = nn.Linear(4, 4)
        self.image_encoder = SimpleNamespace(dtype=torch.bfloat16)
        self.image_encoder_path = "unused"
        if checkpoint_path is not None:
            self.ip_ckpt = str(checkpoint_path)
            self.load_ip_adapter()


class _CheckpointAdapter(
    OrthogonalizedIPAdapterMixin,
    _CheckpointUpstreamAdapter,
):
    pass


def test_off_path_delegates_to_upstream_and_hook_captures_real_text_length():
    adapter = _ComposedAdapter()

    assert adapter.get_image_embeds().item() == 123.0

    adapter.set_mask_single_text_rows(True)
    adapter.pipe.transformer(
        torch.randn(1, 3, 128),
        encoder_hidden_states=torch.randn(1, 7, 128),
    )
    single_processor = adapter.pipe.transformer.attn_processors[
        "single_transformer_blocks.0.attn.processor"
    ]
    assert adapter.last_single_stream_text_seq_len == 7
    assert single_processor.mask_single_text_rows is True
    assert single_processor.text_seq_len == 7


def test_factorial_request_requires_all_counterfactual_prompts():
    with pytest.raises(ValueError, match="all four"):
        OrthogonalizationRequest(
            mode="factorial",
            prompts=PromptSet(
                object_only="a dog",
                object_behavior="a dog running",
            ),
        )


def test_removed_class_only_absolute_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="mode must be one of"):
        OrthogonalizationRequest(
            mode="class_absolute",
            prompts=PromptSet(object_only="a dog"),
        )


def test_class_absolute_request_builds_one_anchor_direction(monkeypatch) -> None:
    adapter = _CheckpointAdapter()
    monkeypatch.setattr(
        adapter,
        "_encode_siglip_text_pool",
        lambda prompts: torch.stack(
            [{"a dog": torch.tensor([3.0, 4.0])}[prompt] for prompt in prompts]
        ),
    )
    request = OrthogonalizationRequest(
        mode="local",
        prompts=PromptSet(
            object_only="a dog",
            object_behavior="a dog flying",
        ),
        include_behavior=True,
        include_background=False,
        class_absolute_prompt="a dog",
        class_strength=0.5,
    )

    directions = adapter._build_hierarchical_class_directions(request)

    assert directions.names == ("class_anchor_absolute",)
    assert directions.values.shape == (1, 2)
    torch.testing.assert_close(directions.values[0], torch.tensor([0.6, 0.8]))
    assert adapter._last_direction_build_diagnostics["class_anchor_prompt"] == (
        "a dog"
    )


def test_class_contextual_explicit_one_pair_builds_one_direction(
    monkeypatch,
) -> None:
    adapter = _CheckpointAdapter()
    vectors = {
        "a dog flying": torch.tensor([1.0, 1.0, 0.0]),
        "flying": torch.tensor([1.0, 0.0, 0.0]),
    }
    seen_prompts = []

    def encode(prompts):
        seen_prompts.append(tuple(prompts))
        return torch.stack([vectors[prompt] for prompt in prompts])

    monkeypatch.setattr(adapter, "_encode_siglip_text_pool", encode)
    request = OrthogonalizationRequest(
        mode="class_contextual",
        class_contextual_pairs=(
            PromptContrastPair(
                positive="a dog flying",
                negative="flying",
                label="behavior_context",
            ),
        ),
    )

    directions = adapter._build_text_directions(request)
    expected = torch.nn.functional.normalize(
        torch.nn.functional.normalize(vectors["a dog flying"], dim=0)
        - torch.nn.functional.normalize(vectors["flying"], dim=0),
        dim=0,
    )

    assert directions.names == ("class_context_mean",)
    assert directions.values.shape == (1, 3)
    torch.testing.assert_close(directions.values[0], expected)
    assert seen_prompts == [("a dog flying", "flying")]
    diagnostics = adapter._last_direction_build_diagnostics
    assert diagnostics["class_pair_labels"] == ["behavior_context"]
    assert diagnostics["class_pair_count"] == 1
    assert diagnostics["class_positive_prompts"] == ["a dog flying"]
    assert diagnostics["class_negative_prompts"] == ["flying"]
    assert len(diagnostics["class_pair_raw_norms"]) == 1
    assert len(diagnostics["class_pair_direction_cosine_matrix"]) == 1


def test_class_contextual_explicit_and_legacy_three_pair_requests_are_numeric_twins(
    monkeypatch,
) -> None:
    adapter = _CheckpointAdapter()
    positive_prompts = (
        "a dog sitting",
        "a dog on a beach",
        "a dog sitting on a beach",
    )
    negative_prompts = (
        "an animal sitting",
        "an animal on a beach",
        "an animal sitting on a beach",
    )
    vectors = {
        positive_prompts[0]: torch.tensor([1.0, 1.0, 0.0]),
        positive_prompts[1]: torch.tensor([1.0, 0.0, 1.0]),
        positive_prompts[2]: torch.tensor([1.0, 2.0, 2.0]),
        negative_prompts[0]: torch.tensor([1.0, 0.0, 0.0]),
        negative_prompts[1]: torch.tensor([1.0, 0.0, 0.0]),
        negative_prompts[2]: torch.tensor([1.0, 1.0, 1.0]),
    }
    seen_prompts = []

    def encode(prompts):
        seen_prompts.append(tuple(prompts))
        return torch.stack([vectors[prompt] for prompt in prompts])

    monkeypatch.setattr(adapter, "_encode_siglip_text_pool", encode)
    explicit_request = OrthogonalizationRequest(
        mode="class_contextual",
        class_contextual_pairs=tuple(
            PromptContrastPair(
                positive=positive,
                negative=negative,
                label=label,
            )
            for positive, negative, label in zip(
                positive_prompts,
                negative_prompts,
                (
                    "behavior_context",
                    "background_context",
                    "joint_context",
                ),
            )
        ),
    )
    legacy_request = OrthogonalizationRequest(
        mode="class_contextual",
        prompts=PromptSet(
            object_only="a dog",
            object_behavior=positive_prompts[0],
            object_background=positive_prompts[1],
            object_behavior_background=positive_prompts[2],
        ),
        comparison_prompts=PromptSet(
            object_only="an animal",
            object_behavior=negative_prompts[0],
            object_background=negative_prompts[1],
            object_behavior_background=negative_prompts[2],
        ),
        include_behavior=False,
        include_background=False,
    )

    explicit_directions = adapter._build_text_directions(explicit_request)
    explicit_diagnostics = dict(adapter._last_direction_build_diagnostics)
    legacy_directions = adapter._build_text_directions(legacy_request)

    assert explicit_directions.names == ("class_context_mean",)
    assert legacy_directions.names == ("source_class_context_mean",)
    assert explicit_directions.values.shape == (1, 3)
    torch.testing.assert_close(
        explicit_directions.values,
        legacy_directions.values,
    )
    assert seen_prompts == [
        positive_prompts + negative_prompts,
        positive_prompts + negative_prompts,
    ]
    assert explicit_diagnostics["class_pair_labels"] == [
        "behavior_context",
        "background_context",
        "joint_context",
    ]
    assert explicit_diagnostics["class_pair_count"] == 3
    diagnostics = adapter._last_direction_build_diagnostics
    assert diagnostics["class_positive_prompts"] == list(positive_prompts)
    assert diagnostics["class_negative_prompts"] == list(negative_prompts)
    assert len(diagnostics["class_pair_raw_norms"]) == 3
    assert len(diagnostics["class_pair_direction_cosine_matrix"]) == 3


def test_prompt_contrast_pair_requires_normalized_nonempty_strings() -> None:
    for field_name, field_value in (
        ("positive", ""),
        ("negative", " flying"),
        ("label", "behavior  context"),
    ):
        values = {
            "positive": "a dog flying",
            "negative": "flying",
            "label": "behavior_context",
        }
        values[field_name] = field_value
        with pytest.raises(ValueError, match=field_name):
            PromptContrastPair(**values)


def test_class_requests_validate_explicit_pair_cardinality_and_exclusivity() -> None:
    pair = PromptContrastPair(
        positive="a dog flying",
        negative="flying",
        label="behavior_context",
    )

    with pytest.raises(TypeError, match="must be a tuple"):
        OrthogonalizationRequest(
            mode="class_contextual",
            class_contextual_pairs=[pair],
        )

    with pytest.raises(ValueError, match="exactly one or three"):
        OrthogonalizationRequest(
            mode="class_contextual",
            class_contextual_pairs=(
                pair,
                PromptContrastPair(
                    positive="a dog on a stage",
                    negative="on a stage",
                    label="background_context",
                ),
            ),
        )

    with pytest.raises(ValueError, match="labels must be unique"):
        OrthogonalizationRequest(
            mode="class_contextual",
            class_contextual_pairs=(
                pair,
                PromptContrastPair(
                    positive="a dog on a stage",
                    negative="on a stage",
                    label="behavior_context",
                ),
                PromptContrastPair(
                    positive="a dog flying on a stage",
                    negative="flying on a stage",
                    label="joint_context",
                ),
            ),
        )

    with pytest.raises(ValueError, match="either explicit pairs or legacy"):
        OrthogonalizationRequest(
            mode="class_contextual",
            prompts=PromptSet(object_only="a dog"),
            class_contextual_pairs=(pair,),
        )


def test_class_requests_reject_missing_or_extraneous_prompt_sets() -> None:
    with pytest.raises(ValueError, match="either contextual class pairs"):
        OrthogonalizationRequest(
            mode="local",
            prompts=PromptSet(
                object_only="a dog",
                object_behavior="a dog running",
            ),
            class_contextual_pairs=(
                PromptContrastPair("a dog running", "running", "context"),
            ),
            class_absolute_prompt="a dog",
            class_strength=0.5,
        )

    with pytest.raises(ValueError, match="requires explicit pairs"):
        OrthogonalizationRequest(
            mode="class_contextual",
            prompts=PromptSet(object_only="a dog"),
        )


def test_hierarchical_request_requires_pairs_and_independent_class_strength() -> None:
    pair = PromptContrastPair(
        positive="a dog flying",
        negative="flying",
        label="behavior_context",
    )
    prompts = PromptSet(
        object_only="a dog",
        object_behavior="a dog flying",
    )

    with pytest.raises(ValueError, match="class_strength is required"):
        OrthogonalizationRequest(
            mode="local",
            prompts=prompts,
            include_behavior=True,
            include_background=False,
            class_contextual_pairs=(pair,),
        )
    with pytest.raises(ValueError, match="requires contextual class pairs"):
        OrthogonalizationRequest(
            mode="local",
            prompts=prompts,
            include_behavior=True,
            include_background=False,
            class_strength=0.25,
        )

    request = OrthogonalizationRequest(
        mode="local",
        strength=1.0,
        prompts=prompts,
        include_behavior=True,
        include_background=False,
        class_contextual_pairs=(pair,),
        class_strength=0.25,
    )
    assert request.hierarchical_class_enabled is True


def test_adapter_hierarchical_path_reports_separate_nuisance_and_class_edits(
    monkeypatch,
) -> None:
    adapter = _CheckpointAdapter()
    adapter.image_proj_model = nn.Identity()
    embeddings = {
        "a dog": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        "a dog flying": torch.tensor([0.0, 1.0, 0.0, 0.0]),
        "flying": torch.tensor([0.0, 0.0, 1.0, 0.0]),
    }
    monkeypatch.setattr(
        adapter,
        "_encode_siglip_text_pool",
        lambda prompts: torch.stack([embeddings[prompt] for prompt in prompts]),
    )
    request = OrthogonalizationRequest(
        mode="local",
        strength=1.0,
        prompts=PromptSet(
            object_only="a dog",
            object_behavior="a dog flying",
        ),
        include_behavior=True,
        include_background=False,
        class_contextual_pairs=(
            PromptContrastPair(
                positive="a dog flying",
                negative="flying",
                label="behavior_context",
            ),
        ),
        class_strength=0.25,
    )
    visual = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    adapter._active_orthogonalization = request
    output = adapter.get_image_embeds(clip_image_embeds=visual)

    assert output.shape == visual.shape
    diagnostics = adapter.last_orthogonalization_diagnostics
    assert diagnostics["hierarchical_class_projection"] is True
    assert diagnostics["nuisance_strength"] == pytest.approx(1.0)
    assert diagnostics["class_strength"] == pytest.approx(0.25)
    assert diagnostics["nuisance_basis_rank"] == 1
    assert diagnostics["class_basis_rank"] == 1
    assert diagnostics["basis_rank"] == 2
    assert diagnostics["class_residual_ratio"] > 0.0
    assert diagnostics["nuisance_component_norm"] > 0.0
    assert diagnostics["class_component_norm"] > 0.0
    torch.testing.assert_close(
        torch.linalg.vector_norm(output.float()),
        torch.linalg.vector_norm(visual.float()),
        rtol=2e-3,
        atol=2e-3,
    )


def test_adapter_hierarchical_path_builds_target_source_class_subspace(
    monkeypatch,
) -> None:
    adapter = _CheckpointAdapter()
    adapter.image_proj_model = nn.Identity()
    embeddings = {
        "a dog": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        "a dog flying": torch.tensor([0.0, 1.0, 0.0, 0.0]),
        "flying": torch.tensor([0.0, 0.0, 1.0, 0.0]),
        "a dog sitting": torch.tensor([0.0, 0.0, 0.0, 1.0]),
        "sitting": torch.tensor([0.5, 0.0, 0.0, 0.5]),
    }
    monkeypatch.setattr(
        adapter,
        "_encode_siglip_text_pool",
        lambda prompts: torch.stack([embeddings[prompt] for prompt in prompts]),
    )
    request = OrthogonalizationRequest(
        mode="local",
        strength=0.8,
        prompts=PromptSet(
            object_only="a dog",
            object_behavior="a dog flying",
        ),
        include_behavior=True,
        include_background=False,
        class_contextual_pairs=(
            PromptContrastPair("a dog flying", "flying", "target_behavior"),
        ),
        class_strength=0.25,
        primary_class_label="target_class_context",
        additional_class_direction_sets=(
            ClassDirectionSet(
                label="source_class_context",
                pairs=(
                    PromptContrastPair("a dog sitting", "sitting", "source_behavior"),
                ),
            ),
        ),
    )

    adapter._active_orthogonalization = request
    adapter.get_image_embeds(
        clip_image_embeds=torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    )
    diagnostics = adapter.last_orthogonalization_diagnostics

    assert diagnostics["class_direction_count"] == 2
    assert diagnostics["class_pair_count"] == 2
    assert diagnostics["class_direction_set_count"] == 2
    assert len(diagnostics["class_direction_sets"]) == 2
    assert diagnostics["class_basis_rank"] in {1, 2}
    assert diagnostics["basis_rank"] == (
        diagnostics["nuisance_basis_rank"] + diagnostics["class_basis_rank"]
    )


@pytest.mark.parametrize("strength", [-0.1, 1.1, float("nan")])
def test_request_rejects_invalid_strength(strength):
    with pytest.raises(ValueError, match="strength"):
        OrthogonalizationRequest(mode="off", strength=strength)


def test_new_request_field_preserves_legacy_positional_signature() -> None:
    prompts = PromptSet(
        object_only="a dog",
        object_background="a dog on a stage",
    )

    request = OrthogonalizationRequest(
        "local",
        0.5,
        prompts,
        False,
        True,
    )

    assert request.prompts is prompts
    assert request.include_behavior is False
    assert request.include_background is True
    assert request.comparison_prompts is None


def test_off_request_preserves_legacy_unused_prompt_tolerance() -> None:
    prompts = PromptSet(object_only="a dog")

    request = OrthogonalizationRequest("off", 0.5, prompts)

    assert request.enabled is False
    assert request.prompts is prompts


def test_checkpoint_loader_requires_exact_57_processor_mapping(tmp_path):
    source = _CheckpointAdapter()
    ip_layers = nn.ModuleList(source.pipe.transformer.attn_processors.values())
    checkpoint = {
        "image_proj": source.image_proj_model.state_dict(),
        "ip_adapter": ip_layers.state_dict(),
    }
    checkpoint_path = tmp_path / "ip-adapter.bin"
    torch.save(checkpoint, checkpoint_path)

    loaded = _CheckpointAdapter(checkpoint_path)

    assert loaded.checkpoint_load_diagnostics == {
        "processor_count": 57,
        "missing_keys": [],
        "unexpected_keys": [],
    }


def test_checkpoint_loader_rejects_silent_missing_key(tmp_path):
    source = _CheckpointAdapter()
    ip_layers = nn.ModuleList(source.pipe.transformer.attn_processors.values())
    ip_state = ip_layers.state_dict()
    ip_state.pop(next(iter(ip_state)))
    checkpoint_path = tmp_path / "bad-ip-adapter.bin"
    torch.save(
        {
            "image_proj": source.image_proj_model.state_dict(),
            "ip_adapter": ip_state,
        },
        checkpoint_path,
    )

    with pytest.raises(RuntimeError, match="missing="):
        _CheckpointAdapter(checkpoint_path)


def test_source_target_request_concatenates_all_rows_before_projection(
    monkeypatch,
) -> None:
    adapter = _CheckpointAdapter()
    adapter.image_proj_model = nn.Identity()
    embeddings = {
        "a dog": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        "a dog running": torch.tensor([1.0, 1.0, 0.0, 0.0]),
        "a dog sitting": torch.tensor([1.0, 0.0, 1.0, 0.0]),
        "a dog on a beach": torch.tensor([1.0, 0.0, 0.0, 1.0]),
        "a dog sitting on a beach": torch.tensor([1.0, 0.0, 2.0, 2.0]),
    }
    monkeypatch.setattr(
        adapter,
        "_encode_siglip_text_pool",
        lambda prompts: torch.stack([embeddings[prompt] for prompt in prompts]),
    )
    request = OrthogonalizationRequest(
        mode="local",
        prompts=PromptSet(
            object_only="a dog",
            object_behavior="a dog running",
        ),
        include_behavior=True,
        include_background=False,
        primary_label="target",
        additional_direction_sets=(
            TextDirectionSet(
                mode="factorial",
                prompts=PromptSet(
                    object_only="a dog",
                    object_behavior="a dog sitting",
                    object_background="a dog on a beach",
                    object_behavior_background=(
                        "a dog sitting on a beach"
                    ),
                ),
                include_behavior=True,
                include_background=False,
                include_interaction=False,
                label="source",
            ),
        ),
    )

    directions = adapter._build_text_directions(request)
    expected_target = build_local_directions(
        embeddings["a dog"],
        object_behavior=embeddings["a dog running"],
        include_behavior=True,
        include_background=False,
    )
    expected_source = build_factorial_directions(
        embeddings["a dog"],
        embeddings["a dog sitting"],
        embeddings["a dog on a beach"],
        embeddings["a dog sitting on a beach"],
    )

    assert directions.values.shape == (2, 4)
    torch.testing.assert_close(
        directions.values,
        torch.cat(
            [expected_target.values, expected_source.values[:1]],
            dim=0,
        ),
    )
    assert directions.names[0].startswith("target")
    assert directions.names[1].startswith("source")


def test_source_target_union_reaches_orthogonalizer_in_one_call(
    monkeypatch,
) -> None:
    import orthogonalization_instantx_ip_adapter.adapter as adapter_module

    adapter = _CheckpointAdapter()
    adapter.image_proj_model = nn.Identity()
    embeddings = {
        "a dog": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        "a dog running": torch.tensor([1.0, 1.0, 0.0, 0.0]),
        # Deliberately make the source behavior contrast collinear with the
        # target behavior contrast. The union must still enter one SVD and
        # collapse to a rank-one basis instead of subtracting twice.
        "a dog sitting": torch.tensor([1.0, 1.0, 0.0, 0.0]),
        "a dog on a beach": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        "a dog sitting on a beach": torch.tensor([1.0, 1.0, 0.0, 0.0]),
    }
    monkeypatch.setattr(
        adapter,
        "_encode_siglip_text_pool",
        lambda prompts: torch.stack([embeddings[prompt] for prompt in prompts]),
    )
    real_orthogonalize = adapter_module.orthogonalize_pooled_embedding
    captured_directions = []

    def spy_orthogonalize(embedding, directions, **kwargs):
        captured_directions.append(directions.detach().clone())
        return real_orthogonalize(embedding, directions, **kwargs)

    monkeypatch.setattr(
        adapter_module,
        "orthogonalize_pooled_embedding",
        spy_orthogonalize,
    )
    adapter._active_orthogonalization = OrthogonalizationRequest(
        mode="local",
        prompts=PromptSet(
            object_only="a dog",
            object_behavior="a dog running",
        ),
        include_behavior=True,
        include_background=False,
        primary_label="target",
        additional_direction_sets=(
            TextDirectionSet(
                mode="factorial",
                prompts=PromptSet(
                    object_only="a dog",
                    object_behavior="a dog sitting",
                    object_background="a dog on a beach",
                    object_behavior_background=(
                        "a dog sitting on a beach"
                    ),
                ),
                include_behavior=True,
                include_background=False,
                label="source",
            ),
        ),
    )

    adapter.get_image_embeds(
        clip_image_embeds=torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    )

    assert len(captured_directions) == 1
    assert captured_directions[0].shape == (2, 4)
    assert adapter.last_orthogonalization_diagnostics[
        "input_direction_count"
    ] == 2
    assert adapter.last_orthogonalization_diagnostics["basis_rank"] == 1
    assert adapter.last_orthogonalization_diagnostics["dropped_rank"] == 1
    assert adapter.last_orthogonalization_diagnostics[
        "direction_normalization"
    ] == "l2_per_row"
    assert len(
        adapter.last_orthogonalization_diagnostics["direction_raw_norms"]
    ) == 2
    assert adapter.last_orthogonalization_diagnostics[
        "projection_component_norm"
    ] >= 0.0
    assert adapter.last_orthogonalization_diagnostics["degenerate"] is False
    cosine_matrix = adapter.last_orthogonalization_diagnostics[
        "direction_cosine_matrix"
    ]
    assert cosine_matrix[0][1] == pytest.approx(1.0)


def test_legacy_request_keeps_unprefixed_direction_names(monkeypatch) -> None:
    adapter = _CheckpointAdapter()
    embeddings = {
        "a dog": torch.tensor([1.0, 0.0]),
        "a dog running": torch.tensor([1.0, 1.0]),
    }
    monkeypatch.setattr(
        adapter,
        "_encode_siglip_text_pool",
        lambda prompts: torch.stack([embeddings[prompt] for prompt in prompts]),
    )
    request = OrthogonalizationRequest(
        mode="local",
        prompts=PromptSet(
            object_only="a dog",
            object_behavior="a dog running",
        ),
        include_behavior=True,
        include_background=False,
    )

    directions = adapter._build_text_directions(request)

    assert request.primary_label is None
    assert request.additional_direction_sets == ()
    assert directions.names == ("behavior_local",)
