from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from orthogonalization_instantx_ip_adapter.adapter import (
    OrthogonalizationRequest,
    OrthogonalizedIPAdapterMixin,
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


@pytest.mark.parametrize("strength", [-0.1, 1.1, float("nan")])
def test_request_rejects_invalid_strength(strength):
    with pytest.raises(ValueError, match="strength"):
        OrthogonalizationRequest(mode="off", strength=strength)


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
