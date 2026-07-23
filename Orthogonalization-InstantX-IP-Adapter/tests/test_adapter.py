from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from orthogonalization_instantx_ip_adapter.adapter import (
    OrthogonalizationRequest,
    OrthogonalizedIPAdapterMixin,
)
from orthogonalization_instantx_ip_adapter.orthogonalization import PromptSet


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
