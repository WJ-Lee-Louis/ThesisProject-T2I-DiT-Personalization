import pytest
import torch
import torch.nn as nn

from orthogonalization_instantx_ip_adapter.attention_processor import (
    IPAFluxAttnProcessor2_0,
    build_single_stream_ip_row_gate,
    mask_single_stream_ip_residual,
)


def test_row_gate_zeros_text_prefix_and_keeps_latent_suffix():
    residual = torch.arange(2 * 7 * 3, dtype=torch.float32).reshape(2, 7, 3)

    gate = build_single_stream_ip_row_gate(residual, text_seq_len=3)
    masked = mask_single_stream_ip_residual(residual, text_seq_len=3)

    assert gate.shape == (1, 7, 1)
    assert gate.dtype == residual.dtype
    assert gate.device == residual.device
    torch.testing.assert_close(gate[:, :3], torch.zeros_like(gate[:, :3]))
    torch.testing.assert_close(gate[:, 3:], torch.ones_like(gate[:, 3:]))
    torch.testing.assert_close(masked[:, :3], torch.zeros_like(masked[:, :3]))
    torch.testing.assert_close(masked[:, 3:], residual[:, 3:])


@pytest.mark.parametrize("text_seq_len", [0, 5])
def test_row_gate_supports_empty_or_full_text_prefix(text_seq_len):
    residual = torch.randn(2, 5, 4)

    masked = mask_single_stream_ip_residual(residual, text_seq_len)

    expected = residual if text_seq_len == 0 else torch.zeros_like(residual)
    torch.testing.assert_close(masked, expected)


@pytest.mark.parametrize("invalid_length", [-1, 6])
def test_row_gate_rejects_out_of_range_text_length(invalid_length):
    residual = torch.randn(1, 5, 4)

    with pytest.raises(ValueError):
        build_single_stream_ip_row_gate(residual, invalid_length)


@pytest.mark.parametrize("invalid_length", [True, 1.5, "3", None])
def test_row_gate_rejects_invalid_text_length_types(invalid_length):
    residual = torch.randn(1, 5, 4)

    with pytest.raises((TypeError, ValueError)):
        build_single_stream_ip_row_gate(residual, invalid_length)


def test_row_gate_rejects_non_sequence_tensor():
    with pytest.raises(ValueError, match=r"\[batch, sequence, channels\]"):
        build_single_stream_ip_row_gate(torch.randn(5, 4), 2)


def test_processor_setters_are_chainable_and_not_persistent_state():
    processor = IPAFluxAttnProcessor2_0(
        hidden_size=16,
        cross_attention_dim=12,
    )
    upstream_compatible_keys = {"to_k_ip.weight", "to_v_ip.weight"}

    result = processor.set_mask_single_text_rows(True).set_text_seq_len(5)

    assert result is processor
    assert processor.mask_single_text_rows is True
    assert processor.text_seq_len == 5
    assert set(processor.state_dict()) == upstream_compatible_keys


def test_processor_loads_upstream_style_state_dict_strictly():
    source = IPAFluxAttnProcessor2_0(
        hidden_size=16,
        cross_attention_dim=12,
    )
    target = IPAFluxAttnProcessor2_0(
        hidden_size=16,
        cross_attention_dim=12,
        mask_single_text_rows=True,
        text_seq_len=5,
    )

    incompatible = target.load_state_dict(source.state_dict(), strict=True)

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []


@pytest.mark.parametrize("invalid_flag", [0, 1, "true", None])
def test_processor_rejects_non_boolean_mask_flag(invalid_flag):
    processor = IPAFluxAttnProcessor2_0(hidden_size=16)

    with pytest.raises(TypeError):
        processor.set_mask_single_text_rows(invalid_flag)


class _SingleStreamAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 1
        self.to_q = nn.Linear(128, 128, bias=False)
        self.to_k = nn.Linear(128, 128, bias=False)
        self.to_v = nn.Linear(128, 128, bias=False)
        self.norm_q = None
        self.norm_k = None


class _DoubleStreamAttention(_SingleStreamAttention):
    def __init__(self):
        super().__init__()
        self.add_q_proj = nn.Linear(128, 128, bias=False)
        self.add_k_proj = nn.Linear(128, 128, bias=False)
        self.add_v_proj = nn.Linear(128, 128, bias=False)
        self.norm_added_q = None
        self.norm_added_k = None
        self.to_out = nn.ModuleList([nn.Identity(), nn.Identity()])
        self.to_add_out = nn.Identity()


def test_processor_masks_only_direct_single_stream_ip_delta():
    torch.manual_seed(7)
    attention = _SingleStreamAttention()
    processor = IPAFluxAttnProcessor2_0(
        hidden_size=128,
        cross_attention_dim=128,
        scale=0.7,
    )
    hidden_states = torch.randn(1, 5, 128)
    image_emb = torch.randn(1, 2, 128)

    without_ip = processor(attention, hidden_states, image_emb=None)
    with_ip = processor(attention, hidden_states, image_emb=image_emb)
    processor.set_mask_single_text_rows(True).set_text_seq_len(2)
    masked = processor(attention, hidden_states, image_emb=image_emb)

    direct_ip_delta = with_ip - without_ip
    masked_delta = masked - without_ip
    torch.testing.assert_close(
        masked_delta[:, :2],
        torch.zeros_like(masked_delta[:, :2]),
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        masked_delta[:, 2:],
        direct_ip_delta[:, 2:],
        atol=1e-6,
        rtol=1e-6,
    )


def test_text_row_mask_does_not_change_double_stream_outputs():
    torch.manual_seed(11)
    attention = _DoubleStreamAttention()
    processor = IPAFluxAttnProcessor2_0(
        hidden_size=128,
        cross_attention_dim=128,
        scale=0.7,
        text_seq_len=2,
    )
    latent_hidden_states = torch.randn(1, 3, 128)
    text_hidden_states = torch.randn(1, 2, 128)
    image_emb = torch.randn(1, 2, 128)

    unmasked = processor(
        attention,
        latent_hidden_states,
        image_emb=image_emb,
        encoder_hidden_states=text_hidden_states,
    )
    processor.set_mask_single_text_rows(True)
    masked = processor(
        attention,
        latent_hidden_states,
        image_emb=image_emb,
        encoder_hidden_states=text_hidden_states,
    )

    torch.testing.assert_close(masked[0], unmasked[0])
    torch.testing.assert_close(masked[1], unmasked[1])
