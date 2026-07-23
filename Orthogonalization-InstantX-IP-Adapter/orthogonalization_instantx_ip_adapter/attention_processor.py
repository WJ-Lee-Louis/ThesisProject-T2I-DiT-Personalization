"""InstantX FLUX IP-Adapter attention processor with an optional text-row gate.

The model parameters and the default execution path intentionally match
InstantX's ``IPAFluxAttnProcessor2_0``.  The only intervention is an optional
gate on the IP-attention residual in a FLUX single-stream block: the text-row
prefix is zeroed while the latent-image-row suffix is left unchanged.
"""

from numbers import Integral
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.normalization import RMSNorm


def _validate_mask_flag(enabled: bool) -> bool:
    if not isinstance(enabled, bool):
        raise TypeError(
            "mask_single_text_rows must be a bool, "
            f"but received {type(enabled).__name__}."
        )
    return enabled


def _validate_text_seq_len(text_seq_len: Optional[int]) -> Optional[int]:
    if text_seq_len is None:
        return None
    if isinstance(text_seq_len, bool) or not isinstance(text_seq_len, Integral):
        raise TypeError(
            "text_seq_len must be a non-negative integer or None, "
            f"but received {type(text_seq_len).__name__}."
        )
    text_seq_len = int(text_seq_len)
    if text_seq_len < 0:
        raise ValueError(
            f"text_seq_len must be non-negative, but received {text_seq_len}."
        )
    return text_seq_len


def build_single_stream_ip_row_gate(
    ip_hidden_states: torch.Tensor,
    text_seq_len: int,
) -> torch.Tensor:
    """Build ``[1, sequence, 1]`` gate for ``[text rows; latent rows]``.

    The returned tensor has the same device and dtype as ``ip_hidden_states``.
    Its text prefix is zero and its latent-image suffix is one, so it broadcasts
    over batch and channel dimensions without allocating a full residual-sized
    mask.
    """

    if not isinstance(ip_hidden_states, torch.Tensor):
        raise TypeError(
            "ip_hidden_states must be a torch.Tensor, "
            f"but received {type(ip_hidden_states).__name__}."
        )
    if ip_hidden_states.ndim != 3:
        raise ValueError(
            "ip_hidden_states must have shape [batch, sequence, channels], "
            f"but received shape {tuple(ip_hidden_states.shape)}."
        )

    text_seq_len = _validate_text_seq_len(text_seq_len)
    if text_seq_len is None:
        raise ValueError("text_seq_len is required when the text-row mask is enabled.")

    sequence_length = ip_hidden_states.shape[1]
    if text_seq_len > sequence_length:
        raise ValueError(
            "text_seq_len cannot exceed the single-stream sequence length: "
            f"received {text_seq_len} for sequence length {sequence_length}."
        )

    gate = torch.ones(
        (1, sequence_length, 1),
        dtype=ip_hidden_states.dtype,
        device=ip_hidden_states.device,
    )
    if text_seq_len:
        gate[:, :text_seq_len] = 0
    return gate


def mask_single_stream_ip_residual(
    ip_hidden_states: torch.Tensor,
    text_seq_len: int,
) -> torch.Tensor:
    """Zero direct IP residual updates to text rows in a single-stream block."""

    return ip_hidden_states * build_single_stream_ip_row_gate(
        ip_hidden_states, text_seq_len
    )


class IPAFluxAttnProcessor2_0(nn.Module):
    """InstantX attention processor with an optional single-stream row mask.

    ``mask_single_text_rows`` and ``text_seq_len`` are ordinary Python
    attributes rather than persistent buffers.  Consequently, this processor's
    state-dict keys remain identical to the upstream InstantX processor:
    ``to_k_ip.weight`` and ``to_v_ip.weight``.
    """

    def __init__(
        self,
        hidden_size,
        cross_attention_dim=None,
        scale=1.0,
        num_tokens=4,
        *,
        mask_single_text_rows: bool = False,
        text_seq_len: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens
        self.to_k_ip = nn.Linear(
            cross_attention_dim or hidden_size, hidden_size, bias=False
        )
        self.to_v_ip = nn.Linear(
            cross_attention_dim or hidden_size, hidden_size, bias=False
        )
        # Keep this exactly aligned with the InstantX processor/checkpoint.
        self.norm_added_k = RMSNorm(128, eps=1e-5, elementwise_affine=False)

        self.mask_single_text_rows = _validate_mask_flag(mask_single_text_rows)
        self.text_seq_len = _validate_text_seq_len(text_seq_len)

    def set_mask_single_text_rows(
        self, enabled: bool
    ) -> "IPAFluxAttnProcessor2_0":
        """Enable or disable the per-processor single-stream text-row gate."""

        self.mask_single_text_rows = _validate_mask_flag(enabled)
        return self

    def set_text_seq_len(
        self, text_seq_len: Optional[int]
    ) -> "IPAFluxAttnProcessor2_0":
        """Set the text-prefix length used by the single-stream row gate."""

        self.text_seq_len = _validate_text_seq_len(text_seq_len)
        return self

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        image_emb: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        mask_single_text_rows: Optional[bool] = None,
        text_seq_len: Optional[int] = None,
    ) -> torch.FloatTensor:
        # These arguments are retained for drop-in compatibility with upstream.
        del attention_mask, mask

        batch_size, _, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if image_emb is not None:
            # `ip-adapter` projections.
            ip_hidden_states = image_emb
            ip_hidden_states_key_proj = self.to_k_ip(ip_hidden_states)
            ip_hidden_states_value_proj = self.to_v_ip(ip_hidden_states)

            ip_hidden_states_key_proj = ip_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            ip_hidden_states_value_proj = ip_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            ip_hidden_states_key_proj = self.norm_added_k(
                ip_hidden_states_key_proj
            )

            ip_hidden_states = F.scaled_dot_product_attention(
                query,
                ip_hidden_states_key_proj,
                ip_hidden_states_value_proj,
                dropout_p=0.0,
                is_causal=False,
            )
            ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(
                batch_size, -1, attn.heads * head_dim
            )
            ip_hidden_states = ip_hidden_states.to(query.dtype)

        # FluxSingleTransformerBlock does not pass `encoder_hidden_states`.
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(
                encoder_hidden_states
            )
            encoder_hidden_states_key_proj = attn.add_k_proj(
                encoder_hidden_states
            )
            encoder_hidden_states_value_proj = attn.add_v_proj(
                encoder_hidden_states
            )

            encoder_hidden_states_query_proj = (
                encoder_hidden_states_query_proj.view(
                    batch_size, -1, attn.heads, head_dim
                ).transpose(1, 2)
            )
            encoder_hidden_states_key_proj = (
                encoder_hidden_states_key_proj.view(
                    batch_size, -1, attn.heads, head_dim
                ).transpose(1, 2)
            )
            encoder_hidden_states_value_proj = (
                encoder_hidden_states_value_proj.view(
                    batch_size, -1, attn.heads, head_dim
                ).transpose(1, 2)
            )

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(
                    encoder_hidden_states_query_proj
                )
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(
                    encoder_hidden_states_key_proj
                )

            # Joint text/image attention.
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # Deliberately unchanged from upstream: in a double-stream block the
            # IP residual updates image hidden states only.
            if image_emb is not None:
                hidden_states = hidden_states + self.scale * ip_hidden_states

            # Linear projection and dropout.
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states

        if image_emb is not None:
            effective_mask = (
                self.mask_single_text_rows
                if mask_single_text_rows is None
                else _validate_mask_flag(mask_single_text_rows)
            )
            if effective_mask:
                effective_text_seq_len = (
                    self.text_seq_len
                    if text_seq_len is None
                    else _validate_text_seq_len(text_seq_len)
                )
                if effective_text_seq_len is None:
                    raise ValueError(
                        "text_seq_len must be set when "
                        "mask_single_text_rows=True."
                    )
                ip_hidden_states = mask_single_stream_ip_residual(
                    ip_hidden_states, effective_text_seq_len
                )

            hidden_states = hidden_states + self.scale * ip_hidden_states

        return hidden_states


__all__ = [
    "IPAFluxAttnProcessor2_0",
    "build_single_stream_ip_row_gate",
    "mask_single_stream_ip_residual",
]
