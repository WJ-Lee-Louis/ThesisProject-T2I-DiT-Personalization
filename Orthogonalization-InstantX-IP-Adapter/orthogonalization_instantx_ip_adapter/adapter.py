"""Minimal InstantX adapter overlay for the two training-free interventions."""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
from typing import Iterable, Optional

import torch
from PIL import Image

from .attention_processor import IPAFluxAttnProcessor2_0
from .orthogonalization import (
    ContrastDirections,
    PromptSet,
    build_factorial_directions,
    build_local_directions,
    orthogonalize_pooled_embedding,
)


ORTHOGONALIZATION_MODES = {"off", "local", "factorial"}


@dataclass(frozen=True)
class TextDirectionSet:
    """One named prompt-contrast set contributing rows to a shared SVD."""

    mode: str
    prompts: PromptSet
    include_behavior: bool = True
    include_background: bool = True
    include_interaction: bool = False
    label: Optional[str] = None

    def __post_init__(self) -> None:
        _validate_direction_set(
            mode=self.mode,
            prompts=self.prompts,
            include_behavior=self.include_behavior,
            include_background=self.include_background,
            label=self.label,
        )


@dataclass(frozen=True)
class OrthogonalizationRequest:
    """Per-generation settings for editing the SigLIP visual pooled vector."""

    mode: str = "off"
    strength: float = 0.5
    prompts: Optional[PromptSet] = None
    include_behavior: bool = True
    include_background: bool = True
    include_interaction: bool = False
    restore_norm: bool = True
    relative_tolerance: float = 1e-3
    primary_label: Optional[str] = None
    additional_direction_sets: tuple[TextDirectionSet, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in ORTHOGONALIZATION_MODES:
            choices = ", ".join(sorted(ORTHOGONALIZATION_MODES))
            raise ValueError(f"mode must be one of: {choices}")
        if not math.isfinite(self.strength) or not 0.0 <= self.strength <= 1.0:
            raise ValueError("strength must be finite and in [0, 1]")
        if not isinstance(self.additional_direction_sets, tuple):
            raise TypeError("additional_direction_sets must be a tuple")
        if not all(
            isinstance(direction_set, TextDirectionSet)
            for direction_set in self.additional_direction_sets
        ):
            raise TypeError(
                "additional_direction_sets must contain TextDirectionSet values"
            )
        if self.mode == "off":
            if self.primary_label is not None or self.additional_direction_sets:
                raise ValueError(
                    "direction labels/sets are not valid when orthogonalization is off"
                )
            return
        if self.prompts is None:
            raise ValueError("prompts are required when orthogonalization is enabled")
        _validate_direction_set(
            mode=self.mode,
            prompts=self.prompts,
            include_behavior=self.include_behavior,
            include_background=self.include_background,
            label=self.primary_label,
        )

        labels = [
            label
            for label in (
                self.primary_label,
                *(value.label for value in self.additional_direction_sets),
            )
            if label is not None
        ]
        if len(labels) != len(set(labels)):
            raise ValueError("direction-set labels must be unique")

    @property
    def enabled(self) -> bool:
        return self.mode != "off"


def _validate_direction_set(
    *,
    mode: str,
    prompts: PromptSet,
    include_behavior: bool,
    include_background: bool,
    label: Optional[str],
) -> None:
    if mode not in ORTHOGONALIZATION_MODES - {"off"}:
        raise ValueError("direction-set mode must be local or factorial")
    if not isinstance(prompts, PromptSet):
        raise TypeError("direction-set prompts must be a PromptSet")
    if not include_behavior and not include_background:
        raise ValueError(
            "at least one of include_behavior/include_background must be enabled"
        )
    if label is not None and (not isinstance(label, str) or not label.strip()):
        raise ValueError("direction-set label must be a non-empty string or None")
    if mode == "factorial" and any(
        value is None
        for value in (
            prompts.object_behavior,
            prompts.object_background,
            prompts.object_behavior_background,
        )
    ):
        raise ValueError("factorial mode requires all four counterfactual prompts")


def _build_direction_set(
    direction_set: TextDirectionSet,
    embedding_by_prompt: dict[str, torch.Tensor],
) -> ContrastDirections:
    def get(prompt: Optional[str]):
        return None if prompt is None else embedding_by_prompt[prompt]

    prompts = direction_set.prompts
    if direction_set.mode == "factorial":
        factorial = build_factorial_directions(
            get(prompts.object_only),
            get(prompts.object_behavior),
            get(prompts.object_background),
            get(prompts.object_behavior_background),
            include_interaction=direction_set.include_interaction,
        )
        retained_indices = []
        if direction_set.include_behavior:
            retained_indices.append(0)
        if direction_set.include_background:
            retained_indices.append(1)
        if direction_set.include_interaction:
            retained_indices.append(2)
        result = ContrastDirections(
            values=factorial.values[retained_indices],
            names=tuple(factorial.names[index] for index in retained_indices),
            raw_norms=(
                factorial.raw_norms[retained_indices]
                if factorial.raw_norms is not None
                else None
            ),
        )
    else:
        result = build_local_directions(
            get(prompts.object_only),
            object_behavior=get(prompts.object_behavior),
            object_background=get(prompts.object_background),
            object_behavior_background=get(prompts.object_behavior_background),
            include_behavior=direction_set.include_behavior,
            include_background=direction_set.include_background,
            include_joint=direction_set.include_interaction,
        )

    if direction_set.label is None:
        return result
    return ContrastDirections(
        values=result.values,
        names=tuple(f"{direction_set.label}/{name}" for name in result.names),
        raw_norms=result.raw_norms,
    )


class OrthogonalizedIPAdapterMixin:
    """Mixin composed with InstantX's published ``IPAdapter`` at load time.

    The upstream class is kept outside this package because InstantX distributes
    it as a model-repository source file. ``model_loader`` downloads a pinned
    revision and creates ``(this mixin, upstream IPAdapter)`` as the runtime MRO.
    """

    def __init__(self, *args, **kwargs):
        self._active_orthogonalization = OrthogonalizationRequest()
        self._mask_single_text_rows = False
        self._generation_lock = threading.RLock()
        self._siglip_text_encoder = None
        self._text_pool_cache: dict[str, torch.Tensor] = {}
        self.last_orthogonalization_diagnostics: dict = {"enabled": False}
        self.last_single_stream_text_seq_len: Optional[int] = None
        super().__init__(*args, **kwargs)

        # Capture the true FLUX text-row count on every denoising call. This
        # avoids hard-coding 512 and avoids copying InstantX's full transformer
        # and pipeline files solely to pass one scalar to the processors.
        self._text_length_hook_handle = (
            self.pipe.transformer.register_forward_pre_hook(
                self._capture_text_seq_len,
                with_kwargs=True,
            )
        )

    def set_ip_adapter(self):
        """Install checkpoint-compatible processors with the optional row gate."""

        transformer = self.pipe.transformer
        ip_attn_procs = {}
        for name, current_processor in transformer.attn_processors.items():
            if name.startswith("transformer_blocks.") or name.startswith(
                "single_transformer_blocks"
            ):
                ip_attn_procs[name] = IPAFluxAttnProcessor2_0(
                    hidden_size=(
                        transformer.config.num_attention_heads
                        * transformer.config.attention_head_dim
                    ),
                    cross_attention_dim=transformer.config.joint_attention_dim,
                    num_tokens=self.num_tokens,
                    mask_single_text_rows=self._mask_single_text_rows,
                ).to(self.device, dtype=torch.bfloat16)
            else:
                ip_attn_procs[name] = current_processor
        transformer.set_attn_processor(ip_attn_procs)

    def set_scale(self, scale):
        """Set the IP scale on this package's processors."""

        for processor in self._iter_experiment_processors():
            processor.scale = scale

    def load_ip_adapter(self):
        """Load the published checkpoint and fail on silent processor mismatch."""

        try:
            state_dict = torch.load(
                self.ip_ckpt,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")

        self.image_proj_model.load_state_dict(
            state_dict["image_proj"],
            strict=True,
        )
        ip_layers = torch.nn.ModuleList(
            self.pipe.transformer.attn_processors.values()
        )
        incompatible = ip_layers.load_state_dict(
            state_dict["ip_adapter"],
            strict=False,
        )
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "InstantX IP checkpoint does not exactly match the installed "
                "attention processors. "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )

        processor_count = sum(1 for _ in self._iter_experiment_processors())
        if processor_count != 57:
            raise RuntimeError(
                "Expected 57 InstantX IP processors (19 double + 38 single), "
                f"but loaded {processor_count}."
            )
        self.checkpoint_load_diagnostics = {
            "processor_count": processor_count,
            "missing_keys": [],
            "unexpected_keys": [],
        }

    def set_mask_single_text_rows(self, enabled: bool) -> None:
        """Toggle the direct single-stream IP-to-text residual gate."""

        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a bool")
        self._mask_single_text_rows = enabled
        for processor in self._iter_experiment_processors(single_stream_only=True):
            processor.set_mask_single_text_rows(enabled)

    def _iter_experiment_processors(
        self, *, single_stream_only: bool = False
    ) -> Iterable[IPAFluxAttnProcessor2_0]:
        for name, processor in self.pipe.transformer.attn_processors.items():
            if single_stream_only and not name.startswith(
                "single_transformer_blocks"
            ):
                continue
            if isinstance(processor, IPAFluxAttnProcessor2_0):
                yield processor

    def _capture_text_seq_len(self, module, args, kwargs):
        encoder_hidden_states = kwargs.get("encoder_hidden_states")
        if encoder_hidden_states is None and len(args) >= 2:
            encoder_hidden_states = args[1]
        if encoder_hidden_states is None:
            if self._mask_single_text_rows:
                raise RuntimeError(
                    "Could not determine FLUX text_seq_len while the text-row "
                    "gate is enabled."
                )
            return None

        text_seq_len = int(encoder_hidden_states.shape[1])
        self.last_single_stream_text_seq_len = text_seq_len
        for processor in self._iter_experiment_processors(single_stream_only=True):
            processor.set_text_seq_len(text_seq_len)
        return None

    def _get_siglip_text_encoder(self):
        if self._siglip_text_encoder is None:
            from transformers import SiglipTextModel

            self._siglip_text_encoder = SiglipTextModel.from_pretrained(
                self.image_encoder_path,
                torch_dtype=self.image_encoder.dtype,
            ).to(self.device)
            self._siglip_text_encoder.eval()
            for parameter in self._siglip_text_encoder.parameters():
                parameter.requires_grad_(False)
        return self._siglip_text_encoder

    @torch.inference_mode()
    def _encode_siglip_text_pool(self, prompts: list[str]) -> torch.Tensor:
        """Encode and cache paired SigLIP text ``pooler_output`` in FP32."""

        missing = [prompt for prompt in prompts if prompt not in self._text_pool_cache]
        if missing:
            processor_inputs = self.clip_image_processor(
                text=missing,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            model_inputs = {
                key: value.to(self.device)
                for key, value in processor_inputs.items()
                if isinstance(value, torch.Tensor)
            }
            pooled = (
                self._get_siglip_text_encoder()(**model_inputs)
                .pooler_output.detach()
                .float()
                .cpu()
            )
            for prompt, embedding in zip(missing, pooled):
                self._text_pool_cache[prompt] = embedding

        return torch.stack(
            [self._text_pool_cache[prompt] for prompt in prompts], dim=0
        ).to(self.device)

    def _build_text_directions(self, request: OrthogonalizationRequest):
        assert request.prompts is not None
        direction_sets = (
            TextDirectionSet(
                mode=request.mode,
                prompts=request.prompts,
                include_behavior=request.include_behavior,
                include_background=request.include_background,
                include_interaction=request.include_interaction,
                label=request.primary_label,
            ),
            *request.additional_direction_sets,
        )
        present_prompts = list(
            dict.fromkeys(
                prompt
                for direction_set in direction_sets
                for prompt in (
                    direction_set.prompts.object_only,
                    direction_set.prompts.object_behavior,
                    direction_set.prompts.object_background,
                    direction_set.prompts.object_behavior_background,
                )
                if prompt is not None
            )
        )
        encoded = self._encode_siglip_text_pool(present_prompts)
        embedding_by_prompt = dict(zip(present_prompts, encoded))

        built_sets = [
            _build_direction_set(direction_set, embedding_by_prompt)
            for direction_set in direction_sets
        ]
        return ContrastDirections(
            values=torch.cat(
                [directions.values for directions in built_sets], dim=0
            ),
            names=tuple(
                name
                for directions in built_sets
                for name in directions.names
            ),
            raw_norms=(
                torch.cat(
                    [directions.raw_norms for directions in built_sets], dim=0
                )
                if all(
                    directions.raw_norms is not None
                    for directions in built_sets
                )
                else None
            ),
        )

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        request = self._active_orthogonalization
        if not request.enabled:
            self.last_orthogonalization_diagnostics = {"enabled": False}
            # Preserve the exact upstream path for the Off/Off parity condition.
            return super().get_image_embeds(
                pil_image=pil_image,
                clip_image_embeds=clip_image_embeds,
            )

        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(
                images=pil_image,
                return_tensors="pt",
            ).pixel_values
            clip_image_embeds = self.image_encoder(
                clip_image.to(self.device, dtype=self.image_encoder.dtype)
            ).pooler_output
        elif clip_image_embeds is None:
            raise ValueError("pil_image or clip_image_embeds must be provided")
        else:
            clip_image_embeds = clip_image_embeds.to(self.device)

        directions = self._build_text_directions(request)
        result = orthogonalize_pooled_embedding(
            clip_image_embeds,
            directions.values,
            strength=request.strength,
            relative_tolerance=request.relative_tolerance,
            restore_norm=request.restore_norm,
            preserve_dtype=False,
        )
        edited_visual_pool = result.embedding.to(
            device=self.device,
            dtype=torch.bfloat16,
        )
        image_prompt_embeds = self.image_proj_model(edited_visual_pool)

        self.last_orthogonalization_diagnostics = {
            "enabled": True,
            "mode": request.mode,
            "strength": request.strength,
            "direction_set_count": 1 + len(request.additional_direction_sets),
            "direction_names": list(directions.names),
            "direction_normalization": "l2_per_row",
            "direction_raw_norms": (
                _tensor_to_python(directions.raw_norms)
                if directions.raw_norms is not None
                else None
            ),
            "input_direction_count": result.basis_diagnostics.direction_count,
            "direction_cosine_matrix": _tensor_to_python(
                directions.values @ directions.values.transpose(0, 1)
            ),
            "basis_rank": result.basis_diagnostics.retained_rank,
            "dropped_rank": result.basis_diagnostics.dropped_rank,
            "basis_threshold": result.basis_diagnostics.threshold,
            "singular_values": _tensor_to_python(
                result.basis_diagnostics.singular_values
            ),
            "relative_singular_values": _tensor_to_python(
                result.basis_diagnostics.relative_singular_values
            ),
            "original_norm": _tensor_to_python(
                result.projection_diagnostics.original_norm
            ),
            "projection_component_norm": _tensor_to_python(
                result.projection_diagnostics.projection_component_norm
            ),
            "output_norm": _tensor_to_python(
                result.projection_diagnostics.output_norm
            ),
            "removed_component_norm": _tensor_to_python(
                result.projection_diagnostics.removed_component_norm
            ),
            "adjusted_unit_norm": _tensor_to_python(
                result.projection_diagnostics.adjusted_unit_norm
            ),
            "degenerate": _tensor_to_python(
                result.projection_diagnostics.degenerate
            ),
            "edit_norm": _tensor_to_python(
                result.projection_diagnostics.edit_norm
            ),
            "cosine_to_input": _tensor_to_python(
                result.projection_diagnostics.cosine_to_input
            ),
            "max_abs_basis_dot": _tensor_to_python(
                result.projection_diagnostics.max_abs_basis_dot
            ),
        }
        return image_prompt_embeds

    def generate(
        self,
        *args,
        orthogonalization: Optional[OrthogonalizationRequest] = None,
        mask_single_text_rows: bool = False,
        **kwargs,
    ):
        """Generate with independently switchable content and routing edits."""

        request = orthogonalization or OrthogonalizationRequest()
        if not isinstance(request, OrthogonalizationRequest):
            raise TypeError("orthogonalization must be OrthogonalizationRequest")

        with self._generation_lock:
            previous_request = self._active_orthogonalization
            previous_mask = self._mask_single_text_rows
            self._active_orthogonalization = request
            self.set_mask_single_text_rows(mask_single_text_rows)
            try:
                return super().generate(*args, **kwargs)
            finally:
                self._active_orthogonalization = previous_request
                self.set_mask_single_text_rows(previous_mask)


def _tensor_to_python(value: torch.Tensor):
    if value.dtype == torch.bool:
        value = value.detach().cpu()
        if value.numel() == 1:
            return bool(value.item())
        return value.tolist()
    value = value.detach().float().cpu()
    if value.numel() == 1:
        return float(value.item())
    return value.tolist()


__all__ = [
    "ORTHOGONALIZATION_MODES",
    "OrthogonalizationRequest",
    "OrthogonalizedIPAdapterMixin",
    "TextDirectionSet",
]
