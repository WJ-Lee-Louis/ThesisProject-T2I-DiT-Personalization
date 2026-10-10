"""Minimal InstantX adapter overlay for the two training-free interventions."""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
from typing import Iterable, Optional

import torch
from PIL import Image

from .attention_processor import IPAFluxAttnProcessor2_0
from .semantic_components import SemanticConditioning
from .component_directions import build_component_directions
from .orthogonalization import (
    ContrastDirections,
    PromptSet,
    build_absolute_direction,
    build_factorial_directions,
    build_local_directions,
    build_mean_paired_contrast_direction,
    hierarchical_orthogonalize_pooled_embedding,
    normalize_fp32,
    orthogonalize_pooled_embedding,
)


ORTHOGONALIZATION_MODES = {
    "components",
    "off",
    "local",
    "factorial",
    "class_contextual",
}
CONTRAST_SET_MODES = {"local", "factorial"}


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
class PromptContrastPair:
    """One positive-minus-counterfactual prompt pair for class projection."""

    positive: str
    negative: str
    label: str

    def __post_init__(self) -> None:
        for field_name in ("positive", "negative", "label"):
            value = getattr(self, field_name)
            if (
                not isinstance(value, str)
                or not value
                or value != " ".join(value.split())
            ):
                raise ValueError(
                    f"{field_name} must be a normalized non-empty string"
                )


@dataclass(frozen=True)
class ClassDirectionSet:
    """One independently averaged class direction built from matched pairs."""

    label: str
    pairs: tuple[PromptContrastPair, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.label, str)
            or not self.label
            or self.label != " ".join(self.label.split())
        ):
            raise ValueError("class direction-set label must be normalized and non-empty")
        if not isinstance(self.pairs, tuple):
            raise TypeError("class direction-set pairs must be a tuple")
        if not all(isinstance(pair, PromptContrastPair) for pair in self.pairs):
            raise TypeError(
                "class direction-set pairs must contain PromptContrastPair values"
            )
        _validate_class_pairs(self.pairs)


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
    # Appended after every legacy field to preserve positional-call behavior.
    comparison_prompts: Optional[PromptSet] = None
    class_contextual_pairs: tuple[PromptContrastPair, ...] = ()
    class_strength: Optional[float] = None
    primary_class_label: Optional[str] = None
    additional_class_direction_sets: tuple[ClassDirectionSet, ...] = ()
    class_absolute_prompt: Optional[str] = None
    residualize_class_against_nuisance: bool = True
    semantic_conditioning: Optional[SemanticConditioning] = None

    def __post_init__(self) -> None:
        if self.mode not in ORTHOGONALIZATION_MODES:
            choices = ", ".join(sorted(ORTHOGONALIZATION_MODES))
            raise ValueError(f"mode must be one of: {choices}")
        if not math.isfinite(self.strength) or not 0.0 <= self.strength <= 1.0:
            raise ValueError("strength must be finite and in [0, 1]")
        if self.class_strength is not None and (
            not math.isfinite(self.class_strength)
            or not 0.0 <= self.class_strength <= 1.0
        ):
            raise ValueError("class_strength must be finite and in [0, 1]")
        if self.mode == "components":
            if not isinstance(self.semantic_conditioning, SemanticConditioning):
                raise ValueError("components mode requires SemanticConditioning")
            if (self.prompts is not None or self.additional_direction_sets or self.comparison_prompts is not None
                    or self.class_contextual_pairs or self.class_absolute_prompt is not None
                    or self.class_strength is not None or self.additional_class_direction_sets
                    or self.primary_label is not None or self.primary_class_label is not None
                    or self.include_interaction or not self.residualize_class_against_nuisance):
                raise ValueError("components mode cannot mix legacy contrast/class settings")
            if not math.isfinite(self.relative_tolerance) or self.relative_tolerance < 0:
                raise ValueError("relative_tolerance must be finite and non-negative")
            return
        if self.semantic_conditioning is not None:
            raise ValueError("semantic_conditioning is only valid in components mode")
        if not isinstance(self.additional_direction_sets, tuple):
            raise TypeError("additional_direction_sets must be a tuple")
        if not all(
            isinstance(direction_set, TextDirectionSet)
            for direction_set in self.additional_direction_sets
        ):
            raise TypeError(
                "additional_direction_sets must contain TextDirectionSet values"
            )
        if not isinstance(self.class_contextual_pairs, tuple):
            raise TypeError("class_contextual_pairs must be a tuple")
        if not all(
            isinstance(pair, PromptContrastPair)
            for pair in self.class_contextual_pairs
        ):
            raise TypeError(
                "class_contextual_pairs must contain PromptContrastPair values"
            )
        if self.class_contextual_pairs:
            _validate_class_pairs(self.class_contextual_pairs)
        if self.class_absolute_prompt is not None and (
            not isinstance(self.class_absolute_prompt, str)
            or not self.class_absolute_prompt
            or self.class_absolute_prompt
            != " ".join(self.class_absolute_prompt.split())
        ):
            raise ValueError(
                "class_absolute_prompt must be normalized and non-empty"
            )
        if not isinstance(self.residualize_class_against_nuisance, bool):
            raise TypeError(
                "residualize_class_against_nuisance must be a boolean"
            )
        if self.primary_class_label is not None and (
            not isinstance(self.primary_class_label, str)
            or not self.primary_class_label
            or self.primary_class_label
            != " ".join(self.primary_class_label.split())
        ):
            raise ValueError("primary_class_label must be normalized and non-empty")
        if not isinstance(self.additional_class_direction_sets, tuple):
            raise TypeError("additional_class_direction_sets must be a tuple")
        if not all(
            isinstance(direction_set, ClassDirectionSet)
            for direction_set in self.additional_class_direction_sets
        ):
            raise TypeError(
                "additional_class_direction_sets must contain ClassDirectionSet values"
            )
        class_set_labels = [
            label
            for label in (
                self.primary_class_label,
                *(value.label for value in self.additional_class_direction_sets),
            )
            if label is not None
        ]
        if len(class_set_labels) != len(set(class_set_labels)):
            raise ValueError("class direction-set labels must be unique")
        if self.mode == "off":
            if self.comparison_prompts is not None:
                raise ValueError(
                    "comparison_prompts are not valid when orthogonalization is off"
                )
            # Preserve the legacy behavior in which an unused primary prompt
            # set was tolerated while mode='off'.
            if self.primary_label is not None or self.additional_direction_sets:
                raise ValueError(
                    "direction labels/sets are not valid when orthogonalization is off"
                )
            if self.class_contextual_pairs:
                raise ValueError(
                    "class_contextual_pairs are not valid when orthogonalization is off"
                )
            if self.class_absolute_prompt is not None:
                raise ValueError(
                    "class_absolute_prompt is not valid when orthogonalization is off"
                )
            if self.class_strength is not None:
                raise ValueError(
                    "class_strength is not valid when orthogonalization is off"
                )
            if self.primary_class_label is not None or self.additional_class_direction_sets:
                raise ValueError(
                    "class direction sets are not valid when orthogonalization is off"
                )
            if not self.residualize_class_against_nuisance:
                raise ValueError(
                    "class residualization control is not valid when orthogonalization is off"
                )
            return

        if self.mode == "class_contextual":
            if self.class_absolute_prompt is not None:
                raise ValueError(
                    "class_absolute_prompt is not used by class_contextual mode"
                )
            if not self.residualize_class_against_nuisance:
                raise ValueError(
                    "class residualization control is only used by hierarchical requests"
                )
            if self.class_strength is not None:
                raise ValueError(
                    "class_contextual mode uses strength directly; "
                    "class_strength is only for hierarchical requests"
                )
            if self.primary_class_label is not None or self.additional_class_direction_sets:
                raise ValueError(
                    "class_contextual mode does not accept hierarchical class sets"
                )
            if self.primary_label is not None or self.additional_direction_sets:
                raise ValueError(
                    "class projection modes do not accept direction labels/sets"
                )
            if self.class_contextual_pairs:
                if self.prompts is not None or self.comparison_prompts is not None:
                    raise ValueError(
                        "class_contextual mode accepts either explicit pairs or "
                        "legacy prompt sets, not both"
                    )
                return
            if self.prompts is None or self.comparison_prompts is None:
                raise ValueError(
                    "class_contextual requires explicit pairs, or "
                    "comparison_prompts are required with legacy prompts"
                )
            required_fields = (
                "object_behavior",
                "object_background",
                "object_behavior_background",
            )
            if any(
                getattr(prompt_set, field_name) is None
                for prompt_set in (self.prompts, self.comparison_prompts)
                for field_name in required_fields
            ):
                raise ValueError(
                    "class_contextual mode requires behavior, background, and "
                    "behavior-background prompts for both classes"
                )
            return

        if self.prompts is None:
            raise ValueError("prompts are required when orthogonalization is enabled")
        if self.comparison_prompts is not None:
            raise ValueError(
                "comparison_prompts are only valid for class_contextual mode"
            )
        has_contextual_class = bool(self.class_contextual_pairs)
        has_absolute_class = self.class_absolute_prompt is not None
        if has_contextual_class and has_absolute_class:
            raise ValueError(
                "hierarchical requests accept either contextual class pairs or "
                "an absolute class prompt, not both"
            )
        has_hierarchical_class = has_contextual_class or has_absolute_class
        if has_hierarchical_class and self.class_strength is None:
            raise ValueError(
                "class_strength is required when nuisance and class projections are combined"
            )
        if not has_hierarchical_class and self.class_strength is not None:
            raise ValueError(
                "class_strength requires contextual class pairs or an absolute class prompt"
            )
        if not has_contextual_class and self.additional_class_direction_sets:
            raise ValueError(
                "additional class direction sets require contextual class pairs"
            )
        if not has_hierarchical_class and (
            self.primary_class_label is not None
        ):
            raise ValueError(
                "primary_class_label requires a hierarchical class direction"
            )
        if not has_hierarchical_class and not self.residualize_class_against_nuisance:
            raise ValueError(
                "class residualization control requires a hierarchical class direction"
            )
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

    @property
    def hierarchical_class_enabled(self) -> bool:
        return (
            bool(self.class_contextual_pairs)
            or self.class_absolute_prompt is not None
        ) and self.mode in CONTRAST_SET_MODES


def _validate_direction_set(
    *,
    mode: str,
    prompts: PromptSet,
    include_behavior: bool,
    include_background: bool,
    label: Optional[str],
) -> None:
    if mode not in CONTRAST_SET_MODES:
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


def _validate_class_pairs(pairs: tuple[PromptContrastPair, ...]) -> None:
    if len(pairs) not in {1, 3}:
        raise ValueError(
            "class_contextual_pairs must contain exactly one or three pairs"
        )
    labels = [pair.label for pair in pairs]
    if len(labels) != len(set(labels)):
        raise ValueError("class_contextual pair labels must be unique")


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
        self._last_direction_build_diagnostics: dict = {}
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
        if request.mode == "components":
            conditioning = request.semantic_conditioning
            self._last_direction_build_diagnostics = {
                "direction_strategy": "semantic_" + conditioning.strategy,
                "direction_raw_norm_kind": "pre_row_normalization_contrast_norm",
                "semantic_conditioning": conditioning.to_dict(),
            }
            return build_component_directions(conditioning, self._encode_component_text_pool)
        if request.mode == "class_contextual":
            return self._build_class_direction(request)

        assert request.prompts is not None
        self._last_direction_build_diagnostics = {
            "direction_strategy": "legacy_prompt_contrast_rows",
            "direction_raw_norm_kind": "pre_row_normalization_contrast_norm",
        }
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

    def _encode_component_text_pool(self, prompts):
        # Legacy experiments retain their old truncation behavior. Free-text input
        # must not silently lose a component beyond the encoder's positional limit.
        tokenizer = self.clip_image_processor.tokenizer
        encoder = self._get_siglip_text_encoder()
        limit = encoder.config.max_position_embeddings
        token_ids = tokenizer(prompts, truncation=False, padding=False)["input_ids"]
        if any(len(ids) > limit for ids in token_ids):
            raise ValueError(f"semantic contrast exceeds SigLIP's {limit}-token limit; shorten input text")
        return self._encode_siglip_text_pool(prompts)

    def _build_class_direction(self, request: OrthogonalizationRequest):
        """Build the single contextual class direction for either use mode."""

        if request.class_contextual_pairs:
            return self._build_class_direction_from_pairs(
                request.class_contextual_pairs,
                direction_name=(
                    request.primary_class_label or "class_context_mean"
                ),
            )
        else:
            assert request.mode == "class_contextual"
            assert request.prompts is not None
            assert request.comparison_prompts is not None
            positive_prompts = (
                request.prompts.object_behavior,
                request.prompts.object_background,
                request.prompts.object_behavior_background,
            )
            negative_prompts = (
                request.comparison_prompts.object_behavior,
                request.comparison_prompts.object_background,
                request.comparison_prompts.object_behavior_background,
            )
            assert all(prompt is not None for prompt in positive_prompts)
            assert all(prompt is not None for prompt in negative_prompts)
            pair_labels = (
                "behavior_context",
                "background_context",
                "joint_context",
            )
            direction_name = "source_class_context_mean"

        pairs = tuple(
            PromptContrastPair(
                positive=positive,
                negative=negative,
                label=label,
            )
            for positive, negative, label in zip(
                positive_prompts,
                negative_prompts,
                pair_labels,
            )
        )
        return self._build_class_direction_from_pairs(
            pairs,
            direction_name=direction_name,
        )

    def _build_hierarchical_class_directions(
        self,
        request: OrthogonalizationRequest,
    ) -> ContrastDirections:
        """Build one equally weighted mean direction per class context source."""

        if request.class_absolute_prompt is not None:
            encoded = self._encode_siglip_text_pool(
                [request.class_absolute_prompt]
            )
            directions = build_absolute_direction(
                encoded[0],
                name=(request.primary_class_label or "class_anchor_absolute"),
            )
            self._last_direction_build_diagnostics = {
                "direction_strategy": "absolute_class_anchor",
                "direction_raw_norm_kind": "absolute_embedding_norm",
                "class_direction_set_count": 1,
                "class_pair_count": 0,
                "class_pair_labels": [],
                "class_positive_prompts": [request.class_absolute_prompt],
                "class_negative_prompts": [],
                "class_pair_raw_norms": [],
                "class_anchor_prompt": request.class_absolute_prompt,
            }
            return directions

        primary_label = request.primary_class_label or "class_context_mean"
        direction_sets = (
            ClassDirectionSet(
                label=primary_label,
                pairs=request.class_contextual_pairs,
            ),
            *request.additional_class_direction_sets,
        )
        built_directions = []
        set_diagnostics = []
        for direction_set in direction_sets:
            built = self._build_class_direction_from_pairs(
                direction_set.pairs,
                direction_name=direction_set.label,
            )
            built_directions.append(built)
            set_diagnostics.append(
                {
                    "label": direction_set.label,
                    **self._last_direction_build_diagnostics,
                }
            )
        aggregate_diagnostics = {
            "direction_strategy": "contextual_class_direction_set",
            "direction_raw_norm_kind": (
                "pre_final_normalization_mean_direction_norm"
            ),
            "class_direction_set_count": len(direction_sets),
            "class_direction_sets": set_diagnostics,
            "class_pair_count": sum(
                len(direction_set.pairs) for direction_set in direction_sets
            ),
            "class_pair_labels": [
                f"{direction_set.label}:{pair.label}"
                for direction_set in direction_sets
                for pair in direction_set.pairs
            ],
            "class_positive_prompts": [
                pair.positive
                for direction_set in direction_sets
                for pair in direction_set.pairs
            ],
            "class_negative_prompts": [
                pair.negative
                for direction_set in direction_sets
                for pair in direction_set.pairs
            ],
            "class_pair_raw_norms": [
                norm
                for diagnostics in set_diagnostics
                for norm in diagnostics["class_pair_raw_norms"]
            ],
        }
        if len(set_diagnostics) == 1:
            aggregate_diagnostics["class_pair_direction_cosine_matrix"] = (
                set_diagnostics[0]["class_pair_direction_cosine_matrix"]
            )
        self._last_direction_build_diagnostics = aggregate_diagnostics
        return ContrastDirections(
            values=torch.cat(
                [directions.values for directions in built_directions],
                dim=0,
            ),
            names=tuple(
                name
                for directions in built_directions
                for name in directions.names
            ),
            raw_norms=torch.cat(
                [directions.raw_norms for directions in built_directions],
                dim=0,
            ),
        )

    def _build_class_direction_from_pairs(
        self,
        pairs: tuple[PromptContrastPair, ...],
        *,
        direction_name: str,
    ) -> ContrastDirections:
        positive_prompts = tuple(pair.positive for pair in pairs)
        negative_prompts = tuple(pair.negative for pair in pairs)
        pair_labels = tuple(pair.label for pair in pairs)

        present_prompts = list(
            dict.fromkeys((*positive_prompts, *negative_prompts))
        )
        encoded = self._encode_siglip_text_pool(present_prompts)
        embedding_by_prompt = dict(zip(present_prompts, encoded))
        positive_embeddings = torch.stack(
            [embedding_by_prompt[prompt] for prompt in positive_prompts]
        )
        negative_embeddings = torch.stack(
            [embedding_by_prompt[prompt] for prompt in negative_prompts]
        )
        paired_raw = normalize_fp32(positive_embeddings) - normalize_fp32(
            negative_embeddings
        )
        paired_unit = normalize_fp32(paired_raw)
        self._last_direction_build_diagnostics = {
            "direction_strategy": "contextual_class_contrast_mean",
            "direction_raw_norm_kind": (
                "pre_final_normalization_mean_direction_norm"
            ),
            "class_pair_labels": list(pair_labels),
            "class_pair_count": len(pair_labels),
            "class_positive_prompts": list(positive_prompts),
            "class_negative_prompts": list(negative_prompts),
            # Keep stable list/matrix shapes even for a single pair.
            "class_pair_raw_norms": (
                torch.linalg.vector_norm(paired_raw, dim=-1)
                .detach()
                .float()
                .cpu()
                .tolist()
            ),
            "class_pair_direction_cosine_matrix": (
                (paired_unit @ paired_unit.transpose(0, 1))
                .detach()
                .float()
                .cpu()
                .tolist()
            ),
        }
        return build_mean_paired_contrast_direction(
            positive_embeddings,
            negative_embeddings,
            name=direction_name,
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

        if request.hierarchical_class_enabled:
            nuisance_directions = self._build_text_directions(request)
            nuisance_build_diagnostics = dict(
                self._last_direction_build_diagnostics
            )
            class_directions = self._build_hierarchical_class_directions(
                request
            )
            class_build_diagnostics = dict(
                self._last_direction_build_diagnostics
            )
            assert request.class_strength is not None
            hierarchical_result = hierarchical_orthogonalize_pooled_embedding(
                clip_image_embeds,
                nuisance_directions.values,
                class_directions.values,
                nuisance_strength=request.strength,
                class_strength=request.class_strength,
                residualize_class_against_nuisance=(
                    request.residualize_class_against_nuisance
                ),
                relative_tolerance=request.relative_tolerance,
                restore_norm=request.restore_norm,
                preserve_dtype=False,
            )
            edited_visual_pool = hierarchical_result.embedding.to(
                device=self.device,
                dtype=torch.bfloat16,
            )
            image_prompt_embeds = self.image_proj_model(edited_visual_pool)
            nuisance_basis_diagnostics = (
                hierarchical_result.nuisance_basis_diagnostics
            )
            class_basis_diagnostics = (
                hierarchical_result.class_basis_diagnostics
            )
            projection_diagnostics = (
                hierarchical_result.projection_diagnostics
            )
            combined_input_directions = torch.cat(
                (nuisance_directions.values, class_directions.values),
                dim=0,
            )
            nuisance_raw_norms = (
                nuisance_directions.raw_norms.detach().float().cpu().tolist()
                if nuisance_directions.raw_norms is not None
                else []
            )
            class_raw_norms = (
                class_directions.raw_norms.detach().float().cpu().tolist()
                if class_directions.raw_norms is not None
                else []
            )
            singular_values = (
                nuisance_basis_diagnostics.singular_values
                .detach()
                .float()
                .cpu()
                .tolist()
                + class_basis_diagnostics.singular_values
                .detach()
                .float()
                .cpu()
                .tolist()
            )
            relative_singular_values = (
                nuisance_basis_diagnostics.relative_singular_values
                .detach()
                .float()
                .cpu()
                .tolist()
                + class_basis_diagnostics.relative_singular_values
                .detach()
                .float()
                .cpu()
                .tolist()
            )
            self.last_orthogonalization_diagnostics = {
                "enabled": True,
                "mode": request.mode,
                "strength": request.strength,
                "nuisance_strength": request.strength,
                "class_strength": request.class_strength,
                "hierarchical_class_projection": True,
                "class_overlap_policy": (
                    "residual"
                    if request.residualize_class_against_nuisance
                    else "direct"
                ),
                "direction_strategy": (
                    "nuisance_svd_then_class_residual"
                    if request.residualize_class_against_nuisance
                    else "nuisance_svd_plus_direct_class_basis"
                ),
                "nuisance_direction_build": nuisance_build_diagnostics,
                **class_build_diagnostics,
                "direction_set_count": (
                    2
                    + len(request.additional_direction_sets)
                    + len(request.additional_class_direction_sets)
                ),
                "direction_names": [
                    *nuisance_directions.names,
                    *class_directions.names,
                ],
                "direction_normalization": (
                    "nuisance_l2_rows_then_class_pair_mean_then_residual"
                ),
                "direction_raw_norms": [
                    *nuisance_raw_norms,
                    *class_raw_norms,
                ],
                "input_direction_count": (
                    nuisance_basis_diagnostics.direction_count
                    + hierarchical_result.class_direction_count
                ),
                "direction_cosine_matrix": _tensor_to_python(
                    combined_input_directions
                    @ combined_input_directions.transpose(0, 1)
                ),
                "basis_rank": (
                    nuisance_basis_diagnostics.retained_rank
                    + class_basis_diagnostics.retained_rank
                ),
                "dropped_rank": (
                    nuisance_basis_diagnostics.dropped_rank
                    + class_basis_diagnostics.dropped_rank
                ),
                "basis_threshold": nuisance_basis_diagnostics.threshold,
                "singular_values": singular_values,
                "relative_singular_values": relative_singular_values,
                "nuisance_basis_rank": (
                    nuisance_basis_diagnostics.retained_rank
                ),
                "nuisance_dropped_rank": (
                    nuisance_basis_diagnostics.dropped_rank
                ),
                "nuisance_singular_values": (
                    nuisance_basis_diagnostics.singular_values
                    .detach()
                    .float()
                    .cpu()
                    .tolist()
                ),
                "class_direction_count": (
                    hierarchical_result.class_direction_count
                ),
                "class_basis_rank": class_basis_diagnostics.retained_rank,
                "class_dropped_rank": class_basis_diagnostics.dropped_rank,
                "class_singular_values": (
                    class_basis_diagnostics.singular_values
                    .detach()
                    .float()
                    .cpu()
                    .tolist()
                ),
                "class_input_norm": hierarchical_result.class_input_norm,
                "class_overlap_norm": hierarchical_result.class_overlap_norm,
                "class_residual_norm": hierarchical_result.class_residual_norm,
                "class_residual_ratio": (
                    hierarchical_result.class_residual_ratio
                ),
                "class_input_norms": _tensor_to_python(
                    hierarchical_result.class_input_norms
                ),
                "class_overlap_norms": _tensor_to_python(
                    hierarchical_result.class_overlap_norms
                ),
                "class_residual_norms": _tensor_to_python(
                    hierarchical_result.class_residual_norms
                ),
                "class_residual_ratios": _tensor_to_python(
                    hierarchical_result.class_residual_ratios
                ),
                "class_basis_threshold": hierarchical_result.class_threshold,
                "class_direction_dropped": (
                    class_basis_diagnostics.retained_rank == 0
                ),
                "original_norm": _tensor_to_python(
                    projection_diagnostics.original_norm
                ),
                "nuisance_component_norm": _tensor_to_python(
                    projection_diagnostics.nuisance_component_norm
                ),
                "class_component_norm": _tensor_to_python(
                    projection_diagnostics.class_component_norm
                ),
                "projection_component_norm": _tensor_to_python(
                    projection_diagnostics.projection_component_norm
                ),
                "nuisance_removed_component_norm": _tensor_to_python(
                    projection_diagnostics.nuisance_removed_component_norm
                ),
                "class_removed_component_norm": _tensor_to_python(
                    projection_diagnostics.class_removed_component_norm
                ),
                "removed_component_norm": _tensor_to_python(
                    projection_diagnostics.removed_component_norm
                ),
                "adjusted_unit_norm": _tensor_to_python(
                    projection_diagnostics.adjusted_unit_norm
                ),
                "output_norm": _tensor_to_python(
                    projection_diagnostics.output_norm
                ),
                "degenerate": _tensor_to_python(
                    projection_diagnostics.degenerate
                ),
                "edit_norm": _tensor_to_python(
                    projection_diagnostics.edit_norm
                ),
                "cosine_to_input": _tensor_to_python(
                    projection_diagnostics.cosine_to_input
                ),
                "max_abs_nuisance_basis_dot": _tensor_to_python(
                    projection_diagnostics.max_abs_nuisance_basis_dot
                ),
                "max_abs_class_basis_dot": _tensor_to_python(
                    projection_diagnostics.max_abs_class_basis_dot
                ),
                "max_abs_basis_dot": _tensor_to_python(
                    projection_diagnostics.max_abs_basis_dot
                ),
            }
            return image_prompt_embeds

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
            **self._last_direction_build_diagnostics,
            "direction_set_count": (len(request.semantic_conditioning.selected)
                                    if request.mode == "components"
                                    else 1 + len(request.additional_direction_sets)),
            "direction_names": list(directions.names),
            "direction_normalization": (
                "normalize_prompts_then_pairs_then_mean"
                if request.mode == "class_contextual"
                else "l2_per_row"
            ),
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
    "ClassDirectionSet",
    "ORTHOGONALIZATION_MODES",
    "OrthogonalizationRequest",
    "OrthogonalizedIPAdapterMixin",
    "PromptContrastPair",
    "TextDirectionSet",
]
