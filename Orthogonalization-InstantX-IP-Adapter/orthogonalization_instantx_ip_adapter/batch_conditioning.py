"""Map one batch job and experiment condition to model conditioning controls."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from .adapter import (
    OrthogonalizationRequest,
    PromptContrastPair,
    TextDirectionSet,
)
from .batch_config import BatchGenerationConfig
from .batch_manifest import BatchJob
from .orthogonalization import PromptSet
from .prompt_catalog import PromptFactor, PromptSpec, get_prompt_spec
from .source_captions import SourceCaptionRecord


@dataclass(frozen=True, slots=True)
class BatchConditioning:
    """Conditioning arguments consumed by ``OrthogonalizedIPAdapterMixin``."""

    orthogonalization: OrthogonalizationRequest
    mask_single_text_rows: bool
    basis_origin: Literal[
        "none",
        "target_prompt",
        "reference_caption",
        "source_target_union",
        "class_absolute",
        "class_target_contextual",
        "class_source_contextual",
    ]
    removed_factors: tuple[PromptFactor, ...]
    target_factors: tuple[PromptFactor, ...]
    source_factors: tuple[PromptFactor, ...]
    # Appended after every legacy field to preserve positional-call behavior.
    class_projection_mode: Literal[
        "none",
        "absolute",
        "target_superclass",
        "target_no_object",
        "source_superclass",
        "source_no_object",
    ] = "none"


def build_batch_conditioning(
    job: BatchJob,
    config: BatchGenerationConfig,
    source_caption: SourceCaptionRecord | None = None,
) -> BatchConditioning:
    """Build projection prompts and the single-stream row-gate flag.

    The interventions are selected by non-overlapping experiment
    conditions:

    * ``baseline``: neither intervention;
    * ``projection``: target-prompt pooled-space projection only;
    * ``gate``: single-stream text-row gate only;
    * ``combined``: target-prompt projection plus the gate;
    * ``sa_projection``: reference-caption projection only;
    * ``sa_combined``: reference-caption projection plus the gate.
    * ``st_projection``: target and reference-caption directions combined in
      one SVD projection;
    * ``st_combined``: the same single-SVD projection plus the gate.
    * ``st_all_combined``: target directions plus both source-caption main
      effects in one SVD projection, with the gate.
    * ``class_absolute_combined``: absolute ``a dog/cat`` class direction plus
      the single-stream gate;
    * ``class_target_*_combined``: target nuisance SVD, an orthogonal residual
      target-context class direction with its own strength, and the gate;
    * ``class_source_*_combined``: matched source-caption nuisance SVD, an
      orthogonal residual source-context class direction with its own strength,
      and the gate;
    * ``class_contextual_combined``: backward-compatible alias for
      ``class_source_superclass_combined``.

    Target projection directions are local for a one-factor prompt and
    factorial for a prompt containing both behavior and background.
    Source-aware directions are always estimated from the reference caption's
    complete 2-by-2 factorial prompt set, while the target suffix selects which
    source factor(s) are attenuated. Contextual class conditions preserve this
    nuisance basis, residualize the class direction against it, and use
    ``orthogonalization_strength`` and ``class_projection_strength``
    independently. ``st_all_combined`` is the explicit
    ablation that includes both source factors regardless of target taxonomy.
    """

    if not isinstance(job, BatchJob):
        raise TypeError("job must be a BatchJob")
    if not isinstance(config, BatchGenerationConfig):
        raise TypeError("config must be a BatchGenerationConfig")

    if source_caption is not None and not isinstance(
        source_caption, SourceCaptionRecord
    ):
        raise TypeError("source_caption must be a SourceCaptionRecord or None")

    if config.class_absolute_projection_enabled:
        if source_caption is not None:
            raise ValueError(
                "source_caption must not be supplied for class-absolute conditions"
            )
        request = _build_class_absolute_projection_request(
            job,
            strength=config.class_projection_strength,
        )
        basis_origin = "class_absolute"
        class_projection_mode = "absolute"
        removed_factors = ()
        target_factors = ()
        source_factors = ()
    elif config.class_contextual_projection_enabled:
        contrast_type: Literal["superclass", "no_object"] = (
            "superclass"
            if config.class_superclass_projection_enabled
            else "no_object"
        )
        if config.class_target_contextual_projection_enabled:
            if source_caption is not None:
                raise ValueError(
                    "source_caption must not be supplied for target-context "
                    "class conditions"
                )
            request = _build_class_target_contextual_projection_request(
                job,
                contrast_type=contrast_type,
                superclass_prompt=config.class_superclass_prompt,
                nuisance_strength=config.orthogonalization_strength,
                class_strength=config.class_projection_strength,
            )
            basis_origin = "class_target_contextual"
            class_projection_mode = f"target_{contrast_type}"
            target_factors = _target_factors(job.prompt_suffix)
            source_factors = ()
            removed_factors = target_factors
        else:
            if source_caption is None:
                raise ValueError(
                    "source_caption is required for source-context class conditions"
                )
            request = _build_class_source_contextual_projection_request(
                job,
                source_caption=source_caption,
                contrast_type=contrast_type,
                superclass_prompt=config.class_superclass_prompt,
                nuisance_strength=config.orthogonalization_strength,
                class_strength=config.class_projection_strength,
            )
            basis_origin = "class_source_contextual"
            class_projection_mode = f"source_{contrast_type}"
            target_factors = ()
            source_factors = _target_factors(job.prompt_suffix)
            removed_factors = source_factors
    elif config.source_target_projection_enabled:
        if source_caption is None:
            raise ValueError(
                "source_caption is required for source-target conditions"
            )
        request = _build_source_target_projection_request(
            job,
            source_caption=source_caption,
            strength=config.orthogonalization_strength,
            include_all_source_factors=config.all_source_factors_enabled,
        )
        basis_origin = "source_target_union"
        class_projection_mode = "none"
        target_factors = _target_factors(job.prompt_suffix)
        source_factors = (
            ("behavior", "background")
            if config.all_source_factors_enabled
            else target_factors
        )
        removed_factors = _merge_factors(target_factors, source_factors)
    elif config.target_projection_enabled:
        if source_caption is not None:
            raise ValueError(
                "source_caption must not be supplied for target-local conditions"
            )
        request = _build_projection_request(
            job,
            strength=config.orthogonalization_strength,
        )
        basis_origin = "target_prompt"
        class_projection_mode = "none"
        target_factors = _target_factors(job.prompt_suffix)
        source_factors = ()
        removed_factors = target_factors
    elif config.source_aware_projection_enabled:
        if source_caption is None:
            raise ValueError(
                "source_caption is required for source-aware conditions"
            )
        request = _build_source_aware_projection_request(
            job,
            source_caption=source_caption,
            strength=config.orthogonalization_strength,
        )
        basis_origin = "reference_caption"
        class_projection_mode = "none"
        target_factors = ()
        source_factors = _target_factors(job.prompt_suffix)
        removed_factors = source_factors
    else:
        if source_caption is not None:
            raise ValueError(
                "source_caption is only valid for source-aware or "
                "source-target/class-contextual conditions"
            )
        request = OrthogonalizationRequest(
            mode="off",
            strength=config.orthogonalization_strength,
        )
        basis_origin = "none"
        class_projection_mode = "none"
        removed_factors = ()
        target_factors = ()
        source_factors = ()

    return BatchConditioning(
        orthogonalization=request,
        mask_single_text_rows=config.mask_single_text_rows,
        basis_origin=basis_origin,
        class_projection_mode=class_projection_mode,
        removed_factors=removed_factors,
        target_factors=target_factors,
        source_factors=source_factors,
    )


def _build_class_absolute_projection_request(
    job: BatchJob,
    *,
    strength: float,
) -> OrthogonalizationRequest:
    """Build one absolute generic-class anchor from the job species."""

    prompt_spec = get_prompt_spec(job.prompt_suffix)
    _validate_job_prompt(job, prompt_spec)
    return OrthogonalizationRequest(
        mode="class_absolute",
        strength=strength,
        prompts=PromptSet(object_only=f"a {job.species}"),
        include_behavior=False,
        include_background=False,
    )


def _build_class_target_contextual_projection_request(
    job: BatchJob,
    *,
    contrast_type: Literal["superclass", "no_object"],
    superclass_prompt: str,
    nuisance_strength: float,
    class_strength: float,
) -> OrthogonalizationRequest:
    """Combine target nuisance rows with a residualized class direction."""

    prompt_spec = get_prompt_spec(job.prompt_suffix)
    expected_prompt = _validate_job_prompt(job, prompt_spec)
    object_only = f"a {job.species}"

    pairs: list[PromptContrastPair] = []
    if prompt_spec.behavior is not None:
        pairs.append(
            _make_class_context_pair(
                positive=f"{object_only} {prompt_spec.behavior}",
                context=prompt_spec.behavior,
                label="behavior_context",
                contrast_type=contrast_type,
                superclass_prompt=superclass_prompt,
            )
        )
    if prompt_spec.background is not None:
        pairs.append(
            _make_class_context_pair(
                positive=f"{object_only} {prompt_spec.background}",
                context=prompt_spec.background,
                label="background_context",
                contrast_type=contrast_type,
                superclass_prompt=superclass_prompt,
            )
        )
    if prompt_spec.is_factorial:
        pairs.append(
            _make_class_context_pair(
                positive=expected_prompt,
                context=(
                    f"{prompt_spec.behavior} {prompt_spec.background}"
                ),
                label="joint_context",
                contrast_type=contrast_type,
                superclass_prompt=superclass_prompt,
            )
        )

    nuisance_request = _build_projection_request(
        job,
        strength=nuisance_strength,
    )
    return replace(
        nuisance_request,
        class_contextual_pairs=tuple(pairs),
        class_strength=class_strength,
    )


def _build_class_source_contextual_projection_request(
    job: BatchJob,
    *,
    source_caption: SourceCaptionRecord,
    contrast_type: Literal["superclass", "no_object"],
    superclass_prompt: str,
    nuisance_strength: float,
    class_strength: float,
) -> OrthogonalizationRequest:
    """Combine source nuisance rows with a residualized class direction."""

    prompt_spec = get_prompt_spec(job.prompt_suffix)
    _validate_job_prompt(job, prompt_spec)
    _validate_source_caption_matches_job(job, source_caption)

    contexts = (
        source_caption.source_behavior,
        source_caption.source_background,
        f"{source_caption.source_behavior} {source_caption.source_background}",
    )
    positives = (
        source_caption.object_behavior,
        source_caption.object_background,
        source_caption.object_behavior_background,
    )
    labels = (
        "behavior_context",
        "background_context",
        "joint_context",
    )
    pairs = tuple(
        _make_class_context_pair(
            positive=positive,
            context=context,
            label=label,
            contrast_type=contrast_type,
            superclass_prompt=superclass_prompt,
        )
        for positive, context, label in zip(positives, contexts, labels)
    )

    nuisance_request = _build_source_aware_projection_request(
        job,
        source_caption=source_caption,
        strength=nuisance_strength,
    )
    return replace(
        nuisance_request,
        class_contextual_pairs=pairs,
        class_strength=class_strength,
    )


def _make_class_context_pair(
    *,
    positive: str,
    context: str,
    label: str,
    contrast_type: Literal["superclass", "no_object"],
    superclass_prompt: str,
) -> PromptContrastPair:
    """Create O+C minus either S+C or C while preserving the same context."""

    if contrast_type == "superclass":
        negative = f"{superclass_prompt} {context}"
    elif contrast_type == "no_object":
        negative = context
    else:  # pragma: no cover - guarded by the Literal-producing dispatch.
        raise ValueError(f"Unknown class contrast type: {contrast_type!r}")
    return PromptContrastPair(
        positive=positive,
        negative=negative,
        label=label,
    )


def _build_projection_request(
    job: BatchJob,
    *,
    strength: float,
) -> OrthogonalizationRequest:
    prompt_spec = get_prompt_spec(job.prompt_suffix)
    object_only = f"a {job.species}"
    expected_prompt = _validate_job_prompt(job, prompt_spec)

    if prompt_spec.is_factorial:
        return OrthogonalizationRequest(
            mode="factorial",
            strength=strength,
            prompts=PromptSet(
                object_only=object_only,
                object_behavior=f"{object_only} {prompt_spec.behavior}",
                object_background=f"{object_only} {prompt_spec.background}",
                object_behavior_background=expected_prompt,
            ),
            include_behavior=True,
            include_background=True,
        )

    if prompt_spec.background is not None:
        return OrthogonalizationRequest(
            mode="local",
            strength=strength,
            prompts=PromptSet(
                object_only=object_only,
                object_background=expected_prompt,
            ),
            include_behavior=False,
            include_background=True,
        )

    assert prompt_spec.behavior is not None
    return OrthogonalizationRequest(
        mode="local",
        strength=strength,
        prompts=PromptSet(
            object_only=object_only,
            object_behavior=expected_prompt,
        ),
        include_behavior=True,
        include_background=False,
    )


def _build_source_aware_projection_request(
    job: BatchJob,
    *,
    source_caption: SourceCaptionRecord,
    strength: float,
) -> OrthogonalizationRequest:
    prompt_spec = get_prompt_spec(job.prompt_suffix)
    _validate_job_prompt(job, prompt_spec)
    _validate_source_caption_matches_job(job, source_caption)

    factors = _target_factors(job.prompt_suffix)
    return OrthogonalizationRequest(
        mode="factorial",
        strength=strength,
        prompts=PromptSet(
            object_only=source_caption.object_only,
            object_behavior=source_caption.object_behavior,
            object_background=source_caption.object_background,
            object_behavior_background=(
                source_caption.object_behavior_background
            ),
        ),
        include_behavior="behavior" in factors,
        include_background="background" in factors,
        include_interaction=False,
    )


def _build_source_target_projection_request(
    job: BatchJob,
    *,
    source_caption: SourceCaptionRecord,
    strength: float,
    include_all_source_factors: bool = False,
) -> OrthogonalizationRequest:
    """Build target and source rows for one shared downstream SVD."""

    target_request = _build_projection_request(job, strength=strength)
    source_request = _build_source_aware_projection_request(
        job,
        source_caption=source_caption,
        strength=strength,
    )
    if include_all_source_factors:
        source_request = replace(
            source_request,
            include_behavior=True,
            include_background=True,
        )
    assert source_request.prompts is not None
    source_direction_set = TextDirectionSet(
        mode=source_request.mode,
        prompts=source_request.prompts,
        include_behavior=source_request.include_behavior,
        include_background=source_request.include_background,
        include_interaction=source_request.include_interaction,
        label="source",
    )
    return replace(
        target_request,
        primary_label="target",
        additional_direction_sets=(source_direction_set,),
    )


def _target_factors(prompt_suffix: str) -> tuple[PromptFactor, ...]:
    return get_prompt_spec(prompt_suffix).factors


def _validate_job_prompt(job: BatchJob, prompt_spec: PromptSpec) -> str:
    expected_prompt = f"a {job.species} {prompt_spec.suffix}"
    if job.prompt != expected_prompt:
        raise ValueError(
            "BatchJob prompt does not match its species and prompt_suffix: "
            f"expected {expected_prompt!r}, got {job.prompt!r}"
        )
    return expected_prompt


def _validate_source_caption_matches_job(
    job: BatchJob,
    source_caption: SourceCaptionRecord,
) -> None:
    if source_caption.reference_stem != job.reference_stem:
        raise ValueError(
            "Source caption/reference mismatch: "
            f"caption={source_caption.reference_stem!r}, "
            f"job={job.reference_stem!r}"
        )
    if source_caption.species != job.species:
        raise ValueError(
            "Source caption species does not match the batch job: "
            f"caption={source_caption.species!r}, job={job.species!r}"
        )


def _merge_factors(
    target_factors: tuple[PromptFactor, ...],
    source_factors: tuple[PromptFactor, ...],
) -> tuple[PromptFactor, ...]:
    present = set(target_factors) | set(source_factors)
    factor_order: tuple[PromptFactor, ...] = ("behavior", "background")
    return tuple(factor for factor in factor_order if factor in present)


__all__ = [
    "BatchConditioning",
    "build_batch_conditioning",
]
