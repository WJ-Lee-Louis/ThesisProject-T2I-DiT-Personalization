"""Projection and SSG controls for XVerse evaluation jobs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from .adapter import OrthogonalizationRequest, TextDirectionSet
from .batch_config import BatchGenerationConfig
from .evaluation_manifest import EvaluationJob, EvaluationSourceCaption
from .orthogonalization import PromptSet


@dataclass(frozen=True, slots=True)
class EvaluationConditioning:
    orthogonalization: OrthogonalizationRequest
    mask_single_text_rows: bool
    basis_origin: Literal[
        "none",
        "target_prompt",
        "reference_caption",
        "source_target_matched",
        "source_target_all",
    ]
    target_factors: tuple[str, ...]
    source_factors: tuple[str, ...]


def build_evaluation_conditioning(
    job: EvaluationJob,
    config: BatchGenerationConfig,
    source_caption: EvaluationSourceCaption | None = None,
) -> EvaluationConditioning:
    factors = job.target_factors
    matched_source_factors = _matched_source_factors(factors)
    requires_source = config.condition in {
        "source_projection_ssg",
        "st_matched_projection_ssg",
        "st_all_projection_ssg",
    }
    if requires_source and source_caption is None:
        raise ValueError("A source caption is required for this condition")
    if not requires_source and source_caption is not None:
        raise ValueError("A source caption is only valid for source-aware conditions")
    if source_caption is not None:
        _validate_source_match(job, source_caption)

    if config.condition in {"baseline", "ssg"}:
        request = _off_request(config.orthogonalization_strength)
        origin = "none"
        target_factors = ()
        source_factors = ()
    elif config.condition == "target_projection_ssg":
        request = _target_request(job, config.orthogonalization_strength)
        origin = "target_prompt"
        target_factors = factors
        source_factors = ()
    elif config.condition == "source_projection_ssg":
        assert source_caption is not None
        request = _source_request(
            source_caption,
            factors=matched_source_factors,
            strength=config.orthogonalization_strength,
        )
        origin = "reference_caption"
        target_factors = ()
        source_factors = matched_source_factors
    elif config.condition == "st_matched_projection_ssg":
        assert source_caption is not None
        request = _joint_request(
            job,
            source_caption,
            include_all_source_factors=False,
            strength=config.orthogonalization_strength,
        )
        origin = "source_target_matched"
        target_factors = factors
        source_factors = matched_source_factors
    elif config.condition == "st_all_projection_ssg":
        assert source_caption is not None
        request = _joint_request(
            job,
            source_caption,
            include_all_source_factors=True,
            strength=config.orthogonalization_strength,
        )
        origin = "source_target_all"
        target_factors = factors
        source_factors = ("behavior", "background")
    else:
        raise ValueError(
            f"Condition is not part of the six-condition evaluation: "
            f"{config.condition!r}"
        )

    return EvaluationConditioning(
        orthogonalization=request,
        mask_single_text_rows=config.mask_single_text_rows,
        basis_origin=origin,
        target_factors=target_factors,
        source_factors=source_factors,
    )


def _target_request(job: EvaluationJob, strength: float) -> OrthogonalizationRequest:
    has_action = job.action is not None
    has_background = job.background is not None
    has_attribute = job.attribute is not None
    if has_attribute:
        if has_action or has_background:
            raise ValueError(
                "Evaluation attribute projection currently requires an "
                "attribute-only target prompt"
            )
        # The existing local one-factor primitive is semantic-agnostic. Here
        # its positive-minus-object contrast represents attribute, not action:
        # e(O+attribute) - e(O).
        return OrthogonalizationRequest(
            mode="local",
            strength=strength,
            prompts=PromptSet(
                object_only=job.target_object_prompt,
                object_behavior=job.prompt.strip(),
            ),
            include_behavior=True,
            include_background=False,
            primary_label="target_attribute",
        )
    if not has_action and not has_background:
        return _off_request(strength)
    if has_action and has_background:
        return OrthogonalizationRequest(
            mode="factorial",
            strength=strength,
            prompts=PromptSet(
                object_only=job.target_object_prompt,
                object_behavior=f"{job.target_object_prompt} {job.action}",
                object_background=f"{job.target_object_prompt} {job.background}",
                object_behavior_background=job.prompt.strip(),
            ),
            include_behavior=True,
            include_background=True,
        )
    if has_action:
        return OrthogonalizationRequest(
            mode="local",
            strength=strength,
            prompts=PromptSet(
                object_only=job.target_object_prompt,
                object_behavior=job.prompt.strip(),
            ),
            include_behavior=True,
            include_background=False,
        )
    return OrthogonalizationRequest(
        mode="local",
        strength=strength,
        prompts=PromptSet(
            object_only=job.target_object_prompt,
            object_background=job.prompt.strip(),
        ),
        include_behavior=False,
        include_background=True,
    )


def _source_request(
    source: EvaluationSourceCaption,
    *,
    factors: tuple[str, ...],
    strength: float,
) -> OrthogonalizationRequest:
    unknown = set(factors) - {"behavior", "background"}
    if unknown:
        raise ValueError(
            f"Source captions do not define these target factors: {sorted(unknown)}"
        )
    if not factors:
        return _off_request(strength)
    return OrthogonalizationRequest(
        mode="factorial",
        strength=strength,
        prompts=_source_prompts(source),
        include_behavior="behavior" in factors,
        include_background="background" in factors,
        include_interaction=False,
    )


def _joint_request(
    job: EvaluationJob,
    source: EvaluationSourceCaption,
    *,
    include_all_source_factors: bool,
    strength: float,
) -> OrthogonalizationRequest:
    target_request = _target_request(job, strength)
    source_factors = (
        ("behavior", "background")
        if include_all_source_factors
        else _matched_source_factors(job.target_factors)
    )
    source_request = _source_request(
        source,
        factors=source_factors,
        strength=strength,
    )
    if target_request.mode == "off":
        if source_request.mode == "off":
            return target_request
        return replace(source_request, primary_label="source")
    if source_request.mode == "off":
        # This is the attribute-only matched case: there is no corresponding
        # source attribute, so the request is intentionally identical to the
        # target-only projection.
        return target_request
    assert source_request.prompts is not None
    return replace(
        target_request,
        primary_label="target",
        additional_direction_sets=(
            TextDirectionSet(
                mode=source_request.mode,
                prompts=source_request.prompts,
                include_behavior=source_request.include_behavior,
                include_background=source_request.include_background,
                include_interaction=source_request.include_interaction,
                label="source",
            ),
        ),
    )


def _source_prompts(source: EvaluationSourceCaption) -> PromptSet:
    return PromptSet(
        object_only=source.object_only,
        object_behavior=source.object_behavior,
        object_background=source.object_background,
        object_behavior_background=source.object_behavior_background,
    )


def _off_request(strength: float) -> OrthogonalizationRequest:
    return OrthogonalizationRequest(mode="off", strength=strength)


def _matched_source_factors(
    target_factors: tuple[str, ...],
) -> tuple[str, ...]:
    """Return factors that have a defined source-caption counterpart."""

    return tuple(
        factor
        for factor in target_factors
        if factor in {"behavior", "background"}
    )


def _validate_source_match(
    job: EvaluationJob,
    source: EvaluationSourceCaption,
) -> None:
    if source.reference_stem != job.reference_stem:
        raise ValueError("Source caption and evaluation job reference do not match")


__all__ = ["EvaluationConditioning", "build_evaluation_conditioning"]
