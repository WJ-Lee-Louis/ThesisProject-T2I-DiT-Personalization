"""Map one batch job and experiment condition to model conditioning controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .adapter import OrthogonalizationRequest
from .batch_config import BatchGenerationConfig
from .batch_manifest import BatchJob, PROMPT_SUFFIXES
from .orthogonalization import PromptSet
from .source_captions import SourceCaptionRecord


BACKGROUND_ONLY_SUFFIX = "on a street"
FACTORIAL_SUFFIX = "running on a street"


@dataclass(frozen=True, slots=True)
class BatchConditioning:
    """Conditioning arguments consumed by ``OrthogonalizedIPAdapterMixin``."""

    orthogonalization: OrthogonalizationRequest
    mask_single_text_rows: bool
    basis_origin: Literal["none", "target_prompt", "reference_caption"]
    removed_factors: tuple[str, ...]


def build_batch_conditioning(
    job: BatchJob,
    config: BatchGenerationConfig,
    source_caption: SourceCaptionRecord | None = None,
) -> BatchConditioning:
    """Build projection prompts and the single-stream row-gate flag.

    The two interventions are selected by six non-overlapping experiment
    conditions:

    * ``baseline``: neither intervention;
    * ``projection``: target-prompt pooled-space projection only;
    * ``gate``: single-stream text-row gate only;
    * ``combined``: target-prompt projection plus the gate;
    * ``sa_projection``: reference-caption projection only;
    * ``sa_combined``: reference-caption projection plus the gate.

    The original projection directions are target-local for a single
    behavior/background suffix and factorial for ``running on a street``.
    Source-aware directions are always estimated from the reference caption's
    complete 2-by-2 factorial prompt set, while the target suffix selects which
    source factor(s) are attenuated.
    """

    if not isinstance(job, BatchJob):
        raise TypeError("job must be a BatchJob")
    if not isinstance(config, BatchGenerationConfig):
        raise TypeError("config must be a BatchGenerationConfig")

    if source_caption is not None and not isinstance(
        source_caption, SourceCaptionRecord
    ):
        raise TypeError("source_caption must be a SourceCaptionRecord or None")

    if config.target_projection_enabled:
        if source_caption is not None:
            raise ValueError(
                "source_caption must not be supplied for target-local conditions"
            )
        request = _build_projection_request(
            job,
            strength=config.orthogonalization_strength,
        )
        basis_origin = "target_prompt"
        removed_factors = _target_factors(job.prompt_suffix)
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
        removed_factors = _target_factors(job.prompt_suffix)
    else:
        if source_caption is not None:
            raise ValueError(
                "source_caption is only valid for source-aware conditions"
            )
        request = OrthogonalizationRequest(
            mode="off",
            strength=config.orthogonalization_strength,
        )
        basis_origin = "none"
        removed_factors = ()

    return BatchConditioning(
        orthogonalization=request,
        mask_single_text_rows=config.mask_single_text_rows,
        basis_origin=basis_origin,
        removed_factors=removed_factors,
    )


def _build_projection_request(
    job: BatchJob,
    *,
    strength: float,
) -> OrthogonalizationRequest:
    if job.prompt_suffix not in PROMPT_SUFFIXES:
        raise ValueError(f"Unknown prompt suffix: {job.prompt_suffix!r}")

    object_only = f"a {job.species}"
    expected_prompt = f"{object_only} {job.prompt_suffix}"
    if job.prompt != expected_prompt:
        raise ValueError(
            "BatchJob prompt does not match its species and prompt_suffix: "
            f"expected {expected_prompt!r}, got {job.prompt!r}"
        )

    if job.prompt_suffix == BACKGROUND_ONLY_SUFFIX:
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

    if job.prompt_suffix == FACTORIAL_SUFFIX:
        return OrthogonalizationRequest(
            mode="factorial",
            strength=strength,
            prompts=PromptSet(
                object_only=object_only,
                object_behavior=f"{object_only} running",
                object_background=f"{object_only} on a street",
                object_behavior_background=expected_prompt,
            ),
            include_behavior=True,
            include_background=True,
        )

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
    expected_prompt = f"a {job.species} {job.prompt_suffix}"
    if job.prompt != expected_prompt:
        raise ValueError(
            "BatchJob prompt does not match its species and prompt_suffix: "
            f"expected {expected_prompt!r}, got {job.prompt!r}"
        )
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


def _target_factors(prompt_suffix: str) -> tuple[str, ...]:
    if prompt_suffix not in PROMPT_SUFFIXES:
        raise ValueError(f"Unknown prompt suffix: {prompt_suffix!r}")
    if prompt_suffix == BACKGROUND_ONLY_SUFFIX:
        return ("background",)
    if prompt_suffix == FACTORIAL_SUFFIX:
        return ("behavior", "background")
    return ("behavior",)


__all__ = [
    "BACKGROUND_ONLY_SUFFIX",
    "FACTORIAL_SUFFIX",
    "BatchConditioning",
    "build_batch_conditioning",
]
