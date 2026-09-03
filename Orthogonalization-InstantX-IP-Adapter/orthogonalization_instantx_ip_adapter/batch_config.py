"""Validated, JSON-serializable settings for one remote batch run."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, ClassVar, Literal, Mapping


BatchCondition = Literal[
    "baseline",
    "ssg",
    "target_projection_ssg",
    "source_projection_ssg",
    "st_matched_projection_ssg",
    "st_all_projection_ssg",
    "class_target_absolute_direct_ssg",
    "class_source_absolute_direct_ssg",
    "class_st_matched_absolute_direct_ssg",
    "class_st_all_absolute_direct_ssg",
    "class_target_absolute_residual_ssg",
    "class_source_absolute_residual_ssg",
    "class_st_matched_absolute_residual_ssg",
    "class_st_all_absolute_residual_ssg",
    "class_target_no_object_ssg",
    "class_source_no_object_ssg",
    "class_st_matched_no_object_ssg",
    "class_st_all_no_object_ssg",
    "class_target_superclass_ssg",
    "class_source_superclass_ssg",
    "class_st_matched_superclass_ssg",
    "class_st_all_superclass_ssg",
]

BASE_TARGET_CONDITIONS = frozenset({"target_projection_ssg"})
BASE_SOURCE_CONDITIONS = frozenset({"source_projection_ssg"})
BASE_ST_MATCHED_CONDITIONS = frozenset({"st_matched_projection_ssg"})
BASE_ST_ALL_CONDITIONS = frozenset({"st_all_projection_ssg"})

CLASS_ABSOLUTE_DIRECT_CONDITIONS = frozenset(
    {
        "class_target_absolute_direct_ssg",
        "class_source_absolute_direct_ssg",
        "class_st_matched_absolute_direct_ssg",
        "class_st_all_absolute_direct_ssg",
    }
)
CLASS_ABSOLUTE_RESIDUAL_CONDITIONS = frozenset(
    {
        "class_target_absolute_residual_ssg",
        "class_source_absolute_residual_ssg",
        "class_st_matched_absolute_residual_ssg",
        "class_st_all_absolute_residual_ssg",
    }
)
CLASS_NO_OBJECT_CONDITIONS = frozenset(
    {
        "class_target_no_object_ssg",
        "class_source_no_object_ssg",
        "class_st_matched_no_object_ssg",
        "class_st_all_no_object_ssg",
    }
)
CLASS_SUPERCLASS_CONDITIONS = frozenset(
    {
        "class_target_superclass_ssg",
        "class_source_superclass_ssg",
        "class_st_matched_superclass_ssg",
        "class_st_all_superclass_ssg",
    }
)
CLASS_CONTEXTUAL_CONDITIONS = frozenset(
    CLASS_NO_OBJECT_CONDITIONS | CLASS_SUPERCLASS_CONDITIONS
)
CLASS_ABSOLUTE_CONDITIONS = frozenset(
    CLASS_ABSOLUTE_DIRECT_CONDITIONS | CLASS_ABSOLUTE_RESIDUAL_CONDITIONS
)
CLASS_CONDITIONS = frozenset(CLASS_ABSOLUTE_CONDITIONS | CLASS_CONTEXTUAL_CONDITIONS)
CLASS_TARGET_CONTEXTUAL_CONDITIONS = frozenset(
    {
        "class_target_no_object_ssg",
        "class_target_superclass_ssg",
    }
)
CLASS_SOURCE_CONTEXTUAL_CONDITIONS = frozenset(
    {
        "class_source_no_object_ssg",
        "class_source_superclass_ssg",
    }
)
CLASS_SOURCE_TARGET_CONTEXTUAL_CONDITIONS = frozenset(
    {
        "class_st_matched_no_object_ssg",
        "class_st_all_no_object_ssg",
        "class_st_matched_superclass_ssg",
        "class_st_all_superclass_ssg",
    }
)
CLASS_TARGET_CONDITIONS = frozenset(
    {
        condition for condition in CLASS_CONDITIONS if condition.startswith("class_target_")
    }
)
CLASS_SOURCE_CONDITIONS = frozenset(
    {
        condition for condition in CLASS_CONDITIONS if condition.startswith("class_source_")
    }
)
CLASS_ST_MATCHED_CONDITIONS = frozenset(
    {
        condition for condition in CLASS_CONDITIONS if condition.startswith("class_st_matched_")
    }
)
CLASS_ST_ALL_CONDITIONS = frozenset(
    {
        condition for condition in CLASS_CONDITIONS if condition.startswith("class_st_all_")
    }
)

SOURCE_CAPTION_CONDITIONS = frozenset(
    BASE_SOURCE_CONDITIONS
    | BASE_ST_MATCHED_CONDITIONS
    | BASE_ST_ALL_CONDITIONS
    | CLASS_SOURCE_CONDITIONS
    | CLASS_ST_MATCHED_CONDITIONS
    | CLASS_ST_ALL_CONDITIONS
)

FINAL_CONDITIONS: tuple[BatchCondition, ...] = (
    "baseline",
    "ssg",
    "target_projection_ssg",
    "source_projection_ssg",
    "st_matched_projection_ssg",
    "st_all_projection_ssg",
    "class_target_absolute_direct_ssg",
    "class_source_absolute_direct_ssg",
    "class_st_matched_absolute_direct_ssg",
    "class_st_all_absolute_direct_ssg",
    "class_target_absolute_residual_ssg",
    "class_source_absolute_residual_ssg",
    "class_st_matched_absolute_residual_ssg",
    "class_st_all_absolute_residual_ssg",
    "class_target_no_object_ssg",
    "class_source_no_object_ssg",
    "class_st_matched_no_object_ssg",
    "class_st_all_no_object_ssg",
    "class_target_superclass_ssg",
    "class_source_superclass_ssg",
    "class_st_matched_superclass_ssg",
    "class_st_all_superclass_ssg",
)
ALL_CONDITIONS = frozenset(FINAL_CONDITIONS)

_SAFE_BATCH_ID = re.compile(r"[A-Za-z0-9._-]+\Z")
_HEX_DIGEST = re.compile(r"[0-9A-Fa-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class BatchGenerationConfig:
    """Generation controls shared by every job in a batch.

    ``batch_id`` is intentionally restricted to one safe path component because
    it is also used as the Modal Volume output directory.
    """

    batch_id: str
    condition: BatchCondition = "baseline"
    orthogonalization_strength: float = 0.5
    ip_adapter_scale: float = 0.7
    guidance_scale: float = 3.5
    steps: int = 24
    seed: int = 42
    width: int = 1024
    height: int = 1024
    overwrite: bool = False
    expected_manifest_digest: str | None = None
    implementation_digest: str | None = None
    source_caption_digest: str | None = None
    # Appended after every legacy field to preserve positional-call behavior.
    class_projection_strength: float = 0.5
    class_superclass_prompt: str = "an animal"

    _SERIALIZED_FIELDS: ClassVar[tuple[str, ...]] = (
        "batch_id",
        "condition",
        "orthogonalization_strength",
        "ip_adapter_scale",
        "guidance_scale",
        "steps",
        "seed",
        "width",
        "height",
        "overwrite",
        "expected_manifest_digest",
        "implementation_digest",
        "source_caption_digest",
        "class_projection_strength",
        "class_superclass_prompt",
    )
    _CONDITIONS: ClassVar[frozenset[str]] = ALL_CONDITIONS

    def __post_init__(self) -> None:
        if not isinstance(self.batch_id, str):
            raise TypeError("batch_id must be a string")
        if (
            not self.batch_id
            or not _SAFE_BATCH_ID.fullmatch(self.batch_id)
            or self.batch_id in {".", ".."}
        ):
            raise ValueError(
                "batch_id must be one safe path component containing only "
                "ASCII letters, digits, '.', '_', or '-'"
            )

        if not isinstance(self.condition, str):
            raise TypeError("condition must be a string")
        if self.condition not in self._CONDITIONS:
            raise ValueError(
                "condition must be one of the 22 finalized SSG experiment conditions"
            )

        self._validate_real(
            "orthogonalization_strength",
            self.orthogonalization_strength,
            minimum=0.0,
            maximum=1.0,
        )
        if self.class_projection_enabled and self.class_projection_strength not in {
            0.5,
            1.0,
        }:
            raise ValueError(
                "class_projection_strength must be either 0.5 or 1.0 for "
                "generic-class conditions"
            )
        self._validate_real(
            "class_projection_strength",
            self.class_projection_strength,
            minimum=0.0,
            maximum=1.0,
        )
        if (
            not isinstance(self.class_superclass_prompt, str)
            or not self.class_superclass_prompt
            or self.class_superclass_prompt
            != " ".join(self.class_superclass_prompt.split())
        ):
            raise ValueError(
                "class_superclass_prompt must be a normalized non-empty string"
            )
        self._validate_real(
            "ip_adapter_scale",
            self.ip_adapter_scale,
            minimum=0.0,
        )
        self._validate_real(
            "guidance_scale",
            self.guidance_scale,
            minimum=0.0,
        )

        self._validate_integer("steps", self.steps, positive=True)
        self._validate_integer("seed", self.seed)
        self._validate_dimension("width", self.width)
        self._validate_dimension("height", self.height)

        if not isinstance(self.overwrite, bool):
            raise TypeError("overwrite must be a boolean")

        digest = self.expected_manifest_digest
        if digest is not None:
            if not isinstance(digest, str):
                raise TypeError("expected_manifest_digest must be a string or None")
            if not digest or not _HEX_DIGEST.fullmatch(digest):
                raise ValueError(
                    "expected_manifest_digest must be a 64-character SHA-256 hex digest"
                )

        implementation_digest = self.implementation_digest
        if implementation_digest is not None:
            if not isinstance(implementation_digest, str):
                raise TypeError("implementation_digest must be a string or None")
            if not _HEX_DIGEST.fullmatch(implementation_digest):
                raise ValueError(
                    "implementation_digest must be a 64-character SHA-256 hex digest"
                )

        source_caption_digest = self.source_caption_digest
        if source_caption_digest is not None:
            if not isinstance(source_caption_digest, str):
                raise TypeError("source_caption_digest must be a string or None")
            if not _HEX_DIGEST.fullmatch(source_caption_digest):
                raise ValueError(
                    "source_caption_digest must be a 64-character SHA-256 hex digest"
                )
        if self.requires_source_captions:
            if source_caption_digest is None:
                raise ValueError(
                    "source_caption_digest is required for source-aware, "
                    "source-target, or class-contextual conditions"
                )
        elif source_caption_digest is not None:
            raise ValueError(
                "source_caption_digest is only valid for caption-aware "
                "projection conditions"
            )

    @property
    def projection_enabled(self) -> bool:
        """Whether pooled-space projection is active for this condition."""

        return (
            self.target_projection_enabled
            or self.source_aware_projection_enabled
            or self.source_target_projection_enabled
            or self.class_projection_enabled
        )

    @property
    def target_projection_enabled(self) -> bool:
        """Whether a non-class target-only projection is active."""

        return self.condition in BASE_TARGET_CONDITIONS

    @property
    def source_aware_projection_enabled(self) -> bool:
        """Whether reference-caption directions control the projection."""

        return self.condition in BASE_SOURCE_CONDITIONS

    @property
    def source_target_projection_enabled(self) -> bool:
        """Whether target and reference directions share one SVD basis."""

        return self.condition in BASE_ST_MATCHED_CONDITIONS | BASE_ST_ALL_CONDITIONS

    @property
    def all_source_factors_enabled(self) -> bool:
        """Whether both source-caption factors enter the union basis."""

        return self.condition in BASE_ST_ALL_CONDITIONS | CLASS_ST_ALL_CONDITIONS

    @property
    def class_projection_enabled(self) -> bool:
        """Whether one of the generic-class projection experiments is active."""

        return self.condition in CLASS_CONDITIONS

    @property
    def class_absolute_projection_enabled(self) -> bool:
        """Whether a nuisance-combined absolute class anchor is active."""

        return self.condition in CLASS_ABSOLUTE_CONDITIONS

    @property
    def class_absolute_direct_projection_enabled(self) -> bool:
        """Whether the absolute anchor is used without nuisance residualization."""

        return self.condition in CLASS_ABSOLUTE_DIRECT_CONDITIONS

    @property
    def class_absolute_residual_projection_enabled(self) -> bool:
        """Whether the absolute anchor is residualized against nuisance."""

        return self.condition in CLASS_ABSOLUTE_RESIDUAL_CONDITIONS

    @property
    def class_contextual_projection_enabled(self) -> bool:
        """Whether a target/source contextual class direction is projected."""

        return self.condition in CLASS_CONTEXTUAL_CONDITIONS

    @property
    def class_target_contextual_projection_enabled(self) -> bool:
        """Whether target-prompt context defines the class direction."""

        return self.condition in CLASS_TARGET_CONTEXTUAL_CONDITIONS

    @property
    def class_source_contextual_projection_enabled(self) -> bool:
        """Whether reference-caption context defines the class direction."""

        return self.condition in CLASS_SOURCE_CONTEXTUAL_CONDITIONS

    @property
    def class_source_target_contextual_projection_enabled(self) -> bool:
        """Whether target and source jointly define nuisance/class subspaces."""

        return self.condition in CLASS_SOURCE_TARGET_CONTEXTUAL_CONDITIONS

    @property
    def class_target_projection_enabled(self) -> bool:
        """Whether target prompts define the nuisance branch."""

        return self.condition in CLASS_TARGET_CONDITIONS

    @property
    def class_source_projection_enabled(self) -> bool:
        """Whether the reference caption defines the nuisance branch."""

        return self.condition in CLASS_SOURCE_CONDITIONS

    @property
    def class_source_target_projection_enabled(self) -> bool:
        """Whether target and source directions share a nuisance SVD."""

        return self.condition in CLASS_ST_MATCHED_CONDITIONS | CLASS_ST_ALL_CONDITIONS

    @property
    def class_superclass_projection_enabled(self) -> bool:
        """Whether the counterfactual keeps context and replaces O by S."""

        return self.condition in CLASS_SUPERCLASS_CONDITIONS

    @property
    def class_no_object_projection_enabled(self) -> bool:
        """Whether the counterfactual removes O and keeps only context."""

        return self.condition in CLASS_NO_OBJECT_CONDITIONS

    @property
    def requires_source_captions(self) -> bool:
        """Whether the remote run requires per-reference source captions."""

        return self.condition in SOURCE_CAPTION_CONDITIONS

    @property
    def mask_single_text_rows(self) -> bool:
        """Whether direct single-stream IP residuals into text rows are masked."""

        return self.condition != "baseline"

    def to_dict(self) -> dict[str, object]:
        """Return the stable JSON-compatible representation."""

        return {
            field_name: getattr(self, field_name)
            for field_name in self._SERIALIZED_FIELDS
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BatchGenerationConfig":
        """Construct a validated config while rejecting unknown keys."""

        if not isinstance(value, Mapping):
            raise TypeError("Batch generation config must be a mapping")

        unknown = set(value) - set(cls._SERIALIZED_FIELDS)
        if unknown:
            names = ", ".join(sorted(str(name) for name in unknown))
            raise ValueError(f"Unknown batch generation config field(s): {names}")

        return cls(**dict(value))

    @staticmethod
    def _validate_real(
        name: str,
        value: object,
        *,
        minimum: float,
        maximum: float | None = None,
    ) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a finite real number")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{name} must be finite")
        if numeric < minimum or (maximum is not None and numeric > maximum):
            interval = (
                f"[{minimum:g}, {maximum:g}]"
                if maximum is not None
                else f">= {minimum:g}"
            )
            raise ValueError(f"{name} must be {interval}")

    @staticmethod
    def _validate_integer(
        name: str,
        value: object,
        *,
        positive: bool = False,
    ) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if positive and value <= 0:
            raise ValueError(f"{name} must be positive")

    @classmethod
    def _validate_dimension(cls, name: str, value: object) -> None:
        cls._validate_integer(name, value, positive=True)
        if value % 16:
            raise ValueError(f"{name} must be a multiple of 16")
