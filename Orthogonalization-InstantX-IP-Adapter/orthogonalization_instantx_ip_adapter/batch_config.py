"""Validated, JSON-serializable settings for one remote batch run."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, ClassVar, Literal, Mapping


BatchCondition = Literal[
    "baseline",
    "projection",
    "gate",
    "combined",
    "sa_projection",
    "sa_combined",
]

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
    width: int = 960
    height: int = 1280
    overwrite: bool = False
    expected_manifest_digest: str | None = None
    implementation_digest: str | None = None
    source_caption_digest: str | None = None

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
    )
    _CONDITIONS: ClassVar[frozenset[str]] = frozenset(
        {
            "baseline",
            "projection",
            "gate",
            "combined",
            "sa_projection",
            "sa_combined",
        }
    )

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
                "condition must be one of: baseline, projection, gate, "
                "combined, sa_projection, sa_combined"
            )

        self._validate_real(
            "orthogonalization_strength",
            self.orthogonalization_strength,
            minimum=0.0,
            maximum=1.0,
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
        if self.source_aware_projection_enabled:
            if source_caption_digest is None:
                raise ValueError(
                    "source_caption_digest is required for source-aware conditions"
                )
        elif source_caption_digest is not None:
            raise ValueError(
                "source_caption_digest is only valid for source-aware conditions"
            )

    @property
    def projection_enabled(self) -> bool:
        """Whether pooled-space projection is active for this condition."""

        return self.target_projection_enabled or self.source_aware_projection_enabled

    @property
    def target_projection_enabled(self) -> bool:
        """Whether the original target-local projection is active."""

        return self.condition in {"projection", "combined"}

    @property
    def source_aware_projection_enabled(self) -> bool:
        """Whether reference-caption directions control the projection."""

        return self.condition in {"sa_projection", "sa_combined"}

    @property
    def mask_single_text_rows(self) -> bool:
        """Whether direct single-stream IP residuals into text rows are masked."""

        return self.condition in {"gate", "combined", "sa_combined"}

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
