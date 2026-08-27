"""Validated, JSON-safe configuration for one FreeGraftor batch."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, ClassVar, Mapping


_SAFE_BATCH_ID = re.compile(r"[A-Za-z0-9._-]+\Z")
_HEX_DIGEST = re.compile(r"[0-9a-fA-F]{64}\Z")


@dataclass(frozen=True, slots=True)
class BatchConfig:
    batch_id: str
    seed: int = 42
    width: int = 512
    height: int = 512
    num_steps: int = 25
    guidance: float = 3.0
    start_inject_step: int = 0
    end_inject_step: int = 25
    start_inject_block: int = 0
    end_inject_block: int = 56
    sim_threshold: float = 0.2
    cyc_threshold: float = 1.5
    inject_match_dropout: float = 0.5
    alignment: str = "center"
    requires_offload: bool = True
    shard_count: int = 2
    overwrite: bool = False
    expected_manifest_digest: str | None = None
    implementation_digest: str | None = None

    _FIELDS: ClassVar[tuple[str, ...]] = (
        "batch_id",
        "seed",
        "width",
        "height",
        "num_steps",
        "guidance",
        "start_inject_step",
        "end_inject_step",
        "start_inject_block",
        "end_inject_block",
        "sim_threshold",
        "cyc_threshold",
        "inject_match_dropout",
        "alignment",
        "requires_offload",
        "shard_count",
        "overwrite",
        "expected_manifest_digest",
        "implementation_digest",
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.batch_id, str)
            or not self.batch_id
            or len(self.batch_id) > 64
            or not _SAFE_BATCH_ID.fullmatch(self.batch_id)
            or self.batch_id in {".", ".."}
        ):
            raise ValueError(
                "batch_id must be at most 64 characters and may contain only "
                "ASCII letters, digits, '.', '_', '-'."
            )
        for name in (
            "seed",
            "width",
            "height",
            "num_steps",
            "start_inject_step",
            "end_inject_step",
            "start_inject_block",
            "end_inject_block",
            "shard_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.width % 16 or self.height % 16:
            raise ValueError("width and height must be multiples of 16")
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if self.shard_count not in {1, 2}:
            raise ValueError("shard_count must be 1 or 2")
        if not 0 <= self.start_inject_step <= self.end_inject_step <= self.num_steps:
            raise ValueError("Injection steps must satisfy 0 <= start <= end <= steps")
        if not 0 <= self.start_inject_block <= self.end_inject_block <= 56:
            raise ValueError("Injection blocks must satisfy 0 <= start <= end <= 56")
        for name in (
            "guidance",
            "sim_threshold",
            "cyc_threshold",
            "inject_match_dropout",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a finite real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.guidance < 0 or self.sim_threshold < 0 or self.cyc_threshold < 0:
            raise ValueError("guidance and matching thresholds must be non-negative")
        if not 0 <= self.inject_match_dropout <= 1:
            raise ValueError("inject_match_dropout must be in [0, 1]")
        if self.alignment not in {
            "center", "top", "bottom", "left", "right",
            "top-left", "top-right", "bottom-left", "bottom-right",
        }:
            raise ValueError(f"Unsupported alignment: {self.alignment!r}")
        if not isinstance(self.requires_offload, bool) or not isinstance(self.overwrite, bool):
            raise TypeError("requires_offload and overwrite must be booleans")
        if not self.requires_offload:
            raise ValueError(
                "The reproducibility baseline is pinned to requires_offload=True"
            )
        for name in ("expected_manifest_digest", "implementation_digest"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or not _HEX_DIGEST.fullmatch(value)
            ):
                raise ValueError(f"{name} must be a 64-character SHA-256 digest")

    @property
    def inject_block_ids(self) -> list[int]:
        return list(range(self.start_inject_block, self.end_inject_block + 1))

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self._FIELDS}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BatchConfig":
        if not isinstance(value, Mapping):
            raise TypeError("Batch config must be a mapping")
        unknown = set(value) - set(cls._FIELDS)
        if unknown:
            raise ValueError(f"Unknown config fields: {sorted(unknown)}")
        return cls(**dict(value))
