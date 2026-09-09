"""Canonical local layout for XVerse evaluation artifacts.

Remote execution identifiers (Modal profile, batch ID, and metric run ID) are
provenance.  They must not determine where research artifacts are stored
locally.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath


DEFAULT_EVALUATION_ID = "animal_compositional_control_6cond_v1"


@dataclass(frozen=True)
class EvaluationLayout:
    dataset_root: Path
    evaluation_id: str = DEFAULT_EVALUATION_ID

    @property
    def root(self) -> Path:
        return self.dataset_root.expanduser().resolve() / "experiments" / self.evaluation_id

    @property
    def generation_root(self) -> Path:
        return self.root / "generation"

    def generation_benchmark_root(self, benchmark: str) -> Path:
        return self.generation_root / benchmark

    def generation_condition_root(self, benchmark: str, condition: str) -> Path:
        return self.generation_benchmark_root(benchmark) / condition

    def generated_manifest(self, benchmark: str) -> Path:
        return self.generation_benchmark_root(benchmark) / "generated_images_manifest.json"

    @property
    def evaluation_root(self) -> Path:
        return self.root / "evaluation"

    def score_root(self, benchmark: str) -> Path:
        return self.evaluation_root / "scores" / benchmark

    def summary_root(self, benchmark: str) -> Path:
        return self.evaluation_root / "summaries" / benchmark

    @property
    def run_metadata_root(self) -> Path:
        return self.evaluation_root / "run_metadata"

    @property
    def reference_artifact_root(self) -> Path:
        return self.evaluation_root / "ip" / "references"

    @property
    def generated_artifact_root(self) -> Path:
        # Keep this segment short enough for Windows' legacy MAX_PATH limit;
        # the remaining hierarchy contains long condition and record IDs.
        return self.evaluation_root / "ip" / "gen"

    def metric_download_path(self, remote_relative_path: PurePosixPath) -> Path:
        """Map a metric-volume path into the research-oriented local layout."""
        parts = remote_relative_path.parts
        if not parts:
            raise ValueError("Empty remote metric path")
        if any(part in {"", ".", ".."} for part in parts):
            raise ValueError(f"Unsafe remote metric path: {remote_relative_path}")
        head, *tail = parts
        if head == "_metadata":
            return self.run_metadata_root.joinpath(*tail)
        if head in {"subset", "custom"}:
            return self.score_root(head).joinpath(*tail)
        if head == "r":
            return self.reference_artifact_root.joinpath(*tail)
        if head == "a":
            # Remote IP artifacts are grouped as
            # ``a/<benchmark>/generated/<condition>/...``.  The local ``gen``
            # directory already communicates that these are generated-image
            # artifacts, so omit the redundant segment.  Besides making the
            # hierarchy easier to scan, this keeps the longest paths below
            # Windows' legacy MAX_PATH limit.
            if len(tail) >= 2 and tail[1] == "generated":
                return self.generated_artifact_root / tail[0] / Path(*tail[2:])
            return self.generated_artifact_root.joinpath(*tail)
        return self.evaluation_root / "unclassified" / head / Path(*tail)
