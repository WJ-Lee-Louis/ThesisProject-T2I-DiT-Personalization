"""Deterministic manifest construction for the 90-image animal experiment."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable, Literal


Species = Literal["dog", "cat"]

PROMPT_SUFFIXES: tuple[str, ...] = (
    "running",
    "on a street",
    "running on a street",
    "sleeping",
    "looking up",
    "jumping",
    "riding a bicycle",
    "playing the guitar",
    "reading a book",
)

EXPECTED_REFERENCE_COUNTS: dict[Species, int] = {
    "dog": 7,
    "cat": 3,
}
EXPECTED_REFERENCE_COUNT = sum(EXPECTED_REFERENCE_COUNTS.values())
EXPECTED_JOB_COUNT = EXPECTED_REFERENCE_COUNT * len(PROMPT_SUFFIXES)

SUPPORTED_IMAGE_SUFFIXES = frozenset(
    {
        ".bmp",
        ".jpeg",
        ".jpg",
        ".png",
        ".webp",
    }
)

DEFAULT_REFERENCE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "dreambooth_reference_dataset"
    / "InstantX-IP-Adapter-DreamBench_filtered"
)

_SPECIES_PATTERNS: dict[Species, re.Pattern[str]] = {
    species: re.compile(rf"(?<![a-z]){species}\d*(?![a-z])", re.IGNORECASE)
    for species in EXPECTED_REFERENCE_COUNTS
}


@dataclass(frozen=True, slots=True)
class ReferenceRecord:
    """One validated local reference image."""

    reference_path: str
    reference_stem: str
    species: Species
    content_sha256: str

    def to_dict(self) -> dict[str, str]:
        return {
            "reference_path": self.reference_path,
            "reference_stem": self.reference_stem,
            "species": self.species,
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True, slots=True)
class BatchJob:
    """One reference/prompt/output combination."""

    reference_path: str
    reference_stem: str
    species: Species
    prompt_suffix: str
    prompt_slug: str
    prompt: str
    output_relative_path: str

    def to_dict(self) -> dict[str, str]:
        return {
            "reference_path": self.reference_path,
            "reference_stem": self.reference_stem,
            "species": self.species,
            "prompt_suffix": self.prompt_suffix,
            "prompt_slug": self.prompt_slug,
            "prompt": self.prompt,
            "output_relative_path": self.output_relative_path,
        }


@dataclass(frozen=True, slots=True)
class BatchManifest:
    """The complete, validated 10-reference by 9-prompt experiment."""

    reference_root: str
    references: tuple[ReferenceRecord, ...]
    jobs: tuple[BatchJob, ...]

    @property
    def semantic_digest(self) -> str:
        """Hash semantic inputs while ignoring machine-specific absolute paths."""

        canonical = {
            "references": [
                {
                    "reference_stem": record.reference_stem,
                    "species": record.species,
                    "content_sha256": record.content_sha256,
                }
                for record in self.references
            ],
            "jobs": [
                {
                    "reference_stem": job.reference_stem,
                    "species": job.species,
                    "prompt_suffix": job.prompt_suffix,
                    "prompt": job.prompt,
                    "output_relative_path": job.output_relative_path,
                }
                for job in self.jobs
            ],
        }
        encoded = json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "reference_root": self.reference_root,
            "semantic_digest": self.semantic_digest,
            "reference_count": len(self.references),
            "job_count": len(self.jobs),
            "prompt_suffixes": list(PROMPT_SUFFIXES),
            "references": [record.to_dict() for record in self.references],
            "jobs": [job.to_dict() for job in self.jobs],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


def prompt_slug(prompt_suffix: str) -> str:
    """Convert one fixed prompt suffix into its stable filename component."""

    if prompt_suffix not in PROMPT_SUFFIXES:
        raise ValueError(f"Unknown prompt suffix: {prompt_suffix!r}")
    return "_".join(prompt_suffix.split())


def infer_species(path: str | Path) -> Species:
    """Infer dog/cat from a filename stem and reject unknown or ambiguous names."""

    stem = Path(path).stem
    matches = [
        species
        for species, pattern in _SPECIES_PATTERNS.items()
        if pattern.search(stem)
    ]
    if len(matches) != 1:
        reason = "ambiguous" if matches else "unknown"
        raise ValueError(
            f"Reference filename has {reason} species: {Path(path).name!r}"
        )
    return matches[0]


def discover_reference_images(
    reference_root: str | Path = DEFAULT_REFERENCE_ROOT,
) -> tuple[Path, ...]:
    """Discover supported image files directly inside the configured root."""

    root = Path(reference_root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Reference root does not exist: {root}")

    paths = tuple(
        sorted(
            (
                path
                for path in root.iterdir()
                if path.is_file()
                and path.suffix.casefold() in SUPPORTED_IMAGE_SUFFIXES
            ),
            key=lambda path: (path.stem.casefold(), path.name.casefold()),
        )
    )
    return paths


def build_reference_records(
    reference_paths: Iterable[str | Path],
    *,
    reference_root: str | Path | None = None,
) -> tuple[ReferenceRecord, ...]:
    """Validate explicit paths and construct deterministically ordered records."""

    root = (
        Path(reference_root).expanduser()
        if reference_root is not None
        else None
    )
    records: list[ReferenceRecord] = []
    seen_stems: dict[str, Path] = {}

    for value in reference_paths:
        path = Path(value).expanduser()
        if root is not None and not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise FileNotFoundError(f"Reference image does not exist: {path}")
        if path.suffix.casefold() not in SUPPORTED_IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported reference image type: {path}")

        stem_key = path.stem.casefold()
        if stem_key in seen_stems:
            raise ValueError(
                "Duplicate reference stem "
                f"{path.stem!r}: {seen_stems[stem_key]} and {path}"
            )
        seen_stems[stem_key] = path

        records.append(
            ReferenceRecord(
                reference_path=str(path.resolve()),
                reference_stem=path.stem,
                species=infer_species(path),
                content_sha256=_sha256_file(path),
            )
        )

    records.sort(
        key=lambda record: (
            record.reference_stem.casefold(),
            record.reference_path.casefold(),
        )
    )
    _validate_reference_counts(records)
    return tuple(records)


def build_batch_manifest_from_paths(
    reference_paths: Iterable[str | Path],
    *,
    reference_root: str | Path | None = None,
) -> BatchManifest:
    """Build the fixed 90-job manifest from explicit local image paths."""

    records = build_reference_records(
        reference_paths,
        reference_root=reference_root,
    )
    jobs = _build_jobs(records)
    _validate_jobs(jobs)

    root_string = (
        str(Path(reference_root).expanduser().resolve())
        if reference_root is not None
        else ""
    )
    return BatchManifest(
        reference_root=root_string,
        references=records,
        jobs=jobs,
    )


def build_batch_manifest(
    reference_root: str | Path = DEFAULT_REFERENCE_ROOT,
) -> BatchManifest:
    """Discover a configured directory and build its fixed 90-job manifest."""

    root = Path(reference_root).expanduser()
    return build_batch_manifest_from_paths(
        discover_reference_images(root),
        reference_root=root,
    )


def _validate_reference_counts(records: Iterable[ReferenceRecord]) -> None:
    records = tuple(records)
    counts = Counter(record.species for record in records)
    actual = {species: counts[species] for species in EXPECTED_REFERENCE_COUNTS}
    if len(records) != EXPECTED_REFERENCE_COUNT or actual != EXPECTED_REFERENCE_COUNTS:
        raise ValueError(
            "Expected exactly 7 dog and 3 cat references "
            f"({EXPECTED_REFERENCE_COUNT} total), got {actual} "
            f"({len(records)} total)"
        )


def _build_jobs(
    records: tuple[ReferenceRecord, ...],
) -> tuple[BatchJob, ...]:
    jobs: list[BatchJob] = []
    for record in records:
        for suffix in PROMPT_SUFFIXES:
            slug = prompt_slug(suffix)
            relative_output = (
                Path(record.reference_stem)
                / f"{record.reference_stem}_{slug}.png"
            ).as_posix()
            jobs.append(
                BatchJob(
                    reference_path=record.reference_path,
                    reference_stem=record.reference_stem,
                    species=record.species,
                    prompt_suffix=suffix,
                    prompt_slug=slug,
                    prompt=f"a {record.species} {suffix}",
                    output_relative_path=relative_output,
                )
            )
    return tuple(jobs)


def _validate_jobs(jobs: tuple[BatchJob, ...]) -> None:
    if len(jobs) != EXPECTED_JOB_COUNT:
        raise ValueError(
            f"Expected exactly {EXPECTED_JOB_COUNT} jobs, got {len(jobs)}"
        )

    outputs = [job.output_relative_path.casefold() for job in jobs]
    if len(outputs) != len(set(outputs)):
        raise ValueError("Batch output paths must be unique")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
