"""Deterministic discovery and naming for 10 references x 16 prompts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath

from .constants import (
    EXPECTED_CAT_COUNT,
    EXPECTED_DOG_COUNT,
    EXPECTED_REFERENCE_COUNT,
)
from .prompt_catalog import PROMPT_CATALOG


_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})


@dataclass(frozen=True, slots=True)
class ReferenceSpec:
    stem: str
    filename: str
    species: str
    sha256: str
    reference_path: str


@dataclass(frozen=True, slots=True)
class JobSpec:
    reference_stem: str
    reference_filename: str
    species: str
    reference_sha256: str
    prompt_suffix: str
    prompt_slug: str
    prompt_category: str
    prompt_composition: str
    full_prompt: str
    output_relative_path: str


@dataclass(frozen=True, slots=True)
class BatchManifest:
    references: tuple[ReferenceSpec, ...]
    jobs: tuple[JobSpec, ...]
    semantic_digest: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "semantic_digest": self.semantic_digest,
            "references": [asdict(item) for item in self.references],
            "jobs": [asdict(item) for item in self.jobs],
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _species_for_stem(stem: str) -> str:
    lowered = stem.casefold()
    if lowered.startswith("dog"):
        return "dog"
    if lowered.startswith("cat") or "white_cat" in lowered:
        return "cat"
    raise ValueError(f"Cannot infer dog/cat species from filename stem {stem!r}")


def build_batch_manifest(reference_root: str | Path) -> BatchManifest:
    root = Path(reference_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Reference directory not found: {root}")
    paths = sorted(
        (
            path
            for path in root.iterdir()
            if path.is_file() and path.suffix.casefold() in _IMAGE_EXTENSIONS
        ),
        key=lambda item: item.name.casefold(),
    )
    if len(paths) != EXPECTED_REFERENCE_COUNT:
        raise ValueError(
            f"Expected exactly {EXPECTED_REFERENCE_COUNT} reference images, "
            f"found {len(paths)} in {root}"
        )
    if len({path.stem.casefold() for path in paths}) != len(paths):
        raise ValueError("Reference filename stems must be unique")

    references = tuple(
        ReferenceSpec(
            stem=path.stem,
            filename=path.name,
            species=_species_for_stem(path.stem),
            sha256=_sha256_file(path),
            reference_path=str(path.resolve()),
        )
        for path in paths
    )
    dog_count = sum(item.species == "dog" for item in references)
    cat_count = sum(item.species == "cat" for item in references)
    if dog_count != EXPECTED_DOG_COUNT or cat_count != EXPECTED_CAT_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_DOG_COUNT} dogs and {EXPECTED_CAT_COUNT} cats, "
            f"found {dog_count} dogs and {cat_count} cats"
        )

    jobs = tuple(
        JobSpec(
            reference_stem=reference.stem,
            reference_filename=reference.filename,
            species=reference.species,
            reference_sha256=reference.sha256,
            prompt_suffix=prompt.suffix,
            prompt_slug=prompt.slug,
            prompt_category=prompt.category,
            prompt_composition=prompt.composition,
            full_prompt=prompt.full_prompt(reference.species),
            output_relative_path=str(
                PurePosixPath(reference.stem)
                / f"{reference.stem}_{prompt.slug}.png"
            ),
        )
        for reference in references
        for prompt in PROMPT_CATALOG
    )

    semantic_value = {
        "schema_version": 1,
        "references": [
            {
                "stem": item.stem,
                "filename": item.filename,
                "species": item.species,
                "sha256": item.sha256,
            }
            for item in references
        ],
        "prompts": [asdict(item) for item in PROMPT_CATALOG],
    }
    encoded = json.dumps(
        semantic_value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return BatchManifest(references=references, jobs=jobs, semantic_digest=digest)
