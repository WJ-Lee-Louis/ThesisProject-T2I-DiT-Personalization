import json
from pathlib import Path

import pytest

from orthogonalization_instantx_ip_adapter.batch_manifest import (
    EXPECTED_JOB_COUNT,
    PROMPT_SUFFIXES,
    build_batch_manifest,
    build_batch_manifest_from_paths,
    build_reference_records,
    infer_species,
)


REFERENCE_NAMES = (
    "cat2_00.jpg",
    "cat_03.jpg",
    "PA_white_cat_background.png",
    "dog2_04.jpg",
    "dog3_04.jpg",
    "dog5_02.jpg",
    "dog6_02.jpg",
    "dog7_03.jpg",
    "dog8_04.jpg",
    "dog_02.jpg",
)


def make_references(root: Path) -> tuple[Path, ...]:
    paths = []
    for name in REFERENCE_NAMES:
        path = root / name
        path.write_bytes(b"test image placeholder")
        paths.append(path)
    return tuple(paths)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("dog_02.jpg", "dog"),
        ("dog2_04.jpg", "dog"),
        ("cat2_00.jpg", "cat"),
        ("PA_white_cat_background.png", "cat"),
    ],
)
def test_infer_species_supports_dataset_filename_patterns(
    name: str,
    expected: str,
) -> None:
    assert infer_species(name) == expected


@pytest.mark.parametrize(
    "name",
    [
        "reference_00.jpg",
        "hotdog_00.jpg",
        "dog_and_cat.jpg",
    ],
)
def test_infer_species_rejects_unknown_or_ambiguous_names(name: str) -> None:
    with pytest.raises(ValueError, match="species"):
        infer_species(name)


def test_build_manifest_discovers_fixed_90_job_design(tmp_path: Path) -> None:
    make_references(tmp_path)
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")

    manifest = build_batch_manifest(tmp_path)

    assert len(manifest.references) == 10
    assert len(manifest.jobs) == EXPECTED_JOB_COUNT == 90
    assert sum(record.species == "dog" for record in manifest.references) == 7
    assert sum(record.species == "cat" for record in manifest.references) == 3

    cat_running = next(
        job
        for job in manifest.jobs
        if job.reference_stem == "cat_03" and job.prompt_suffix == "running"
    )
    assert cat_running.prompt == "a cat running"
    assert cat_running.prompt_slug == "running"
    assert cat_running.output_relative_path == "cat_03/cat_03_running.png"

    dog_street = next(
        job
        for job in manifest.jobs
        if job.reference_stem == "dog_02"
        and job.prompt_suffix == "running on a street"
    )
    assert dog_street.prompt == "a dog running on a street"
    assert (
        dog_street.output_relative_path
        == "dog_02/dog_02_running_on_a_street.png"
    )

    assert tuple(
        job.prompt_suffix
        for job in manifest.jobs
        if job.reference_stem == "cat_03"
    ) == PROMPT_SUFFIXES
    assert len(
        {job.output_relative_path.casefold() for job in manifest.jobs}
    ) == 90


def test_manifest_dataclasses_are_json_serializable(tmp_path: Path) -> None:
    paths = make_references(tmp_path)

    manifest = build_batch_manifest_from_paths(paths)
    payload = manifest.to_dict()

    assert payload["reference_count"] == 10
    assert payload["job_count"] == 90
    assert len(payload["semantic_digest"]) == 64
    assert all(len(record["content_sha256"]) == 64 for record in payload["references"])
    assert len(payload["references"]) == 10
    assert len(payload["jobs"]) == 90
    assert json.loads(manifest.to_json()) == payload
    json.dumps(manifest.references[0].to_dict())
    json.dumps(manifest.jobs[0].to_dict())


def test_semantic_digest_ignores_machine_specific_root(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    make_references(first_root)
    make_references(second_root)

    first = build_batch_manifest(first_root)
    second = build_batch_manifest(second_root)

    assert first.semantic_digest == second.semantic_digest


def test_explicit_relative_paths_resolve_against_reference_root(
    tmp_path: Path,
) -> None:
    make_references(tmp_path)

    manifest = build_batch_manifest_from_paths(
        REFERENCE_NAMES,
        reference_root=tmp_path,
    )

    assert manifest.reference_root == str(tmp_path.resolve())
    assert all(
        Path(record.reference_path).is_absolute()
        for record in manifest.references
    )


def test_missing_reference_file_is_rejected(tmp_path: Path) -> None:
    paths = list(make_references(tmp_path))
    paths[-1].unlink()

    with pytest.raises(FileNotFoundError, match="does not exist"):
        build_batch_manifest_from_paths(paths)


def test_unknown_species_is_rejected(tmp_path: Path) -> None:
    paths = list(make_references(tmp_path))
    paths[-1] = tmp_path / "reference_00.jpg"
    paths[-1].write_bytes(b"test image placeholder")

    with pytest.raises(ValueError, match="unknown species"):
        build_batch_manifest_from_paths(paths)


def test_duplicate_reference_stems_are_rejected(tmp_path: Path) -> None:
    paths = list(make_references(tmp_path))
    duplicate = tmp_path / "dog_02.png"
    duplicate.write_bytes(b"test image placeholder")
    paths.append(duplicate)

    with pytest.raises(ValueError, match="Duplicate reference stem"):
        build_reference_records(paths)


def test_wrong_species_counts_are_rejected(tmp_path: Path) -> None:
    paths = list(make_references(tmp_path))
    paths.pop()

    with pytest.raises(ValueError, match="exactly 7 dog and 3 cat"):
        build_batch_manifest_from_paths(paths)
