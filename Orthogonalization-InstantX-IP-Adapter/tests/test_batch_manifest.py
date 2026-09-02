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
from orthogonalization_instantx_ip_adapter.prompt_catalog import (
    PROMPT_CATALOG,
    PROMPT_SPECS_BY_SUFFIX,
    PromptSpec,
    get_prompt_spec,
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


def test_prompt_catalog_has_expected_order_and_factor_decompositions() -> None:
    assert PROMPT_SUFFIXES == (
        "flying",
        "sleeping",
        "playing the piano",
        "holding an umbrella",
        "eating watermelon",
        "riding a skateboard",
        "flying in the sky",
        "sleeping in the snow",
        "playing the guitar on a stage",
        "swimming in the pool",
        "jumping over a wooden fence",
        "reading a book on a sofa",
        "on a stage",
        "in the snow",
    )
    assert len(PROMPT_CATALOG) == len(PROMPT_SPECS_BY_SUFFIX) == 14

    behavior = get_prompt_spec("playing the piano")
    assert behavior.category == "object_related"
    assert behavior.behavior == "playing the piano"
    assert behavior.background is None
    assert behavior.factors == ("behavior",)
    assert behavior.is_factorial is False

    factorial = get_prompt_spec("playing the guitar on a stage")
    assert factorial.category == "complex"
    assert factorial.behavior == "playing the guitar"
    assert factorial.background == "on a stage"
    assert factorial.factors == ("behavior", "background")
    assert factorial.is_factorial is True

    background = get_prompt_spec("in the snow")
    assert background.category == "background_only"
    assert background.behavior is None
    assert background.background == "in the snow"
    assert background.factors == ("background",)
    assert background.is_factorial is False


def test_relational_fence_prompt_is_decomposed_as_a_factorial_prompt() -> None:
    spec = get_prompt_spec("jumping over a wooden fence")

    assert spec.category == "complex"
    assert spec.behavior == "jumping"
    assert spec.background == "over a wooden fence"
    assert spec.factors == ("behavior", "background")
    assert spec.is_factorial is True


def test_prompt_catalog_is_immutable_and_rejects_invalid_specs() -> None:
    spec = get_prompt_spec("flying")

    with pytest.raises((AttributeError, TypeError)):
        spec.behavior = "sleeping"  # type: ignore[misc]
    with pytest.raises(TypeError):
        PROMPT_SPECS_BY_SUFFIX["new"] = spec  # type: ignore[index]
    with pytest.raises(ValueError, match="Unknown prompt suffix"):
        get_prompt_spec("unknown")
    with pytest.raises(ValueError, match="composition"):
        PromptSpec(
            suffix="flying elsewhere",
            category="single_behavior",
            behavior="flying",
        )
    with pytest.raises(ValueError, match="category"):
        PromptSpec(
            suffix="flying",
            category="unknown",  # type: ignore[arg-type]
            behavior="flying",
        )


def test_build_manifest_discovers_fixed_140_job_design(tmp_path: Path) -> None:
    make_references(tmp_path)
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")

    manifest = build_batch_manifest(tmp_path)

    assert len(manifest.references) == 10
    assert len(manifest.jobs) == EXPECTED_JOB_COUNT == 140
    assert sum(record.species == "dog" for record in manifest.references) == 7
    assert sum(record.species == "cat" for record in manifest.references) == 3

    cat_flying = next(
        job
        for job in manifest.jobs
        if job.reference_stem == "cat_03" and job.prompt_suffix == "flying"
    )
    assert cat_flying.prompt == "a cat flying"
    assert cat_flying.prompt_slug == "flying"
    assert cat_flying.output_relative_path == "cat_03/cat_03_flying.png"

    dog_complex = next(
        job
        for job in manifest.jobs
        if job.reference_stem == "dog_02"
        and job.prompt_suffix == "playing the guitar on a stage"
    )
    assert dog_complex.prompt == "a dog playing the guitar on a stage"
    assert (
        dog_complex.output_relative_path
        == "dog_02/dog_02_playing_the_guitar_on_a_stage.png"
    )

    assert tuple(
        job.prompt_suffix
        for job in manifest.jobs
        if job.reference_stem == "cat_03"
    ) == PROMPT_SUFFIXES
    assert len(
        {job.output_relative_path.casefold() for job in manifest.jobs}
    ) == 140


def test_manifest_dataclasses_are_json_serializable(tmp_path: Path) -> None:
    paths = make_references(tmp_path)

    manifest = build_batch_manifest_from_paths(paths)
    payload = manifest.to_dict()

    assert payload["reference_count"] == 10
    assert payload["job_count"] == 140
    assert payload["prompt_suffixes"] == list(PROMPT_SUFFIXES)
    assert payload["prompt_specs"] == [spec.to_dict() for spec in PROMPT_CATALOG]
    assert len(payload["semantic_digest"]) == 64
    assert all(len(record["content_sha256"]) == 64 for record in payload["references"])
    assert len(payload["references"]) == 10
    assert len(payload["jobs"]) == 140
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


def test_discovered_relative_root_is_not_prefixed_twice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference_root = tmp_path / "references"
    reference_root.mkdir()
    make_references(reference_root)
    monkeypatch.chdir(tmp_path)

    manifest = build_batch_manifest("references")

    assert manifest.reference_root == str(reference_root.resolve())
    assert len(manifest.references) == 10
    assert all(Path(record.reference_path).is_file() for record in manifest.references)


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
