from __future__ import annotations

from pathlib import Path

import pytest

import batch_client
from freegraftor_experiment.batch_config import BatchConfig
from freegraftor_experiment.batch_manifest import build_batch_manifest
from freegraftor_experiment.prompt_catalog import PROMPT_CATALOG


REFERENCE_NAMES = (
    "cat2_00.jpg",
    "cat_03.jpg",
    "dog2_04.jpg",
    "dog3_04.jpg",
    "dog5_02.jpg",
    "dog6_02.jpg",
    "dog7_03.jpg",
    "dog8_04.jpg",
    "dog_02.jpg",
    "PA_white_cat_background.png",
)


def _fake_reference_root(tmp_path: Path) -> Path:
    root = tmp_path / "references"
    root.mkdir(parents=True)
    for index, name in enumerate(REFERENCE_NAMES):
        (root / name).write_bytes(f"fake-image-{index}".encode())
    return root


def test_prompt_catalog_is_the_requested_16() -> None:
    assert len(PROMPT_CATALOG) == 16
    assert [item.suffix for item in PROMPT_CATALOG[:4]] == [
        "flying",
        "sleeping",
        "in the snow",
        "on a stage",
    ]
    assert PROMPT_CATALOG[-1].suffix == "reading a book on a sofa"


def test_manifest_has_7_dogs_3_cats_and_160_jobs(tmp_path: Path) -> None:
    manifest = build_batch_manifest(_fake_reference_root(tmp_path))
    assert len(manifest.references) == 10
    assert sum(item.species == "dog" for item in manifest.references) == 7
    assert sum(item.species == "cat" for item in manifest.references) == 3
    assert len(manifest.jobs) == 160
    cat_job = next(
        item
        for item in manifest.jobs
        if item.reference_stem == "cat_03" and item.prompt_suffix == "flying"
    )
    assert cat_job.full_prompt == "a cat flying"
    assert cat_job.output_relative_path == "cat_03/cat_03_flying.png"


def test_manifest_digest_is_path_independent(tmp_path: Path) -> None:
    first = build_batch_manifest(_fake_reference_root(tmp_path / "a"))
    second = build_batch_manifest(_fake_reference_root(tmp_path / "b"))
    assert first.semantic_digest == second.semantic_digest


def test_paper_defaults_are_explicit() -> None:
    config = BatchConfig(batch_id="baseline")
    assert (config.width, config.height) == (512, 512)
    assert config.num_steps == 25
    assert config.guidance == 3.0
    assert config.inject_match_dropout == 0.5
    assert config.inject_block_ids == list(range(57))
    assert config.requires_offload is True


def test_non_offload_variant_is_rejected() -> None:
    with pytest.raises(ValueError, match="offload"):
        BatchConfig(batch_id="baseline", requires_offload=False)


def test_partition_references_covers_each_stem_once() -> None:
    stems = [Path(name).stem for name in REFERENCE_NAMES]
    groups = batch_client.partition_references(stems, 2)
    assert [len(group) for group in groups] == [5, 5]
    assert sorted(groups[0] + groups[1]) == sorted(stems)


def test_upstream_reproducibility_patches_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    pipeline_source = (root / "src" / "pipeline.py").read_text(encoding="utf-8")
    generate_source = (root / "src" / "generate.py").read_text(encoding="utf-8")
    assert "unpack(gen_x.float(), config.height, config.width)" in pipeline_source
    assert "save_output: bool = True" in pipeline_source
    assert "/workspace/yzb" not in generate_source
