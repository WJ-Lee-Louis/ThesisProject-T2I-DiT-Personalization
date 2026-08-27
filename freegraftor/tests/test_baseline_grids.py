from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

import create_baseline_grids as grids
from freegraftor_experiment.batch_manifest import BatchManifest, ReferenceSpec
from freegraftor_experiment.prompt_catalog import PROMPT_CATALOG


def _one_reference_manifest(reference_path: Path) -> BatchManifest:
    reference = ReferenceSpec(
        stem="dog_01",
        filename=reference_path.name,
        species="dog",
        sha256="a" * 64,
        reference_path=str(reference_path),
    )
    return BatchManifest(
        references=(reference,),
        jobs=(),
        semantic_digest="b" * 64,
    )


def test_layout_matches_existing_grid_convention() -> None:
    assert grids.Layout().canvas_size == (5040, 892)
    assert len(PROMPT_CATALOG) == 16


def test_load_source_captions_builds_object_behavior_background(
    tmp_path: Path,
) -> None:
    reference_path = tmp_path / "dog_01.jpg"
    reference_path.write_bytes(b"reference")
    manifest = _one_reference_manifest(reference_path)
    caption_path = tmp_path / "captions.json"
    caption_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "references": {
                    "dog_01": {
                        "source_behavior": "standing",
                        "source_background": "on the beach",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    captions = grids.load_source_captions(caption_path, manifest)

    assert captions == {"dog_01": "a dog standing on the beach"}


def test_create_reference_grid_renders_all_prompt_columns(tmp_path: Path) -> None:
    reference_path = tmp_path / "dog_01.jpg"
    Image.new("RGB", (64, 48), "orange").save(reference_path)
    batch_root = tmp_path / "batch"
    generated_root = batch_root / "dog_01"
    generated_root.mkdir(parents=True)
    for index, prompt in enumerate(PROMPT_CATALOG):
        Image.new("RGB", (32, 32), (index * 11, 40, 90)).save(
            generated_root / f"dog_01_{prompt.slug}.png"
        )
    output_path = tmp_path / "grid.png"

    grids.create_reference_grid(
        reference_path=reference_path,
        reference_filename=reference_path.name,
        reference_stem="dog_01",
        full_caption="a dog standing on the beach",
        batch_root=batch_root,
        output_path=output_path,
        layout=grids.Layout(),
        fonts=grids.load_fonts(),
    )

    with Image.open(output_path) as image:
        assert image.size == (5040, 892)
        assert image.mode == "RGB"
