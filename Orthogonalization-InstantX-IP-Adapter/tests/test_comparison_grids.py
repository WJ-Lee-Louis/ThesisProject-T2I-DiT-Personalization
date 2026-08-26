import json
from pathlib import Path

from PIL import Image
import pytest

import create_comparison_grids as grids
from orthogonalization_instantx_ip_adapter.batch_manifest import (
    BatchJob,
    BatchManifest,
    ReferenceRecord,
)
from orthogonalization_instantx_ip_adapter.prompt_catalog import (
    PROMPT_SUFFIXES,
    get_prompt_spec,
)
from orthogonalization_instantx_ip_adapter.source_captions import (
    SourceCaptionManifest,
    SourceCaptionRecord,
)


EXPECTED_METHODS = (
    (
        "baseline",
        "baseline",
        "dreambench-v2-baseline-0825",
    ),
    (
        "gate",
        "gate",
        "dreambench-v2-gate-0825",
    ),
    (
        "combined",
        "combined",
        "dreambench-v2-target-gate-l1-0825",
    ),
    (
        "sa_combined",
        "sa_combined",
        "dreambench-v2-source-gate-l1-0825",
    ),
    (
        "st_combined",
        "st_combined",
        "dreambench-v2-matched-union-gate-l1-0825",
    ),
    (
        "st_all_combined",
        "st_all_combined",
        "dreambench-v2-all-source-union-gate-l1-0825",
    ),
)


def _manifest_and_captions(
    tmp_path: Path,
) -> tuple[BatchManifest, SourceCaptionManifest]:
    reference_path = tmp_path / "dog_01.jpg"
    reference_path.write_bytes(b"reference")
    content_digest = "a" * 64
    reference = ReferenceRecord(
        reference_path=str(reference_path),
        reference_stem="dog_01",
        species="dog",
        content_sha256=content_digest,
    )
    jobs = tuple(
        BatchJob(
            reference_path=str(reference_path),
            reference_stem=reference.reference_stem,
            species="dog",
            prompt_suffix=suffix,
            prompt_slug="_".join(suffix.split()),
            prompt=f"a dog {suffix}",
            output_relative_path=(
                f"dog_01/dog_01_{'_'.join(suffix.split())}.png"
            ),
        )
        for suffix in PROMPT_SUFFIXES
    )
    manifest = BatchManifest(
        reference_root=str(tmp_path),
        references=(reference,),
        jobs=jobs,
    )
    captions = SourceCaptionManifest(
        records=(
            SourceCaptionRecord(
                reference_stem="dog_01",
                species="dog",
                content_sha256=content_digest,
                source_behavior="sitting",
                source_background="on grass",
            ),
        )
    )
    return manifest, captions


def _make_valid_batches(
    tmp_path: Path,
    manifest: BatchManifest,
    captions: SourceCaptionManifest,
) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for method in grids.METHODS:
        root = tmp_path / method.default_batch
        metadata = root / "_metadata"
        metadata.mkdir(parents=True)
        config = {
            "condition": method.condition,
            "orthogonalization_strength": 1.0,
            "expected_manifest_digest": manifest.semantic_digest,
            "ip_adapter_scale": 0.7,
            "guidance_scale": 3.5,
            "steps": 24,
            "seed": 42,
            "width": 16,
            "height": 16,
        }
        if method.condition.startswith(("sa_", "st_")):
            config["source_caption_digest"] = captions.semantic_digest
        (metadata / "config.json").write_text(
            json.dumps(config),
            encoding="utf-8",
        )

        generated = frozenset(grids._generated_prompt_suffixes(method))
        for job in manifest.jobs:
            if job.prompt_suffix not in generated:
                continue
            image_path = root / job.output_relative_path
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (16, 16), "white").save(image_path)
        roots[method.variant_id] = root
    return roots


def test_final_methods_and_default_batches_are_stable() -> None:
    assert tuple(
        (method.variant_id, method.condition, method.default_batch)
        for method in grids.METHODS
    ) == EXPECTED_METHODS
    assert all(
        method.orthogonalization_strength == 1.0
        for method in grids.METHODS
    )


def test_all_source_variant_reuses_exactly_factorial_cells() -> None:
    method = grids.METHODS[-1]
    generated = grids._generated_prompt_suffixes(method)
    reused = grids._reused_prompt_suffixes(method)

    assert len(generated) == 10
    assert len(reused) == 6
    assert all(not get_prompt_spec(suffix).is_factorial for suffix in generated)
    assert all(get_prompt_spec(suffix).is_factorial for suffix in reused)
    assert all(
        grids._cell_source_variant_id(method, suffix) == "st_combined"
        for suffix in reused
    )
    assert all(
        grids._cell_source_variant_id(method, suffix) == "st_all_combined"
        for suffix in generated
    )


def test_validate_inputs_accepts_optimized_all_source_batch(
    tmp_path: Path,
) -> None:
    manifest, captions = _manifest_and_captions(tmp_path)
    roots = _make_valid_batches(tmp_path / "outputs", manifest, captions)

    configs = grids.validate_inputs(
        image_manifest=manifest,
        source_captions=captions,
        batch_roots=roots,
    )

    assert set(configs) == {method.variant_id for method in grids.METHODS}
    st_all_pngs = tuple(roots["st_all_combined"].rglob("*.png"))
    assert len(st_all_pngs) == 10


def test_validate_inputs_rejects_missing_selected_all_source_image(
    tmp_path: Path,
) -> None:
    manifest, captions = _manifest_and_captions(tmp_path)
    roots = _make_valid_batches(tmp_path / "outputs", manifest, captions)
    missing_job = next(
        job
        for job in manifest.jobs
        if not get_prompt_spec(job.prompt_suffix).is_factorial
    )
    (roots["st_all_combined"] / missing_job.output_relative_path).unlink()

    with pytest.raises(ValueError, match="Incomplete batch"):
        grids.validate_inputs(
            image_manifest=manifest,
            source_captions=captions,
            batch_roots=roots,
        )


@pytest.mark.parametrize(
    "variant_id",
    ("sa_combined", "st_combined", "st_all_combined"),
)
def test_validate_inputs_requires_caption_digest_for_all_source_conditions(
    tmp_path: Path,
    variant_id: str,
) -> None:
    manifest, captions = _manifest_and_captions(tmp_path)
    roots = _make_valid_batches(tmp_path / "outputs", manifest, captions)
    config_path = roots[variant_id] / "_metadata" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["source_caption_digest"] = "0" * 64
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="Source caption digest mismatch"):
        grids.validate_inputs(
            image_manifest=manifest,
            source_captions=captions,
            batch_roots=roots,
        )
