from collections import Counter

from orthogonalization_instantx_ip_adapter.batch_config import (
    BatchGenerationConfig,
)
from orthogonalization_instantx_ip_adapter.evaluation_conditioning import (
    build_evaluation_conditioning,
)
from orthogonalization_instantx_ip_adapter.evaluation_manifest import (
    DEFAULT_EVALUATION_DATASET_ROOT,
    DEFAULT_EVALUATION_SOURCE_CAPTIONS,
    EVALUATION_CONDITIONS,
    build_evaluation_manifest_from_upload,
    load_evaluation_manifest,
    load_evaluation_source_captions,
)


def test_evaluation_manifests_have_frozen_counts_and_seeds() -> None:
    subset = load_evaluation_manifest(DEFAULT_EVALUATION_DATASET_ROOT, "subset")
    custom = load_evaluation_manifest(DEFAULT_EVALUATION_DATASET_ROOT, "custom")

    assert len(subset.references) == 29
    assert len(subset.jobs) == 116
    assert {job.seed for job in subset.jobs} == {42, 43, 44, 45}
    assert len(custom.references) == 15
    assert len(custom.jobs) == 90
    assert len({job.seed for job in custom.jobs}) == 90


def test_source_captions_match_each_benchmark_reference_pool() -> None:
    for benchmark, expected in (("subset", 29), ("custom", 15)):
        manifest = load_evaluation_manifest(
            DEFAULT_EVALUATION_DATASET_ROOT, benchmark
        )
        captions = load_evaluation_source_captions(
            DEFAULT_EVALUATION_SOURCE_CAPTIONS, manifest
        )
        assert len(captions.records) == expected
        assert {
            record.reference_stem for record in captions.records
        } == {reference.stem for reference in manifest.references}


def test_subset_projection_modes_handle_attribute_only_records_explicitly() -> None:
    manifest = load_evaluation_manifest(DEFAULT_EVALUATION_DATASET_ROOT, "subset")
    captions = load_evaluation_source_captions(
        DEFAULT_EVALUATION_SOURCE_CAPTIONS, manifest
    )
    expected_modes = {
        "baseline": {"off": 116},
        "ssg": {"off": 116},
        "target_projection_ssg": {"factorial": 84, "local": 32},
        "source_projection_ssg": {"factorial": 96, "off": 20},
        "st_matched_projection_ssg": {
            "factorial": 84,
            "local": 32,
        },
        "st_all_projection_ssg": {"factorial": 84, "local": 32},
    }
    for condition in EVALUATION_CONDITIONS:
        source_enabled = condition in {
            "source_projection_ssg",
            "st_matched_projection_ssg",
            "st_all_projection_ssg",
        }
        config = BatchGenerationConfig(
            batch_id="unit-test",
            condition=condition,
            orthogonalization_strength=1.0,
            expected_manifest_digest=manifest.semantic_digest,
            implementation_digest="0" * 64,
            source_caption_digest=(
                captions.semantic_digest if source_enabled else None
            ),
        )
        modes = Counter()
        for job in manifest.jobs:
            source = (
                captions.for_reference(job.reference_stem)
                if source_enabled
                else None
            )
            result = build_evaluation_conditioning(
                job, config, source_caption=source
            )
            modes[result.orthogonalization.mode] += 1
            assert result.mask_single_text_rows == (condition != "baseline")
        assert dict(modes) == expected_modes[condition]


def test_attribute_is_independent_and_matched_equals_target() -> None:
    manifest = load_evaluation_manifest(DEFAULT_EVALUATION_DATASET_ROOT, "subset")
    captions = load_evaluation_source_captions(
        DEFAULT_EVALUATION_SOURCE_CAPTIONS, manifest
    )
    attribute_job = next(
        job for job in manifest.jobs if job.record_id == "official_animal_009"
    )
    assert attribute_job.target_factors == ("attribute",)
    assert attribute_job.target_object_prompt == "a kitten"

    def conditioning(condition: str):
        config = BatchGenerationConfig(
            batch_id="unit-test",
            condition=condition,
            orthogonalization_strength=1.0,
            expected_manifest_digest=manifest.semantic_digest,
            implementation_digest="0" * 64,
            source_caption_digest=(
                captions.semantic_digest
                if condition
                in {
                    "source_projection_ssg",
                    "st_matched_projection_ssg",
                    "st_all_projection_ssg",
                }
                else None
            ),
        )
        source = (
            captions.for_reference(attribute_job.reference_stem)
            if config.requires_source_captions
            else None
        )
        return build_evaluation_conditioning(attribute_job, config, source)

    target = conditioning("target_projection_ssg")
    matched = conditioning("st_matched_projection_ssg")
    source = conditioning("source_projection_ssg")
    all_source = conditioning("st_all_projection_ssg")

    assert target.orthogonalization == matched.orthogonalization
    assert target.target_factors == matched.target_factors == ("attribute",)
    assert matched.source_factors == ()
    assert target.orthogonalization.primary_label == "target_attribute"
    assert target.orthogonalization.prompts.object_only == "a kitten"
    assert (
        target.orthogonalization.prompts.object_behavior
        == "a cute kitten wearing a bowtie"
    )
    assert source.orthogonalization.mode == "off"
    assert source.source_factors == ()
    assert len(all_source.orthogonalization.additional_direction_sets) == 1
    assert all_source.target_factors == ("attribute",)
    assert all_source.source_factors == ("behavior", "background")


def test_attribute_survives_modal_manifest_round_trip() -> None:
    manifest = load_evaluation_manifest(DEFAULT_EVALUATION_DATASET_ROOT, "subset")
    uploaded = build_evaluation_manifest_from_upload(
        manifest.to_upload_dict(),
        DEFAULT_EVALUATION_DATASET_ROOT / "subset_bench",
    )
    assert uploaded.semantic_digest == manifest.semantic_digest
    attribute_jobs = [job for job in uploaded.jobs if job.attribute is not None]
    assert len(attribute_jobs) == 20
    assert {job.target_factors for job in attribute_jobs} == {("attribute",)}


def test_custom_target_prompts_are_balanced_by_projection_mode() -> None:
    manifest = load_evaluation_manifest(DEFAULT_EVALUATION_DATASET_ROOT, "custom")
    config = BatchGenerationConfig(
        batch_id="unit-test",
        condition="target_projection_ssg",
        orthogonalization_strength=1.0,
        expected_manifest_digest=manifest.semantic_digest,
        implementation_digest="0" * 64,
    )
    modes = Counter(
        build_evaluation_conditioning(job, config).orthogonalization.mode
        for job in manifest.jobs
    )
    assert modes == {"local": 60, "factorial": 30}
