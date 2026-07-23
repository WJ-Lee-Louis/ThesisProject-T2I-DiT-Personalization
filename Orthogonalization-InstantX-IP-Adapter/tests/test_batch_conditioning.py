from __future__ import annotations

import pytest

from orthogonalization_instantx_ip_adapter.batch_conditioning import (
    build_batch_conditioning,
)
from orthogonalization_instantx_ip_adapter.batch_config import (
    BatchGenerationConfig,
)
from orthogonalization_instantx_ip_adapter.batch_manifest import (
    BatchJob,
    PROMPT_SUFFIXES,
    prompt_slug,
)
from orthogonalization_instantx_ip_adapter.source_captions import (
    SourceCaptionRecord,
)


def _job(prompt_suffix: str, *, species: str = "dog") -> BatchJob:
    slug = prompt_slug(prompt_suffix)
    return BatchJob(
        reference_path=f"C:/{species}_01.png",
        reference_stem=f"{species}_01",
        species=species,
        prompt_suffix=prompt_suffix,
        prompt_slug=slug,
        prompt=f"a {species} {prompt_suffix}",
        output_relative_path=f"{species}_01/{species}_01_{slug}.png",
    )


def _source_caption(
    *,
    species: str = "dog",
    reference_stem: str | None = None,
) -> SourceCaptionRecord:
    return SourceCaptionRecord(
        reference_stem=reference_stem or f"{species}_01",
        species=species,
        content_sha256="ab" * 32,
        source_behavior="lying down",
        source_background="on a beach",
    )


def _source_config(
    condition: str = "sa_projection",
) -> BatchGenerationConfig:
    return BatchGenerationConfig(
        batch_id=condition,
        condition=condition,
        orthogonalization_strength=0.73,
        source_caption_digest="cd" * 32,
    )


@pytest.mark.parametrize(
    ("condition", "projection_enabled", "mask_enabled"),
    (
        ("baseline", False, False),
        ("projection", True, False),
        ("gate", False, True),
        ("combined", True, True),
    ),
)
@pytest.mark.parametrize("prompt_suffix", PROMPT_SUFFIXES)
def test_every_condition_and_suffix_maps_to_independent_controls(
    condition: str,
    projection_enabled: bool,
    mask_enabled: bool,
    prompt_suffix: str,
) -> None:
    config = BatchGenerationConfig(
        batch_id="test",
        condition=condition,
        orthogonalization_strength=0.73,
    )

    conditioning = build_batch_conditioning(_job(prompt_suffix), config)

    assert conditioning.mask_single_text_rows is mask_enabled
    assert conditioning.orthogonalization.enabled is projection_enabled
    assert conditioning.orthogonalization.strength == pytest.approx(0.73)


@pytest.mark.parametrize(
    ("condition", "basis_origin", "removed_factors"),
    (
        ("baseline", "none", ()),
        ("projection", "target_prompt", ("behavior",)),
        ("gate", "none", ()),
        ("combined", "target_prompt", ("behavior",)),
    ),
)
def test_original_four_conditions_keep_their_basis_origin_and_factor_contract(
    condition: str,
    basis_origin: str,
    removed_factors: tuple[str, ...],
) -> None:
    conditioning = build_batch_conditioning(
        _job("sleeping"),
        BatchGenerationConfig(batch_id=condition, condition=condition),
    )

    assert conditioning.basis_origin == basis_origin
    assert conditioning.removed_factors == removed_factors


@pytest.mark.parametrize(
    "prompt_suffix",
    tuple(
        suffix
        for suffix in PROMPT_SUFFIXES
        if suffix not in {"on a street", "running on a street"}
    ),
)
def test_behavior_suffixes_use_one_local_behavior_contrast(
    prompt_suffix: str,
) -> None:
    conditioning = build_batch_conditioning(
        _job(prompt_suffix, species="cat"),
        BatchGenerationConfig(batch_id="projection", condition="projection"),
    )
    request = conditioning.orthogonalization

    assert request.mode == "local"
    assert request.include_behavior is True
    assert request.include_background is False
    assert request.include_interaction is False
    assert request.prompts is not None
    assert request.prompts.object_only == "a cat"
    assert request.prompts.object_behavior == f"a cat {prompt_suffix}"
    assert request.prompts.object_background is None
    assert request.prompts.object_behavior_background is None


def test_street_suffix_uses_one_local_background_contrast() -> None:
    conditioning = build_batch_conditioning(
        _job("on a street"),
        BatchGenerationConfig(batch_id="projection", condition="projection"),
    )
    request = conditioning.orthogonalization

    assert request.mode == "local"
    assert request.include_behavior is False
    assert request.include_background is True
    assert request.include_interaction is False
    assert request.prompts is not None
    assert request.prompts.object_only == "a dog"
    assert request.prompts.object_behavior is None
    assert request.prompts.object_background == "a dog on a street"
    assert request.prompts.object_behavior_background is None


def test_running_on_street_suffix_uses_factorial_contrasts() -> None:
    conditioning = build_batch_conditioning(
        _job("running on a street"),
        BatchGenerationConfig(batch_id="combined", condition="combined"),
    )
    request = conditioning.orthogonalization

    assert request.mode == "factorial"
    assert request.include_behavior is True
    assert request.include_background is True
    assert request.include_interaction is False
    assert request.prompts is not None
    assert request.prompts.object_only == "a dog"
    assert request.prompts.object_behavior == "a dog running"
    assert request.prompts.object_background == "a dog on a street"
    assert (
        request.prompts.object_behavior_background
        == "a dog running on a street"
    )


def test_non_projection_conditions_do_not_construct_prompt_set() -> None:
    for condition in ("baseline", "gate"):
        conditioning = build_batch_conditioning(
            _job("running on a street"),
            BatchGenerationConfig(batch_id=condition, condition=condition),
        )

        assert conditioning.orthogonalization.mode == "off"
        assert conditioning.orthogonalization.prompts is None


def test_projection_rejects_a_job_with_mismatched_prompt() -> None:
    job = _job("running")
    malformed = BatchJob(
        reference_path=job.reference_path,
        reference_stem=job.reference_stem,
        species=job.species,
        prompt_suffix=job.prompt_suffix,
        prompt_slug=job.prompt_slug,
        prompt="a cat running",
        output_relative_path=job.output_relative_path,
    )

    with pytest.raises(ValueError, match="prompt does not match"):
        build_batch_conditioning(
            malformed,
            BatchGenerationConfig(batch_id="projection", condition="projection"),
        )


@pytest.mark.parametrize(
    ("condition", "mask_enabled"),
    (
        ("sa_projection", False),
        ("sa_combined", True),
    ),
)
@pytest.mark.parametrize("prompt_suffix", PROMPT_SUFFIXES)
def test_source_aware_conditions_share_projection_and_only_gate_differs(
    condition: str,
    mask_enabled: bool,
    prompt_suffix: str,
) -> None:
    conditioning = build_batch_conditioning(
        _job(prompt_suffix),
        _source_config(condition),
        _source_caption(),
    )

    assert conditioning.orthogonalization.enabled is True
    assert conditioning.orthogonalization.mode == "factorial"
    assert conditioning.orthogonalization.strength == pytest.approx(0.73)
    assert conditioning.mask_single_text_rows is mask_enabled
    assert conditioning.basis_origin == "reference_caption"


@pytest.mark.parametrize(
    "prompt_suffix",
    tuple(
        suffix
        for suffix in PROMPT_SUFFIXES
        if suffix not in {"on a street", "running on a street"}
    ),
)
def test_source_aware_behavior_targets_select_only_source_behavior_main_effect(
    prompt_suffix: str,
) -> None:
    conditioning = build_batch_conditioning(
        _job(prompt_suffix),
        _source_config(),
        _source_caption(),
    )
    request = conditioning.orthogonalization

    assert request.mode == "factorial"
    assert request.include_behavior is True
    assert request.include_background is False
    assert request.include_interaction is False
    assert conditioning.removed_factors == ("behavior",)


def test_source_aware_background_target_selects_only_source_background_main_effect() -> None:
    conditioning = build_batch_conditioning(
        _job("on a street"),
        _source_config(),
        _source_caption(),
    )
    request = conditioning.orthogonalization

    assert request.mode == "factorial"
    assert request.include_behavior is False
    assert request.include_background is True
    assert request.include_interaction is False
    assert conditioning.removed_factors == ("background",)


def test_source_aware_joint_target_selects_both_source_main_effects() -> None:
    conditioning = build_batch_conditioning(
        _job("running on a street"),
        _source_config("sa_combined"),
        _source_caption(),
    )
    request = conditioning.orthogonalization

    assert request.mode == "factorial"
    assert request.include_behavior is True
    assert request.include_background is True
    assert request.include_interaction is False
    assert conditioning.removed_factors == ("behavior", "background")


def test_source_aware_basis_contains_only_reference_caption_prompts() -> None:
    job = _job("playing the guitar")
    source_caption = _source_caption()

    conditioning = build_batch_conditioning(
        job,
        _source_config(),
        source_caption,
    )
    prompts = conditioning.orthogonalization.prompts

    assert prompts is not None
    assert prompts.object_only == "a dog"
    assert prompts.object_behavior == "a dog lying down"
    assert prompts.object_background == "a dog on a beach"
    assert (
        prompts.object_behavior_background
        == "a dog lying down on a beach"
    )
    basis_prompts = {
        prompts.object_only,
        prompts.object_behavior,
        prompts.object_background,
        prompts.object_behavior_background,
    }
    assert job.prompt not in basis_prompts
    assert all("playing the guitar" not in prompt for prompt in basis_prompts)


def test_source_aware_projection_is_identical_with_and_without_gate() -> None:
    job = _job("running on a street")
    source_caption = _source_caption()

    projection = build_batch_conditioning(
        job,
        _source_config("sa_projection"),
        source_caption,
    )
    combined = build_batch_conditioning(
        job,
        _source_config("sa_combined"),
        source_caption,
    )

    assert projection.orthogonalization == combined.orthogonalization
    assert projection.removed_factors == combined.removed_factors
    assert projection.mask_single_text_rows is False
    assert combined.mask_single_text_rows is True


@pytest.mark.parametrize("condition", ["sa_projection", "sa_combined"])
def test_source_aware_conditions_require_a_source_caption(
    condition: str,
) -> None:
    with pytest.raises(ValueError, match="source_caption is required"):
        build_batch_conditioning(
            _job("running"),
            _source_config(condition),
        )


@pytest.mark.parametrize(
    "condition",
    ["baseline", "projection", "gate", "combined"],
)
def test_original_conditions_reject_a_source_caption(condition: str) -> None:
    with pytest.raises(ValueError, match="source_caption"):
        build_batch_conditioning(
            _job("running"),
            BatchGenerationConfig(batch_id=condition, condition=condition),
            _source_caption(),
        )


def test_source_caption_must_match_reference_stem_and_species() -> None:
    with pytest.raises(ValueError, match="reference mismatch"):
        build_batch_conditioning(
            _job("running"),
            _source_config(),
            _source_caption(reference_stem="dog_99"),
        )

    with pytest.raises(ValueError, match="species"):
        build_batch_conditioning(
            _job("running"),
            _source_config(),
            _source_caption(species="cat", reference_stem="dog_01"),
        )
