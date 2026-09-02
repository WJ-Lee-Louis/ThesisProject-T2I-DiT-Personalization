from __future__ import annotations

import pytest

from orthogonalization_instantx_ip_adapter.batch_conditioning import (
    build_batch_conditioning,
)
from orthogonalization_instantx_ip_adapter.batch_config import (
    CLASS_CONDITIONS,
    FINAL_CONDITIONS,
    SOURCE_CAPTION_CONDITIONS,
    BatchGenerationConfig,
)
from orthogonalization_instantx_ip_adapter.batch_manifest import (
    BatchJob,
    PROMPT_SUFFIXES,
    prompt_slug,
)
from orthogonalization_instantx_ip_adapter.prompt_catalog import get_prompt_spec
from orthogonalization_instantx_ip_adapter.source_captions import (
    SourceCaptionRecord,
)


def _job(suffix: str, species: str = "dog") -> BatchJob:
    slug = prompt_slug(suffix)
    return BatchJob(
        reference_path=f"C:/{species}_01.jpg",
        reference_stem=f"{species}_01",
        species=species,
        prompt_suffix=suffix,
        prompt_slug=slug,
        prompt=f"a {species} {suffix}",
        output_relative_path=f"{species}_01/{species}_01_{slug}.png",
    )


def _caption(species: str = "dog") -> SourceCaptionRecord:
    return SourceCaptionRecord(
        reference_stem=f"{species}_01",
        species=species,
        content_sha256="ab" * 32,
        source_behavior="lying down",
        source_background="on a beach",
    )


def _config(condition: str, class_strength: float = 0.5) -> BatchGenerationConfig:
    return BatchGenerationConfig(
        batch_id=condition,
        condition=condition,
        orthogonalization_strength=1.0,
        class_projection_strength=class_strength,
        source_caption_digest=(
            "cd" * 32 if condition in SOURCE_CAPTION_CONDITIONS else None
        ),
    )


@pytest.mark.parametrize("condition", FINAL_CONDITIONS)
@pytest.mark.parametrize("suffix", PROMPT_SUFFIXES)
def test_all_22_conditions_build_for_every_prompt(
    condition: str,
    suffix: str,
) -> None:
    source = _caption() if condition in SOURCE_CAPTION_CONDITIONS else None
    result = build_batch_conditioning(_job(suffix), _config(condition), source)

    assert result.mask_single_text_rows is (condition != "baseline")
    assert result.orthogonalization.enabled is (
        condition not in {"baseline", "ssg"}
    )
    assert result.orthogonalization.hierarchical_class_enabled is (
        condition in CLASS_CONDITIONS
    )


@pytest.mark.parametrize(
    ("condition", "origin", "target_factors", "source_factors", "extra_sets"),
    (
        ("target_projection_ssg", "target_prompt", ("behavior",), (), 0),
        ("source_projection_ssg", "reference_caption", (), ("behavior",), 0),
        (
            "st_matched_projection_ssg",
            "source_target_matched",
            ("behavior",),
            ("behavior",),
            1,
        ),
        (
            "st_all_projection_ssg",
            "source_target_all",
            ("behavior",),
            ("behavior", "background"),
            1,
        ),
    ),
)
def test_four_base_nuisance_origins(
    condition: str,
    origin: str,
    target_factors: tuple[str, ...],
    source_factors: tuple[str, ...],
    extra_sets: int,
) -> None:
    source = _caption() if condition in SOURCE_CAPTION_CONDITIONS else None
    result = build_batch_conditioning(_job("flying"), _config(condition), source)
    request = result.orthogonalization

    assert result.basis_origin == origin
    assert result.class_projection_mode == "none"
    assert result.target_factors == target_factors
    assert result.source_factors == source_factors
    assert len(request.additional_direction_sets) == extra_sets
    assert request.include_behavior is True
    assert request.include_background is False
    if extra_sets:
        source_set = request.additional_direction_sets[0]
        assert source_set.include_behavior is True
        assert source_set.include_background is (
            condition == "st_all_projection_ssg"
        )


@pytest.mark.parametrize(
    ("condition", "origin", "policy"),
    (
        ("class_target_absolute_direct_ssg", "target_prompt", False),
        ("class_source_absolute_direct_ssg", "reference_caption", False),
        (
            "class_st_matched_absolute_direct_ssg",
            "source_target_matched",
            False,
        ),
        ("class_st_all_absolute_direct_ssg", "source_target_all", False),
        ("class_target_absolute_residual_ssg", "target_prompt", True),
        ("class_source_absolute_residual_ssg", "reference_caption", True),
        (
            "class_st_matched_absolute_residual_ssg",
            "source_target_matched",
            True,
        ),
        ("class_st_all_absolute_residual_ssg", "source_target_all", True),
    ),
)
def test_absolute_strategies_always_keep_nuisance_projection(
    condition: str,
    origin: str,
    policy: bool,
) -> None:
    source = _caption() if condition in SOURCE_CAPTION_CONDITIONS else None
    result = build_batch_conditioning(_job("flying"), _config(condition, 1.0), source)
    request = result.orthogonalization

    assert request.mode == (
        "factorial" if condition.startswith("class_source_") else "local"
    )
    assert request.prompts is not None
    assert request.include_behavior is True
    assert request.class_absolute_prompt == "a dog"
    assert request.class_contextual_pairs == ()
    assert request.class_strength == 1.0
    assert request.residualize_class_against_nuisance is policy
    assert result.basis_origin == origin
    assert result.class_projection_mode == (
        "absolute_residual" if policy else "absolute_direct"
    )


@pytest.mark.parametrize("kind", ("no_object", "superclass"))
@pytest.mark.parametrize("origin", ("target", "source", "st_matched", "st_all"))
def test_contextual_four_by_two_matrix(kind: str, origin: str) -> None:
    condition = f"class_{origin}_{kind}_ssg"
    source = _caption() if condition in SOURCE_CAPTION_CONDITIONS else None
    result = build_batch_conditioning(
        _job("flying in the sky"),
        _config(condition, 1.0),
        source,
    )
    request = result.orthogonalization

    assert result.class_projection_mode == kind
    assert request.class_strength == 1.0
    assert request.residualize_class_against_nuisance is True
    assert request.class_absolute_prompt is None
    if origin == "target":
        assert len(request.class_contextual_pairs) == 3
        assert request.additional_class_direction_sets == ()
    elif origin == "source":
        assert len(request.class_contextual_pairs) == 3
        assert request.additional_class_direction_sets == ()
    else:
        assert len(request.class_contextual_pairs) == 3
        assert len(request.additional_class_direction_sets) == 1
        assert len(request.additional_class_direction_sets[0].pairs) == 3

    negatives = tuple(pair.negative for pair in request.class_contextual_pairs)
    if kind == "superclass":
        assert all(value.startswith("an animal ") for value in negatives)
    else:
        assert all(not value.startswith("an animal ") for value in negatives)


@pytest.mark.parametrize(
    "condition",
    (
        "class_st_all_absolute_direct_ssg",
        "class_st_all_absolute_residual_ssg",
        "class_st_all_no_object_ssg",
        "class_st_all_superclass_ssg",
    ),
)
def test_all_source_generic_conditions_include_both_source_nuisance_factors(
    condition: str,
) -> None:
    result = build_batch_conditioning(
        _job("flying"),
        _config(condition),
        _caption(),
    )
    request = result.orthogonalization
    source_set = request.additional_direction_sets[0]

    assert result.source_factors == ("behavior", "background")
    assert source_set.include_behavior is True
    assert source_set.include_background is True


def test_target_factorial_uses_three_class_pairs_but_single_factor_uses_one() -> None:
    config = _config("class_target_no_object_ssg")
    single = build_batch_conditioning(_job("flying"), config)
    factorial = build_batch_conditioning(_job("flying in the sky"), config)
    assert len(single.orthogonalization.class_contextual_pairs) == 1
    assert len(factorial.orthogonalization.class_contextual_pairs) == 3


@pytest.mark.parametrize("condition", tuple(sorted(SOURCE_CAPTION_CONDITIONS)))
def test_caption_aware_conditions_require_source_caption(condition: str) -> None:
    with pytest.raises(ValueError, match="source_caption is required"):
        build_batch_conditioning(_job("flying"), _config(condition))


def test_non_caption_condition_rejects_source_caption() -> None:
    with pytest.raises(ValueError, match="must not be supplied"):
        build_batch_conditioning(_job("flying"), _config("target_projection_ssg"), _caption())


def test_caption_must_match_reference_and_species() -> None:
    bad = SourceCaptionRecord(
        reference_stem="dog_99",
        species="dog",
        content_sha256="ab" * 32,
        source_behavior="lying",
        source_background="on a beach",
    )
    with pytest.raises(ValueError, match="reference mismatch"):
        build_batch_conditioning(
            _job("flying"),
            _config("source_projection_ssg"),
            bad,
        )


def test_job_prompt_is_validated() -> None:
    job = _job("flying")
    invalid = BatchJob(
        reference_path=job.reference_path,
        reference_stem=job.reference_stem,
        species=job.species,
        prompt_suffix=job.prompt_suffix,
        prompt_slug=job.prompt_slug,
        prompt="a dog sleeping",
        output_relative_path=job.output_relative_path,
    )
    with pytest.raises(ValueError, match="does not match"):
        build_batch_conditioning(
            invalid,
            _config("target_projection_ssg"),
        )


def test_prompt_taxonomy_still_covers_behavior_background_and_factorial() -> None:
    assert get_prompt_spec("flying").factors == ("behavior",)
    assert get_prompt_spec("on a stage").factors == ("background",)
    assert get_prompt_spec("flying in the sky").factors == (
        "behavior",
        "background",
    )
