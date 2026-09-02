from __future__ import annotations

import json

import pytest

from orthogonalization_instantx_ip_adapter.batch_config import (
    CLASS_ABSOLUTE_DIRECT_CONDITIONS,
    CLASS_ABSOLUTE_RESIDUAL_CONDITIONS,
    CLASS_CONDITIONS,
    CLASS_NO_OBJECT_CONDITIONS,
    CLASS_SUPERCLASS_CONDITIONS,
    FINAL_CONDITIONS,
    SOURCE_CAPTION_CONDITIONS,
    BatchGenerationConfig,
)


def _config(condition: str, **kwargs) -> BatchGenerationConfig:
    return BatchGenerationConfig(
        batch_id=condition,
        condition=condition,
        source_caption_digest=(
            "ab" * 32 if condition in SOURCE_CAPTION_CONDITIONS else None
        ),
        **kwargs,
    )


def test_final_condition_catalog_is_exactly_22_unique_names() -> None:
    assert len(FINAL_CONDITIONS) == 22
    assert len(set(FINAL_CONDITIONS)) == 22
    assert FINAL_CONDITIONS[:2] == ("baseline", "ssg")
    assert not any("combined" in value for value in FINAL_CONDITIONS)
    assert "class_absolute_combined" not in FINAL_CONDITIONS


@pytest.mark.parametrize("condition", FINAL_CONDITIONS)
def test_every_final_condition_round_trips_and_uses_expected_ssg(
    condition: str,
) -> None:
    config = _config(condition)
    payload = json.loads(json.dumps(config.to_dict()))

    assert BatchGenerationConfig.from_dict(payload) == config
    assert config.mask_single_text_rows is (condition != "baseline")
    assert config.class_projection_enabled is (condition in CLASS_CONDITIONS)
    assert config.requires_source_captions is (
        condition in SOURCE_CAPTION_CONDITIONS
    )


@pytest.mark.parametrize(
    ("condition", "target", "source", "source_target", "all_source"),
    (
        ("target_projection_ssg", True, False, False, False),
        ("source_projection_ssg", False, True, False, False),
        ("st_matched_projection_ssg", False, False, True, False),
        ("st_all_projection_ssg", False, False, True, True),
    ),
)
def test_four_base_projection_origins(
    condition: str,
    target: bool,
    source: bool,
    source_target: bool,
    all_source: bool,
) -> None:
    config = _config(condition)
    assert config.projection_enabled is True
    assert config.target_projection_enabled is target
    assert config.source_aware_projection_enabled is source
    assert config.source_target_projection_enabled is source_target
    assert config.all_source_factors_enabled is all_source


@pytest.mark.parametrize("condition", tuple(sorted(CLASS_CONDITIONS)))
@pytest.mark.parametrize("strength", (0.5, 1.0))
def test_generic_conditions_accept_only_two_class_lambdas(
    condition: str,
    strength: float,
) -> None:
    config = _config(condition, class_projection_strength=strength)
    assert config.class_projection_strength == strength


@pytest.mark.parametrize("strength", (0.0, 0.25, 0.75, 0.99))
def test_generic_conditions_reject_unplanned_class_lambdas(strength: float) -> None:
    with pytest.raises(ValueError, match="either 0.5 or 1.0"):
        _config(
            "class_target_absolute_direct_ssg",
            class_projection_strength=strength,
        )


def test_class_strategy_sets_form_four_by_four_matrix() -> None:
    strategy_sets = (
        CLASS_ABSOLUTE_DIRECT_CONDITIONS,
        CLASS_ABSOLUTE_RESIDUAL_CONDITIONS,
        CLASS_NO_OBJECT_CONDITIONS,
        CLASS_SUPERCLASS_CONDITIONS,
    )
    assert all(len(values) == 4 for values in strategy_sets)
    assert len(set().union(*strategy_sets)) == 16


@pytest.mark.parametrize("condition", tuple(sorted(CLASS_ABSOLUTE_DIRECT_CONDITIONS)))
def test_absolute_direct_flags(condition: str) -> None:
    config = _config(condition)
    assert config.class_absolute_projection_enabled is True
    assert config.class_absolute_direct_projection_enabled is True
    assert config.class_absolute_residual_projection_enabled is False
    assert config.class_contextual_projection_enabled is False


@pytest.mark.parametrize("condition", tuple(sorted(CLASS_ABSOLUTE_RESIDUAL_CONDITIONS)))
def test_absolute_residual_flags(condition: str) -> None:
    config = _config(condition)
    assert config.class_absolute_projection_enabled is True
    assert config.class_absolute_direct_projection_enabled is False
    assert config.class_absolute_residual_projection_enabled is True


@pytest.mark.parametrize("condition", tuple(sorted(CLASS_NO_OBJECT_CONDITIONS)))
def test_no_object_flags(condition: str) -> None:
    config = _config(condition)
    assert config.class_contextual_projection_enabled is True
    assert config.class_no_object_projection_enabled is True
    assert config.class_superclass_projection_enabled is False


@pytest.mark.parametrize("condition", tuple(sorted(CLASS_SUPERCLASS_CONDITIONS)))
def test_superclass_flags(condition: str) -> None:
    config = _config(condition)
    assert config.class_contextual_projection_enabled is True
    assert config.class_superclass_projection_enabled is True
    assert config.class_no_object_projection_enabled is False


@pytest.mark.parametrize(
    "condition",
    (
        "projection",
        "gate",
        "combined",
        "sa_combined",
        "st_combined",
        "class_absolute_combined",
        "class_contextual_combined",
    ),
)
def test_legacy_ambiguous_condition_names_are_rejected(condition: str) -> None:
    with pytest.raises(ValueError, match="22 finalized"):
        BatchGenerationConfig(batch_id="legacy", condition=condition)


@pytest.mark.parametrize("condition", tuple(sorted(SOURCE_CAPTION_CONDITIONS)))
def test_caption_aware_conditions_require_digest(condition: str) -> None:
    with pytest.raises(ValueError, match="source_caption_digest is required"):
        BatchGenerationConfig(batch_id="missing-caption", condition=condition)


def test_unknown_serialized_field_is_rejected() -> None:
    payload = _config("baseline").to_dict()
    payload["unknown"] = True
    with pytest.raises(ValueError, match="Unknown"):
        BatchGenerationConfig.from_dict(payload)
