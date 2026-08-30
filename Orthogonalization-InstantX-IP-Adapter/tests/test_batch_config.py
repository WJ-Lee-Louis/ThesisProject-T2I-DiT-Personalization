import json

import pytest

from orthogonalization_instantx_ip_adapter.batch_config import (
    BatchGenerationConfig,
)


def test_config_round_trips_through_dict_and_json() -> None:
    config = BatchGenerationConfig(
        batch_id="animals.2026-07-23_run-01",
        condition="combined",
        orthogonalization_strength=0.75,
        ip_adapter_scale=0.7,
        guidance_scale=3.5,
        steps=24,
        seed=-17,
        width=960,
        height=1280,
        overwrite=True,
        expected_manifest_digest="aB" * 32,
        implementation_digest="Cd" * 32,
    )

    payload = config.to_dict()
    assert BatchGenerationConfig.from_dict(payload) == config
    assert BatchGenerationConfig.from_dict(json.loads(json.dumps(payload))) == config
    assert "projection_enabled" not in payload
    assert "mask_single_text_rows" not in payload


def test_new_config_fields_preserve_legacy_positional_signature() -> None:
    config = BatchGenerationConfig(
        "legacy-positional",
        "projection",
        0.6,
        0.9,
    )

    assert config.orthogonalization_strength == pytest.approx(0.6)
    assert config.ip_adapter_scale == pytest.approx(0.9)
    assert config.class_projection_strength == pytest.approx(0.5)
    assert config.class_superclass_prompt == "an animal"


@pytest.mark.parametrize(
    ("condition", "projection_enabled", "mask_single_text_rows"),
    [
        ("baseline", False, False),
        ("projection", True, False),
        ("gate", False, True),
        ("combined", True, True),
    ],
)
def test_condition_derives_intervention_flags(
    condition: str,
    projection_enabled: bool,
    mask_single_text_rows: bool,
) -> None:
    config = BatchGenerationConfig(batch_id="batch", condition=condition)

    assert config.projection_enabled is projection_enabled
    assert config.mask_single_text_rows is mask_single_text_rows
    assert config.target_projection_enabled is projection_enabled
    assert config.source_aware_projection_enabled is False
    assert config.source_target_projection_enabled is False
    assert config.all_source_factors_enabled is False
    assert config.requires_source_captions is False


@pytest.mark.parametrize(
    (
        "condition",
        "projection_enabled",
        "target_projection_enabled",
        "source_aware_projection_enabled",
        "mask_single_text_rows",
    ),
    [
        ("sa_projection", True, False, True, False),
        ("sa_combined", True, False, True, True),
    ],
)
def test_source_aware_conditions_derive_independent_intervention_flags(
    condition: str,
    projection_enabled: bool,
    target_projection_enabled: bool,
    source_aware_projection_enabled: bool,
    mask_single_text_rows: bool,
) -> None:
    config = BatchGenerationConfig(
        batch_id="source-aware",
        condition=condition,
        source_caption_digest="ab" * 32,
    )

    assert config.projection_enabled is projection_enabled
    assert config.target_projection_enabled is target_projection_enabled
    assert (
        config.source_aware_projection_enabled
        is source_aware_projection_enabled
    )
    assert config.source_target_projection_enabled is False
    assert config.all_source_factors_enabled is False
    assert config.requires_source_captions is True
    assert config.mask_single_text_rows is mask_single_text_rows


@pytest.mark.parametrize(
    (
        "condition",
        "mask_single_text_rows",
    ),
    [
        ("st_projection", False),
        ("st_combined", True),
        ("st_all_combined", True),
    ],
)
def test_source_target_union_conditions_derive_independent_intervention_flags(
    condition: str,
    mask_single_text_rows: bool,
) -> None:
    config = BatchGenerationConfig(
        batch_id="source-target-union",
        condition=condition,
        source_caption_digest="ab" * 32,
    )

    assert config.projection_enabled is True
    assert config.target_projection_enabled is False
    assert config.source_aware_projection_enabled is False
    assert config.source_target_projection_enabled is True
    assert config.all_source_factors_enabled is (
        condition == "st_all_combined"
    )
    assert config.requires_source_captions is True
    assert config.mask_single_text_rows is mask_single_text_rows


@pytest.mark.parametrize(
    (
        "condition",
        "requires_source_captions",
        "class_mode",
        "context_origin",
        "counterfactual",
    ),
    [
        ("class_absolute_combined", False, "absolute", None, None),
        (
            "class_contextual_combined",
            True,
            "contextual",
            "source",
            "superclass",
        ),
        (
            "class_target_superclass_combined",
            False,
            "contextual",
            "target",
            "superclass",
        ),
        (
            "class_target_no_object_combined",
            False,
            "contextual",
            "target",
            "no_object",
        ),
        (
            "class_source_superclass_combined",
            True,
            "contextual",
            "source",
            "superclass",
        ),
        (
            "class_source_no_object_combined",
            True,
            "contextual",
            "source",
            "no_object",
        ),
    ],
)
def test_class_conditions_have_independent_strength_and_always_use_gate(
    condition: str,
    requires_source_captions: bool,
    class_mode: str,
    context_origin: str | None,
    counterfactual: str | None,
) -> None:
    config = BatchGenerationConfig(
        batch_id=condition,
        condition=condition,
        orthogonalization_strength=0.9,
        class_projection_strength=0.35,
        class_superclass_prompt="an animal",
        source_caption_digest=(
            "ab" * 32 if requires_source_captions else None
        ),
    )

    assert config.projection_enabled is True
    assert config.class_projection_enabled is True
    assert config.class_absolute_projection_enabled is (class_mode == "absolute")
    assert config.class_contextual_projection_enabled is (
        class_mode == "contextual"
    )
    assert config.class_target_contextual_projection_enabled is (
        context_origin == "target"
    )
    assert config.class_source_contextual_projection_enabled is (
        context_origin == "source"
    )
    assert config.class_superclass_projection_enabled is (
        counterfactual == "superclass"
    )
    assert config.class_no_object_projection_enabled is (
        counterfactual == "no_object"
    )
    assert config.target_projection_enabled is False
    assert config.source_aware_projection_enabled is False
    assert config.source_target_projection_enabled is False
    assert config.requires_source_captions is requires_source_captions
    assert config.mask_single_text_rows is True
    assert config.class_projection_strength == pytest.approx(0.35)

    serialized = json.loads(json.dumps(config.to_dict()))
    assert BatchGenerationConfig.from_dict(serialized) == config
    assert serialized["class_projection_strength"] == pytest.approx(0.35)
    assert serialized["class_superclass_prompt"] == "an animal"


@pytest.mark.parametrize(
    "condition",
    [
        "class_contextual_combined",
        "class_source_superclass_combined",
        "class_source_no_object_combined",
    ],
)
def test_source_contextual_class_condition_requires_caption_digest(
    condition: str,
) -> None:
    with pytest.raises(ValueError, match="source_caption_digest is required"):
        BatchGenerationConfig(
            batch_id="class-contextual",
            condition=condition,
        )


@pytest.mark.parametrize("condition", ["sa_projection", "sa_combined"])
def test_source_aware_config_round_trips_with_caption_digest(
    condition: str,
) -> None:
    config = BatchGenerationConfig(
        batch_id=f"{condition}-run",
        condition=condition,
        expected_manifest_digest="12" * 32,
        implementation_digest="34" * 32,
        source_caption_digest="56" * 32,
    )

    serialized = json.loads(json.dumps(config.to_dict()))

    assert BatchGenerationConfig.from_dict(serialized) == config
    assert serialized["source_caption_digest"] == "56" * 32


@pytest.mark.parametrize(
    "condition", ["st_projection", "st_combined", "st_all_combined"]
)
def test_source_target_union_config_round_trips_with_caption_digest(
    condition: str,
) -> None:
    config = BatchGenerationConfig(
        batch_id=f"{condition}-run",
        condition=condition,
        expected_manifest_digest="12" * 32,
        implementation_digest="34" * 32,
        source_caption_digest="56" * 32,
    )

    serialized = json.loads(json.dumps(config.to_dict()))

    assert BatchGenerationConfig.from_dict(serialized) == config
    assert serialized["source_caption_digest"] == "56" * 32


@pytest.mark.parametrize("condition", ["sa_projection", "sa_combined"])
def test_source_aware_condition_requires_caption_digest(condition: str) -> None:
    with pytest.raises(ValueError, match="source_caption_digest is required"):
        BatchGenerationConfig(batch_id="batch", condition=condition)


@pytest.mark.parametrize(
    "condition", ["st_projection", "st_combined", "st_all_combined"]
)
def test_source_target_union_condition_requires_caption_digest(
    condition: str,
) -> None:
    with pytest.raises(ValueError, match="source_caption_digest is required"):
        BatchGenerationConfig(batch_id="batch", condition=condition)


@pytest.mark.parametrize(
    "condition",
    [
        "baseline",
        "projection",
        "gate",
        "combined",
        "class_absolute_combined",
        "class_target_superclass_combined",
        "class_target_no_object_combined",
    ],
)
def test_original_conditions_reject_source_caption_digest(
    condition: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="only valid for caption-aware",
    ):
        BatchGenerationConfig(
            batch_id="batch",
            condition=condition,
            source_caption_digest="ab" * 32,
        )


@pytest.mark.parametrize(
    "digest",
    ["", "not-hex", "ab" * 31, "ab" * 33],
)
def test_invalid_source_caption_digest_is_rejected(digest: str) -> None:
    with pytest.raises(ValueError, match="source_caption_digest"):
        BatchGenerationConfig(
            batch_id="batch",
            condition="sa_projection",
            source_caption_digest=digest,
        )


def test_non_string_source_caption_digest_is_rejected() -> None:
    with pytest.raises(TypeError, match="source_caption_digest"):
        BatchGenerationConfig(
            batch_id="batch",
            condition="sa_projection",
            source_caption_digest=1234,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "batch_id",
    [
        "",
        ".",
        "..",
        "../escape",
        r"..\escape",
        "/absolute",
        "two parts",
        "동물",
        "batch:01",
    ],
)
def test_unsafe_batch_ids_are_rejected(batch_id: str) -> None:
    with pytest.raises(ValueError, match="batch_id"):
        BatchGenerationConfig(batch_id=batch_id)


@pytest.mark.parametrize("batch_id", [None, 1, b"batch"])
def test_non_string_batch_ids_are_rejected(batch_id: object) -> None:
    with pytest.raises(TypeError, match="batch_id"):
        BatchGenerationConfig(batch_id=batch_id)  # type: ignore[arg-type]


def test_invalid_condition_is_rejected() -> None:
    with pytest.raises(ValueError, match="condition"):
        BatchGenerationConfig(batch_id="batch", condition="all")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="condition"):
        BatchGenerationConfig(batch_id="batch", condition=1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("orthogonalization_strength", -0.01),
        ("orthogonalization_strength", 1.01),
        ("orthogonalization_strength", float("nan")),
        ("orthogonalization_strength", True),
        ("class_projection_strength", -0.01),
        ("class_projection_strength", 1.01),
        ("class_projection_strength", float("nan")),
        ("class_projection_strength", True),
        ("ip_adapter_scale", -0.01),
        ("ip_adapter_scale", float("inf")),
        ("guidance_scale", -1),
        ("guidance_scale", "3.5"),
    ],
)
def test_invalid_real_controls_are_rejected(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError), match=field):
        BatchGenerationConfig(batch_id="batch", **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("steps", 0),
        ("steps", -1),
        ("steps", 1.5),
        ("steps", True),
        ("seed", 1.5),
        ("seed", False),
        ("width", 0),
        ("width", 17),
        ("width", 960.0),
        ("height", -16),
        ("height", 1000),
        ("height", True),
    ],
)
def test_invalid_integer_controls_are_rejected(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError), match=field):
        BatchGenerationConfig(batch_id="batch", **{field: value})


def test_overwrite_requires_a_boolean() -> None:
    with pytest.raises(TypeError, match="overwrite"):
        BatchGenerationConfig(batch_id="batch", overwrite=1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    ["", " an animal", "an animal ", "an  animal", 123],
)
def test_invalid_class_superclass_prompt_is_rejected(value: object) -> None:
    with pytest.raises(ValueError, match="class_superclass_prompt"):
        BatchGenerationConfig(
            batch_id="batch",
            class_superclass_prompt=value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "digest",
    ["", "not-hex", "0x1234", "12 34", "ab" * 31, "ab" * 33],
)
def test_non_hex_manifest_digest_is_rejected(digest: str) -> None:
    with pytest.raises(ValueError, match="expected_manifest_digest"):
        BatchGenerationConfig(
            batch_id="batch",
            expected_manifest_digest=digest,
        )


def test_non_string_manifest_digest_is_rejected() -> None:
    with pytest.raises(TypeError, match="expected_manifest_digest"):
        BatchGenerationConfig(
            batch_id="batch",
            expected_manifest_digest=1234,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "digest",
    ["", "not-hex", "ab" * 31, "ab" * 33],
)
def test_invalid_implementation_digest_is_rejected(digest: str) -> None:
    with pytest.raises(ValueError, match="implementation_digest"):
        BatchGenerationConfig(
            batch_id="batch",
            implementation_digest=digest,
        )


def test_non_string_implementation_digest_is_rejected() -> None:
    with pytest.raises(TypeError, match="implementation_digest"):
        BatchGenerationConfig(
            batch_id="batch",
            implementation_digest=1234,  # type: ignore[arg-type]
        )


def test_from_dict_rejects_non_mapping_and_unknown_fields() -> None:
    with pytest.raises(TypeError, match="mapping"):
        BatchGenerationConfig.from_dict([])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unknown"):
        BatchGenerationConfig.from_dict(
            {"batch_id": "batch", "output_dir": "../escape"}
        )
