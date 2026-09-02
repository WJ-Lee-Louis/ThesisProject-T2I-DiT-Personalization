from pathlib import Path, PurePosixPath

import pytest

import batch_client
import batch_modal
from orthogonalization_instantx_ip_adapter.adapter import (
    OrthogonalizationRequest,
    PromptContrastPair,
)
from orthogonalization_instantx_ip_adapter.batch_config import (
    FINAL_CONDITIONS,
    SOURCE_CAPTION_CONDITIONS,
    BatchGenerationConfig,
)
from orthogonalization_instantx_ip_adapter.batch_manifest import (
    DEFAULT_REFERENCE_ROOT,
    build_batch_manifest,
)
from orthogonalization_instantx_ip_adapter.orthogonalization import PromptSet
from orthogonalization_instantx_ip_adapter.source_captions import (
    DEFAULT_SOURCE_CAPTIONS_PATH,
)


def test_default_batch_id_is_a_valid_safe_component() -> None:
    value = batch_client.default_batch_id("target_projection_ssg")
    BatchGenerationConfig(batch_id=value, condition="target_projection_ssg")
    assert value.startswith("dreambench-target_projection_ssg-")


@pytest.mark.parametrize(
    ("remote_path", "expected"),
    (
        (
            "run-01/cat_03/cat_03_running.png",
            PurePosixPath("cat_03/cat_03_running.png"),
        ),
        (
            "/run-01/_metadata/status.json",
            PurePosixPath("_metadata/status.json"),
        ),
    ),
)
def test_relative_remote_path_accepts_selected_batch(
    remote_path: str,
    expected: PurePosixPath,
) -> None:
    assert batch_client._relative_remote_path(remote_path, "run-01") == expected


def test_relative_remote_path_rejects_another_batch() -> None:
    with pytest.raises(ValueError, match="outside batch"):
        batch_client._relative_remote_path(
            "run-02/cat_03/cat_03_running.png",
            "run-01",
        )


def test_safe_local_child_rejects_parent_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes"):
        batch_client._safe_local_child(tmp_path, "../outside.png")


def test_submit_parser_exposes_exact_22_conditions() -> None:
    parser = batch_client.build_parser()
    for condition in FINAL_CONDITIONS:
        args = parser.parse_args(["submit", "--condition", condition])
        assert args.condition == condition
        assert args.source_captions == DEFAULT_SOURCE_CAPTIONS_PATH

    with pytest.raises(SystemExit):
        parser.parse_args(["submit", "--condition", "combined"])


@pytest.mark.parametrize("condition", tuple(sorted(SOURCE_CAPTION_CONDITIONS)))
def test_caption_aware_default_batch_id_is_safe(condition: str) -> None:
    value = batch_client.default_batch_id(condition)
    BatchGenerationConfig(
        batch_id=value,
        condition=condition,
        source_caption_digest="ab" * 32,
    )


@pytest.mark.parametrize("module", (batch_client, batch_modal))
@pytest.mark.parametrize(
    "condition",
    (
        "st_all_projection_ssg",
        "class_st_all_absolute_direct_ssg",
        "class_st_all_absolute_residual_ssg",
        "class_st_all_no_object_ssg",
        "class_st_all_superclass_ssg",
    ),
)
def test_new_st_all_conditions_are_self_contained(
    module,
    condition: str,
) -> None:
    manifest = build_batch_manifest(DEFAULT_REFERENCE_ROOT)
    selected, skipped = module._select_jobs_for_condition(
        manifest.jobs,
        condition,
    )
    assert len(selected) == 140
    assert len(skipped) == 0


@pytest.mark.parametrize("module", (batch_client, batch_modal))
def test_non_all_conditions_keep_140_jobs(module) -> None:
    manifest = build_batch_manifest(DEFAULT_REFERENCE_ROOT)
    selected, skipped = module._select_jobs_for_condition(
        manifest.jobs,
        "class_st_matched_superclass_ssg",
    )
    assert len(selected) == 140
    assert skipped == ()


@pytest.mark.parametrize(
    ("condition", "expected"),
    (
        ("baseline", 140),
        ("target_projection_ssg", 140),
        ("st_all_projection_ssg", 140),
        ("class_st_matched_no_object_ssg", 140),
        ("class_st_all_no_object_ssg", 140),
    ),
)
def test_download_count_uses_condition_selection(
    condition: str,
    expected: int,
) -> None:
    assert batch_client._resolve_expected_png_count(
        None,
        {"condition": condition},
    ) == expected


def test_download_count_prefers_remote_status() -> None:
    assert batch_client._resolve_expected_png_count(
        {"total_count": 90},
        {"condition": "baseline"},
    ) == 90


@pytest.mark.parametrize("value", (0, -1, True, 1.5, "100"))
def test_download_count_rejects_invalid_status(value) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        batch_client._resolve_expected_png_count(
            {"total_count": value},
            {"condition": "st_all_projection_ssg"},
        )


def test_download_temporary_name_is_short(tmp_path: Path) -> None:
    destination = tmp_path / "_metadata" / "images" / ("x" * 180 + ".json")
    temporary = batch_client._download_temporary_path(destination)
    assert temporary.parent == destination.parent
    assert destination.name not in temporary.name
    assert temporary.name.startswith(".download-")
    assert temporary.name.endswith(".tmp")


def test_modal_serializer_records_absolute_policy() -> None:
    request = OrthogonalizationRequest(
        mode="local",
        strength=1.0,
        prompts=PromptSet(
            object_only="a dog",
            object_behavior="a dog flying",
        ),
        include_behavior=True,
        include_background=False,
        class_absolute_prompt="a dog",
        class_strength=0.5,
        residualize_class_against_nuisance=False,
    )
    payload = batch_modal._request_to_dict(request)
    assert payload["class_absolute_prompt"] == "a dog"
    assert payload["class_strength"] == 0.5
    assert payload["residualize_class_against_nuisance"] is False


def test_modal_serializer_records_contextual_pairs() -> None:
    request = OrthogonalizationRequest(
        mode="local",
        prompts=PromptSet(
            object_only="a dog",
            object_behavior="a dog flying",
        ),
        include_behavior=True,
        include_background=False,
        class_contextual_pairs=(
            PromptContrastPair("a dog flying", "flying", "behavior_context"),
        ),
        class_strength=1.0,
    )
    payload = batch_modal._request_to_dict(request)
    assert payload["class_contextual_pairs"] == [
        {
            "positive": "a dog flying",
            "negative": "flying",
            "label": "behavior_context",
        }
    ]
    assert payload["residualize_class_against_nuisance"] is True
