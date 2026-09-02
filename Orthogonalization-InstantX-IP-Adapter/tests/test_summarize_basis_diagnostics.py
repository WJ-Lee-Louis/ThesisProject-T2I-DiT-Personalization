from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import summarize_basis_diagnostics as diagnostics


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_batch(
    root: Path,
    *,
    batch_id: str,
    condition: str,
    prompt_specs: list[tuple[str, str, list[str]]],
    references: int = 1,
    enabled: bool = True,
    gate: bool = True,
    full_count: int | None = None,
    redundant_count: int = 0,
) -> Path:
    batch_root = root / batch_id
    selected_count = len(prompt_specs) * references
    if full_count is None:
        full_count = selected_count + redundant_count
    metadata = batch_root / "_metadata"
    _write_json(
        metadata / "config.json",
        {
            "batch_id": batch_id,
            "condition": condition,
            "orthogonalization_strength": 1.0,
        },
    )
    _write_json(
        metadata / "status.json",
        {
            "batch_id": batch_id,
            "condition": condition,
            "state": "completed",
            "total_count": selected_count,
            "completed_count": selected_count,
            "failed_count": 0,
        },
    )
    _write_json(
        metadata / "job_selection.json",
        {
            "condition": condition,
            "full_manifest_count": full_count,
            "selected_job_count": selected_count,
            "selected_prompt_suffixes": [value[0] for value in prompt_specs],
            "skipped_as_redundant_job_count": redundant_count,
            "skipped_as_redundant_prompt_suffixes": [],
        },
    )

    for reference_index in range(references):
        reference_stem = f"dog_{reference_index:02d}"
        for prompt_index, (suffix, category, factors) in enumerate(prompt_specs):
            slug = suffix.replace(" ", "_")
            relative_png = f"{reference_stem}/{reference_stem}_{slug}.png"
            result = {"enabled": enabled}
            if enabled:
                # Scalars exercise the one-direction representation emitted by
                # adapter._tensor_to_python; the report normalizes them to lists.
                result.update(
                    {
                        "direction_names": ["target_behavior_local"],
                        "direction_raw_norms": 0.7,
                        "basis_rank": 1,
                        "dropped_rank": 0,
                        "singular_values": 1.0,
                        "relative_singular_values": 1.0,
                        "direction_cosine_matrix": 1.0,
                        "projection_component_norm": 0.25 + prompt_index,
                        "removed_component_norm": 0.25 + prompt_index,
                        "edit_norm": 0.5 + prompt_index,
                        "cosine_to_input": 0.99,
                        "degenerate": False,
                    }
                )
            _write_json(
                metadata
                / "images"
                / reference_stem
                / f"{reference_stem}_{slug}.json",
                {
                    "batch_id": batch_id,
                    "condition": condition,
                    "job": {
                        "reference_stem": reference_stem,
                        "species": "dog",
                        "prompt_suffix": suffix,
                        "output_relative_path": relative_png,
                    },
                    "conditioning": {
                        "prompt_spec": {
                            "category": category,
                            "factors": factors,
                        },
                        "basis_origin": "target_prompt" if enabled else "none",
                        "source_factor_policy": "none",
                        "mask_single_text_rows": gate,
                    },
                    "orthogonalization_result": result,
                },
            )
    return batch_root


def test_cli_writes_per_image_csv_and_grouped_json(tmp_path: Path) -> None:
    batch = _make_batch(
        tmp_path,
        batch_id="combined-lambda1",
        condition="combined",
        prompt_specs=[
            ("flying", "single_behavior", ["behavior"]),
            ("flying in the sky", "background_related", ["behavior", "background"]),
        ],
        references=2,
    )
    output = tmp_path / "analysis"

    assert diagnostics.main([str(batch), "--output-dir", str(output)]) == 0

    with (output / diagnostics.PER_IMAGE_FILENAME).open(
        encoding="utf-8",
        newline="",
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert json.loads(rows[0]["direction_raw_norms"]) == [0.7]
    assert json.loads(rows[0]["singular_values"]) == [1.0]
    assert json.loads(rows[0]["direction_cosine_matrix"]) == [[1.0]]
    assert rows[0]["gate"] == "True"

    summary = json.loads(
        (output / diagnostics.SUMMARY_FILENAME).read_text(encoding="utf-8")
    )
    assert summary["image_count"] == 4
    by_condition = summary["groups"]["by_condition"]
    assert by_condition[0]["condition"] == "combined"
    assert by_condition[0]["metrics"]["basis_rank"] == {
        "count": 4,
        "mean": 1.0,
        "min": 1.0,
        "max": 1.0,
    }
    categories = summary["groups"]["by_condition_and_prompt_category"]
    assert {value["prompt_category"]: value["image_count"] for value in categories} == {
        "background_related": 2,
        "single_behavior": 2,
    }
    factor_groups = summary["groups"]["by_condition_and_prompt_factors"]
    assert {tuple(value["prompt_factors"]) for value in factor_groups} == {
        ("behavior",),
        ("behavior", "background"),
    }


def test_baseline_and_gate_disabled_projection_are_retained(tmp_path: Path) -> None:
    baseline = _make_batch(
        tmp_path,
        batch_id="baseline",
        condition="baseline",
        prompt_specs=[("flying", "single_behavior", ["behavior"])],
        enabled=False,
        gate=False,
    )
    gate = _make_batch(
        tmp_path,
        batch_id="gate",
        condition="gate",
        prompt_specs=[("flying", "single_behavior", ["behavior"])],
        enabled=False,
        gate=True,
    )

    rows, _ = diagnostics.analyze_batches([baseline, gate])

    assert [(row["condition"], row["basis_rank"], row["gate"]) for row in rows] == [
        ("baseline", 0, False),
        ("gate", 0, True),
    ]
    assert all(row["singular_values"] == [] for row in rows)
    assert all(row["projection_component_norm"] is None for row in rows)


def test_st_all_completed_partial_selection_accepts_80_sidecars(tmp_path: Path) -> None:
    prompt_specs = [
        (f"behavior {index}", "single_behavior", ["behavior"])
        for index in range(8)
    ]
    batch = _make_batch(
        tmp_path,
        batch_id="st-all",
        condition="st_all_projection_ssg",
        prompt_specs=prompt_specs,
        references=10,
        full_count=140,
        redundant_count=60,
    )

    rows, batches = diagnostics.analyze_batches([batch])

    assert len(rows) == 80
    assert batches[0]["full_manifest_count"] == 140
    assert batches[0]["selected_job_count"] == 80
    assert batches[0]["skipped_as_redundant_job_count"] == 60


def test_completed_batch_rejects_missing_expected_sidecar(tmp_path: Path) -> None:
    batch = _make_batch(
        tmp_path,
        batch_id="incomplete-download",
        condition="combined",
        prompt_specs=[
            ("flying", "single_behavior", ["behavior"]),
            ("sleeping", "single_behavior", ["behavior"]),
        ],
    )
    missing = next((batch / "_metadata" / "images").rglob("*.json"))
    missing.unlink()

    with pytest.raises(
        diagnostics.DiagnosticsValidationError,
        match=r"1/2 expected image sidecars",
    ):
        diagnostics.analyze_batches([batch])
