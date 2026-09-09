"""Tests for metric-suite completion guards."""

# ruff: noqa: E402

import json
from pathlib import Path

import sys


METRIC_ROOT = (
    Path(__file__).resolve().parents[1] / "datasets" / "XVerseBench_0907" / "metrics"
)
if str(METRIC_ROOT) not in sys.path:
    sys.path.insert(0, str(METRIC_ROOT))

from run_metric_suite import valid_reference_artifacts, valid_score_output


def _write_reference_artifacts(root: Path, *, fallback_method=None) -> None:
    rows = []
    for index in range(29):
        crop = root / "crops" / f"{index}.png"
        mask = root / "masks" / f"{index}.png"
        crop.parent.mkdir(parents=True, exist_ok=True)
        mask.parent.mkdir(parents=True, exist_ok=True)
        crop.write_bytes(b"crop")
        mask.write_bytes(b"mask")
        rows.append(
            {
                "segmentation_method": "florence2_open_vocabulary+sam2",
                "segmentation_query_attempts": ["animal"],
                "foreground_crop": f"crops/{index}.png",
                "binary_mask": f"masks/{index}.png",
            }
        )
    (root / "segmentation_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 4,
                "fallback_method": fallback_method,
                "florence2_compatibility": {
                    "after": {"lm_head_tied_to_shared": True}
                },
                "records": rows,
            }
        ),
        encoding="utf-8",
    )


def test_valid_score_output_rejects_duplicate_rows(tmp_path: Path) -> None:
    path = tmp_path / "dpg.json"
    row = {"record_id": "r", "condition": "baseline", "seed": 42}
    path.write_text(
        json.dumps(
            {
                "benchmark_id": "xverse_ref_custom_animal_15x6",
                "scores": [row] * 540,
            }
        ),
        encoding="utf-8",
    )

    assert not valid_score_output(path, "custom")


def test_valid_score_output_rejects_all_zero_ip_detection(tmp_path: Path) -> None:
    path = tmp_path / "ip.json"
    scores = []
    for index in range(90):
        for condition in (
            "baseline",
            "ssg",
            "target_projection_ssg",
            "source_projection_ssg",
            "st_matched_projection_ssg",
            "st_all_projection_ssg",
        ):
            scores.append(
                {
                    "record_id": f"custom_{index:03d}",
                    "condition": condition,
                    "seed": 42 + index,
                    "detected_candidate_count": 0,
                }
            )
    path.write_text(
        json.dumps(
            {
                "metric": "IP-Sim",
                "benchmark_id": "xverse_ref_custom_animal_15x6",
                "scores": scores,
            }
        ),
        encoding="utf-8",
    )

    assert not valid_score_output(path, "custom")


def test_reference_artifact_validation_checks_files(tmp_path: Path) -> None:
    rows = [
        {"foreground_crop": f"crops/{index}.png", "binary_mask": f"masks/{index}.png"}
        for index in range(29)
    ]
    (tmp_path / "segmentation_manifest.json").write_text(
        json.dumps({"schema_version": 3, "records": rows}), encoding="utf-8"
    )

    assert not valid_reference_artifacts(tmp_path)


def test_reference_artifact_validation_requires_audited_queries(
    tmp_path: Path,
) -> None:
    rows = []
    for index in range(29):
        crop = tmp_path / "crops" / f"{index}.png"
        mask = tmp_path / "masks" / f"{index}.png"
        crop.parent.mkdir(parents=True, exist_ok=True)
        mask.parent.mkdir(parents=True, exist_ok=True)
        crop.write_bytes(b"crop")
        mask.write_bytes(b"mask")
        rows.append(
            {
                "foreground_crop": f"crops/{index}.png",
                "binary_mask": f"masks/{index}.png",
            }
        )
    (tmp_path / "segmentation_manifest.json").write_text(
        json.dumps({"schema_version": 2, "records": rows}), encoding="utf-8"
    )

    assert not valid_reference_artifacts(tmp_path)


def test_reference_artifact_validation_accepts_repaired_florence_only(
    tmp_path: Path,
) -> None:
    _write_reference_artifacts(tmp_path)

    assert valid_reference_artifacts(tmp_path)


def test_reference_artifact_validation_rejects_manual_bbox_fallback(
    tmp_path: Path,
) -> None:
    _write_reference_artifacts(tmp_path, fallback_method="audited_bbox")

    assert not valid_reference_artifacts(tmp_path)
