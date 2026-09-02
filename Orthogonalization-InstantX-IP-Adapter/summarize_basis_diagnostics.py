"""Summarize downloaded batch sidecars without loading any ML dependencies.

The Modal worker writes one JSON sidecar for every generated image under
``_metadata/images``.  This script validates that a downloaded, completed
batch contains exactly the sidecars selected by ``job_selection.json`` and
then emits:

* ``basis_diagnostics_per_image.csv`` -- lossless JSON-in-cell vectors and
  matrices plus scalar diagnostics for every image; and
* ``basis_diagnostics_summary.json`` -- aggregate metrics grouped by
  condition, prompt category, and semantic prompt factors.

Only the Python standard library is used.  In particular, running this file
does not import torch, diffusers, transformers, Modal, or model code.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
from typing import Iterable, Mapping, Sequence
from uuid import uuid4


PER_IMAGE_FILENAME = "basis_diagnostics_per_image.csv"
SUMMARY_FILENAME = "basis_diagnostics_summary.json"


class DiagnosticsValidationError(ValueError):
    """A downloaded batch is incomplete or its metadata is inconsistent."""


def analyze_batches(batch_directories: Iterable[str | Path]) -> tuple[list[dict], list[dict]]:
    """Validate batches and return normalized per-image rows and batch records."""

    paths = tuple(Path(value).expanduser().resolve() for value in batch_directories)
    if not paths:
        raise DiagnosticsValidationError("At least one batch directory is required")
    if len(set(paths)) != len(paths):
        raise DiagnosticsValidationError("Batch directories must be unique")

    rows: list[dict] = []
    batches: list[dict] = []
    seen_batch_ids: dict[str, Path] = {}
    for batch_path in paths:
        batch_rows, batch_record = _read_batch(batch_path)
        batch_id = batch_record["batch_id"]
        if batch_id in seen_batch_ids:
            raise DiagnosticsValidationError(
                f"Duplicate batch_id {batch_id!r}: {seen_batch_ids[batch_id]} and {batch_path}"
            )
        seen_batch_ids[batch_id] = batch_path
        rows.extend(batch_rows)
        batches.append(batch_record)

    rows.sort(
        key=lambda row: (
            row["condition"],
            row["batch_id"],
            row["reference_stem"].casefold(),
            row["prompt_suffix"],
        )
    )
    batches.sort(key=lambda value: (value["condition"], value["batch_id"]))
    return rows, batches


def write_reports(
    rows: Sequence[Mapping[str, object]],
    batches: Sequence[Mapping[str, object]],
    output_directory: str | Path,
) -> tuple[Path, Path]:
    """Write the per-image CSV and grouped JSON report."""

    output_root = Path(output_directory).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / PER_IMAGE_FILENAME
    summary_path = output_root / SUMMARY_FILENAME

    csv_rows = [_csv_row(row) for row in rows]
    _atomic_write_csv(csv_path, csv_rows)
    summary = build_summary(rows, batches)
    _atomic_write_json(summary_path, summary)
    return csv_path, summary_path


def build_summary(
    rows: Sequence[Mapping[str, object]],
    batches: Sequence[Mapping[str, object]],
) -> dict:
    """Build JSON-compatible aggregate groups from normalized image rows."""

    return {
        "schema_version": 1,
        "batch_count": len(batches),
        "image_count": len(rows),
        "batches": [dict(value) for value in batches],
        "groups": {
            "by_condition": _group_rows(rows, ("condition",)),
            "by_condition_and_prompt_category": _group_rows(
                rows,
                ("condition", "prompt_category"),
            ),
            "by_condition_and_prompt_factors": _group_rows(
                rows,
                ("condition", "prompt_factors"),
            ),
            "by_condition_prompt_category_and_factors": _group_rows(
                rows,
                ("condition", "prompt_category", "prompt_factors"),
            ),
        },
    }


def _read_batch(batch_root: Path) -> tuple[list[dict], dict]:
    if not batch_root.is_dir():
        raise DiagnosticsValidationError(f"Batch directory does not exist: {batch_root}")

    metadata_root = batch_root / "_metadata"
    config = _read_object(metadata_root / "config.json")
    status = _read_object(metadata_root / "status.json")
    selection = _read_object(metadata_root / "job_selection.json")

    batch_id = _required_text(config, "batch_id", metadata_root / "config.json")
    condition = _required_text(config, "condition", metadata_root / "config.json")
    _require_equal(status.get("batch_id"), batch_id, "status batch_id", batch_root)
    _require_equal(status.get("condition"), condition, "status condition", batch_root)
    _require_equal(selection.get("condition"), condition, "job-selection condition", batch_root)

    state = _required_text(status, "state", metadata_root / "status.json")
    if state != "completed":
        raise DiagnosticsValidationError(
            f"Batch {batch_id!r} is not completed (state={state!r})"
        )

    total_count = _nonnegative_integer(status.get("total_count"), "status.total_count")
    completed_count = _nonnegative_integer(
        status.get("completed_count"),
        "status.completed_count",
    )
    if completed_count != total_count:
        raise DiagnosticsValidationError(
            f"Batch {batch_id!r} reports {completed_count}/{total_count} completed jobs"
        )
    failed_count = _nonnegative_integer(status.get("failed_count", 0), "status.failed_count")
    if failed_count:
        raise DiagnosticsValidationError(
            f"Batch {batch_id!r} reports {failed_count} failed job(s)"
        )

    selected_count = _nonnegative_integer(
        selection.get("selected_job_count"),
        "job_selection.selected_job_count",
    )
    if selected_count != total_count:
        raise DiagnosticsValidationError(
            f"Batch {batch_id!r} selected_job_count={selected_count} but total_count={total_count}"
        )
    full_count = _nonnegative_integer(
        selection.get("full_manifest_count"),
        "job_selection.full_manifest_count",
    )
    redundant_count = _nonnegative_integer(
        selection.get("skipped_as_redundant_job_count", 0),
        "job_selection.skipped_as_redundant_job_count",
    )
    if selected_count + redundant_count != full_count:
        raise DiagnosticsValidationError(
            f"Batch {batch_id!r} selection does not partition its full manifest: "
            f"{selected_count} selected + {redundant_count} redundant != {full_count}"
        )

    selected_suffixes = _text_list(
        selection.get("selected_prompt_suffixes"),
        "job_selection.selected_prompt_suffixes",
    )
    if not selected_suffixes:
        raise DiagnosticsValidationError(
            f"Batch {batch_id!r} has no selected prompt suffixes"
        )
    if len(set(selected_suffixes)) != len(selected_suffixes):
        raise DiagnosticsValidationError(
            f"Batch {batch_id!r} has duplicate selected prompt suffixes"
        )
    if selected_count % len(selected_suffixes):
        raise DiagnosticsValidationError(
            f"Batch {batch_id!r} selected count is not divisible by its prompt count"
        )
    expected_reference_count = selected_count // len(selected_suffixes)

    image_metadata_root = metadata_root / "images"
    sidecar_paths = tuple(sorted(image_metadata_root.rglob("*.json"))) if image_metadata_root.is_dir() else ()
    if len(sidecar_paths) != selected_count:
        raise DiagnosticsValidationError(
            f"Batch {batch_id!r} has {len(sidecar_paths)}/{selected_count} expected image sidecars"
        )

    rows: list[dict] = []
    seen_jobs: set[tuple[str, str]] = set()
    suffix_counts: Counter[str] = Counter()
    for sidecar_path in sidecar_paths:
        payload = _read_object(sidecar_path)
        row = _normalize_sidecar(
            payload,
            sidecar_path=sidecar_path,
            image_metadata_root=image_metadata_root,
            batch_root=batch_root,
            expected_batch_id=batch_id,
            expected_condition=condition,
            config=config,
        )
        job_key = (row["reference_stem"], row["prompt_suffix"])
        if job_key in seen_jobs:
            raise DiagnosticsValidationError(
                f"Batch {batch_id!r} has duplicate sidecar job {job_key!r}"
            )
        seen_jobs.add(job_key)
        suffix_counts[row["prompt_suffix"]] += 1
        rows.append(row)

    selected_suffix_set = set(selected_suffixes)
    if set(suffix_counts) != selected_suffix_set:
        missing = sorted(selected_suffix_set - set(suffix_counts))
        unexpected = sorted(set(suffix_counts) - selected_suffix_set)
        raise DiagnosticsValidationError(
            f"Batch {batch_id!r} prompt suffix mismatch; missing={missing}, unexpected={unexpected}"
        )
    bad_counts = {
        suffix: suffix_counts[suffix]
        for suffix in selected_suffixes
        if suffix_counts[suffix] != expected_reference_count
    }
    if bad_counts:
        raise DiagnosticsValidationError(
            f"Batch {batch_id!r} does not have {expected_reference_count} references "
            f"for every selected prompt: {bad_counts}"
        )

    batch_record = {
        "batch_directory": str(batch_root),
        "batch_id": batch_id,
        "condition": condition,
        "orthogonalization_strength": _optional_finite_number(
            config.get("orthogonalization_strength"),
            "config.orthogonalization_strength",
        ),
        "class_projection_strength": _optional_finite_number(
            config.get("class_projection_strength"),
            "config.class_projection_strength",
        ),
        "state": state,
        "full_manifest_count": full_count,
        "selected_job_count": selected_count,
        "skipped_as_redundant_job_count": redundant_count,
        "selected_prompt_suffixes": selected_suffixes,
        "sidecar_count": len(rows),
    }
    return rows, batch_record


def _normalize_sidecar(
    payload: Mapping[str, object],
    *,
    sidecar_path: Path,
    image_metadata_root: Path,
    batch_root: Path,
    expected_batch_id: str,
    expected_condition: str,
    config: Mapping[str, object],
) -> dict:
    _require_equal(payload.get("batch_id"), expected_batch_id, "sidecar batch_id", sidecar_path)
    _require_equal(payload.get("condition"), expected_condition, "sidecar condition", sidecar_path)
    job = _required_mapping(payload, "job", sidecar_path)
    conditioning = _required_mapping(payload, "conditioning", sidecar_path)
    result = _required_mapping(payload, "orthogonalization_result", sidecar_path)

    reference_stem = _required_text(job, "reference_stem", sidecar_path)
    prompt_suffix = _required_text(job, "prompt_suffix", sidecar_path)
    species = _required_text(job, "species", sidecar_path)
    output_relative_path = _required_text(job, "output_relative_path", sidecar_path)
    output_path = PurePosixPath(output_relative_path.replace("\\", "/"))
    if output_path.is_absolute() or ".." in output_path.parts:
        raise DiagnosticsValidationError(
            f"Unsafe output_relative_path in {sidecar_path}: {output_relative_path!r}"
        )
    expected_sidecar_relative = output_path.with_suffix(".json").as_posix()
    actual_sidecar_relative = sidecar_path.relative_to(image_metadata_root).as_posix()
    if actual_sidecar_relative != expected_sidecar_relative:
        raise DiagnosticsValidationError(
            f"Sidecar path does not match job output in {sidecar_path}: "
            f"expected {expected_sidecar_relative!r}, got {actual_sidecar_relative!r}"
        )

    enabled = result.get("enabled")
    if not isinstance(enabled, bool):
        raise DiagnosticsValidationError(
            f"orthogonalization_result.enabled must be boolean in {sidecar_path}"
        )
    gate = conditioning.get("mask_single_text_rows")
    if not isinstance(gate, bool):
        raise DiagnosticsValidationError(
            f"conditioning.mask_single_text_rows must be boolean in {sidecar_path}"
        )

    prompt_spec_value = conditioning.get("prompt_spec")
    if prompt_spec_value is None:
        prompt_spec: Mapping[str, object] = {}
    elif isinstance(prompt_spec_value, Mapping):
        prompt_spec = prompt_spec_value
    else:
        raise DiagnosticsValidationError(f"conditioning.prompt_spec must be an object in {sidecar_path}")
    prompt_category = prompt_spec.get("category", "unknown")
    if not isinstance(prompt_category, str) or not prompt_category:
        raise DiagnosticsValidationError(f"Invalid prompt category in {sidecar_path}")
    factor_value = prompt_spec.get("factors", conditioning.get("target_factors", []))
    prompt_factors = _text_list(factor_value, f"prompt factors in {sidecar_path}")

    direction_names = _text_list(
        result.get("direction_names", []),
        f"direction_names in {sidecar_path}",
    )
    direction_raw_norms = _number_list(
        result.get("direction_raw_norms"),
        f"direction_raw_norms in {sidecar_path}",
    )
    singular_values = _number_list(
        result.get("singular_values"),
        f"singular_values in {sidecar_path}",
    )
    relative_singular_values = _number_list(
        result.get("relative_singular_values"),
        f"relative_singular_values in {sidecar_path}",
    )
    direction_cosine_matrix = _number_matrix(
        result.get("direction_cosine_matrix"),
        f"direction_cosine_matrix in {sidecar_path}",
    )
    class_pair_raw_norms = _number_list(
        result.get("class_pair_raw_norms"),
        f"class_pair_raw_norms in {sidecar_path}",
    )
    class_pair_direction_cosine_matrix = _number_matrix(
        result.get("class_pair_direction_cosine_matrix"),
        f"class_pair_direction_cosine_matrix in {sidecar_path}",
    )
    class_pair_labels = _text_list(
        result.get("class_pair_labels", []),
        f"class_pair_labels in {sidecar_path}",
    )
    class_pair_count = _optional_nonnegative_integer(
        result.get("class_pair_count"),
        "class_pair_count",
    )
    class_singular_values = _number_list(
        result.get("class_singular_values"),
        f"class_singular_values in {sidecar_path}",
    )
    class_overlap_norms = _number_list(
        result.get("class_overlap_norms"),
        f"class_overlap_norms in {sidecar_path}",
    )
    class_residual_norms = _number_list(
        result.get("class_residual_norms"),
        f"class_residual_norms in {sidecar_path}",
    )
    class_residual_ratios = _number_list(
        result.get("class_residual_ratios"),
        f"class_residual_ratios in {sidecar_path}",
    )
    if class_pair_count is not None:
        if class_pair_labels and len(class_pair_labels) != class_pair_count:
            raise DiagnosticsValidationError(
                f"Class pair label count differs in {sidecar_path}"
            )
        if class_pair_raw_norms and len(class_pair_raw_norms) != class_pair_count:
            raise DiagnosticsValidationError(
                f"Class pair raw-norm count differs in {sidecar_path}"
            )

    basis_rank = _optional_nonnegative_integer(result.get("basis_rank"), "basis_rank")
    dropped_rank = _optional_nonnegative_integer(result.get("dropped_rank"), "dropped_rank")
    if not enabled:
        basis_rank = 0 if basis_rank is None else basis_rank
        dropped_rank = 0 if dropped_rank is None else dropped_rank
    if direction_names and direction_raw_norms and len(direction_names) != len(direction_raw_norms):
        raise DiagnosticsValidationError(
            f"Direction-name/raw-norm lengths differ in {sidecar_path}"
        )
    if direction_names and direction_cosine_matrix:
        size = len(direction_names)
        if len(direction_cosine_matrix) != size or any(len(row) != size for row in direction_cosine_matrix):
            raise DiagnosticsValidationError(
                f"Direction cosine matrix is not {size}x{size} in {sidecar_path}"
            )
    if (
        basis_rank is not None
        and dropped_rank is not None
        and singular_values
        and len(singular_values) != basis_rank + dropped_rank
    ):
        raise DiagnosticsValidationError(
            f"Singular-value count does not equal retained+dropped rank in {sidecar_path}"
        )
    if relative_singular_values and len(relative_singular_values) != len(singular_values):
        raise DiagnosticsValidationError(
            f"Relative singular-value count differs in {sidecar_path}"
        )

    degenerate = result.get("degenerate")
    if degenerate is not None and not isinstance(degenerate, bool):
        raise DiagnosticsValidationError(f"degenerate must be boolean or null in {sidecar_path}")
    hierarchical_class_projection = result.get("hierarchical_class_projection")
    if hierarchical_class_projection is not None and not isinstance(
        hierarchical_class_projection, bool
    ):
        raise DiagnosticsValidationError(
            f"hierarchical_class_projection must be boolean or null in {sidecar_path}"
        )
    class_direction_dropped = result.get("class_direction_dropped")
    if class_direction_dropped is not None and not isinstance(
        class_direction_dropped, bool
    ):
        raise DiagnosticsValidationError(
            f"class_direction_dropped must be boolean or null in {sidecar_path}"
        )

    return {
        "batch_directory": str(batch_root),
        "batch_id": expected_batch_id,
        "condition": expected_condition,
        "orthogonalization_strength": _optional_finite_number(
            config.get("orthogonalization_strength"),
            "config.orthogonalization_strength",
        ),
        "class_projection_strength": _optional_finite_number(
            config.get("class_projection_strength"),
            "config.class_projection_strength",
        ),
        "reference_stem": reference_stem,
        "species": species,
        "prompt_suffix": prompt_suffix,
        "prompt_category": prompt_category,
        "prompt_factors": prompt_factors,
        "output_relative_path": output_relative_path,
        "sidecar_relative_path": str(sidecar_path.relative_to(batch_root).as_posix()),
        "basis_origin": _optional_text(conditioning.get("basis_origin")),
        "class_projection_mode": _optional_text(
            conditioning.get("class_projection_mode")
        ),
        "source_factor_policy": _optional_text(conditioning.get("source_factor_policy")),
        "orthogonalization_enabled": enabled,
        "gate": gate,
        "hierarchical_class_projection": hierarchical_class_projection,
        "class_overlap_policy": _optional_text(
            result.get("class_overlap_policy")
        ),
        "direction_names": direction_names,
        "direction_raw_norms": direction_raw_norms,
        "direction_raw_norm_kind": _optional_text(
            result.get("direction_raw_norm_kind")
        ),
        "class_pair_raw_norms": class_pair_raw_norms,
        "class_pair_count": class_pair_count,
        "class_pair_labels": class_pair_labels,
        "class_pair_direction_cosine_matrix": (
            class_pair_direction_cosine_matrix
        ),
        "basis_rank": basis_rank,
        "dropped_rank": dropped_rank,
        "nuisance_basis_rank": _optional_nonnegative_integer(
            result.get("nuisance_basis_rank"),
            "nuisance_basis_rank",
        ),
        "class_basis_rank": _optional_nonnegative_integer(
            result.get("class_basis_rank"),
            "class_basis_rank",
        ),
        "class_direction_count": _optional_nonnegative_integer(
            result.get("class_direction_count"),
            "class_direction_count",
        ),
        "class_dropped_rank": _optional_nonnegative_integer(
            result.get("class_dropped_rank"),
            "class_dropped_rank",
        ),
        "class_singular_values": class_singular_values,
        "class_direction_dropped": class_direction_dropped,
        "class_overlap_norm": _optional_finite_number(
            result.get("class_overlap_norm"),
            f"class_overlap_norm in {sidecar_path}",
        ),
        "class_residual_norm": _optional_finite_number(
            result.get("class_residual_norm"),
            f"class_residual_norm in {sidecar_path}",
        ),
        "class_residual_ratio": _optional_finite_number(
            result.get("class_residual_ratio"),
            f"class_residual_ratio in {sidecar_path}",
        ),
        "class_overlap_norms": class_overlap_norms,
        "class_residual_norms": class_residual_norms,
        "class_residual_ratios": class_residual_ratios,
        "singular_values": singular_values,
        "relative_singular_values": relative_singular_values,
        "direction_cosine_matrix": direction_cosine_matrix,
        "projection_component_norm": _optional_finite_number(
            result.get("projection_component_norm"),
            f"projection_component_norm in {sidecar_path}",
        ),
        "nuisance_component_norm": _optional_finite_number(
            result.get("nuisance_component_norm"),
            f"nuisance_component_norm in {sidecar_path}",
        ),
        "class_component_norm": _optional_finite_number(
            result.get("class_component_norm"),
            f"class_component_norm in {sidecar_path}",
        ),
        "nuisance_removed_component_norm": _optional_finite_number(
            result.get("nuisance_removed_component_norm"),
            f"nuisance_removed_component_norm in {sidecar_path}",
        ),
        "class_removed_component_norm": _optional_finite_number(
            result.get("class_removed_component_norm"),
            f"class_removed_component_norm in {sidecar_path}",
        ),
        "removed_component_norm": _optional_finite_number(
            result.get("removed_component_norm"),
            f"removed_component_norm in {sidecar_path}",
        ),
        "edit_norm": _optional_finite_number(
            result.get("edit_norm"),
            f"edit_norm in {sidecar_path}",
        ),
        "cosine_to_input": _optional_finite_number(
            result.get("cosine_to_input"),
            f"cosine_to_input in {sidecar_path}",
        ),
        "degenerate": degenerate,
    }


def _group_rows(rows: Sequence[Mapping[str, object]], keys: tuple[str, ...]) -> list[dict]:
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    for row in rows:
        group_key = tuple(_hashable_group_value(row[key]) for key in keys)
        grouped.setdefault(group_key, []).append(row)

    values: list[dict] = []
    for group_key, members in sorted(grouped.items(), key=lambda item: repr(item[0])):
        identity = {
            key: _json_group_value(value)
            for key, value in zip(keys, group_key)
        }
        identity.update(_aggregate(members))
        values.append(identity)
    return values


def _aggregate(rows: Sequence[Mapping[str, object]]) -> dict:
    direction_names = sorted(
        {
            name
            for row in rows
            for name in row.get("direction_names", [])
            if isinstance(name, str)
        }
    )
    direction_name_sets = sorted(
        {tuple(row.get("direction_names", [])) for row in rows},
        key=repr,
    )
    rank_histogram = Counter(
        row["basis_rank"] for row in rows if row.get("basis_rank") is not None
    )
    return {
        "image_count": len(rows),
        "batch_ids": sorted({str(row["batch_id"]) for row in rows}),
        "orthogonalization_enabled_count": sum(
            bool(row.get("orthogonalization_enabled")) for row in rows
        ),
        "gate_enabled_count": sum(bool(row.get("gate")) for row in rows),
        "degenerate_count": sum(row.get("degenerate") is True for row in rows),
        "degenerate_available_count": sum(
            row.get("degenerate") is not None for row in rows
        ),
        "direction_names": direction_names,
        "direction_name_sets": [list(value) for value in direction_name_sets],
        "basis_rank_histogram": {
            str(key): rank_histogram[key] for key in sorted(rank_histogram)
        },
        "metrics": {
            name: _metric_summary(rows, name)
            for name in (
                "basis_rank",
                "dropped_rank",
                "projection_component_norm",
                "removed_component_norm",
                "edit_norm",
                "cosine_to_input",
            )
        },
    }


def _metric_summary(rows: Sequence[Mapping[str, object]], name: str) -> dict:
    values = [
        float(row[name])
        for row in rows
        if row.get(name) is not None
    ]
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": math.fsum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def _csv_row(row: Mapping[str, object]) -> dict[str, object]:
    return {
        name: _compact_json(value) if isinstance(value, (list, dict)) else value
        for name, value in row.items()
    }


def _atomic_write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise DiagnosticsValidationError("Cannot write an empty per-image report")
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_object(path: Path) -> dict:
    if not path.is_file():
        raise DiagnosticsValidationError(f"Required metadata file is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DiagnosticsValidationError(f"Could not read JSON metadata {path}: {error}") from error
    if not isinstance(value, dict):
        raise DiagnosticsValidationError(f"JSON metadata must be an object: {path}")
    return value


def _required_mapping(value: Mapping[str, object], key: str, path: Path) -> Mapping[str, object]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise DiagnosticsValidationError(f"{key} must be an object in {path}")
    return item


def _required_text(value: Mapping[str, object], key: str, path: Path) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise DiagnosticsValidationError(f"{key} must be a non-empty string in {path}")
    return item


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise DiagnosticsValidationError("Optional text metadata must be null or a non-empty string")
    return value


def _text_list(value: object, name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise DiagnosticsValidationError(f"{name} must be a list of non-empty strings")
    return list(value)


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DiagnosticsValidationError(f"{name} must be a non-negative integer")
    return value


def _optional_nonnegative_integer(value: object, name: str) -> int | None:
    if value is None:
        return None
    return _nonnegative_integer(value, name)


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DiagnosticsValidationError(f"{name} must contain finite numbers")
    result = float(value)
    if not math.isfinite(result):
        raise DiagnosticsValidationError(f"{name} must contain finite numbers")
    return result


def _optional_finite_number(value: object, name: str) -> float | None:
    if value is None:
        return None
    return _finite_number(value, name)


def _number_list(value: object, name: str) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [_finite_number(value, name)]
    if not isinstance(value, list):
        raise DiagnosticsValidationError(f"{name} must be a number, list, or null")
    return [_finite_number(item, name) for item in value]


def _number_matrix(value: object, name: str) -> list[list[float]]:
    if value is None:
        return []
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [[_finite_number(value, name)]]
    if not isinstance(value, list):
        raise DiagnosticsValidationError(f"{name} must be a numeric matrix or null")
    if not value:
        return []
    if all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
        return [[_finite_number(item, name) for item in value]]
    rows: list[list[float]] = []
    for item in value:
        if not isinstance(item, list):
            raise DiagnosticsValidationError(f"{name} must be a rectangular numeric matrix")
        rows.append([_finite_number(number, name) for number in item])
    if rows and any(len(row) != len(rows[0]) for row in rows):
        raise DiagnosticsValidationError(f"{name} must be a rectangular numeric matrix")
    return rows


def _require_equal(actual: object, expected: object, name: str, path: Path) -> None:
    if actual != expected:
        raise DiagnosticsValidationError(
            f"Inconsistent {name} in {path}: expected {expected!r}, got {actual!r}"
        )


def _hashable_group_value(value: object) -> object:
    return tuple(value) if isinstance(value, list) else value


def _json_group_value(value: object) -> object:
    return list(value) if isinstance(value, tuple) else value


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate completed downloaded batches and summarize pooled-basis "
            "diagnostics without loading a model."
        )
    )
    parser.add_argument(
        "batch_directories",
        nargs="+",
        help="One or more downloaded batch directories containing _metadata",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory in which the per-image CSV and grouped JSON are written",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rows, batches = analyze_batches(args.batch_directories)
        csv_path, summary_path = write_reports(rows, batches, args.output_dir)
    except (DiagnosticsValidationError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"Validated {len(rows)} sidecars from {len(batches)} completed batch(es).")
    print(f"Per-image CSV: {csv_path}")
    print(f"Grouped JSON: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
