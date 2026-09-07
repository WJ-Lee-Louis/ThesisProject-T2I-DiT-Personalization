"""Merge XVerseBench_0907 metric outputs and create analysis-ready tables."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from common import load_benchmark, load_json, mean, write_json


IDENTITY_FIELDS = ("record_id", "condition", "seed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--benchmark", choices=("subset", "custom"), required=True)
    parser.add_argument("--dpg", type=Path)
    parser.add_argument("--ip", type=Path)
    parser.add_argument("--aes", type=Path)
    parser.add_argument("--baseline-condition", default="baseline")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (str(row["record_id"]), str(row["condition"]), int(row["seed"]))


def _flatten_metric_row(row: dict[str, Any]) -> dict[str, Any]:
    flattened = {
        key: row[key]
        for key in ("dpg_with_dependency", "dpg_without_dependency", "ip_similarity", "aes")
        if key in row
    }
    for prefix, nested_key in (
        ("dpg_factor_", "factor_scores"),
        ("dpg_component_", "component_scores"),
    ):
        for name, value in row.get(nested_key, {}).items():
            flattened[f"{prefix}{name}"] = value
    if row.get("joint_action_background_score") is not None:
        flattened["dpg_joint_action_background"] = row[
            "joint_action_background_score"
        ]
    return flattened


def _load_metric(path: Path | None, expected_benchmark_id: str) -> list[dict[str, Any]]:
    if path is None:
        return []
    payload = load_json(path)
    if payload.get("benchmark_id") != expected_benchmark_id:
        raise ValueError(f"Metric benchmark_id mismatch in {path}")
    return payload.get("scores", [])


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    preferred = [
        "condition",
        "composition",
        "prompt_id",
        "record_id",
        "reference_filename",
        "seed",
        "n",
    ]
    fields = [key for key in preferred if key in fields] + [
        key for key in fields if key not in preferred
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summarize(
    rows: Iterable[dict[str, Any]],
    group_fields: tuple[str, ...],
    metric_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field) for field in group_fields)].append(row)
    summaries = []
    for key, items in sorted(groups.items(), key=lambda item: str(item[0])):
        result = dict(zip(group_fields, key))
        result["n"] = len(items)
        for metric in metric_fields:
            values = [float(item[metric]) for item in items if item.get(metric) is not None]
            if values:
                result[metric] = mean(values)
        summaries.append(result)
    return summaries


def _paired_deltas(
    rows: list[dict[str, Any]],
    baseline_condition: str,
    metric_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    baseline = {
        (row["record_id"], int(row["seed"])): row
        for row in rows
        if row["condition"] == baseline_condition
    }
    delta_rows = []
    for row in rows:
        if row["condition"] == baseline_condition:
            continue
        base = baseline.get((row["record_id"], int(row["seed"])))
        if base is None:
            continue
        delta = {
            "condition": row["condition"],
            "record_id": row["record_id"],
            "seed": row["seed"],
            "composition": row.get("composition"),
            "prompt_id": row.get("prompt_id"),
        }
        for metric in metric_fields:
            if row.get(metric) is not None and base.get(metric) is not None:
                delta[f"delta_{metric}"] = float(row[metric]) - float(base[metric])
        delta_rows.append(delta)
    delta_metrics = tuple(
        sorted({key for row in delta_rows for key in row if key.startswith("delta_")})
    )
    return _summarize(
        delta_rows,
        ("condition", "composition"),
        delta_metrics,
    )


def main() -> int:
    args = build_parser().parse_args()
    manifest, _ = load_benchmark(args.benchmark_root, args.benchmark)
    records = {record["record_id"]: record for record in manifest["records"]}
    merged: dict[tuple[str, str, int], dict[str, Any]] = {}
    for metric_path in (args.dpg, args.ip, args.aes):
        for score in _load_metric(metric_path, manifest["benchmark_id"]):
            key = _key(score)
            target = merged.setdefault(
                key,
                {
                    "record_id": key[0],
                    "condition": key[1],
                    "seed": key[2],
                },
            )
            target.update(_flatten_metric_row(score))

    if not merged:
        raise SystemExit("No metric score files were provided")
    per_image = []
    for row in merged.values():
        record = records[row["record_id"]]
        row.update(
            {
                "reference_filename": record["reference_filename"],
                "composition": record.get("composition"),
                "prompt_id": record.get("prompt_id"),
                "prompt": record["prompt"],
            }
        )
        per_image.append(row)
    per_image.sort(key=lambda row: (row["condition"], row["record_id"], row["seed"]))
    metric_fields = tuple(
        sorted(
            {
                key
                for row in per_image
                for key in row
                if key.startswith("dpg_") or key in {"ip_similarity", "aes"}
            }
        )
    )

    by_condition = _summarize(per_image, ("condition",), metric_fields)
    by_composition = _summarize(
        per_image, ("condition", "composition"), metric_fields
    )
    by_prompt = _summarize(
        per_image, ("condition", "composition", "prompt_id"), metric_fields
    )
    paired = _paired_deltas(per_image, args.baseline_condition, metric_fields)

    output_dir = args.output_dir.resolve()
    _write_csv(output_dir / "per_image_scores.csv", per_image)
    _write_csv(output_dir / "summary_by_condition.csv", by_condition)
    _write_csv(output_dir / "summary_by_composition.csv", by_composition)
    _write_csv(output_dir / "summary_by_prompt.csv", by_prompt)
    _write_csv(output_dir / "paired_delta_vs_baseline.csv", paired)
    write_json(
        output_dir / "summary.json",
        {
            "schema_version": 1,
            "benchmark_id": manifest["benchmark_id"],
            "baseline_condition": args.baseline_condition,
            "metric_fields": list(metric_fields),
            "summary_by_condition": by_condition,
            "summary_by_composition": by_composition,
            "summary_by_prompt": by_prompt,
            "paired_delta_vs_baseline": paired,
        },
    )
    print(f"Aggregated {len(per_image)} image rows -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
