"""Shared manifest utilities for XVerseBench_0907 metric runners."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


BENCHMARK_FILES = {
    "subset": (
        "subset_bench/benchmark_manifest.json",
        "subset_bench/XVerseBench_single_animal_29_DSG.json",
    ),
    "custom": (
        "custom_bench/benchmark_manifest.json",
        "custom_bench/XVerseBench_custom_animal_15x6_DSG.json",
    ),
}


def load_json(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_benchmark(
    benchmark_root: str | Path,
    benchmark: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    root = Path(benchmark_root).resolve()
    try:
        manifest_rel, dsg_rel = BENCHMARK_FILES[benchmark]
    except KeyError as exc:
        raise ValueError(f"Unknown benchmark: {benchmark!r}") from exc
    manifest = load_json(root / manifest_rel)
    dsg_list = load_json(root / dsg_rel)
    records = manifest["records"]
    if len(records) != len(dsg_list):
        raise ValueError("Benchmark manifest and DSG lengths differ")

    dsg_by_record: dict[str, dict[str, Any]] = {}
    for record, dsg in zip(records, dsg_list):
        if record["prompt"] != dsg["prompt"]:
            raise ValueError(
                f"Prompt/DSG mismatch for {record['record_id']}: "
                f"{record['prompt']!r} != {dsg['prompt']!r}"
            )
        dsg_by_record[record["record_id"]] = dsg
    return manifest, dsg_by_record


def load_generated_images(
    generated_manifest_path: str | Path,
    benchmark_manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    manifest_path = Path(generated_manifest_path).resolve()
    generated = load_json(manifest_path)
    if generated.get("schema_version") != 1:
        raise ValueError("Generated-image manifest schema_version must be 1")
    if generated.get("benchmark_id") != benchmark_manifest["benchmark_id"]:
        raise ValueError(
            "Generated-image benchmark_id does not match benchmark manifest"
        )

    benchmark_records = {
        record["record_id"]: record for record in benchmark_manifest["records"]
    }
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for raw in generated.get("images", []):
        missing = {
            key
            for key in ("record_id", "condition", "seed", "image_path")
            if key not in raw
        }
        if missing:
            raise ValueError(f"Generated-image row is missing fields: {missing}")
        record_id = str(raw["record_id"])
        if record_id not in benchmark_records:
            raise ValueError(f"Unknown benchmark record_id: {record_id!r}")
        key = (record_id, str(raw["condition"]), int(raw["seed"]))
        if key in seen:
            raise ValueError(f"Duplicate generated-image key: {key}")
        seen.add(key)

        image_path = Path(raw["image_path"])
        if not image_path.is_absolute():
            image_path = manifest_path.parent / image_path
        if not image_path.is_file():
            raise FileNotFoundError(f"Generated image does not exist: {image_path}")
        row = dict(raw)
        row["record_id"] = record_id
        row["condition"] = str(raw["condition"])
        row["seed"] = int(raw["seed"])
        row["image_path"] = str(image_path.resolve())
        row["benchmark_record"] = benchmark_records[record_id]
        rows.append(row)
    if not rows:
        raise ValueError("Generated-image manifest contains no images")
    return rows


def resolve_reference_path(
    benchmark_root: str | Path,
    benchmark_record: dict[str, Any],
) -> Path:
    path = Path(benchmark_record["reference_path"])
    if not path.is_absolute():
        path = Path(benchmark_root).resolve() / path
    if not path.is_file():
        raise FileNotFoundError(f"Reference image does not exist: {path}")
    return path.resolve()


def import_official_xverse(xverse_root: str | Path) -> Path:
    """Put a pinned official XVerse checkout on sys.path for metric imports."""

    import sys

    root = Path(xverse_root).resolve()
    required = (
        root / "eval" / "tools" / "dpg_score.py",
        root / "eval" / "tools" / "dino.py",
        root / "eval" / "tools" / "florence_sam.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "XVerse checkout lacks official metric files: " + ", ".join(missing)
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def dependency_filtered_scores(result: dict[str, Any]) -> dict[str, float]:
    scores = {str(key): float(value) for key, value in result["qid2scores"].items()}
    validity = {
        str(key): bool(value) for key, value in result["qid2validity"].items()
    }
    return {
        qid: score if validity.get(qid, True) else 0.0
        for qid, score in scores.items()
    }


def grouped_question_scores(
    qid_scores: dict[str, float],
    qid_to_group: dict[str, str] | None,
) -> dict[str, float]:
    if not qid_to_group:
        return {}
    values: dict[str, list[float]] = {}
    for qid, group in qid_to_group.items():
        values.setdefault(group, []).append(float(qid_scores[str(qid)]))
    return {name: 100.0 * mean(items) for name, items in values.items()}


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")
