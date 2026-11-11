"""Durable cloud reports and verification; no generation or LLM changes."""
import csv
import json
import math
from pathlib import Path
import shutil

from .config import MODELS, SETTINGS
from .io import read, sha, write
from .vendor.artifact_bundle import verify
from .vendor.benchmark_metrics import METRICS, PROTOCOL

FIELDS = ["condition", "benchmark", "case_id", "reference_id", "category", "target_prompt",
          "reference_sha256", "generated_sha256", *METRICS]


def csv_scores(path, scores):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(scores)


def ready_file(folder, **metadata):
    files = {p.relative_to(folder).as_posix(): dict(size=p.stat().st_size, sha256=sha(p))
             for p in folder.rglob("*") if p.is_file() and p.name != "ready.json"}
    result = dict(status="complete", directory=folder.relative_to("/work").as_posix(), files=files, **metadata)
    write(folder / "ready.json", result)
    return result


def finish_shard(bundle_id, cases, shard, workflow, runs):
    runs.reload()
    if workflow["status"] != "complete":
        raise ValueError("Cannot finalize an incomplete workflow: " + workflow["status"])
    plan = read(Path("/work/bundles") / bundle_id / "plan.json")
    rows = [r for r in plan["rows"] if r["case_id"] in set(cases)]
    if len(rows) != len(set(cases)) or len({r["benchmark"] for r in rows}) != 1:
        raise ValueError("Invalid shard case membership")
    benchmark = rows[0]["benchmark"]
    index_path = Path("/work") / workflow["export"]["index"]
    index = read(index_path)
    if (set(index["case_ids"]) != set(cases) or index["generated_images"] != len(cases) * 2
            or index["benchmark"] != benchmark or index["bundle_id"] != bundle_id):
        raise ValueError("Export shard identity differs")
    verify(index_path.parent, index)
    metric_path = Path("/work") / workflow["evaluation"]["path"]
    metric = read(metric_path)
    if metric["protocol"] != PROTOCOL or len(metric["scores"]) != 2 * len(cases):
        raise ValueError("Metric protocol/count differs")
    if metric["generation_provenance"] != dict(code=plan["code"], gpu="H100", settings=SETTINGS, models=MODELS):
        raise ValueError("Metric generation provenance differs")
    expected = {(condition, r["case_id"]): r for r in rows for condition in ("baseline", "source_target")}
    seen = set()
    for score in metric["scores"]:
        key = score["condition"], score["case_id"]
        if key not in expected or key in seen:
            raise ValueError("Duplicate or unexpected metric row")
        row = expected[key]
        if any(score[k] != row[k] for k in ("reference_id", "target_prompt", "reference_sha256")):
            raise ValueError("Metric input binding differs")
        member = f"results/{score['condition']}/{benchmark}/{score['case_id']}.png"
        if score["generated_sha256"] != index["members"][member]["sha256"]:
            raise ValueError("Metric and archived PNG differ")
        if not all(math.isfinite(score[k]) for k in METRICS):
            raise ValueError("Nonfinite metric")
        seen.add(key)
    if seen != set(expected):
        raise ValueError("Missing metric rows")
    out = Path("/work/deliverables") / bundle_id / benchmark / shard
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(metric_path, out / "metrics.json")
    shutil.copyfile(index_path, out / "bundle_index.json")
    csv_scores(out / "per_image.csv", metric["scores"])
    metadata = Path("/work/offline_inputs") / bundle_id
    for name in ("generation_source.zip", "source_hashes.json", "plan.json", "orchestration_source.zip",
                 "orchestration_hashes.json", benchmark + "_seed42.jsonl"):
        shutil.copyfile(metadata / name, out / name)
    write(out / "workflow.json", workflow)
    write(out / "verification.json", dict(archive_hashes_verified=True, member_hashes_and_crc_verified=True,
                                          metric_png_binding_verified=True, generated_images=len(cases) * 2))
    result = ready_file(out, benchmark=benchmark, shard=shard, bundle_id=bundle_id,
                        generated_images=len(cases) * 2, archive_index=index_path.relative_to("/work").as_posix())
    runs.commit()
    return result


def merge_cloud(bundle_id, benchmark, shards, runs):
    """Merge only XLabs shards, preserving each workspace's archive location."""
    from .evaluation import merge
    runs.reload()
    folder = Path("/work/deliverables") / bundle_id / benchmark / "combined"
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for shard in shards:
        path = folder / "inputs" / (shard["profile"] + "-" + shard["ready"]["shard"] + ".json")
        write(path, shard["metrics"])
        paths.append(path)
    plan = read(Path("/work/bundles") / bundle_id / "plan.json")
    result = merge(paths, plan, benchmark, folder / "scores.json")
    csv_scores(folder / "per_image.csv", result["scores"])
    write(folder / "summary.json", dict(protocol=result["protocol"], summary=result["summary"],
          generation_provenance=result["generation_provenance"], scored_images=len(result["scores"])))
    write(folder / "shards.json", [dict(profile=s["profile"], ready=s["ready"]) for s in shards])
    result = ready_file(folder, benchmark=benchmark, bundle_id=bundle_id, scored_images=len(result["scores"]))
    runs.commit()
    return result
