"""Publish benchmark bundles and restore tracked ZIPs without Modal credentials."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from .evaluation import merge
from .io import extract, read, sha, write
from .vendor.artifact_bundle import verify
from .cloud_runtime import csv_scores

BASE = Path(__file__).resolve().parent
BENCHMARKS = {"dreambench": 750, "dreambench_plus": 1350}
SHARDS = {"dreambench": ["method-comparison", "wj-lee-louis"],
          "dreambench_plus": ["dreambench-plus-0", "dreambench-plus-1"]}


def validate_catalog(folder, index):
    rows = [json.loads(line) for line in (folder / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line]
    benchmark = index["benchmark"]
    if len(rows) != BENCHMARKS[benchmark] or len({r["case_id"] for r in rows}) != len(rows):
        raise ValueError("Manifest completeness differs")
    if any(r["benchmark"] != benchmark for r in rows):
        raise ValueError("Unexpected benchmark in manifest")
    expected = {(condition, r["case_id"]): r for r in rows for condition in ("baseline", "source_target")}
    archived = {}
    for lane in index["lanes"]:
        part = read(folder / lane["index"])
        if part["bundle_id"] != index["bundle_id"] or part["benchmark"] != benchmark:
            raise ValueError("Archive identity differs")
        if part["generated_images"] != len(part["case_ids"]) * 2:
            raise ValueError("Archive image count differs")
        for case in part["case_ids"]:
            for condition in ("baseline", "source_target"):
                key = condition, case
                if key not in expected or key in archived:
                    raise ValueError("Unexpected/duplicate archived case")
                row = expected[key]
                for name, digest in ((row["reference"], row["reference_sha256"]),
                                     (row["conditioning"], row["conditioning_sha256"])):
                    if part["members"][name]["sha256"] != digest:
                        raise ValueError("Archived reference/LLM cache differs from manifest")
                png = f"results/{condition}/{benchmark}/{case}.png"
                archived[key] = part["members"][png]["sha256"]
    if set(archived) != set(expected):
        raise ValueError("Incomplete archived benchmark")
    metrics = read(folder / "metrics.json")
    seen = set()
    for score in metrics["scores"]:
        key = score["condition"], score["case_id"]
        if key not in expected or key in seen or score["generated_sha256"] != archived[key]:
            raise ValueError("Missing/duplicate score or metric/PNG hash mismatch")
        if any(score[k] != expected[key][k] for k in ("reference_id", "reference_sha256", "target_prompt")):
            raise ValueError("Metric prompt/reference differs")
        seen.add(key)
    if seen != set(expected):
        raise ValueError("Incomplete metric benchmark")
    return dict(generated_images=len(archived), scored_images=len(seen), manifest_cases=len(rows),
                by_condition=dict(Counter(condition for condition, _ in seen)))


def assemble(benchmark, bundle):
    folder = BASE / "results" / benchmark / bundle
    lanes = [folder / "lanes" / shard for shard in SHARDS[benchmark]]
    if not (folder / "ready.json").exists() or not all((lane / "local_verification.json").exists() for lane in lanes):
        raise ValueError("Download all completed cloud reports and lanes first")
    cloud_ready = read(folder / "ready.json")
    if cloud_ready["status"] != "complete" or cloud_ready["scored_images"] != BENCHMARKS[benchmark] * 2:
        raise ValueError("Cloud final report is incomplete")
    plan = read(lanes[0] / "plan.json")
    temp = BASE / "outputs/assembly" / bundle / (benchmark + ".json")
    combined = merge([lane / "metrics.json" for lane in lanes], plan, benchmark, temp)
    if combined != read(folder / "scores.json"):
        raise ValueError("Local metric merge differs from cloud final result")
    index = dict(schema="xlabs-lane-archives-v1", method="xlabs_application", benchmark=benchmark,
                 bundle_id=bundle, generated_images=BENCHMARKS[benchmark] * 2,
                 manifest_cases=BENCHMARKS[benchmark], missing_case_ids=[], generation_fingerprint=plan["code"],
                 lanes=[], parts=[])
    for lane in lanes:
        part = read(lane / "bundle_index.json")
        verification = read(lane / "local_verification.json")
        if (not verification["archives_verified"] or not verification["restored_bytes_verified"]
                or verification["bundle_index_sha256"] != sha(lane / "bundle_index.json")
                or verification["parts"] != {p["file"]: p["sha256"] for p in part["parts"]}):
            raise ValueError("Local verification is not bound to this archive index")
        index["lanes"].append(dict(shard=lane.name, index=f"lanes/{lane.name}/bundle_index.json",
                                   archive_directory=lane.name, generated_images=part["generated_images"]))
        for entry in part["parts"]:
            archive = BASE / "artifacts" / benchmark / bundle / lane.name / entry["file"]
            if archive.stat().st_size != entry["size_bytes"] or entry["size_bytes"] > 1024**3:
                raise ValueError("Archive size differs or exceeds 1 GiB")
            index["parts"].append(dict(entry, file=lane.name + "/" + entry["file"]))
    shutil.copyfile(temp, folder / "metrics.json")
    csv_scores(folder / "per_image.csv", combined["scores"])
    for name in ("generation_source.zip", "source_hashes.json", "orchestration_source.zip", "orchestration_hashes.json"):
        if len({sha(lane / name) for lane in lanes}) != 1:
            raise ValueError("Source snapshots differ between archive lanes")
        shutil.copyfile(lanes[0] / name, folder / name)
    manifest_name = benchmark + "_seed42.jsonl"
    if len({sha(lane / manifest_name) for lane in lanes}) != 1:
        raise ValueError("Manifest differs between archive lanes")
    shutil.copyfile(lanes[0] / manifest_name, folder / "manifest.jsonl")
    index["manifest_sha256"] = sha(folder / "manifest.jsonl")
    counts = validate_catalog(folder, index)
    write(folder / "bundle_index.json", index)
    write(folder / "run.json", dict(bundle_id=bundle, benchmark=benchmark, code=plan["code"],
          settings=plan["settings"], models=plan["models"], conditions=["baseline", "source_target"],
          mask_single_text_rows=False, generation_gpu="H100"))
    write(folder / "provenance.json", dict(method="xlabs_application", benchmark=benchmark, bundle_id=bundle,
          source_archive_sha256=sha(folder / "generation_source.zip"), metrics_sha256=sha(folder / "metrics.json"),
          manifest_sha256=index["manifest_sha256"], metric_protocol=combined["protocol"],
          layout="artifacts/<benchmark>/<bundle>/<shard> ZIPs; results/<benchmark>/<bundle> scores and lanes"))
    report = dict(status="complete", observed_at=datetime.now(timezone.utc).isoformat(),
                  benchmark=benchmark, bundle_id=bundle, **counts, archive_parts=len(index["parts"]),
                  archive_bytes=sum(p["size_bytes"] for p in index["parts"]),
                  archive_and_member_sha256_crc_verified=True, restored_bytes_verified=True,
                  metric_png_reference_cache_binding_verified=True, cloud_local_metric_merge_identical=True)
    write(folder / "verification.json", report)
    write(BASE / "verification" / (benchmark + ".json"), report)
    shutil.copyfile(folder / "metrics.json", BASE / "results/reports" / (benchmark + "_scores.json"))
    write(BASE / "results/reports" / (benchmark + "_summary.json"),
          dict(protocol=combined["protocol"], summary=combined["summary"], scored_images=counts["scored_images"]))
    print(json.dumps(report, indent=2), flush=True)


def verify_or_restore(benchmark, bundle, restore=False):
    folder = BASE / "results" / benchmark / bundle
    index = read(folder / "bundle_index.json")
    counts = validate_catalog(folder, index)
    for lane in index["lanes"]:
        part = read(folder / lane["index"])
        archives = BASE / "artifacts" / benchmark / bundle / lane["archive_directory"]
        verify(archives, part)
        if restore:
            target = BASE / "outputs/restored" / benchmark / bundle / lane["shard"]
            if any(not (target / name).is_file() or sha(target / name) != member["sha256"] for name, member in part["members"].items()):
                for entry in part["parts"]:
                    extract(archives / entry["file"], target)
            if any(sha(target / name) != member["sha256"] for name, member in part["members"].items()):
                raise ValueError("Restored member bytes differ")
    print(json.dumps(dict(status="verified", benchmark=benchmark, restored=restore, **counts)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["assemble", "verify", "restore"])
    parser.add_argument("--benchmark", choices=[*BENCHMARKS, "all"], default="all")
    parser.add_argument("--bundle-id")
    args = parser.parse_args()
    for benchmark in BENCHMARKS if args.benchmark == "all" else [args.benchmark]:
        candidates = [p for p in (BASE / "results" / benchmark).iterdir() if p.is_dir() and len(p.name) == 24]
        bundle = args.bundle_id or (candidates[0].name if len(candidates) == 1 else None)
        if bundle is None:
            raise ValueError("Specify the bundle ID")
        if args.command == "assemble":
            assemble(benchmark, bundle)
        else:
            verify_or_restore(benchmark, bundle, args.command == "restore")


if __name__ == "__main__":
    main()
