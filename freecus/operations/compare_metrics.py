"""Reaggregate existing methods on exactly the FreeCus evaluation case sets."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from freecus.evaluate import project_evaluator
from freecus.manifest import PROJECT, load, sha256
from freecus.run import write_json


def export_csv(report, folder):
    with (folder / "comparison_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["benchmark", "condition", "cases", "references", "CLIP-T", "CLIP-I", "DINO-I"])
        writer.writeheader()
        for key, item in sorted(report["summary"].items()):
            benchmark, condition = key.split("/")
            writer.writerow(dict(benchmark=benchmark, condition=condition, cases=item["cases"],
                                 references=item["subjects"], **item["subject_macro"]))
    with (folder / "comparison_case_scores.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["benchmark", "condition", "case_id", "reference_id", "category",
                                                   "CLIP-T", "CLIP-I", "DINO-I", "target_prompt"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(report["scores"], key=lambda r: (r["benchmark"], r["case_id"], r["condition"])))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prepare-only", action="store_true", help="Prepare matched existing-method scores before FreeCus scoring")
    a = p.parse_args()
    base = Path(__file__).resolve().parents[1]
    evaluator = project_evaluator()
    rows = []
    for benchmark in ["dreambench", "dreambench_plus"]:
        rows += [r for r in load(base / f"manifests/{benchmark}_project_seed42.jsonl", PROJECT)
                 if not (benchmark == "dreambench_plus" and r["category"] == "style")]
    canonical = {(r["benchmark"], r["case_id"]): r for r in rows}
    paths = [PROJECT / "Orthogonalization-InstantX-IP-Adapter/outputs/benchmark_metrics_v1" / f"scores_{part}.json"
             for part in ["A", "B"]]
    if not a.prepare_only:
        paths += [base / "outputs/metrics" / f"{name}_scores.json"
                  for name in ["dreambench", "dreambench_plus_no_style"]]
    selected, seen, sources = [], set(), []
    for path in paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["protocol"] != evaluator.PROTOCOL:
            raise ValueError(f"Metric protocol differs: {path}")
        sources.append(dict(path=str(path.resolve()), sha256=sha256(path),
                            environment={k: result.get(k) for k in ["device", "device_name", "versions"]}))
        for score in result["scores"]:
            key = (score["benchmark"], score["case_id"])
            if key not in canonical:
                if score["benchmark"] == "dreambench_plus" and score["category"] == "style":
                    continue
                raise ValueError("Unknown case in score file")
            row = canonical[key]
            if (score["reference_id"] != row["reference_id"] or score["reference_sha256"] != row["reference_sha256"]
                    or score["target_prompt"] != row["prompt"]):
                raise ValueError(f"Reference/prompt mismatch: {key}")
            identity = key + (score["condition"],)
            if identity in seen:
                raise ValueError(f"Duplicate score: {identity}")
            seen.add(identity)
            selected.append(score)
    counts = Counter((s["benchmark"], s["condition"]) for s in selected)
    conditions = ["baseline", "source_target_ssg"] + ([] if a.prepare_only else ["freecus"])
    for benchmark, expected in [("dreambench", 750), ("dreambench_plus", 1170)]:
        for condition in conditions:
            if counts[benchmark, condition] != expected:
                raise ValueError(f"Missing or extra scores: {benchmark}/{condition}")
    summary = evaluator.summarize(selected)
    deltas = {}
    if not a.prepare_only:
        for benchmark in ["dreambench", "dreambench_plus"]:
            ours = summary[benchmark + "/source_target_ssg"]["subject_macro"]
            freecus = summary[benchmark + "/freecus"]["subject_macro"]
            deltas[benchmark] = {metric: ours[metric] - freecus[metric] for metric in evaluator.METRICS}
    output = base / "outputs/metrics" / ("existing_matched_cohorts.json" if a.prepare_only else "comparison.json")
    if output.exists():
        raise FileExistsError(f"Preserving existing comparison: {output}")
    write_json(output, dict(protocol=evaluator.PROTOCOL, selection=dict(dreambench="all 750 cases",
               dreambench_plus="non-style only: 130 references / 1170 cases; excludes all 20 style references / 180 cases"),
               sources=sources, summary=summary, ours_minus_freecus_subject_macro=deltas, scores=selected))
    if not a.prepare_only:
        export_csv(json.loads(output.read_text(encoding="utf-8")), output.parent)
    print(json.dumps(dict(output=str(output.resolve()), summary=summary), indent=2))


if __name__ == "__main__":
    main()
