"""Evaluate FreeCus with the exact established local selected-reference evaluator."""
import argparse
import importlib.util
import json
from pathlib import Path

from .manifest import load, sha256
from .run import artifact_paths, complete


def project_evaluator():
    folder = Path(__file__).parent / "project_metrics"
    source = json.loads((folder / "SOURCE.json").read_text(encoding="utf-8"))
    path = folder / "benchmark_metrics.py"
    if sha256(path) != source["sha256"]:
        raise ValueError("Established evaluator source changed")
    spec = importlib.util.spec_from_file_location("freecus_project_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(manifest, root, run, evaluator, selection="all"):
    rows = load(manifest, root)
    from collections import Counter
    counts = Counter(row["benchmark"] for row in rows)
    expected = {"dreambench": 750, "dreambench_plus": 1350}
    if any(name not in expected or count != expected[name] for name, count in counts.items()):
        raise ValueError("Primary evaluation requires every case in each selected benchmark")
    if len({row.get("case_id") for row in rows}) != len(rows):
        raise ValueError("Primary evaluation requires unique project case IDs")
    excluded = []
    if selection == "dbpp-no-style":
        if counts != {"dreambench_plus": 1350}:
            raise ValueError("dbpp-no-style requires the original complete 1350-case manifest")
        excluded = [r for r in rows if r["category"] == "style"]
        if len(excluded) != 180 or len({r["reference_id"] for r in excluded}) != 20:
            raise ValueError("Expected exactly 20 style references / 180 cases")
        rows = [r for r in rows if r["category"] != "style"]
    elif selection != "all":
        raise ValueError("Unknown evaluation selection")
    records = []
    for row in rows:
        if row["seed"] != 42 or row.get("reference_policy") != "selected_one":
            raise ValueError("Primary project comparison requires seed 42 and selected_one references")
        if row["evaluation_references"] != [row["reference"]]:
            raise ValueError("Primary project evaluation requires exactly the conditioning reference")
        metric_row = dict(case_id=row["case_id"], reference_id=row["reference_id"],
                          benchmark=row["benchmark"], target_prompt=row["prompt"])
        paths = artifact_paths(run, row)
        records.append(dict(condition="freecus", category=evaluator.category(metric_row),
                            row=metric_row, reference=str((root / row["reference"]).resolve()),
                            reference_sha256=row["reference_sha256"],
                            generated=str(paths["tgt_image"].resolve()), freecus_id=row["id"]))
    return dict(schema_version=1, protocol=evaluator.PROTOCOL, records=records,
                selection=dict(name=selection, included_cases=len(rows), excluded_cases=len(excluded),
                               excluded_case_ids=[r["case_id"] for r in excluded],
                               excluded_reference_ids=sorted({r["reference_id"] for r in excluded})),
                manifest=str(manifest.resolve()), manifest_sha256=sha256(manifest),
                freecus_run=str(run.resolve()), data_root=str(root.resolve()))


def validate(plan, account=None):
    if account is not None:
        raise ValueError("Account selection belongs to the later Modal integration")
    run = Path(plan["freecus_run"])
    info = json.loads((run / "run.json").read_text(encoding="utf-8"))
    if sha256(plan["manifest"]) != plan["manifest_sha256"] or info["manifest_sha256"] != plan["manifest_sha256"]:
        raise ValueError("Run and evaluation manifests differ")
    if (info["config"]["width"], info["config"]["height"]) != (1024, 1024) or info["config"]["seeds"] != [42]:
        raise ValueError("Primary comparison requires 1024x1024, seed 42")
    expected = prepare(Path(plan["manifest"]), Path(plan["data_root"]), run,
                       project_evaluator(), plan["selection"]["name"])
    if expected["records"] != plan["records"] or expected["selection"] != plan["selection"]:
        raise ValueError("Evaluation selection differs from the canonical manifest")
    from PIL import Image
    for record in plan["records"]:
        paths = artifact_paths(run, dict(id=record["freecus_id"]))
        if not complete(paths):
            raise ValueError(f"Missing or corrupt output: {record['freecus_id']}")
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        row = record["row"]
        if (metadata["fingerprint"] != info["fingerprint"] or metadata["seed"] != 42
                or metadata["case_id"] != row["case_id"] or metadata["prompt"] != row["target_prompt"]
                or metadata["reference_id"] != row["reference_id"]):
            raise ValueError("Output does not belong to the expected project case")
        if sha256(record["reference"]) != record["reference_sha256"]:
            raise ValueError("Reference changed")
        with Image.open(record["generated"]) as image:
            image.verify()
            if image.size != (1024, 1024):
                raise ValueError("Generated image is not 1024x1024")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--validate-only", action="store_true", help="Check every artifact without loading metric models")
    p.add_argument("--selection", choices=["all", "dbpp-no-style"], default="all")
    a = p.parse_args()
    if a.batch_size < 1:
        p.error("batch-size must be positive")
    evaluator = project_evaluator()
    plan = prepare(a.manifest, a.data_root, a.run, evaluator, a.selection)
    plan["data_root"] = str(a.data_root.resolve())
    if a.dry_run:
        print(json.dumps(dict(protocol=plan["protocol"], selection=plan["selection"], expected=len(plan["records"]),
                              generated_present=sum(Path(r["generated"]).is_file() for r in plan["records"])), indent=2))
        return
    if a.validate_only:
        validate(plan)
        print(json.dumps(dict(status="validated", selection=a.selection, cases=len(plan["records"]))))
        return
    if a.output.exists():
        raise FileExistsError("Choose a new output path to preserve previous scores")
    from .run import write_json
    plan_path = a.output.with_name(a.output.stem + ".plan.json")
    write_json(plan_path, plan)
    result = evaluator.evaluate(plan, a.output, a.device, a.batch_size, validator=validate,
                               checkpoint=a.output.with_name(a.output.stem + ".checkpoint.json"))
    result["evaluation_selection"] = plan["selection"]
    result["evaluation_plan"] = str(plan_path.resolve())
    write_json(a.output, result)


if __name__ == "__main__":
    main()
