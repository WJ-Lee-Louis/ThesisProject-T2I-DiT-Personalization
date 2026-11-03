"""Reuse FreeCus' exact evaluator, with full DreamBench++ by default and optional style exclusion."""
import argparse
import copy
from pathlib import Path
from freecus.evaluate import prepare, project_evaluator, validate as shared_validate
from freecus.manifest import PROJECT
from freecus.run import write_json


def validate(plan, account=None):
    view = copy.deepcopy(plan)
    for record in view["records"]:
        if record["condition"] != "uno":
            raise ValueError("Unexpected condition in UNO scores")
        record["condition"] = "freecus"
    shared_validate(view, account)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", choices=["dreambench", "dreambench_plus"], required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=PROJECT)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--selection", choices=["all", "dbpp-no-style"], default="all")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.batch_size < 1:
        p.error("batch-size must be positive")
    manifest = a.manifest or PROJECT / f"freecus/manifests/{a.benchmark}_project_seed42.jsonl"
    evaluator = project_evaluator()
    plan = prepare(manifest, a.data_root, a.run, evaluator, a.selection)
    for record in plan["records"]:
        record["condition"] = "uno"
    if a.dry_run:
        print(dict(expected=len(plan["records"]), selection=plan["selection"]["name"]))
        return
    if a.validate_only:
        validate(plan)
        print("Validated", len(plan["records"]))
        return
    if a.output.exists():
        raise FileExistsError(a.output)
    write_json(a.output.with_suffix(".plan.json"), plan)
    result = evaluator.evaluate(plan, a.output, a.device, a.batch_size, validator=validate,
                                checkpoint=a.output.with_suffix(".checkpoint.json"))
    result["evaluation_selection"] = plan["selection"]
    write_json(a.output, result)


if __name__ == "__main__":
    main()
