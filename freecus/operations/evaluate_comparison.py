"""Detached scoring of completed DreamO/UNO runs with the exact FreeCus evaluator."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROFILES = {
    ("dreamo", "dreambench"): "method-comparison",
    ("dreamo", "dreambench_plus"): "method-comparison-dbpp",
    ("uno", "dreambench"): "wj-lee-louis",
    ("uno", "dreambench_plus"): "woojin716",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["dreamo", "uno"], required=True)
    parser.add_argument("--benchmark", choices=["dreambench", "dreambench_plus"], required=True)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    profile = PROFILES[args.method, args.benchmark]
    state = ROOT / args.method / "outputs/modal" / profile
    plan = json.loads((state / (args.benchmark + ".json")).read_text())
    workflow = json.loads((state / (args.benchmark + "-workflow-status.json")).read_text())
    if workflow["status"] != "complete" or workflow["bundle_id"] != plan["bundle_id"]:
        raise ValueError("Inspect generation workflow-status; complete matching generation required")
    from freecus.modal_client import connect
    from freecus.run import write_json
    modal = connect(profile)
    runs = modal.Volume.from_name(args.method + "-runs")
    cache = modal.Volume.from_name("freecus-method-comparison-models" if args.method == "dreamo" else "uno-models")
    receipt_path = state / (args.benchmark + "-evaluation.json")
    if args.status:
        receipt = json.loads(receipt_path.read_text())
        try:
            result = modal.FunctionCall.from_id(receipt["call_id"]).get(timeout=0)
        except TimeoutError:
            progress = dict(status="running", **receipt)
            checkpoint = "metrics/" + plan["bundle_id"] + "/" + args.benchmark + "-all/checkpoint.json"
            try:
                partial = json.loads(b"".join(runs.read_file(checkpoint)))
                progress.update(phase="scoring", scored=len(partial["scores"]))
            except FileNotFoundError:
                progress["phase"] = "artifact_validation_or_model_setup"
            except json.JSONDecodeError:
                progress["phase"] = "checkpoint_publication"
            print(json.dumps(progress))
            return
        destination = ROOT / args.method / "outputs/metrics" / (args.benchmark + "_scores.json")
        data = b"".join(runs.read_file(result["scores"]))
        if destination.exists() and destination.read_bytes() != data:
            raise ValueError("Preserve existing local scores; output differs")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        write_json(state / (args.benchmark + "-evaluation-result.json"), result)
        print(json.dumps(dict(status="complete", output=str(destination.resolve()), **result), indent=2))
        return
    if receipt_path.exists():
        raise ValueError("Already submitted; use --status")
    image = (modal.Image.debian_slim(python_version="3.11").apt_install("git")
             .pip_install("torch==2.7.1", "torchvision==0.22.1", "numpy==1.26.4", "Pillow==11.3.0",
                          "ftfy==6.3.1", "regex", "tqdm", "packaging")
             .pip_install("git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6")
             .env(dict(TORCH_HOME="/cache/eval-torch", CLIP_DOWNLOAD_ROOT="/cache/eval-clip", PYTHONUNBUFFERED="1")))
    paths = ["freecus/" + name for name in ["__init__.py", "manifest.py", "run.py", "evaluate.py",
             "operations/assemble_run.py", "project_metrics/benchmark_metrics.py", "project_metrics/SOURCE.json"]]
    paths += [args.method + "/__init__.py", args.method + "/evaluate.py"]
    for relative in paths:
        image = image.add_local_file(ROOT / relative, "/root/" + relative)
    app = modal.App(args.method + "-metric-evaluation")

    @app.function(name="evaluate", serialized=True, image=image, gpu="T4", cpu=(2, 2),
                  memory=(8192, 8192), volumes={"/work": runs, "/cache": cache},
                  timeout=3600, retries=0, max_containers=1, scaledown_window=2,
                  single_use_containers=True)
    def evaluate(bundle_id, generation_gpu, method, benchmark):
        import importlib
        import json
        from pathlib import Path
        from freecus.operations.assemble_run import assemble_lanes
        from freecus.evaluate import project_evaluator
        from freecus.run import write_json
        adapter = importlib.import_module(method + ".evaluate")
        runs.reload()
        cache.reload()
        bundle = Path("/work/bundles") / bundle_id
        folder = Path("/work/metrics") / bundle_id / (benchmark + "-all")
        scores = folder / "scores.json"
        if scores.exists():
            result = json.loads(scores.read_text())
            return dict(scores=scores.relative_to("/work").as_posix(), cases=len(result["scores"]), summary=result["summary"])
        assembled = Path("/work/evaluation_inputs") / bundle_id / (benchmark + "-all")
        evaluation = assemble_lanes(Path("/work/results") / bundle_id / generation_gpu,
                                    bundle / "manifest.jsonl", bundle / "input", assembled, "all")
        for record in evaluation["records"]:
            record["condition"] = method
        write_json(folder / "plan.json", evaluation)
        runs.commit()
        evaluator = project_evaluator()
        result = evaluator.evaluate(evaluation, scores, "cuda", 32, validator=adapter.validate,
                                    checkpoint=folder / "checkpoint.json", commit=lambda result: runs.commit())
        result["evaluation_selection"] = evaluation["selection"]
        result["generation_fingerprint"] = json.loads((assembled / "run.json").read_text())["fingerprint"]
        result["method"] = method
        write_json(scores, result)
        cache.commit()
        runs.commit()
        return dict(scores=scores.relative_to("/work").as_posix(), cases=len(result["scores"]), summary=result["summary"])

    with modal.enable_output():
        app.deploy()
    call = evaluate.spawn(plan["bundle_id"], workflow["selected_gpu"], args.method, args.benchmark)
    receipt = dict(call_id=call.object_id, benchmark=args.benchmark, method=args.method, profile=profile,
                   gpu="T4", selection="all", samples=750 if args.benchmark == "dreambench" else 1350, app=app.name)
    write_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
