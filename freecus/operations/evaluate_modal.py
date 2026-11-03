"""Submit selected-reference CLIP/DINO scoring in the established T4 environment."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", choices=["dreambench", "dreambench_plus"], required=True)
    p.add_argument("--status", action="store_true", help="Inspect and download scores when complete")
    a = p.parse_args()
    from freecus.modal_client import connect, BASE, read
    from freecus.modal_remote import RUNS_NAME, CACHE_NAME
    primary = a.benchmark == "dreambench"
    profile = "method-comparison" if primary else "method-comparison-dbpp"
    state = BASE / ("outputs/modal" if primary else f"outputs/modal-{profile}")
    plan = read(state / f"latest-{a.benchmark}.json")
    name = "dreambench" if primary else "dreambench_plus_no_style"
    receipt_path = state / "evaluation.json"
    modal = connect(profile)
    runs = modal.Volume.from_name(RUNS_NAME)
    cache = modal.Volume.from_name(CACHE_NAME)
    if a.status:
        receipt = read(receipt_path)
        try:
            result = modal.FunctionCall.from_id(receipt["call_id"]).get(timeout=0)
        except TimeoutError:
            print("Evaluation is running remotely:", receipt["call_id"])
            return
        folder = BASE / "outputs/metrics"
        folder.mkdir(parents=True, exist_ok=True)
        destination = folder / (name + "_scores.json")
        data = b"".join(runs.read_file(result["scores"]))
        if destination.exists() and destination.read_bytes() != data:
            raise ValueError("Local score output already exists and differs")
        destination.write_bytes(data)
        print(json.dumps(dict(status="complete", output=str(destination.resolve()), **result), indent=2))
        return
    if receipt_path.exists():
        raise ValueError("Evaluation already submitted; use --status")
    image = (modal.Image.debian_slim(python_version="3.11").apt_install("git")
             .pip_install("torch==2.7.1", "torchvision==0.22.1", "numpy==1.26.4", "Pillow==11.3.0",
                          "ftfy==6.3.1", "regex", "tqdm", "packaging")
             .pip_install("git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6")
             .env(dict(TORCH_HOME="/cache/eval-torch", CLIP_DOWNLOAD_ROOT="/cache/eval-clip", PYTHONUNBUFFERED="1")))
    for relative in ["__init__.py", "manifest.py", "run.py", "evaluate.py", "operations/assemble_run.py",
                     "project_metrics/benchmark_metrics.py", "project_metrics/SOURCE.json"]:
        image = image.add_local_file(BASE / relative, "/root/freecus/" + relative)
    app = modal.App("freecus-metric-evaluation")

    @app.function(name="evaluate", serialized=True, image=image, gpu="T4", cpu=(2, 2),
                  memory=(8192, 8192), volumes={"/work": runs, "/cache": cache},
                  timeout=3600, retries=0, max_containers=1, scaledown_window=2,
                  single_use_containers=True)
    def evaluate(bundle_id, selection, name):
        import json
        from pathlib import Path
        from freecus.operations.assemble_run import assemble_lanes
        from freecus.evaluate import project_evaluator, validate
        from freecus.run import write_json
        runs.reload()
        cache.reload()
        bundle = Path("/work/bundles") / bundle_id
        folder = Path("/work/metrics") / bundle_id / name
        scores = folder / "scores.json"
        if scores.exists():
            result = json.loads(scores.read_text())
            return dict(scores=scores.relative_to("/work").as_posix(), cases=len(result["scores"]), summary=result["summary"])
        plan = assemble_lanes(Path("/work/results") / bundle_id, bundle / "manifest.jsonl", bundle / "input",
                              Path("/work/evaluation_inputs") / bundle_id / name, selection)
        write_json(folder / "plan.json", plan)
        runs.commit()
        evaluator = project_evaluator()
        result = evaluator.evaluate(plan, scores, "cuda", 32, validator=validate,
                                    checkpoint=folder / "checkpoint.json", commit=lambda result: runs.commit())
        result["evaluation_selection"] = plan["selection"]
        result["generation_fingerprint"] = json.loads((Path(plan["freecus_run"]) / "run.json").read_text())["fingerprint"]
        write_json(scores, result)
        cache.commit()
        runs.commit()
        return dict(scores=scores.relative_to("/work").as_posix(), cases=len(result["scores"]), summary=result["summary"])

    with modal.enable_output():
        app.deploy()
    call = evaluate.spawn(plan["bundle_id"], "all" if primary else "dbpp-no-style", name)
    receipt = dict(call_id=call.object_id, benchmark=a.benchmark, profile=profile, gpu="T4",
                   samples=750 if primary else 1170, app=app.name)
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    print("Submitted remotely; the local PC can be turned off. Use the same command with --status to collect scores.")


if __name__ == "__main__":
    main()
