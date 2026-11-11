"""Additional H100 workers using the exact frozen DreamBench renderer."""
from pathlib import Path

PLUS_APP = "xlabs-application-plus-v1"


def build_app(source, cache_name):
    import modal
    from .config import RUNS, SECRET
    source = Path(source)
    runs = modal.Volume.from_name(RUNS)
    cache = modal.Volume.from_name(cache_name)
    secret = modal.Secret.from_name(SECRET, required_keys=["HF_TOKEN"])
    env = dict(HF_HOME="/cache/huggingface", HF_HUB_ENABLE_HF_TRANSFER="1", PYTHONUNBUFFERED="1")
    cpu = (modal.Image.debian_slim(python_version="3.11").pip_install("Pillow==11.3.0")
           .add_local_dir(source, "/root", copy=True))
    image = (modal.Image.debian_slim(python_version="3.11").apt_install("git")
             .pip_install_from_requirements(source / "xlabs_application/requirements-generation.txt")
             .env(env).add_local_dir(source, "/root", copy=True))
    eval_image = (modal.Image.debian_slim(python_version="3.11").apt_install("git")
                  .pip_install_from_requirements(source / "xlabs_application/requirements-evaluation.txt")
                  .env(dict(PYTHONUNBUFFERED="1", TORCH_HOME="/cache/eval-torch", CLIP_DOWNLOAD_ROOT="/cache/eval-clip"))
                  .add_local_dir(source, "/root", copy=True))
    app = modal.App(PLUS_APP)

    @app.function(name="generate_h100", serialized=True, image=image, secrets=[secret],
                  volumes={"/cache": cache, "/work": runs}, gpu="H100", cpu=(1, 1),
                  memory=(49152, 65536), timeout=21600, retries=0, max_containers=2,
                  scaledown_window=2, single_use_containers=True)
    def generate(bundle_id, cases, invocation, seconds_cap):
        from xlabs_application.runtime import generate as execute
        return execute(bundle_id, cases, invocation, seconds_cap, "H100", runs, cache, False, False)

    @app.function(name="evaluate", serialized=True, image=eval_image,
                  volumes={"/cache": cache, "/work": runs}, gpu="T4", cpu=(1, 1),
                  memory=(8192, 8192), timeout=7200, retries=0, max_containers=1,
                  scaledown_window=2, single_use_containers=True)
    def evaluate(bundle_id, cases):
        from xlabs_application.evaluation import evaluate as execute
        return execute(bundle_id, cases, runs, cache)

    @app.function(name="export", serialized=True, image=cpu, volumes={"/work": runs},
                  cpu=(1, 1), memory=(2048, 4096), timeout=7200, retries=0,
                  max_containers=1, scaledown_window=2)
    def export(bundle_id, cases, shard):
        from xlabs_application.artifacts import export as execute
        return execute(bundle_id, cases, shard, runs)

    @app.function(name="workflow", serialized=True, image=cpu, volumes={"/work": runs},
                  cpu=(.125, .125), memory=(512, 512), timeout=86400, retries=0,
                  max_containers=2, scaledown_window=2)
    def workflow(bundle_id, cases, shard, admission=30.0):
        import traceback
        from xlabs_application.config import worker_rate
        from xlabs_application.io import write
        state = dict(status="generating", shard=shard, children=[], generations=[], generation_cap_usd=admission)
        path = Path("/work/plus_workflows") / bundle_id / (shard + ".json")

        def publish():
            write(path, state)
            runs.commit()

        try:
            spent = 0
            for attempt in range(2):
                invocation = shard + "-" + str(attempt)
                state["active_invocation"] = invocation
                seconds_cap = min(21000, (admission - spent) / worker_rate("H100") - 30)
                if seconds_cap < 120:
                    state["status"] = "budget_checkpoint"
                    publish()
                    return state
                child = generate.spawn(bundle_id, cases, invocation, seconds_cap)
                state["children"].append(child.object_id)
                publish()
                result = child.get()
                state["generations"].append({k: v for k, v in result.items() if k != "events"})
                spent += result["estimated_compute_usd"]
                state["generation_estimate_usd"] = spent
                if result["status"] == "complete":
                    break
                if result["status"] != "budget_checkpoint" or attempt == 1:
                    state["status"] = result["status"]
                    publish()
                    return state
            state["status"] = "evaluating"
            child = evaluate.spawn(bundle_id, cases)
            state["children"].append(child.object_id)
            publish()
            state["evaluation"] = child.get()
            state["status"] = "exporting"
            child = export.spawn(bundle_id, cases, shard)
            state["children"].append(child.object_id)
            publish()
            state["export"] = child.get()
            state["status"] = "complete"
            publish()
            return state
        except Exception:
            state.update(status="failed", error=traceback.format_exc()[-10000:])
            publish()
            raise

    return app
