"""XLabs-specific apps; existing model/result apps are never changed."""
from pathlib import Path

from .config import APP, RUNS, SECRET


def build_app(source, cache_name):
    import modal
    runs = modal.Volume.from_name(RUNS, create_if_missing=True, version=2)
    cache = modal.Volume.from_name(cache_name)
    secret = modal.Secret.from_name(SECRET, required_keys=["HF_TOKEN"])
    env = dict(HF_HOME="/cache/huggingface", HF_HUB_ENABLE_HF_TRANSFER="1", PYTHONUNBUFFERED="1")
    source = Path(source)
    cpu_image = (modal.Image.debian_slim(python_version="3.11")
                 .pip_install("huggingface-hub==0.34.3", "hf-transfer==0.1.9", "Pillow==11.3.0")
                 .env(env).add_local_dir(source, "/root"))
    image = (modal.Image.debian_slim(python_version="3.11").apt_install("git")
             .pip_install_from_requirements(source / "xlabs_application/requirements-generation.txt")
             .env(env).add_local_dir(source, "/root"))
    app = modal.App(APP)

    @app.function(name="stage", serialized=True, image=cpu_image, secrets=[secret],
                  volumes={"/cache": cache, "/work": runs}, cpu=(1, 1), memory=(4096, 8192),
                  timeout=3600, retries=0, max_containers=1, scaledown_window=2)
    def stage(bundle_id):
        from xlabs_application.runtime import stage as execute
        return execute(bundle_id, runs, cache)

    def register_generator(gpu, name):
        @app.function(name=name, serialized=True, image=image, secrets=[secret],
                      volumes={"/cache": cache, "/work": runs}, gpu=gpu, cpu=(1, 1),
                      memory=(49152, 65536), timeout=21600, retries=0, max_containers=1,
                      scaledown_window=2, single_use_containers=True)
        def generate(bundle_id, cases, invocation, seconds_cap, pilot=False, parity=False):
            from xlabs_application.runtime import generate as execute
            return execute(bundle_id, cases, invocation, seconds_cap, gpu, runs, cache, pilot, parity)
        return generate

    register_generator("L40S", "generate_l40s")
    register_generator("H100", "generate_h100")

    eval_image = (modal.Image.debian_slim(python_version="3.11").apt_install("git")
                  .pip_install_from_requirements(source / "xlabs_application/requirements-evaluation.txt")
                  .env(dict(PYTHONUNBUFFERED="1", TORCH_HOME="/cache/eval-torch", CLIP_DOWNLOAD_ROOT="/cache/eval-clip"))
                  .add_local_dir(source, "/root"))

    @app.function(name="evaluate", serialized=True, image=eval_image,
                  volumes={"/cache": cache, "/work": runs}, gpu="T4", cpu=(1, 1),
                  memory=(8192, 8192), timeout=3600, retries=0, max_containers=1,
                  scaledown_window=2, single_use_containers=True)
    def evaluate(bundle_id, cases):
        from xlabs_application.evaluation import evaluate as execute
        return execute(bundle_id, cases, runs, cache)

    @app.function(name="export", serialized=True, image=cpu_image, volumes={"/work": runs},
                  cpu=(1, 1), memory=(2048, 4096), timeout=3600, retries=0,
                  max_containers=1, scaledown_window=2)
    def export(bundle_id, cases, shard):
        from xlabs_application.artifacts import export as execute
        return execute(bundle_id, cases, shard, runs)

    @app.function(name="workflow", serialized=True, image=cpu_image, volumes={"/work": runs},
                  cpu=(0.125, 0.125), memory=(512, 512), timeout=86400, retries=0,
                  max_containers=1, scaledown_window=2)
    def workflow(bundle_id, cases, invocation, seconds_cap, shard, gpu="L40S"):
        from xlabs_application.io import read, write
        path = Path("/work/workflows") / bundle_id / (invocation + ".json")
        status = dict(status="generating", invocation=invocation, children=[])

        def publish():
            write(path, status)
            runs.commit()

        try:
            generator = modal.Function.from_name(APP, "generate_" + gpu.lower())
            child = generator.spawn(bundle_id, cases, invocation, seconds_cap, False, False)
            status["children"].append(child.object_id)
            publish()
            result = child.get()
            status["generation"] = {k: v for k, v in result.items() if k != "events"}
            if result["status"] != "complete":
                status["status"] = result["status"]
                publish()
                return status
            status["status"] = "evaluating"
            child = evaluate.spawn(bundle_id, cases)
            status["children"].append(child.object_id)
            publish()
            status["evaluation"] = child.get()
            status["status"] = "exporting"
            child = export.spawn(bundle_id, cases, shard)
            status["children"].append(child.object_id)
            publish()
            status["export"] = child.get()
            status["status"] = "complete"
            publish()
            return status
        except Exception:
            import traceback
            status["status"] = "failed"
            status["error"] = traceback.format_exc()[-10000:]
            publish()
            raise

    return app
