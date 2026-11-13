"""Detached Modal staging and single-GPU lanes. Importing this module starts no GPU."""
from pathlib import Path

APP = "uno-method-comparison"
CACHE = "uno-models"
RUNS = "uno-runs"
SECRET = "uno-huggingface"


def build_app(source, gpu="A100-80GB", *, app_name=APP,cache_name=CACHE,runs_name=RUNS,
              secret_name=SECRET,config_filename=None):
    import modal
    source = Path(source)
    cache = modal.Volume.from_name(cache_name, create_if_missing=True,
                                   **({'version':2} if cache_name==CACHE else {}))
    runs = modal.Volume.from_name(runs_name, create_if_missing=True, version=2)
    secret = modal.Secret.from_name(secret_name, required_keys=["HF_TOKEN"])
    env = dict(HF_HOME="/cache/huggingface", HF_HUB_ENABLE_HF_TRANSFER="1", PYTHONUNBUFFERED="1")
    cpu_image = (modal.Image.debian_slim(python_version="3.11")
                 .pip_install("huggingface-hub==0.29.3", "hf_transfer==0.1.9", "Pillow==11.1.0")
                 .env(env).add_local_dir(source, "/root"))
    image = (modal.Image.debian_slim(python_version="3.11")
             .add_local_dir(source, "/root", copy=True)
             .run_commands("pip install --no-cache-dir -r /root/uno/requirements-gpu.txt")
             .pip_install("hf_transfer==0.1.9").env(env))
    app = modal.App(app_name)

    @app.function(name="stage", serialized=True, image=cpu_image, secrets=[secret],
                  volumes={"/cache": cache, "/work": runs}, cpu=(2, 2), memory=(4096, 8192),
                  timeout=3600, max_containers=1, retries=0, scaledown_window=2)
    def stage(bundle_id):
        import json
        import zipfile
        from uno.models import stage as stage_models
        from freecus.manifest import sha256, load
        from freecus.run import write_json
        runs.reload()
        cache.reload()
        folder = Path("/work/bundles") / bundle_id
        folder.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(Path("/work/uploads") / (bundle_id + ".zip")) as z:
            for member in z.infolist():
                if not (folder / member.filename).resolve().is_relative_to(folder.resolve()):
                    raise ValueError("Unsafe archive path")
            z.extractall(folder)
        info = json.loads((folder / "bundle.json").read_text())
        for name, digest in info["files"].items():
            if sha256(folder / name) != digest:
                raise ValueError("Input archive integrity check failed")
        load(folder / "manifest.jsonl", folder / "input")
        lock = json.loads(Path("/root/uno/models.lock.json").read_text())
        try:
            stage_models(lock, folder / "resolved_models.json")
        finally:
            cache.commit()
        write_json(folder / "stage.json", dict(status="ready", bundle_id=bundle_id))
        runs.commit()
        return dict(status="ready", bundle_id=bundle_id)

    def register_lane(gpu_kind, name):
        @app.function(name=name, serialized=True, image=image, secrets=[secret],
                      volumes={"/cache": cache, "/work": runs}, gpu=gpu_kind, cpu=(2, 2), memory=(65536, 65536),
                      timeout=21600, max_containers=2, retries=0, scaledown_window=2, single_use_containers=True)
        def lane(bundle_id, benchmark, invocation, lane, cases, seconds_cap):
            from uno.modal_app import execute_lane
            return execute_lane(bundle_id, benchmark, invocation, lane, cases, seconds_cap, gpu_kind, runs, cache,config_filename)
        return lane

    a100_lane = register_lane(gpu, "gpu_lane")
    l40s_lane = register_lane("L40S", "gpu_lane_l40s")

    @app.function(name="workflow", serialized=True, image=cpu_image, volumes={"/work": runs},
                  cpu=(0.125, 0.125), memory=(512, 512), timeout=86400,
                  max_containers=1, retries=0, scaledown_window=2)
    def workflow(workflow_id, plan, admission_usd):
        import json
        import time
        import uuid
        from PIL import Image
        from uno.efficiency import estimate_phase, choose_gpu, peak_vram
        from freecus.modal_cost import rate
        from freecus.run import write_json
        if gpu != "A100-80GB" or not 0 < admission_usd <= 38:
            raise ValueError("Autonomous baseline requires A100-80GB and at most $38 compute")
        report = Path("/work/workflows") / (workflow_id + ".json")
        state = dict(workflow_id=workflow_id, status="starting", benchmark=plan["benchmark"],
                     bundle_id=plan["bundle_id"], compute_admission_usd=admission_usd,
                     accounted_compute_usd=0.0, phases=[], started_at=time.time())

        def save():
            state["updated_at"] = time.time()
            write_json(report, state)
            runs.commit()

        def phase(mode, gpu_kind, lanes, budget):
            invocation = uuid.uuid4().hex
            item = dict(mode=mode, gpu=gpu_kind, invocation_id=invocation, budget_usd=budget, calls=[])
            state["phases"].append(item)
            state["status"] = mode + "_" + gpu_kind
            save()
            calls = []
            function = a100_lane if gpu_kind == "A100-80GB" else l40s_lane
            for lane, cases in enumerate(lanes):
                cap = min(21000, budget / 2 / rate(gpu_kind) - 25)
                call = function.spawn(plan["bundle_id"], plan["benchmark"], invocation, lane, cases, cap)
                item["calls"].append(dict(id=call.object_id, lane=lane, cases=len(cases)))
                calls.append(call)
                save()
            for index, call in enumerate(calls):
                try:
                    result = call.get()
                except Exception as exc:
                    result = dict(status="failed", error=repr(exc), estimated_compute_usd=budget / 2)
                item["calls"][index]["result"] = result
                state["accounted_compute_usd"] += result["estimated_compute_usd"]
                save()
            return item

        save()
        try:
            state["status"] = "staging"
            call = stage.spawn(plan["bundle_id"])
            state["stage_call"] = call.object_id
            save()
            state["stage"] = call.get()
            runs.reload()
            folder = Path("/work/bundles") / plan["bundle_id"]
            rows = [json.loads(line) for line in (folder / "manifest.jsonl").read_text().splitlines()]
            sizes = {}
            for row in rows:
                if row["reference_id"] not in sizes:
                    with Image.open(folder / "input" / row["reference"]) as image:
                        sizes[row["reference_id"]] = image.width * image.height
            chosen = set()
            for cases in plan["lanes"]:
                case_set = set(cases)
                available = [r for r in rows if r["case_id"] in case_set]
                for category in sorted({r["category"] for r in available}):
                    candidates = [r for r in available if r["category"] == category]
                    ref = max(candidates, key=lambda r: sizes[r["reference_id"]])["reference_id"]
                    chosen.update(r["case_id"] for r in candidates if r["reference_id"] == ref and r["prompt_index"] in {0, 4})
            pilot_lanes = [[c for c in lane if c in chosen] for lane in plan["lanes"]]
            baseline = phase("pilot", "A100-80GB", pilot_lanes, 3.0)
            if not all(c["result"]["status"] == "complete" for c in baseline["calls"]):
                state["status"] = "a100_pilot_failed"
                save()
                return state
            estimates = {"A100-80GB": estimate_phase(baseline, len(rows))}
            if peak_vram(baseline) <= 42:
                cheaper = phase("pilot", "L40S", pilot_lanes, 3.0)
                if all(c["result"]["status"] == "complete" for c in cheaper["calls"]):
                    estimates["L40S"] = estimate_phase(cheaper, len(rows))
            else:
                state["l40s_skipped"] = "A100 peak exceeds conservative 42 GiB admission threshold"
            selected = choose_gpu(estimates)
            state.update(estimates=estimates, selected_gpu=selected)
            remaining = admission_usd - state["accounted_compute_usd"]
            cost = estimates[selected]["compute_usd_with_margin"]
            if cost > remaining:
                state["status"] = "budget_review_required"
                save()
                return state
            complete = phase("full", selected, plan["lanes"], min(remaining, cost))
            state["output"] = str(Path("/work/results") / plan["bundle_id"] / selected)
            state["status"] = "complete" if all(c["result"]["status"] == "complete" for c in complete["calls"]) else "generation_failed_or_budget_paused"
            save()
            return state
        except BaseException as exc:
            state.update(status="failed", error=repr(exc))
            save()
            raise

    return app


def execute_lane(bundle_id, benchmark, invocation, lane, cases, seconds_cap, gpu, runs, cache,config_filename=None):
    import json
    import subprocess
    import resource
    import sys
    import time
    from freecus.modal_cost import rate
    from freecus.run import write_json
    if not 0 < seconds_cap <= 21000:
        raise ValueError("Invalid worker time cap")
    runs.reload()
    cache.reload()
    bundle = Path("/work/bundles") / bundle_id
    if json.loads((bundle / "stage.json").read_text())["status"] != "ready":
        raise ValueError("Complete CPU staging first")
    journal = Path("/work/invocations") / invocation / f"lane-{lane}"
    journal.mkdir(parents=True, exist_ok=False)
    # Never mix A100 and L40S images into a run labelled with only one GPU.
    output = Path("/work/results") / bundle_id / gpu / f"lane-{lane}"
    command = [sys.executable, "-m", "uno.run", "--benchmark", benchmark,
               "--manifest", str(bundle / "manifest.jsonl"), "--data-root", str(bundle / "input"),
               "--resolved-models", str(bundle / "resolved_models.json"), "--output", str(output), "--case-ids", *cases]
    if config_filename is not None:
        command.extend(['--config',str(bundle/config_filename)])
    began = time.monotonic()
    with (journal / "worker.log").open("wb") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    capped = False
    try:
        while child.poll() is None:
            elapsed = time.monotonic() - began
            write_json(journal / "state.json", dict(status="running", gpu=gpu, elapsed_seconds=elapsed, cases=len(cases)))
            runs.commit()
            if elapsed >= seconds_cap:
                child.kill()
                child.wait()
                capped = True
                break
            time.sleep(20)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
    elapsed = time.monotonic() - began
    records = [json.loads(p.read_text()) for p in (output / "metadata").rglob("*.json")]
    result = dict(status="budget_paused" if capped else "complete" if child.returncode == 0 else "failed",
                  exit_code=child.returncode, elapsed_seconds=elapsed, lane=lane, gpu=gpu,
                  estimated_compute_usd=(elapsed + 22) * rate(gpu), requested_cases=len(cases),
                  generated_total=len(records), output=str(output), timings=[r["timing"] for r in records],
                  host_peak_gib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20,
                  initialization=json.loads((output / "initialization.json").read_text())
                                 if (output / "initialization.json").exists() else None)
    if child.returncode:
        result["error_tail"] = (journal / "worker.log").read_text(errors="replace")[-6000:]
    write_json(journal / "state.json", result)
    runs.commit()
    return result
