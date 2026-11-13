"""Modal deployment factory. GPU workers run one sequential engine each."""
from pathlib import Path

APP_NAME = "freecus-method-comparison"
CACHE_NAME = "freecus-method-comparison-models"
RUNS_NAME = "freecus-method-comparison-runs"
SECRET_NAME = "freecus-huggingface"


def build_app(source_dir, gpu_type="A100-80GB", *, app_name=APP_NAME,
              cache_name=CACHE_NAME, runs_name=RUNS_NAME, secret_name=SECRET_NAME,
              worker_module="freecus.modal_worker"):
    import modal
    source_dir = Path(source_dir)
    cache = modal.Volume.from_name(cache_name, create_if_missing=True,
                                   **({'version': 2} if cache_name == CACHE_NAME else {}))
    runs = modal.Volume.from_name(runs_name, create_if_missing=True, version=2)
    secret = modal.Secret.from_name(secret_name, required_keys=["HF_TOKEN"])
    env = dict(HF_HOME="/cache/huggingface", HF_HUB_ENABLE_HF_TRANSFER="1", PYTHONUNBUFFERED="1")
    cpu_image = (modal.Image.debian_slim(python_version="3.11")
                 .pip_install("huggingface-hub==0.29.3", "hf_transfer==0.1.9")
                 .env(env).add_local_dir(source_dir, "/root/freecus"))
    gpu_image = modal.Image.debian_slim(python_version="3.11")
    for name in ["requirements.txt", "requirements-gpu.txt", "constraints.txt"]:
        gpu_image = gpu_image.add_local_file(source_dir / name, f"/tmp/freecus/{name}", copy=True)
    gpu_image = (gpu_image.run_commands("pip install --no-cache-dir -r /tmp/freecus/requirements-gpu.txt")
                 .pip_install("hf_transfer==0.1.9").env(env).add_local_dir(source_dir, "/root/freecus"))
    app = modal.App(app_name)

    @app.function(name="stage", serialized=True, image=cpu_image, secrets=[secret], volumes={"/cache": cache, "/work": runs},
                  cpu=(2, 2), memory=(4096, 8192), timeout=3600, max_containers=1, scaledown_window=2, retries=0)
    def stage(bundle_id):
        import json
        import time
        import zipfile
        from huggingface_hub import snapshot_download
        from freecus.manifest import load, sha256
        from freecus.run import write_json
        began = time.monotonic()
        destination = Path("/work/bundles") / bundle_id
        destination.mkdir(parents=True, exist_ok=True)
        archive = Path("/work/uploads") / f"{bundle_id}.zip"
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                if not (destination / member.filename).resolve().is_relative_to(destination.resolve()):
                    raise ValueError("Unsafe input bundle path")
            z.extractall(destination)
        meta = json.loads((destination / "bundle.json").read_text())
        for name, digest in meta["files"].items():
            if sha256(destination / name) != digest:
                raise ValueError("Input bundle hash mismatch")
        load(destination / "manifest.jsonl", destination / "input")
        cfg = json.loads((destination / "config.json").read_text())
        lock = json.loads((destination / "models.lock.json").read_text())
        paths = {}
        for key, model in lock.items():
            progress = dict(status="downloading", model=key, elapsed_seconds=time.monotonic() - began)
            write_json(destination / "stage.json", progress)
            runs.commit()
            allow = (["model_index.json", "scheduler/*", "tokenizer/*", "tokenizer_2/*", "text_encoder/*",
                      "text_encoder_2/*", "transformer/*", "vae/*"] if key == "flux"
                     else ["*.json", "*.safetensors", "*.txt", "*.model", "*.py", "*.tiktoken"])
            paths[key] = snapshot_download(model["repo"], revision=model["revision"], allow_patterns=allow,
                         ignore_patterns=["*.bin", "*.pt", "*.pth", "*.onnx", "*.h5", "*.msgpack"])
            cache.commit()
        cfg["models"] = paths
        write_json(destination / "resolved_config.json", cfg)
        result = dict(status="ready", bundle_id=bundle_id, elapsed_seconds=time.monotonic() - began,
                      samples=len(load(destination / "manifest.jsonl", destination / "input")))
        write_json(destination / "stage.json", result)
        runs.commit()
        return result

    @app.function(name="gpu_lane", serialized=True, image=gpu_image, secrets=[secret], volumes={"/cache": cache, "/work": runs},
                  gpu=gpu_type, cpu=(2, 2), memory=(65536, 65536), timeout=14400,
                  max_containers=2, min_containers=0, scaledown_window=2, retries=0, single_use_containers=True)
    def gpu_lane(bundle_id, invocation_id, lane, case_ids, worker_seconds_cap):
        import json
        import subprocess
        import sys
        import time
        from freecus.modal_cost import rate
        from freecus.run import write_json
        if worker_seconds_cap <= 0 or worker_seconds_cap > 13800:
            raise ValueError("Invalid worker budget")
        began = time.monotonic()
        runs.reload()
        cache.reload()
        bundle = Path("/work/bundles") / bundle_id
        if json.loads((bundle / "stage.json").read_text())["status"] != "ready":
            raise ValueError("CPU model staging has not completed")
        output = Path("/work/results") / bundle_id / f"lane-{lane}"
        journal = Path("/work/invocations") / invocation_id / f"lane-{lane}"
        journal.mkdir(parents=True, exist_ok=False)
        job = dict(config=str(bundle / "resolved_config.json"), manifest=str(bundle / "manifest.jsonl"),
                   data_root=str(bundle / "input"), output=str(output), case_ids=case_ids)
        write_json(journal / "job.json", job)
        write_json(journal / "state.json", dict(status="running", cases=len(case_ids), seconds_cap=worker_seconds_cap))
        runs.commit()
        child = subprocess.Popen([sys.executable, "-m", worker_module, str(journal / "job.json")])
        last_commit = time.monotonic()
        event_size = 0
        stopped = False
        try:
            while child.poll() is None:
                elapsed = time.monotonic() - began
                if elapsed >= worker_seconds_cap:
                    child.kill()
                    child.wait()
                    stopped = True
                    break
                event_file = journal / "events.jsonl"
                size = event_file.stat().st_size if event_file.exists() else 0
                if size != event_size or time.monotonic() - last_commit > 30:
                    write_json(journal / "state.json", dict(status="running", cases=len(case_ids), elapsed_seconds=elapsed))
                    runs.commit()
                    last_commit, event_size = time.monotonic(), size
                time.sleep(0.5)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
        elapsed = time.monotonic() - began
        result = dict(status="budget_paused" if stopped else "complete" if child.returncode == 0 else "failed",
                      exit_code=child.returncode, elapsed_seconds=elapsed,
                      estimated_compute_usd=(elapsed + 2) * rate(gpu_type), gpu=gpu_type,
                      output=str(output), events=[], invocation_id=invocation_id, lane=lane)
        event_file = journal / "events.jsonl"
        if event_file.exists():
            result["events"] = [json.loads(line) for line in event_file.read_text().splitlines() if line]
        write_json(journal / "state.json", result)
        runs.commit()
        return result

    @app.function(name="workflow", serialized=True, image=cpu_image, volumes={"/work": runs},
                  cpu=(0.125, 0.125), memory=(512, 512), timeout=86400,
                  max_containers=1, scaledown_window=2, retries=0)
    def workflow(workflow_id, plans, admission_usd):
        import json
        import time
        import uuid
        from freecus.modal_client import pilot_projection
        from freecus.modal_cost import rate
        from freecus.run import write_json
        if admission_usd <= 0 or admission_usd > 29 or len(plans) != 1:
            raise ValueError("One benchmark per account; reserve at least $1 from $30")
        report = Path("/work/workflows") / f"{workflow_id}.json"
        if report.exists():
            raise ValueError("Duplicate workflow ID")
        state = dict(workflow_id=workflow_id, status="starting", admission_usd=admission_usd,
                     accounted_compute_usd=0.0, phases=[], started_at=time.time(), gpu=gpu_type)

        def save():
            state["updated_at"] = time.time()
            write_json(report, state)
            runs.commit()

        def run_phase(plan, mode, lanes, budget):
            invocation = uuid.uuid4().hex
            receipt = dict(invocation_id=invocation, mode=mode, benchmark=plan["benchmark"], budget_usd=budget, calls=[])
            state["phases"].append(receipt)
            state["status"] = mode + "_" + plan["benchmark"]
            save()
            pending = []
            for lane, cases in enumerate(lanes):
                seconds = min(13800, budget / 2 / rate(gpu_type) - 10)
                call = gpu_lane.spawn(plan["bundle_id"], invocation, lane, cases, seconds)
                receipt["calls"].append(dict(id=call.object_id, lane=lane, cases=len(cases), budget_usd=budget / 2))
                pending.append(call)
                save()
            for i, call in enumerate(pending):
                try:
                    result = call.get()
                except Exception as exc:
                    result = dict(status="failed", error=repr(exc), estimated_compute_usd=budget / 2)
                receipt["calls"][i]["result"] = result
                state["accounted_compute_usd"] += result["estimated_compute_usd"]
                # Commit only this workflow's file: lane outputs and events are written to disjoint paths.
                save()
            return receipt

        def stage_plan(plan):
            call = stage.spawn(plan["bundle_id"])
            record = dict(id=call.object_id, benchmark=plan["benchmark"], done=False)
            state.setdefault("stage_calls", []).append(record)
            save()
            result = call.get()
            record["done"] = True
            save()
            return result

        save()
        try:
            state["status"] = "staging_" + plans[0]["benchmark"]
            save()
            staged = stage_plan(plans[0])
            state["stage_dreambench"] = staged
            bundle = Path("/work/bundles") / plans[0]["bundle_id"]
            runs.reload()
            rows = [json.loads(line) for line in (bundle / "manifest.jsonl").read_text().splitlines()]
            if plans[0]["benchmark"] == "dreambench":
                chosen = {r["case_id"] for r in rows if r["reference_id"].split("/")[-1] in {"backpack", "cat2", "dog", "teapot"}
                          and r["prompt_index"] in {0, 10, 20}}
            else:
                # Choose two references per lane, spread across available categories.
                selected = set()
                for lane_cases in plans[0]["lanes"]:
                    lane_set = set(lane_cases)
                    candidates = sorted({(r["category"], r["reference_id"]) for r in rows if r["case_id"] in lane_set})
                    categories = sorted({c for c, ref in candidates})
                    for category in (categories[0], categories[-1]):
                        selected.add(next(ref for cat, ref in candidates if cat == category))
                chosen = {r["case_id"] for r in rows if r["reference_id"] in selected and r["prompt_index"] in {0, 4, 8}}
            lanes = [[case for case in lane if case in chosen] for lane in plans[0]["lanes"]]
            if any(len(lane) < 2 for lane in lanes):
                raise ValueError("Invalid two-GPU pilot partition")
            pilot = run_phase(plans[0], "pilot", lanes, 2.5)
            if not all(c["result"]["status"] == "complete" for c in pilot["calls"]):
                state["status"] = "pilot_failed_or_budget_paused"
                save()
                return state
            for plan in plans:
                estimate = pilot_projection([pilot], plan["benchmark"], gpu=gpu_type)
                remaining = admission_usd - state["accounted_compute_usd"]
                state["next_projection"] = estimate
                if estimate["compute_usd_with_margin"] > remaining:
                    state["status"] = "budget_review_required"
                    state["not_started"] = plan["benchmark"]
                    save()
                    return state
                if plan["bundle_id"] != plans[0]["bundle_id"]:
                    state["status"] = "staging_dreambench_plus"
                    save()
                    state["stage_dreambench_plus"] = stage_plan(plan)
                # Two single-GPU workers remain the concurrency ceiling throughout.
                # Give each phase its projected cost + 25% margin, bounded by remaining credits.
                result = run_phase(plan, "full", plan["lanes"], min(remaining, estimate["compute_usd_with_margin"]))
                if not all(c["result"]["status"] == "complete" for c in result["calls"]):
                    state["status"] = "generation_failed_or_budget_paused"
                    save()
                    return state
            state["status"] = "complete"
            save()
            return state
        except BaseException as exc:
            state["status"] = "failed"
            state["error"] = repr(exc)
            save()
            raise

    return app, stage, gpu_lane
