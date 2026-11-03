"""Explicit-profile Modal setup, staging, two-GPU pilot and budget-gated submission."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import statistics
import time
import uuid
import zipfile

from .manifest import PROJECT, load, sha256
from .modal_cost import projection, rate
from .run import write_json

BASE = Path(__file__).parent
PROFILE = os.environ.get("FREECUS_MODAL_PROFILE", "method-comparison")
STATE = BASE / ("outputs/modal" if PROFILE == "method-comparison" else f"outputs/modal-{PROFILE}")
GPU = os.environ.get("FREECUS_MODAL_GPU", "H100" if PROFILE == "method-comparison-dbpp" else "A100-80GB")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def connect(profile):
    os.environ["MODAL_PROFILE"] = profile
    os.environ["PYTHONUTF8"] = "1"
    for name in ["MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "MODAL_OAUTH_REFRESH_TOKEN", "MODAL_OAUTH_CLIENT_ID",
                 "MODAL_OAUTH_CLIENT_SECRET", "MODAL_OAUTH_JWT_KEY"]:
        os.environ.pop(name, None)
    import modal
    from modal.config import config_profiles
    if profile not in config_profiles():
        raise ValueError(f"Authenticate first: python -m modal token new --profile {profile} --no-activate")
    return modal


def summarize_workflow(value):
    summary = {k: v for k, v in value.items() if k != "phases"}
    summary["phases"] = []
    for phase in value.get("phases", []):
        item = {k: v for k, v in phase.items() if k != "calls"}
        item["calls"] = []
        for call in phase["calls"]:
            detail = {k: v for k, v in call.items() if k != "result"}
            if "result" in call:
                result = call["result"]
                detail["result"] = {k: v for k, v in result.items() if k != "events"}
                events = result.get("events", [])
                detail["generated"] = sum(e["phase"] == "generated" for e in events)
                detail["peak_gpu_allocated_gib"] = max((e.get("peak_allocated_gib", 0) for e in events), default=0)
                detail["host_peak_gib"] = max((e.get("host_peak_gib", 0) for e in events), default=0)
            item["calls"].append(detail)
        summary["phases"].append(item)
    return summary


def source_snapshot():
    files = [*BASE.glob("*.py"), *BASE.glob("requirements*.txt"), BASE / "constraints.txt", BASE / "UPSTREAM.json"]
    files += [*BASE.glob("runtime_src/*.py"), *BASE.glob("upstream/*.py")]
    identity = {p.relative_to(BASE).as_posix(): sha256(p) for p in sorted(files)}
    folder = STATE / "source" / digest(identity)
    folder.mkdir(parents=True, exist_ok=True)
    for p in files:
        dest = folder / p.relative_to(BASE)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
    return folder, identity


def prepare(benchmark):
    from huggingface_hub import HfApi
    manifest = BASE / f"manifests/{benchmark}_project_seed42.jsonl"
    rows = load(manifest, PROJECT)
    expected = 750 if benchmark == "dreambench" else 1350
    if len(rows) != expected or {r["seed"] for r in rows} != {42}:
        raise ValueError("Unexpected benchmark inputs")
    config = read(BASE / "config.json")
    if config["implementation"] != "paper":
        raise ValueError("This deployment is the declared paper adaptation")
    lock_path = BASE / "models.lock.json"
    if lock_path.exists():
        lock = read(lock_path)
    else:
        api = HfApi()
        lock = {key: dict(repo=repo, revision=api.model_info(repo).sha) for key, repo in config["models"].items()}
        write_json(lock_path, lock)
    if {key: model["repo"] for key, model in lock.items()} != config["models"]:
        raise ValueError("Model lock differs from config")
    source, code = source_snapshot()
    identity = dict(benchmark=benchmark, manifest_sha256=sha256(manifest), config=config, lock=lock, code=code)
    bundle_id = digest(identity)[:24]
    destination = STATE / "bundles" / bundle_id
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "input.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as z:
        files = {"manifest.jsonl": manifest.read_bytes(), "config.json": json.dumps(config).encode(),
                 "models.lock.json": json.dumps(lock).encode()}
        for name in sorted({r["reference"] for r in rows}):
            files["input/" + name] = (PROJECT / name).read_bytes()
        for name, data in files.items():
            z.writestr(name, data)
        z.writestr("bundle.json", json.dumps(dict(files={name: hashlib.sha256(data).hexdigest() for name, data in files.items()})))
    refs = sorted({r["reference_id"] for r in rows})
    owners = {ref: i % 2 for i, ref in enumerate(refs)}
    lanes = [[r["case_id"] for r in rows if owners[r["reference_id"]] == lane] for lane in range(2)]
    result = dict(bundle_id=bundle_id, benchmark=benchmark, samples=len(rows), source=str(source.resolve()),
                  archive=str(archive.resolve()), manifest=str(manifest.resolve()), identity=identity,
                  profile=PROFILE, lanes=lanes)
    write_json(destination / "plan.json", result)
    write_json(STATE / f"latest-{benchmark}.json", result)
    print(json.dumps(dict(bundle_id=bundle_id, benchmark=benchmark, samples=len(rows), lane_counts=list(map(len, lanes))), indent=2))
    return result


def plan_for(benchmark, require_current=True):
    plan = read(STATE / f"latest-{benchmark}.json")
    _, code = source_snapshot()
    if require_current and code != plan["identity"]["code"]:
        raise ValueError("Code changed; prepare a new bundle and redeploy")
    return plan


def invocation_files():
    return sorted((STATE / "invocations").glob("*.json"))


def poll(modal):
    results = []
    for path in invocation_files():
        receipt = read(path)
        for lane in receipt["calls"]:
            if "result" not in lane:
                try:
                    lane["result"] = modal.FunctionCall.from_id(lane["id"]).get(timeout=0)
                except TimeoutError:
                    continue
                except Exception as exc:
                    # A failed/timed-out worker still consumes its reserved budget in the ledger.
                    lane["result"] = dict(status="failed", error=repr(exc), estimated_compute_usd=lane["budget_usd"])
        write_json(path, receipt)
        results.append(receipt)
    return results


def accounted_usd(receipts):
    return sum(call.get("result", {}).get("estimated_compute_usd", call["budget_usd"])
               for receipt in receipts for call in receipt["calls"])


def pilot_projection(receipts, benchmark, gpu="A100-80GB"):
    successes = [r for r in receipts if r["mode"] == "pilot" and all(c.get("result", {}).get("status") == "complete" for c in r["calls"])]
    if not successes:
        raise ValueError("A successful completed two-GPU pilot is required")
    pilot = successes[-1]
    timings, overhead, caption_times = [], [], []
    caption_count = 0
    for call in pilot["calls"]:
        result = call["result"]
        seconds = [e["seconds"] for e in result["events"] if e["phase"] == "generated"]
        if len(seconds) < 2:
            raise ValueError("Pilot has insufficient measured generated images")
        # First image includes FLUX load; retain it as startup rather than extrapolating it per image.
        timings.extend(seconds[1:])
        overhead.append(max(0, result["elapsed_seconds"] - sum(seconds)) + seconds[0])
        captions = [e["seconds"] for e in result["events"] if e["phase"] == "caption"]
        caption_count += len(captions)
        caption_times.extend(captions[1:])
    seconds = max(statistics.mean(timings), sorted(timings)[int(0.9 * (len(timings) - 1))])
    extra_caption_seconds = max(0, (30 if benchmark == "dreambench" else 150) - caption_count) * (max(caption_times) if caption_times else 0)
    estimate = projection(750 if benchmark == "dreambench" else 1350, seconds,
                          workers=2, startup_seconds=max(overhead) + extra_caption_seconds / 2, safety=1.25, gpu=gpu)
    estimate["extra_caption_seconds"] = extra_caption_seconds
    estimate["pilot_id"] = pilot["invocation_id"]
    estimate["measured_images"] = sum(len([e for e in c["result"]["events"] if e["phase"] == "generated"]) for c in pilot["calls"])
    return estimate


def submit(modal, plan, mode, budget):
    from .modal_remote import APP_NAME
    receipts = poll(modal)
    if any("result" not in c for r in receipts for c in r["calls"]):
        raise ValueError("A GPU invocation is still active; inspect status before another submission")
    spent = accounted_usd(receipts)
    if budget <= 0 or spent + budget > 25:
        raise ValueError(f"Admission budget exceeded: accounted ${spent:.2f} + requested ${budget:.2f} > $25. $5 reserved for setup/uncertainty.")
    rows = load(Path(plan["manifest"]), PROJECT)
    if mode == "pilot":
        if plan["benchmark"] != "dreambench":
            raise ValueError("The initial pilot is DreamBench only")
        names = {"backpack", "cat2", "dog", "teapot"}
        chosen = {r["case_id"] for r in rows if r["reference_id"].split("/")[-1] in names and r["prompt_index"] in {0, 10, 20}}
        lanes = [[case for case in lane if case in chosen] for lane in plan["lanes"]]
        if any(len(lane) < 2 for lane in lanes):
            raise ValueError("Pilot partition is unexpectedly empty")
    else:
        estimate = pilot_projection(receipts, plan["benchmark"])
        print(json.dumps(estimate, indent=2))
        if estimate["compute_usd_with_margin"] > budget:
            raise ValueError("Measured projection exceeds requested budget; no full GPU invocation submitted")
        lanes = plan["lanes"]
    invocation = uuid.uuid4().hex
    f = modal.Function.from_name(APP_NAME, "gpu_lane")
    receipt = dict(invocation_id=invocation, bundle_id=plan["bundle_id"], benchmark=plan["benchmark"],
                   mode=mode, profile=PROFILE, budget_usd=budget, calls=[])
    path = STATE / "invocations" / f"{invocation}.json"
    write_json(path, receipt)
    for lane, cases in enumerate(lanes):
        seconds = min(13800, (budget / 2) / rate() - 10)
        call = f.spawn(plan["bundle_id"], invocation, lane, cases, seconds)
        receipt["calls"].append(dict(id=call.object_id, lane=lane, cases=len(cases), budget_usd=budget / 2))
        write_json(path, receipt)
    print(json.dumps(receipt, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "bootstrap", "deploy", "stage", "stage-status", "pilot", "submit", "status", "estimate", "download", "stop", "workflow", "workflow-status"])
    p.add_argument("--benchmark", choices=["dreambench", "dreambench_plus"], default="dreambench")
    p.add_argument("--budget-usd", type=float, default=2.5)
    p.add_argument("--include-dbpp", action="store_true")
    a = p.parse_args()
    STATE.mkdir(parents=True, exist_ok=True)
    if a.command == "prepare":
        prepare(a.benchmark)
        return
    modal = connect(PROFILE)
    from .modal_remote import APP_NAME, CACHE_NAME, RUNS_NAME, SECRET_NAME, build_app
    if a.command == "bootstrap":
        from huggingface_hub import get_token
        token = get_token()
        if not token:
            raise ValueError("Log into Hugging Face locally before bootstrap")
        modal.Secret.objects.create(SECRET_NAME, {"HF_TOKEN": token}, allow_existing=True)
        modal.Volume.from_name(CACHE_NAME, create_if_missing=True, version=2).hydrate()
        modal.Volume.from_name(RUNS_NAME, create_if_missing=True, version=2).hydrate()
        print(f"Prepared HF Secret and v2 Volumes in explicit profile {PROFILE}; token values were not printed")
        return
    if a.command == "deploy":
        plan = plan_for(a.benchmark)
        app, _, _ = build_app(plan["source"], gpu_type=GPU)
        with modal.enable_output():
            app.deploy()
        write_json(STATE / "deployment.json", dict(app=APP_NAME, app_id=app.app_id, profile=PROFILE))
        print("Deployed", APP_NAME)
        return
    if a.command == "workflow-status":
        record = read(STATE / "workflow.json")
        call = modal.FunctionCall.from_id(record["call_id"])
        try:
            result = call.get(timeout=0)
        except TimeoutError:
            volume = modal.Volume.from_name(RUNS_NAME)
            data = b"".join(volume.read_file(f"workflows/{record['workflow_id']}.json"))
            result = json.loads(data)
        write_json(STATE / "workflow-status.json", result)
        print(json.dumps(summarize_workflow(result), indent=2))
        return
    if a.command in {"status", "estimate", "stop"}:
        receipts = poll(modal)
        if a.command == "stop":
            workflow_file = STATE / "workflow.json"
            if workflow_file.exists():
                coordinator = read(workflow_file)
                modal.FunctionCall.from_id(coordinator["call_id"]).cancel(terminate_containers=True)
                ancestor = coordinator.get("parent")
                while ancestor:
                    modal.FunctionCall.from_id(ancestor["call_id"]).cancel(terminate_containers=True)
                    ancestor = ancestor.get("parent")
                volume = modal.Volume.from_name(RUNS_NAME)
                state = json.loads(b"".join(volume.read_file(f"workflows/{coordinator['workflow_id']}.json")))
                ids = [c["id"] for phase in state.get("phases", []) for c in phase["calls"]]
                ids += [c["id"] for c in state.get("stage_calls", []) if not c["done"]]
                for call_id in ids:
                    modal.FunctionCall.from_id(call_id).cancel(terminate_containers=True)
            for receipt in receipts:
                for call in receipt["calls"]:
                    if "result" not in call:
                        modal.FunctionCall.from_id(call["id"]).cancel(terminate_containers=True)
            print("Cancellation requested; checkpointed results remain on the Volume")
        elif a.command == "estimate":
            print(json.dumps(dict(accounted_usd=accounted_usd(receipts), projection=pilot_projection(receipts, a.benchmark)), indent=2))
        else:
            summary = [{k: v for k, v in r.items() if k != "calls"} | dict(calls=[{k: v for k, v in c.items() if k != "result"} |
                         dict(result={k: v for k, v in c.get("result", {}).items() if k != "events"}) for c in r["calls"]]) for r in receipts]
            print(json.dumps(dict(accounted_usd=accounted_usd(receipts), invocations=summary), indent=2))
        return
    plan = plan_for(a.benchmark, require_current=a.command != "download")
    if a.command == "workflow":
        if (STATE / "workflow.json").exists():
            raise ValueError("A workflow receipt already exists; inspect it before starting another")
        plans = [plan]
        if a.include_dbpp:
            plans.append(plan_for("dreambench_plus"))
        if any(x["identity"]["code"] != plan["identity"]["code"] for x in plans):
            raise ValueError("Both benchmarks must use the same code snapshot")
        volume = modal.Volume.from_name(RUNS_NAME)
        with volume.batch_upload(force=True) as upload:
            for item in plans:
                upload.put_file(item["archive"], f"/uploads/{item['bundle_id']}.zip")
        workflow_id = uuid.uuid4().hex
        args = [dict(bundle_id=item["bundle_id"], benchmark=item["benchmark"], lanes=item["lanes"]) for item in plans]
        admission = 29.0 if a.budget_usd == 2.5 else a.budget_usd
        call = modal.Function.from_name(APP_NAME, "workflow").spawn(workflow_id, args, admission)
        record = dict(workflow_id=workflow_id, call_id=call.object_id, app=APP_NAME, profile=PROFILE,
                      compute_admission_usd=admission, reserve_usd=30-admission, gpu=GPU, plans=args)
        write_json(STATE / "workflow.json", record)
        print(json.dumps({k: v for k, v in record.items() if k != "plans"}, indent=2))
    elif a.command == "stage":
        volume = modal.Volume.from_name(RUNS_NAME)
        with volume.batch_upload(force=True) as upload:
            upload.put_file(plan["archive"], f"/uploads/{plan['bundle_id']}.zip")
        call = modal.Function.from_name(APP_NAME, "stage").spawn(plan["bundle_id"])
        record = dict(call_id=call.object_id, bundle_id=plan["bundle_id"], profile=PROFILE)
        write_json(STATE / f"stage-{a.benchmark}.json", record)
        print(json.dumps(record, indent=2))
    elif a.command == "stage-status":
        record = read(STATE / f"stage-{a.benchmark}.json")
        try:
            print(json.dumps(modal.FunctionCall.from_id(record["call_id"]).get(timeout=0), indent=2))
        except TimeoutError:
            volume = modal.Volume.from_name(RUNS_NAME)
            try:
                data = b"".join(volume.read_file(f"bundles/{plan['bundle_id']}/stage.json"))
                print(data.decode())
            except FileNotFoundError:
                print("CPU staging queued or starting")
    elif a.command in {"pilot", "submit"}:
        submit(modal, plan, "pilot" if a.command == "pilot" else "full", a.budget_usd)
    elif a.command == "download":
        volume = modal.Volume.from_name(RUNS_NAME)
        folder = STATE / "download" / plan["bundle_id"]
        for entry in volume.iterdir(f"results/{plan['bundle_id']}", recursive=True):
            if entry.type == 1:
                dest = folder / entry.path.lstrip("/")
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open("wb") as stream:
                    for chunk in volume.read_file(entry.path):
                        stream.write(chunk)
        print("Downloaded", folder)


if __name__ == "__main__":
    main()
