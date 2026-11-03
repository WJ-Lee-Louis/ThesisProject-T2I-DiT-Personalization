"""Explicit-profile preparation; pilot and full GPU work require explicit commands."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import uuid
import zipfile

from freecus.manifest import PROJECT, sha256
from freecus.run import write_json
from freecus.modal_cost import rate, projection
from .protocol import BASE, selected_rows, validate_config, verify_upstream
from .modal_app import APP, CACHE, RUNS, SECRET


def snapshot():
    files = [*BASE.glob("*.py"), *BASE.glob("requirements*.txt"), BASE / "constraints.txt", BASE / "config.json", BASE / "models.lock.json", BASE / "UPSTREAM.json"]
    files += [p for p in (BASE / "upstream").rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    files += [PROJECT / "freecus" / name for name in ["__init__.py", "manifest.py", "run.py", "evaluate.py", "modal_cost.py",
              "project_metrics/benchmark_metrics.py", "project_metrics/SOURCE.json"]]
    identity = {p.relative_to(PROJECT).as_posix(): sha256(p) for p in files}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    destination = BASE / "outputs/modal_sources" / digest
    for path in files:
        target = destination / path.relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    return destination.resolve(), digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "bootstrap", "deploy", "stage", "stage-status", "pilot", "submit", "status", "workflow", "workflow-status", "stop"])
    p.add_argument("--profile", required=True)
    p.add_argument("--benchmark", choices=["dreambench", "dreambench_plus"], default="dreambench")
    p.add_argument("--gpu", choices=["A100-80GB", "H100", "L40S"], default="A100-80GB")
    p.add_argument("--budget-usd", type=float, default=2.5)
    p.add_argument("--admission-usd", type=float, default=38)
    a = p.parse_args()
    state = BASE / "outputs/modal" / a.profile
    state.mkdir(parents=True, exist_ok=True)
    plan_path = state / (a.benchmark + ".json")
    if a.command == "prepare":
        verify_upstream()
        validate_config(json.loads((BASE / "config.json").read_text()))
        rows, manifest = selected_rows(a.benchmark)
        source, code = snapshot()
        bundle_id = hashlib.sha256((code + sha256(manifest)).encode()).hexdigest()[:24]
        archive = state / (bundle_id + ".zip")
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as z:
            files = {"manifest.jsonl": manifest.read_bytes()}
            for name in sorted({r["reference"] for r in rows}):
                files["input/" + name] = (PROJECT / name).read_bytes()
            for name, data in files.items():
                z.writestr(name, data)
            z.writestr("bundle.json", json.dumps(dict(files={n: hashlib.sha256(data).hexdigest() for n, data in files.items()})))
        refs = sorted({r["reference_id"] for r in rows})
        owners = {ref: i % 2 for i, ref in enumerate(refs)}
        lanes = [[r["case_id"] for r in rows if owners[r["reference_id"]] == lane] for lane in range(2)]
        write_json(plan_path, dict(bundle_id=bundle_id, source=str(source), code=code, archive=str(archive.resolve()),
                                  benchmark=a.benchmark, profile=a.profile, gpu=a.gpu, lanes=lanes))
        print(dict(bundle_id=bundle_id, gpu=a.gpu, lanes=list(map(len, lanes))))
        return
    from freecus.modal_client import connect
    modal = connect(a.profile)
    if a.command == "bootstrap":
        from huggingface_hub import get_token
        token = get_token()
        if not token:
            raise ValueError("Authenticate to Hugging Face locally before bootstrap")
        modal.Secret.objects.create(SECRET, {"HF_TOKEN": token}, allow_existing=True)
        modal.Volume.from_name(CACHE, create_if_missing=True, version=2).hydrate()
        modal.Volume.from_name(RUNS, create_if_missing=True, version=2).hydrate()
        print("Prepared UNO volumes and HF Secret in explicit profile", a.profile)
        return
    plan = json.loads(plan_path.read_text())
    if a.command == "deploy":
        from .modal_app import build_app
        with modal.enable_output():
            build_app(plan["source"], plan["gpu"]).deploy()
        write_json(state / "deployment.json", dict(code=plan["code"], gpu=plan["gpu"]))
        return
    deployment = json.loads((state / "deployment.json").read_text())
    if deployment != dict(code=plan["code"], gpu=plan["gpu"]):
        raise ValueError("Deployment differs from prepared source/GPU")
    volume = modal.Volume.from_name(RUNS)
    workflow_path = state / (a.benchmark + "-workflow.json")
    if a.command == "workflow":
        if workflow_path.exists():
            raise ValueError("Autonomous workflow already submitted; inspect workflow-status")
        if not 0 < a.admission_usd <= 38:
            raise ValueError("Compute admission must be within $38; reserve $2 for CPU/setup")
        with volume.batch_upload() as upload:
            upload.put_file(plan["archive"], "/uploads/" + plan["bundle_id"] + ".zip")
        workflow_id = uuid.uuid4().hex
        args = {k: plan[k] for k in ["bundle_id", "benchmark", "lanes"]}
        call = modal.Function.from_name(APP, "workflow").spawn(workflow_id, args, a.admission_usd)
        record = dict(workflow_id=workflow_id, call_id=call.object_id, bundle_id=plan["bundle_id"],
                      profile=a.profile, benchmark=a.benchmark, compute_admission_usd=a.admission_usd,
                      workspace_budget_usd=40, app=APP)
        write_json(workflow_path, record)
        print(json.dumps(record, indent=2))
        return
    if a.command in {"workflow-status", "stop"}:
        record = json.loads(workflow_path.read_text())
        call = modal.FunctionCall.from_id(record["call_id"])
        try:
            result = call.get(timeout=0)
        except Exception:
            result = json.loads(b"".join(volume.read_file("workflows/" + record["workflow_id"] + ".json")))
        if a.command == "stop":
            call.cancel(terminate_containers=True)
            ids = [c["id"] for phase in result.get("phases", []) for c in phase["calls"]]
            if result.get("stage_call"):
                ids.append(result["stage_call"])
            for call_id in ids:
                modal.FunctionCall.from_id(call_id).cancel(terminate_containers=True)
            print("Cancellation requested; checkpointed images remain on the Volume")
            return
        write_json(state / (a.benchmark + "-workflow-status.json"), result)
        view = {k:v for k,v in result.items() if k != "phases"}
        view["phases"] = [{k:v for k,v in phase.items() if k != "calls"} | dict(calls=[
            {k:v for k,v in c.items() if k != "result"} | dict(result={k:v for k,v in c.get("result", {}).items()
             if k not in {"timings", "error_tail"}}) for c in phase["calls"]]) for phase in result.get("phases", [])]
        print(json.dumps(view, indent=2))
        return
    if a.command == "stage":
        record_path = state / (a.benchmark + "-stage.json")
        if record_path.exists():
            raise ValueError("Staging already submitted; inspect stage-status")
        with volume.batch_upload() as upload:
            upload.put_file(plan["archive"], "/uploads/" + plan["bundle_id"] + ".zip")
        call = modal.Function.from_name(APP, "stage").spawn(plan["bundle_id"])
        write_json(record_path, dict(call_id=call.object_id, bundle_id=plan["bundle_id"]))
        print(call.object_id)
        return
    if a.command == "stage-status":
        record = json.loads((state / (a.benchmark + "-stage.json")).read_text())
        try:
            print(modal.FunctionCall.from_id(record["call_id"]).get(timeout=0))
        except TimeoutError:
            print("CPU model staging is running remotely")
        return
    receipts = []
    for path in sorted(state.glob("invocations/*.json")):
        record = json.loads(path.read_text())
        for item in record["calls"]:
            if "result" not in item:
                try:
                    item["result"] = modal.FunctionCall.from_id(item["id"]).get(timeout=0)
                except TimeoutError:
                    pass
        write_json(path, record)
        receipts.append(record)
    spent = sum(c.get("result", {}).get("estimated_compute_usd", r["budget_usd"] / 2) for r in receipts for c in r["calls"])
    if a.command == "status":
        print(json.dumps(dict(accounted_compute_usd=spent, invocations=[dict(mode=r["mode"], calls=[dict(id=c["id"],
              result={k:v for k,v in c.get("result", {}).items() if k != "timings"}) for c in r["calls"]]) for r in receipts]), indent=2))
        return
    if any("result" not in c for r in receipts for c in r["calls"]):
        raise ValueError("GPU work is still running; inspect status")
    if not 0 < a.budget_usd <= 38 - spent:
        raise ValueError("Cumulative compute admission is $38; reserve $2 within the previously authorized $40")
    rows, _ = selected_rows(a.benchmark)
    if a.command == "pilot":
        chosen = set()
        for cases in plan["lanes"]:
            available = [r for r in rows if r["case_id"] in set(cases)]
            for category in sorted({r["category"] for r in available}):
                ref = next(r["reference_id"] for r in available if r["category"] == category)
                chosen.update(r["case_id"] for r in available if r["reference_id"] == ref and r["prompt_index"] in {0, 4})
        lanes = [[case for case in cases if case in chosen] for cases in plan["lanes"]]
    else:
        pilot = next((r for r in reversed(receipts) if r["mode"] == "pilot" and r["benchmark"] == a.benchmark
                      and r["gpu"] == plan["gpu"] and all(c["result"]["status"] == "complete" for c in r["calls"])), None)
        if pilot is None:
            raise ValueError("A successful pilot on the same benchmark/GPU is required")
        seconds = max(t["seconds"] for c in pilot["calls"] for t in c["result"]["timings"])
        startup = max(c["result"]["elapsed_seconds"] - sum(t["seconds"] for t in c["result"]["timings"]) for c in pilot["calls"])
        estimate = projection(len(rows), seconds, workers=2, startup_seconds=startup, gpu=plan["gpu"], safety=1.25)
        print(json.dumps(estimate, indent=2))
        if estimate["compute_usd_with_margin"] > a.budget_usd:
            raise ValueError("Measured projection exceeds phase budget; no GPU submitted")
        lanes = plan["lanes"]
    invocation = uuid.uuid4().hex
    record = dict(mode=a.command, benchmark=a.benchmark, gpu=plan["gpu"], budget_usd=a.budget_usd, calls=[])
    path = state / "invocations" / (invocation + ".json")
    write_json(path, record)
    for lane, cases in enumerate(lanes):
        call = modal.Function.from_name(APP, "gpu_lane").spawn(plan["bundle_id"], a.benchmark, invocation, lane, cases,
                                      min(21000, a.budget_usd / 2 / rate(plan["gpu"]) - 25))
        record["calls"].append(dict(id=call.object_id, lane=lane, cases=len(cases)))
        write_json(path, record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
