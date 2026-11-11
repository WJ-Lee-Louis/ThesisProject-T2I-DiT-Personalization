"""Explicit-account CLI; durable receipts prevent accidental duplicate paid work."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid

from .config import ACCOUNTS, APP, RUNS, SECRET, worker_rate
from .io import read, write

BASE = Path(__file__).resolve().parent


def connect(profile):
    # These are credentials for other sessions, not this explicitly chosen profile.
    for name in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "MODAL_OAUTH_REFRESH_TOKEN", "MODAL_OAUTH_CLIENT_ID",
                 "MODAL_OAUTH_CLIENT_SECRET", "MODAL_OAUTH_JWT_KEY"):
        os.environ.pop(name, None)
    os.environ["MODAL_PROFILE"] = profile
    import modal
    from modal.config import config_profiles
    if profile not in config_profiles():
        raise ValueError("Authenticate the explicit profile first")
    return modal


def current():
    info = read(BASE / "outputs/current.json")
    directory = BASE / "outputs" / info["bundle_id"]
    return info, directory, read(directory / "plan.json")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "bootstrap", "deploy", "stage", "status", "pilot", "submit", "stop", "evaluate", "export", "collect", "estimate"])
    p.add_argument("--profile", choices=list(ACCOUNTS))
    p.add_argument("--budget-usd", type=float)
    p.add_argument("--gpu", choices=["L40S", "H100"], default="L40S")
    p.add_argument("--receipt", type=Path)
    args = p.parse_args()
    if args.command == "prepare":
        from .prepare import prepare
        prepare()
        return
    if not args.profile:
        p.error("--profile is required")
    modal = connect(args.profile)
    account = ACCOUNTS[args.profile]
    if args.command == "bootstrap":
        from huggingface_hub import get_token
        token = get_token()
        if not token:
            raise ValueError("Local HF login required (token is never printed)")
        modal.Secret.objects.create(SECRET, {"HF_TOKEN": token}, allow_existing=True)
        modal.Volume.from_name(RUNS, create_if_missing=True, version=2).hydrate()
        print("Secret and XLabs results Volume prepared:", account["workspace"])
        return
    info, directory, plan = current()
    state = directory / args.profile
    state.mkdir(parents=True, exist_ok=True)
    runs = modal.Volume.from_name(RUNS)
    if args.command == "deploy":
        from .modal_app import build_app
        with modal.enable_output():
            build_app(info["source"], account["cache"]).deploy()
        write(state / "deployment.json", dict(code=info["code"], app=APP))
        return
    if read(state / "deployment.json")["code"] != info["code"]:
        raise ValueError("Source changed; redeploy and use matching receipts")
    if args.command == "stage":
        path = state / "stage-receipt.json"
        if path.exists():
            raise ValueError("Stage already submitted; use status --receipt")
        with runs.batch_upload() as upload:
            upload.put_file(directory / "inputs.zip", "/uploads/" + plan["bundle_id"] + ".zip")
        call = modal.Function.from_name(APP, "stage").spawn(plan["bundle_id"])
        write(path, dict(call_id=call.object_id, kind="stage", profile=args.profile, code=info["code"]))
        print(json.dumps(dict(receipt=str(path), call_id=call.object_id)))
        return
    if args.command in ("pilot", "submit"):
        stage_receipt = read(state / "stage-receipt.json")
        staged = modal.FunctionCall.from_id(stage_receipt["call_id"]).get(timeout=0)
        if staged["status"] != "ready":
            raise ValueError("Model staging is not ready")
        path = args.receipt or state / (args.command + "-receipt.json")
        if path.exists():
            raise ValueError("Receipt already exists; inspect status, never resubmit blindly")
        cap = args.budget_usd if args.budget_usd is not None else (.40 if args.command == "pilot" else account["generation_cap"])
        previous = []
        # Paid history belongs to the workspace, including older source bundles.
        for rpath in (BASE / "outputs").glob("*/" + args.profile + "/*-receipt.json"):
            record = read(rpath)
            if "budget_usd" in record:
                try:
                    result = modal.FunctionCall.from_id(record["call_id"]).get(timeout=0)
                except TimeoutError:
                    raise ValueError("An existing GPU invocation is still running")
                except Exception:
                    # Failed calls still cost money; conservatively charge their admission.
                    result = {}
                previous.append(result.get("generation", result).get("estimated_compute_usd", record["budget_usd"]))
        if cap <= 0 or sum(previous) + cap > account["generation_cap"]:
            raise ValueError(f"Generation cap exceeded; previous estimate={sum(previous):.4f}, cap={account['generation_cap']}")
        cases = plan["assignments"][args.profile]
        if args.command == "pilot":
            # One object and one animal, both conditions. Parity adds two diagnostic renders.
            chosen_refs = ["dreambench/backpack", "dreambench/dog"]
            available = [r for r in plan["rows"] if r["case_id"] in cases]
            # Largest input of each assigned broad category; reference ownership never changes.
            animals = {"cat", "cat2", "dog", "dog2", "dog3", "dog5", "dog6", "dog7", "dog8"}
            cases = []
            for animal in (False, True):
                candidates = [r for r in available if (r["reference_id"].split("/")[-1] in animals) == animal]
                if candidates:
                    cases.append(max(candidates, key=lambda r: len(r["target_prompt"]))["case_id"])
        invocation = args.command + "-" + uuid.uuid4().hex[:12]
        seconds_cap = min(21000, cap / worker_rate(args.gpu) - 30)
        if args.command == "pilot":
            call = modal.Function.from_name(APP, "generate_" + args.gpu.lower()).spawn(
                plan["bundle_id"], cases, invocation, seconds_cap, True, True)
        else:
            call = modal.Function.from_name(APP, "workflow").spawn(
                plan["bundle_id"], cases, invocation, seconds_cap, args.profile, args.gpu)
        record = dict(call_id=call.object_id, kind=args.command, profile=args.profile, code=info["code"],
                      bundle_id=plan["bundle_id"], invocation=invocation, cases=cases, budget_usd=cap, gpu=args.gpu,
                      seconds_cap=seconds_cap, submitted_at=datetime.now(timezone.utc).isoformat())
        write(path, record)
        print(json.dumps(dict(receipt=str(path), **record), indent=2))
        return
    if args.command in ("evaluate", "export"):
        path = state / (args.command + "-receipt.json")
        if path.exists():
            raise ValueError("Already submitted; inspect status")
        generated = read(state / "submit-receipt.json")
        completed = modal.FunctionCall.from_id(generated["call_id"]).get(timeout=0)
        if completed["status"] != "complete":
            raise ValueError("Shard generation is not complete; no final evaluation/export")
        cases = plan["assignments"][args.profile]
        function = modal.Function.from_name(APP, args.command)
        call = (function.spawn(plan["bundle_id"], cases) if args.command == "evaluate"
                else function.spawn(plan["bundle_id"], cases, args.profile))
        write(path, dict(call_id=call.object_id, kind=args.command, profile=args.profile, code=info["code"]))
        print(json.dumps(dict(receipt=str(path), call_id=call.object_id)))
        return
    if args.command == "estimate":
        events, consumed, gpu_kinds = [], 0, set()
        for path in state.glob("*-receipt-status.json"):
            data = read(path)
            data = data.get("generation", data)
            if "estimated_compute_usd" in data:
                consumed += data["estimated_compute_usd"]
                events += data.get("events", [])
                gpu_kinds.add(data["gpu"])
        if not events:
            raise ValueError("Collect pilot status first")
        import statistics
        if len(gpu_kinds) != 1:
            raise ValueError("Estimate each GPU pilot separately")
        measured_gpu = next(iter(gpu_kinds))
        seconds = statistics.mean(e["diagnostics"]["seconds"] for e in events)
        print(json.dumps(dict(gpu=measured_gpu, seconds_per_image=seconds, cost_per_image=seconds * worker_rate(measured_gpu),
                              consumed_estimate_usd=consumed,
                              dreambench_1500_images_usd=1500 * seconds * worker_rate(measured_gpu),
                              dreambench_plus_2700_images_usd=2700 * seconds * worker_rate(measured_gpu),
                              note="Steady generation only; add startup, evaluation and safety margin"), indent=2))
        return
    if args.command in ("status", "stop"):
        paths = [args.receipt] if args.receipt else sorted(state.glob("*-receipt.json"))
        for path in paths:
            record = read(path)
            call = modal.FunctionCall.from_id(record["call_id"])
            if args.command == "stop":
                # Stop the controller first so it cannot submit another child.
                call.cancel(terminate_containers=True)
                if record.get("supervised"):
                    try:
                        supervision = json.loads(b"".join(runs.read_file(
                            f"supervision/{plan['bundle_id']}/{args.profile}.json")))
                        for call_id in supervision.get("children", []):
                            modal.FunctionCall.from_id(call_id).cancel(terminate_containers=True)
                    except FileNotFoundError:
                        pass
                    modal.FunctionCall.from_id(record["original_call_id"]).cancel(terminate_containers=True)
                if record.get("invocation"):
                    try:
                        workflow = json.loads(b"".join(runs.read_file(f"workflows/{plan['bundle_id']}/{record['invocation']}.json")))
                        for call_id in workflow.get("children", []):
                            modal.FunctionCall.from_id(call_id).cancel(terminate_containers=True)
                    except FileNotFoundError:
                        pass
                print("Stopped", record["call_id"])
                continue
            try:
                result = call.get(timeout=0)
            except TimeoutError:
                result = dict(status="running")
                if record.get("invocation"):
                    remote = (f"supervision/{plan['bundle_id']}/{args.profile}.json" if record.get("supervised") else
                              f"{'state' if record['kind'] == 'pilot' else 'workflows'}/{plan['bundle_id']}/{record['invocation']}.json")
                    try:
                        result = json.loads(b"".join(runs.read_file(remote)))
                    except FileNotFoundError:
                        pass
            except Exception as error:
                result = dict(status="remote_error", error=str(error))
            if record.get("kind") == "submit" and record.get("invocation"):
                try:
                    active = result.get("active_invocation", record["invocation"])
                    progress = json.loads(b"".join(runs.read_file(f"state/{plan['bundle_id']}/{active}.json")))
                    result["generation_progress"] = {k: v for k, v in progress.items() if k != "events"}
                    result["generation_progress"]["generated"] = len(progress.get("events", []))
                except FileNotFoundError:
                    pass
            write(path.with_name(path.stem + "-status.json"), result)
            if record["kind"] == "submit" and result.get("status") == "complete":
                for phase, child_id in zip(("generation", "evaluate", "export"), result["children"]):
                    if phase != "generation":
                        write(state / (phase + "-receipt.json"), dict(call_id=child_id, kind=phase,
                               profile=args.profile, code=info["code"]))
            print(json.dumps(dict(receipt=path.name, **{k: v for k, v in result.items() if k != "events"},
                                 generated=len(result.get("events", [])) if "generation_progress" not in result
                                 else result["generation_progress"]["generated"]), indent=2))
        return
    if args.command == "collect":
        # Individual archive collection is implemented by artifacts.py, after export.
        from .artifacts import collect
        collect(modal, runs, plan, directory, args.profile)


if __name__ == "__main__":
    main()
