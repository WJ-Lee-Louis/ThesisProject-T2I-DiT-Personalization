"""CPU-only continuation supervisor for already deployed, frozen GPU functions."""
import argparse
import json
from pathlib import Path
import uuid

from .config import APP, RUNS, worker_rate
from .io import read, write

BASE = Path(__file__).resolve().parent


def deploy_and_submit(profile, cap=12.0):
    from .client import connect, current
    modal = connect(profile)
    info, directory, plan = current()
    receipt_path = directory / profile / "submit-receipt.json"
    original = read(receipt_path)
    if original.get("supervised"):
        raise ValueError("Already supervised; do not submit another supervisor")
    if not 0 < cap <= 12:
        raise ValueError("This DreamBench continuation admission is bounded to $12/account")
    pilot_cost = 0
    for path in (BASE / "outputs").glob("*/" + profile + "/pilot-receipt.json"):
        record = read(path)
        result = modal.FunctionCall.from_id(record["call_id"]).get(timeout=0)
        pilot_cost += result["estimated_compute_usd"]
    runs = modal.Volume.from_name(RUNS)
    image = (modal.Image.debian_slim(python_version="3.11")
             .add_local_dir(info["source"], "/root", copy=True)
             .add_local_file(Path(__file__), "/root/xlabs_application/supervisor.py", copy=True))
    app = modal.App("xlabs-application-supervisor-v1")

    @app.function(name="supervise", serialized=True, image=image, volumes={"/work": runs},
                  cpu=(0.125, .125), memory=(512, 512), timeout=86400, retries=0,
                  max_containers=1, scaledown_window=2)
    def supervise(bundle_id, cases, shard, initial_call_id, initial_invocation, admission, paid_pilots):
        import traceback
        import modal
        from xlabs_application.io import write
        from xlabs_application.config import APP, worker_rate
        state_path = Path("/work/supervision") / bundle_id / (shard + ".json")
        state = dict(status="generating", active_invocation=initial_invocation,
                     original_call_id=initial_call_id, children=[], generation_cap_usd=admission,
                     previous_pilot_estimate_usd=paid_pilots)

        def publish():
            write(state_path, state)
            runs.commit()

        publish()
        try:
            initial = modal.FunctionCall.from_id(initial_call_id).get()
            if initial["status"] == "complete":
                state.update(initial)
                publish()
                return state
            if initial["status"] != "budget_checkpoint":
                state.update(initial)
                publish()
                return state
            spent = initial["generation"]["estimated_compute_usd"]
            remaining = admission - paid_pilots - spent
            if remaining <= .1:
                state.update(status="budget_checkpoint", generation=initial["generation"])
                publish()
                return state
            invocation = "continuation-" + uuid.uuid4().hex[:12]
            state.update(status="generating", active_invocation=invocation, initial_generation=initial["generation"])
            child = modal.Function.from_name(APP, "generate_h100").spawn(
                bundle_id, cases, invocation, min(21000, remaining / worker_rate("H100") - 30), False, False)
            state["children"] = [child.object_id]
            publish()
            generation = child.get()
            state["continuation_generation"] = {k: v for k, v in generation.items() if k != "events"}
            state["generation"] = dict(state["continuation_generation"])
            state["generation"]["estimated_compute_usd"] += spent
            if generation["status"] != "complete":
                state["status"] = generation["status"]
                publish()
                return state
            state["status"] = "evaluating"
            child = modal.Function.from_name(APP, "evaluate").spawn(bundle_id, cases)
            state["children"].append(child.object_id)
            publish()
            state["evaluation"] = child.get()
            state["status"] = "exporting"
            child = modal.Function.from_name(APP, "export").spawn(bundle_id, cases, shard)
            state["children"].append(child.object_id)
            publish()
            state["export"] = child.get()
            state["status"] = "complete"
            publish()
            return state
        except Exception:
            state["status"] = "failed"
            state["error"] = traceback.format_exc()[-10000:]
            publish()
            raise

    with modal.enable_output():
        app.deploy()
    call = supervise.spawn(plan["bundle_id"], plan["assignments"][profile], profile,
                           original["call_id"], original["invocation"], cap, pilot_cost)
    write(directory / profile / "initial-workflow.json", original)
    record = dict(original, call_id=call.object_id, original_call_id=original["call_id"],
                  supervised=True, budget_usd=cap, pilot_cost_estimate_usd=pilot_cost)
    write(receipt_path, record)
    print(json.dumps(dict(profile=profile, supervisor_call=call.object_id, generation_cap_usd=cap,
                          pilot_estimate_usd=pilot_cost), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--cap-usd", type=float, default=12)
    args = parser.parse_args()
    deploy_and_submit(args.profile, args.cap_usd)
