"""Continue the measured DreamBench run using its immutable deployed GPU workers."""
import json
from pathlib import Path
import uuid


def main():
    from freecus.modal_client import connect, STATE, read
    from freecus.modal_remote import RUNS_NAME, APP_NAME
    modal = connect("method-comparison")
    previous = read(STATE / "workflow.json")
    if previous.get("continuation"):
        raise ValueError("Continuation already submitted; inspect workflow-status")
    plan = read(STATE / "latest-dreambench.json")
    runs = modal.Volume.from_name(RUNS_NAME)
    app = modal.App("freecus-dreambench-continuation")
    image = (modal.Image.debian_slim(python_version="3.11")
             .pip_install("huggingface-hub==0.29.3")
             .add_local_dir(plan["source"], "/root/freecus"))

    @app.function(name="continue_run", serialized=True, image=image, volumes={"/work": runs},
                  cpu=(0.125, 0.125), memory=(512, 512), timeout=86400,
                  max_containers=1, scaledown_window=2, retries=0)
    def continue_run(workflow_id, parent_id, plan):
        import time
        from freecus.modal_client import pilot_projection
        from freecus.modal_cost import rate
        from freecus.run import write_json
        runs.reload()
        parent = json.loads((Path("/work/workflows") / f"{parent_id}.json").read_text())
        if parent["status"] != "budget_review_required":
            raise ValueError("Parent must be stopped at the measured budget gate")
        pilot = parent["phases"][0]
        estimate = pilot_projection([pilot], "dreambench")
        spent = parent["accounted_compute_usd"]
        budget = min(29 - spent, estimate["compute_usd_with_margin"])
        if estimate["compute_usd_with_margin"] > 29 - spent:
            raise ValueError("Measured estimate exceeds $29 compute admission")
        invocation = uuid.uuid4().hex
        phase = dict(mode="full", benchmark="dreambench", invocation_id=invocation, budget_usd=budget, calls=[])
        state = dict(workflow_id=workflow_id, parent_workflow=parent_id, status="full_dreambench",
                     admission_usd=29, reserve_usd=1, accounted_compute_usd=spent,
                     next_projection=estimate, phases=[pilot, phase], started_at=time.time())
        report = Path("/work/workflows") / f"{workflow_id}.json"

        def save():
            state["updated_at"] = time.time()
            write_json(report, state)
            runs.commit()

        save()
        gpu = modal.Function.from_name(APP_NAME, "gpu_lane")
        pending = []
        for lane, cases in enumerate(plan["lanes"]):
            call = gpu.spawn(plan["bundle_id"], invocation, lane, cases,
                             min(13800, budget / 2 / rate() - 10))
            phase["calls"].append(dict(id=call.object_id, lane=lane, cases=len(cases), budget_usd=budget / 2))
            pending.append(call)
            save()
        for i, call in enumerate(pending):
            try:
                result = call.get()
            except Exception as exc:
                result = dict(status="failed", error=repr(exc), estimated_compute_usd=budget / 2)
            phase["calls"][i]["result"] = result
            state["accounted_compute_usd"] += result["estimated_compute_usd"]
            save()
        state["status"] = "complete" if all(c["result"]["status"] == "complete" for c in phase["calls"]) else "generation_failed_or_budget_paused"
        save()
        return state

    with modal.enable_output():
        app.deploy()
    workflow_id = uuid.uuid4().hex
    call = continue_run.spawn(workflow_id, previous["workflow_id"],
                             {k: plan[k] for k in ["bundle_id", "lanes"]})
    (STATE / "workflow-initial.json").write_text(json.dumps(previous, indent=2), encoding="utf-8")
    record = dict(workflow_id=workflow_id, call_id=call.object_id, app=app.name,
                  profile="method-comparison", compute_admission_usd=29, reserve_usd=1,
                  continuation=True, plans=[previous["plans"][0]])
    (STATE / "workflow.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k != "plans"}, indent=2))


if __name__ == "__main__":
    main()
