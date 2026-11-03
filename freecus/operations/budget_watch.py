"""Remote $40 continuation without replacing in-flight immutable GPU code."""
import argparse
import json
from pathlib import Path
import uuid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    args = parser.parse_args()
    from freecus.modal_client import connect, BASE, read
    from freecus.modal_remote import RUNS_NAME, APP_NAME
    modal = connect(args.profile)
    primary = args.profile == "method-comparison"
    state_dir = BASE / ("outputs/modal" if primary else f"outputs/modal-{args.profile}")
    receipt_path = state_dir / "workflow.json"
    previous = read(receipt_path)
    if previous.get("budget_watch"):
        raise ValueError("Budget supervisor already submitted")
    benchmark = "dreambench" if primary else "dreambench_plus"
    plan = read(state_dir / f"latest-{benchmark}.json")
    runs = modal.Volume.from_name(RUNS_NAME)
    app = modal.App("freecus-budget-supervisor")
    image = (modal.Image.debian_slim(python_version="3.11")
             .pip_install("huggingface-hub==0.29.3")
             .add_local_dir(plan["source"], "/root/freecus"))

    @app.function(name="supervise", serialized=True, image=image, volumes={"/work": runs},
                  cpu=(0.125, 0.125), memory=(512, 512), timeout=86400,
                  max_containers=1, scaledown_window=2, retries=0)
    def supervise(workflow_id, previous, plan, prior_spend):
        import time
        from freecus.modal_client import pilot_projection
        from freecus.modal_cost import rate
        from freecus.run import write_json, artifact_paths, complete
        report = Path("/work/workflows") / f"{workflow_id}.json"
        parent_report = Path("/work/workflows") / f"{previous['workflow_id']}.json"
        parent_call = modal.FunctionCall.from_id(previous["call_id"])
        state = dict(status="awaiting_existing_workflow", phases=[])

        def save():
            state.update(workflow_id=workflow_id, parent_workflow=previous["workflow_id"],
                         parent_call_id=previous["call_id"], workspace_budget_usd=40,
                         admission_usd=39, previous_attempt_compute_usd=prior_spend,
                         updated_at=time.time())
            write_json(report, state)
            runs.commit()

        save()
        while True:
            try:
                state = parent_call.get(timeout=30)
                break
            except TimeoutError:
                runs.reload()
                if parent_report.exists():
                    state = json.loads(parent_report.read_text())
                save()
        state["accounted_compute_usd"] = state.get("accounted_compute_usd", 0) + prior_spend
        save()
        if state["status"] not in {"budget_review_required", "generation_failed_or_budget_paused"}:
            return state
        if state["status"] == "generation_failed_or_budget_paused":
            last = state["phases"][-1]
            if any(c.get("result", {}).get("status") != "budget_paused" for c in last["calls"]):
                # A CUDA/model/data failure needs investigation, not repeated paid attempts.
                return state
        gpu_type = plan["gpu"]
        estimate = pilot_projection(state["phases"], plan["benchmark"], gpu=gpu_type)
        bundle = Path("/work/bundles") / plan["bundle_id"]
        runs.reload()
        rows = [json.loads(line) for line in (bundle / "manifest.jsonl").read_text().splitlines()]
        by_case = {r["case_id"]: r for r in rows}
        lanes = []
        for lane, cases in enumerate(plan["lanes"]):
            output = Path("/work/results") / plan["bundle_id"] / f"lane-{lane}"
            lanes.append([case for case in cases if not complete(artifact_paths(output, by_case[case]))])
        remaining = 39 - state["accounted_compute_usd"]
        pending_count = sum(map(len, lanes))
        expected = estimate["compute_usd"] * pending_count / len(rows)
        state["next_projection"] = dict(**estimate, pending_images=pending_count,
                                        pending_compute_usd=expected, revised_safety_factor=1.10)
        if expected * 1.10 > remaining:
            state["status"] = "budget_review_required_40usd"
            save()
            return state
        invocation = uuid.uuid4().hex
        phase = dict(mode="full_continuation", benchmark=plan["benchmark"], invocation_id=invocation,
                     budget_usd=remaining, calls=[])
        state["phases"].append(phase)
        state["status"] = "full_" + plan["benchmark"]
        save()
        gpu = modal.Function.from_name(APP_NAME, "gpu_lane")
        pending = []
        for lane, cases in enumerate(lanes):
            if not cases:
                continue
            call = gpu.spawn(plan["bundle_id"], invocation, lane, cases,
                             min(13800, remaining / 2 / rate(gpu_type) - 10))
            phase["calls"].append(dict(id=call.object_id, lane=lane, cases=len(cases), budget_usd=remaining / 2))
            pending.append(call)
            save()
        for i, call in enumerate(pending):
            try:
                result = call.get()
            except Exception as exc:
                result = dict(status="failed", error=repr(exc), estimated_compute_usd=remaining / 2)
            phase["calls"][i]["result"] = result
            state["accounted_compute_usd"] += result["estimated_compute_usd"]
            save()
        state["status"] = "complete" if all(c["result"]["status"] == "complete" for c in phase["calls"]) else "generation_failed_or_budget_paused"
        save()
        return state

    with modal.enable_output():
        app.deploy()
    workflow_id = uuid.uuid4().hex
    remote_plan = {k: plan[k] for k in ["benchmark", "bundle_id", "lanes"]}
    remote_plan["gpu"] = "A100-80GB" if primary else "H100"
    call = supervise.spawn(workflow_id, previous, remote_plan, 0.7311197273311324 if primary else 0)
    (state_dir / "workflow-before-40usd.json").write_text(json.dumps(previous, indent=2), encoding="utf-8")
    record = dict(workflow_id=workflow_id, call_id=call.object_id, profile=args.profile,
                  app=app.name, workspace_budget_usd=40, compute_admission_usd=39,
                  reserve_usd=1, budget_watch=True, parent=previous, plans=previous["plans"])
    receipt_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k not in {"plans", "parent"}}, indent=2))


if __name__ == "__main__":
    main()
