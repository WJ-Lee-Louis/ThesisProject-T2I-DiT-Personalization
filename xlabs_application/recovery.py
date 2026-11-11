"""Restart interrupted XLabs queues without changing the frozen renderer."""
from datetime import datetime, timezone
from pathlib import Path
import json
import uuid

from .client import connect, current
from .cloud_control import CLOUD_APP, COMBINE_APP, plus_shards
from .config import APP, RUNS, worker_rate
from .io import read, write
from .plus_app import PLUS_APP


def restart():
    modal = connect("wj-lee-louis")
    info, directory, plan = current()
    cloud = directory / "cloud-control"
    stats = [modal.Function.from_name(app, "generate_h100").get_current_stats() for app in (APP, PLUS_APP)]
    if any(s.num_running_inputs for s in stats):
        print("Existing generation has resumed; no duplicate GPU calls submitted.", flush=True)
        return
    roots = []
    names = ["dreambench-combined.json", "dreambench_plus-combined.json", "wj-lee-louis-finish.json",
             "dreambench-plus-0-finish.json", "dreambench-plus-1-finish.json",
             "dreambench-plus-0-workflow.json", "dreambench-plus-1-workflow.json"]
    paths = [cloud / name for name in names] + [directory / "wj-lee-louis/submit-receipt.json"]
    for path in paths:
        record = read(path)
        if record["profile"] != "wj-lee-louis":
            raise ValueError("Only wj-lee-louis calls may be restarted")
        roots.append(record["call_id"])
        if record.get("original_call_id"):
            roots.append(record["original_call_id"])
    calls = set(roots)

    def walk(nodes):
        for node in nodes:
            calls.add(node.function_call_id)
            walk(node.children)

    for call_id in roots:
        walk(modal.FunctionCall.from_id(call_id).get_call_graph())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    history = directory / "recovery" / stamp
    history.mkdir(parents=True)
    write(history / "cancelled_calls.json", dict(workspace="wj-lee-louis", roots=roots, calls=sorted(calls)))
    # Controllers first, then their old queued GPU inputs; no other workspace is touched.
    for call_id in list(dict.fromkeys(roots)) + sorted(calls - set(roots)):
        modal.FunctionCall.from_id(call_id).cancel(terminate_containers=True)
    for path in paths:
        destination = history / path.parent.name / path.name
        destination.parent.mkdir(exist_ok=True)
        path.replace(destination)
    runs = modal.Volume.from_name(RUNS)
    invocation = "resume-db-" + uuid.uuid4().hex[:12]
    cases = plan["assignments"]["wj-lee-louis"]
    call = modal.Function.from_name(APP, "workflow").spawn(
        plan["bundle_id"], cases, invocation, min(21000, 8.5 / worker_rate("H100") - 30), "wj-lee-louis", "H100")
    record = dict(call_id=call.object_id, kind="submit", profile="wj-lee-louis", code=info["code"],
                  bundle_id=plan["bundle_id"], invocation=invocation, cases=cases, gpu="H100",
                  submitted_at=datetime.now(timezone.utc).isoformat(), recovery_of=str(history))
    write(directory / "wj-lee-louis/submit-receipt.json", record)
    finish = modal.Function.from_name(CLOUD_APP, "finish").spawn(
        plan["bundle_id"], cases, "wj-lee-louis", call.object_id, False, 0)
    write(cloud / "wj-lee-louis-finish.json", dict(call_id=finish.object_id, profile="wj-lee-louis", shard="wj-lee-louis", bundle_id=plan["bundle_id"]))
    for shard, cases in plus_shards(plan).items():
        call = modal.Function.from_name(PLUS_APP, "workflow").spawn(plan["bundle_id"], cases, shard, 30.0)
        write(cloud / (shard + "-workflow.json"), dict(call_id=call.object_id, profile="wj-lee-louis",
              shard=shard, cases=cases, generation_cap_usd=30, recovery_of=str(history), submitted_at=datetime.now(timezone.utc).isoformat()))
        finish = modal.Function.from_name(CLOUD_APP, "finish").spawn(plan["bundle_id"], cases, shard, call.object_id)
        write(cloud / (shard + "-finish.json"), dict(call_id=finish.object_id, profile="wj-lee-louis", shard=shard, bundle_id=plan["bundle_id"]))
    for benchmark, shards in (("dreambench", ["wj-lee-louis", "method-comparison"]),
                              ("dreambench_plus", list(plus_shards(plan)))):
        handles = [read(cloud / (shard + "-finish.json")) for shard in shards]
        call = modal.Function.from_name(COMBINE_APP, "consolidate").spawn(plan["bundle_id"], benchmark, handles)
        write(cloud / (benchmark + "-combined.json"), dict(call_id=call.object_id, profile="wj-lee-louis", benchmark=benchmark, handles=handles))
    # Preserve restart receipts remotely so PC shutdown cannot lose the recovery trail.
    with runs.batch_upload() as upload:
        for path in cloud.glob("*.json"):
            upload.put_file(path, f"/recovery/{plan['bundle_id']}/{stamp}/" + path.name)
        upload.put_file(history / "cancelled_calls.json", f"/recovery/{plan['bundle_id']}/{stamp}/cancelled_calls.json")
        upload.put_file(Path(__file__), f"/recovery/{plan['bundle_id']}/{stamp}/recovery.py")
    print(json.dumps(dict(status="resubmitted", workspace="wj-lee-louis", history=str(history),
                         frozen_generation_code=info["code"]), indent=2), flush=True)


if __name__ == "__main__":
    restart()
