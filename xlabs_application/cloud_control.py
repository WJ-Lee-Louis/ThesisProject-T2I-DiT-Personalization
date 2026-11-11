"""PC-independent finalization, additional DB++ GPUs, and resumable downloads."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import zipfile

from .client import connect, current
from .config import ACCOUNTS, APP, RUNS
from .io import read, sha, write

BASE = Path(__file__).resolve().parent
CLOUD_APP = "xlabs-application-offline-v1"
COMBINE_APP = "xlabs-application-consolidation-v1"
READER_SECRET = "xlabs-application-cross-workspace-results"


def plus_shards(plan):
    rows = [r for r in plan["rows"] if r["benchmark"] == "dreambench_plus"]
    refs = sorted({r["reference_id"] for r in rows})
    owners = {ref: i % 2 for i, ref in enumerate(refs)}
    shards = {f"dreambench-plus-{i}": [r["case_id"] for r in rows if owners[r["reference_id"]] == i] for i in range(2)}
    selected = [case for cases in shards.values() for case in cases]
    if len(selected) != 1350 or len(set(selected)) != 1350 or any(len(c) != 675 for c in shards.values()):
        raise ValueError("Expected two disjoint 675-case DreamBench++ shards")
    return shards


def build_app(info, profile, app_name=CLOUD_APP):
    import modal
    runs = modal.Volume.from_name(RUNS)
    image = (modal.Image.debian_slim(python_version="3.11").pip_install("Pillow==11.3.0")
             .add_local_dir(info["source"], "/root", copy=True)
             .add_local_file(BASE / "cloud_runtime.py", "/root/xlabs_application/cloud_runtime.py", copy=True))
    app = modal.App(app_name)

    @app.function(name="probe", serialized=True, image=image, volumes={"/work": runs},
                  cpu=(.125, .125), memory=(512, 512), timeout=120, scaledown_window=2)
    def probe(bundle_id):
        from xlabs_application.io import read
        root = Path("/work/bundles") / bundle_id
        plan = read(root / "plan.json")
        plus = [r for r in plan["rows"] if r["benchmark"] == "dreambench_plus"]
        assert len(plus) == 1350
        assert all((root / r[k]).is_file() for r in plus for k in ("reference", "conditioning"))
        return dict(status="ready", bundle_id=bundle_id, code=plan["code"], dreambench_plus_cases=len(plus))

    @app.function(name="finish", serialized=True, image=image, volumes={"/work": runs},
                  cpu=(.125, .125), memory=(1024, 1024), timeout=86400, retries=0,
                  max_containers=4, scaledown_window=2)
    def finish(bundle_id, cases, shard, initial_call_id, rescue_dreambench=False, pilot_cost=0):
        import traceback
        from xlabs_application.cloud_runtime import finish_shard
        from xlabs_application.config import APP, worker_rate
        from xlabs_application.io import write
        state = dict(status="waiting", initial_call_id=initial_call_id, children=[])
        path = Path("/work/offline_state") / bundle_id / ("finish-" + shard + ".json")

        def publish():
            write(path, state)
            runs.commit()

        publish()
        try:
            result = modal.FunctionCall.from_id(initial_call_id).get()
            if result["status"] == "budget_checkpoint" and rescue_dreambench:
                # Additional-spend authorization: one bounded recovery after the old $12 admission.
                spent = result["generation"]["estimated_compute_usd"] + pilot_cost
                remaining = 16.0 - spent
                if remaining < .2:
                    raise ValueError("DreamBench $16/account generation admission exhausted")
                state.update(status="generating", generation_cap_usd=16.0)
                child = modal.Function.from_name(APP, "generate_h100").spawn(
                    bundle_id, cases, "offline-recovery-" + shard,
                    min(21000, remaining / worker_rate("H100") - 30), False, False)
                state["children"].append(child.object_id)
                publish()
                generation = child.get()
                if generation["status"] != "complete":
                    raise ValueError("DreamBench recovery incomplete: " + generation["status"])
                result = dict(status="complete", generation={k: v for k, v in generation.items() if k != "events"})
                for phase in ("evaluate", "export"):
                    state["status"] = phase
                    function = modal.Function.from_name(APP, phase)
                    child = function.spawn(bundle_id, cases) if phase == "evaluate" else function.spawn(bundle_id, cases, shard)
                    state["children"].append(child.object_id)
                    publish()
                    result["evaluation" if phase == "evaluate" else "export"] = child.get()
            state["status"] = "verifying"
            publish()
            ready = finish_shard(bundle_id, cases, shard, result, runs)
            state.update(status="complete", ready=ready)
            publish()
            return ready
        except Exception:
            state.update(status="failed", error=traceback.format_exc()[-10000:])
            publish()
            raise

    secrets = [modal.Secret.from_name(READER_SECRET)] if profile == "wj-lee-louis" else []

    @app.function(name="consolidate", serialized=True, image=image, volumes={"/work": runs}, secrets=secrets,
                  cpu=(.125, .125), memory=(1024, 1024), timeout=86400, retries=0,
                  max_containers=2, scaledown_window=2)
    def consolidate(bundle_id, benchmark, handles):
        import os
        import traceback
        from modal.client import Client
        from xlabs_application.cloud_runtime import merge_cloud
        from xlabs_application.io import write
        path = Path("/work/offline_state") / bundle_id / (benchmark + ".json")
        state = dict(status="waiting", handles=handles)

        def publish():
            write(path, state)
            runs.commit()

        publish()
        try:
            shards = []
            for handle in handles:
                if handle["profile"] == "wj-lee-louis":
                    ready = modal.FunctionCall.from_id(handle["call_id"]).get()
                    volume = runs
                else:
                    # from_credentials already opens its connection and registers shutdown cleanup.
                    other = Client.from_credentials(os.environ["XLABS_RESULTS_TOKEN_ID"],
                                                    os.environ["XLABS_RESULTS_TOKEN_SECRET"])
                    ready = modal.FunctionCall.from_id(handle["call_id"], client=other).get()
                    volume = modal.Volume.from_name(RUNS, client=other)
                    data = json.loads(b"".join(volume.read_file(ready["directory"] + "/metrics.json")))
                    shards.append(dict(profile=handle["profile"], ready=ready, metrics=data))
                    continue
                data = json.loads(b"".join(volume.read_file(ready["directory"] + "/metrics.json")))
                shards.append(dict(profile=handle["profile"], ready=ready, metrics=data))
            ready = merge_cloud(bundle_id, benchmark, shards, runs)
            state.update(status="complete", ready=ready)
            publish()
            return ready
        except Exception:
            state.update(status="failed", error=traceback.format_exc()[-10000:])
            publish()
            raise

    return app


def stage_metadata(modal, directory, bundle_id):
    # Explicit profiles may stage concurrently; keep their local archives independent.
    folder = directory / "cloud-control" / os.environ.get("MODAL_PROFILE", "metadata")
    folder.mkdir(parents=True, exist_ok=True)
    names = ["cloud_runtime.py", "cloud_control.py", "plus_app.py"]
    with zipfile.ZipFile(folder / "orchestration_source.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for name in names:
            z.write(BASE / name, "xlabs_application/" + name)
    write(folder / "orchestration_hashes.json", {name: sha(BASE / name) for name in names})
    files = [directory / n for n in ("generation_source.zip", "source_hashes.json", "plan.json")]
    files += [folder / n for n in ("orchestration_source.zip", "orchestration_hashes.json")]
    files += [BASE / "manifests" / (b + "_seed42.jsonl") for b in ("dreambench", "dreambench_plus")]
    with modal.Volume.from_name(RUNS).batch_upload(force=True) as upload:
        for path in files:
            upload.put_file(path, "/offline_inputs/" + bundle_id + "/" + path.name)


def submit_finish(modal, directory, plan, profile, shard, cases, call_id, rescue=False, pilot_cost=0):
    path = directory / "cloud-control" / (shard + "-finish.json")
    if not path.exists():
        call = modal.Function.from_name(CLOUD_APP, "finish").spawn(
            plan["bundle_id"], cases, shard, call_id, rescue, pilot_cost)
        write(path, dict(call_id=call.object_id, profile=profile, shard=shard, bundle_id=plan["bundle_id"]))
    return read(path)


def download_file(volume, remote, path, expected=None):
    path = Path(path)
    if expected and path.is_file() and sha(path) == expected:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".download")
    transferred, reported = 0, 0

    def progress(*, advance=0, **_):
        nonlocal transferred, reported
        transferred += advance
        if transferred - reported >= 64 * 1024**2:
            print(f"Downloading {path.name}: {transferred / 1024**2:.0f} MiB", flush=True)
            reported = transferred

    with temp.open("wb") as stream:
        # The pinned SDK can stream blocks into their file offsets without retaining
        # the whole prefetched archive in RAM. Bound parallel block requests locally.
        volume._read_file_into_fileobj(remote, stream, concurrency=16, progress_cb=progress)
    if expected and sha(temp) != expected:
        raise ValueError("Downloaded file hash differs: " + remote)
    temp.replace(path)


def download_ready(volume, ready, profile):
    from .io import extract
    from .vendor.artifact_bundle import verify
    benchmark, bundle_id = ready["benchmark"], ready["bundle_id"]
    if "shard" in ready:
        local = BASE / "results" / benchmark / bundle_id / "lanes" / ready["shard"]
    else:
        local = BASE / "results" / benchmark / bundle_id
    for name, entry in ready["files"].items():
        download_file(volume, ready["directory"] + "/" + name, local / name, entry["sha256"])
    write(local / "ready.json", ready)
    if "archive_index" not in ready:
        return
    index = read(local / "bundle_index.json")
    archives = BASE / "artifacts" / benchmark / bundle_id / ready["shard"]
    remote = str(Path(ready["archive_index"]).parent).replace("\\", "/")
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = [pool.submit(download_file, volume, remote + "/" + part["file"],
                               archives / part["file"], part["sha256"]) for part in index["parts"]]
        for future in pending:
            future.result()
    write(archives / "bundle_index.json", index)
    verify(archives, index)
    restored = BASE / "outputs/restored" / benchmark / bundle_id / ready["shard"]
    if any(not (restored / name).is_file() or sha(restored / name) != m["sha256"] for name, m in index["members"].items()):
        for part in index["parts"]:
            extract(archives / part["file"], restored)
    if any(sha(restored / name) != m["sha256"] for name, m in index["members"].items()):
        raise ValueError("Restored bytes differ")
    write(local / "local_verification.json", dict(profile=profile, archives_verified=True, restored_bytes_verified=True,
          bundle_index_sha256=sha(local / "bundle_index.json"), generated_images=index["generated_images"],
          parts={part["file"]: part["sha256"] for part in index["parts"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["deploy", "deploy-merge", "submit-dreambench", "deploy-plus", "submit-plus", "merge", "status", "download"])
    parser.add_argument("--profile", choices=list(ACCOUNTS), required=True)
    parser.add_argument("--benchmark", choices=["dreambench", "dreambench_plus"], default="dreambench")
    args = parser.parse_args()
    modal = connect(args.profile)
    info, directory, plan = current()
    folder = directory / "cloud-control"
    folder.mkdir(exist_ok=True)
    if args.command == "deploy":
        if args.profile == "wj-lee-louis":
            from modal.config import config
            token_id = config.get("token_id", profile="method-comparison", use_env=False)
            token_secret = config.get("token_secret", profile="method-comparison", use_env=False)
            if not token_id or not token_secret:
                raise ValueError("Other workspace credentials unavailable; no credentials are printed")
            modal.Secret.objects.create(READER_SECRET,
                {"XLABS_RESULTS_TOKEN_ID": token_id, "XLABS_RESULTS_TOKEN_SECRET": token_secret}, allow_existing=True)
        stage_metadata(modal, directory, plan["bundle_id"])
        with modal.enable_output():
            build_app(info, args.profile).deploy()
        result = modal.Function.from_name(CLOUD_APP, "probe").remote(plan["bundle_id"])
        write(folder / (args.profile + "-deployment.json"), result)
        print(json.dumps(result, indent=2))
    elif args.command == "deploy-merge":
        if args.profile != "wj-lee-louis":
            raise ValueError("Combined reports are stored in wj-lee-louis")
        stage_metadata(modal, directory, plan["bundle_id"])
        with modal.enable_output():
            build_app(info, args.profile, COMBINE_APP).deploy()
    elif args.command == "submit-dreambench":
        record = read(directory / args.profile / "submit-receipt.json")
        print(json.dumps(submit_finish(modal, directory, plan, args.profile, args.profile,
                         plan["assignments"][args.profile], record["call_id"], True,
                         record.get("pilot_cost_estimate_usd", 0)), indent=2))
    elif args.command == "deploy-plus":
        if args.profile != "wj-lee-louis":
            raise ValueError("DreamBench++ is authorized only in wj-lee-louis")
        from .plus_app import build_app as plus_app
        with modal.enable_output():
            plus_app(info["source"], ACCOUNTS[args.profile]["cache"]).deploy()
        write(folder / "plus-deployment.json", dict(code=info["code"], gpu="H100", max_containers=2))
    elif args.command == "submit-plus":
        if args.profile != "wj-lee-louis" or read(folder / "plus-deployment.json")["code"] != plan["code"]:
            raise ValueError("Matching wj-lee-louis DB++ deployment required")
        from .plus_app import PLUS_APP
        for shard, cases in plus_shards(plan).items():
            path = folder / (shard + "-workflow.json")
            if not path.exists():
                call = modal.Function.from_name(PLUS_APP, "workflow").spawn(plan["bundle_id"], cases, shard, 30.0)
                write(path, dict(call_id=call.object_id, profile=args.profile, shard=shard,
                      cases=cases, generation_cap_usd=30, submitted_at=datetime.now(timezone.utc).isoformat()))
            record = submit_finish(modal, directory, plan, args.profile, shard, cases, read(path)["call_id"])
            print(json.dumps(record))
    elif args.command == "merge":
        if args.profile != "wj-lee-louis":
            raise ValueError("Combined reports are stored in wj-lee-louis")
        path = folder / (args.benchmark + "-combined.json")
        if path.exists():
            print(json.dumps(read(path)))
            return
        names = list(ACCOUNTS) if args.benchmark == "dreambench" else list(plus_shards(plan))
        handles = [read(folder / (name + "-finish.json")) for name in names]
        call = modal.Function.from_name(COMBINE_APP, "consolidate").spawn(plan["bundle_id"], args.benchmark, handles)
        record = dict(call_id=call.object_id, profile=args.profile, benchmark=args.benchmark, handles=handles)
        write(path, record)
        print(json.dumps(record, indent=2))
    elif args.command in ("status", "download"):
        volume = modal.Volume.from_name(RUNS)
        for path in sorted(folder.glob("*-finish.json")) + sorted(folder.glob("*-combined.json")):
            record = read(path)
            if record["profile"] != args.profile:
                continue
            state_name = "finish-" + record["shard"] if "shard" in record else record["benchmark"]
            try:
                state = json.loads(b"".join(volume.read_file(f"offline_state/{plan['bundle_id']}/{state_name}.json")))
            except FileNotFoundError:
                state = dict(status="starting")
            if args.command == "download" and state["status"] == "complete":
                download_ready(volume, state["ready"], args.profile)
            print(json.dumps(dict(receipt=path.name, **state), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
