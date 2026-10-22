"""Durable transport for the unchanged benchmark_modal generation implementation.

No inference code or provenance digest in the original runner is modified.
Each long-running lane owns disjoint jobs and commits each PNG/JSON pair.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import io
import json
from pathlib import Path, PurePosixPath
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
VOLUME_NAME = "orthogonalization-dreambench-durable-v1"
LOCK_NAME = "orthogonalization-dreambench-durable-locks-v1"
REMOTE_ROOT = "/benchmark"  # Serialize strings, never a host-specific WindowsPath.
DEFAULT_CONDITIONS = ("baseline", "source_target_ssg")


def legacy():
    return importlib.import_module("benchmark_modal")


def generation_function(workers, gpu):
    """Return Python code only; never serialize a process-local Synchronizer."""
    from types import SimpleNamespace
    _, original = legacy().build_generator(SimpleNamespace(workers=workers, gpu=gpu))
    wrapped = original._get_user_cls().__dict__["generate"]._get_raw_f()
    return inspect.unwrap(wrapped)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def result_paths(root, job):
    return legacy().paths_for(Path(root) / "results", job["settings"]["condition"], job["row"])


def valid_result(root, job):
    png, sidecar = result_paths(root, job)
    if not legacy().complete(png, sidecar, job["signature"]):
        return False
    record = legacy().read(sidecar)
    if record["row"] != job["row"] or record["settings"] != job["settings"]:
        raise ValueError(f"Result provenance mismatch: {sidecar}")
    return True


def save_result(root, job, result, profile, run_id):
    """Publish sidecar last, so interrupted PNG writes are never complete jobs."""
    result = dict(result)
    png_bytes = result.pop("png")
    if result["job_signature"] != job["signature"]:
        raise ValueError("Worker returned a different job signature")
    if hashlib.sha256(png_bytes).hexdigest() != result["png_sha256"]:
        raise ValueError("Worker returned an invalid PNG hash")
    png, sidecar = result_paths(root, job)
    png.parent.mkdir(parents=True, exist_ok=True)
    temp = png.with_suffix(".png.tmp")
    temp.write_bytes(png_bytes)
    temp.replace(png)
    result["modal_profile"] = profile
    result["remote_run_id"] = run_id
    atomic_json(sidecar, result)
    if not valid_result(root, job):
        raise ValueError("Result did not pass the original runner's completion check")


def extract_bundle(bundle, destination):
    """Allow only relative regular files, even for a locally generated archive."""
    destination = Path(destination).resolve()
    with zipfile.ZipFile(bundle) as archive:
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if (path.is_absolute() or ".." in path.parts or "\\" in info.filename
                    or ":" in info.filename or info.is_dir()
                    or (info.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError(f"Unsafe bundle member: {info.filename}")
            target = destination.joinpath(*path.parts).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("Bundle member escapes destination")
        archive.extractall(destination)


def inspect_results(root, plan):
    counts = {c: dict(expected=0, complete=0, pending=0, invalid=[]) for c in plan["conditions"]}
    for job in plan["jobs"]:
        bucket = counts[job["settings"]["condition"]]
        bucket["expected"] += 1
        try:
            if valid_result(root, job):
                bucket["complete"] += 1
            else:
                bucket["pending"] += 1
        except Exception as exc:
            bucket["invalid"].append(dict(case_id=job["row"]["case_id"], error=str(exc)))
    return counts


def run_lane(root, plan, lane, generate, commit, profile, claim=None, release=None):
    """Pure orchestration seam: tests use a fake generator and local filesystem."""
    root = Path(root)
    if not plan['conditions'] or not set(plan['conditions']).issubset(DEFAULT_CONDITIONS):
        raise ValueError('Only baseline and source_target_ssg are enabled; source_target is on hold')
    if not 0 <= lane < plan["workers"]:
        raise ValueError("Invalid lane")
    if claim is not None and not claim():
        raise RuntimeError("This lane is already claimed; use status before resuming")
    state_path = root / "state" / f"lane-{lane}.json"
    state = dict(lane=lane, status="running", saved=0, skipped=0, failed=[], updated_at=time.time())
    consecutive_failures = 0
    try:
        atomic_json(state_path, state)
        commit()
        for index, job in enumerate(plan["jobs"]):
            if index % plan["workers"] != lane:
                continue
            # Integrity/settings errors must stop, never silently mix experiments.
            if valid_result(root, job):
                state["skipped"] += 1
                continue
            job = dict(job)
            image_bytes = (root / job["image_member"]).read_bytes()
            if hashlib.sha256(image_bytes).hexdigest() != job["image_sha256"]:
                raise ValueError("Reference input hash mismatch")
            job["image_bytes"] = image_bytes
            try:
                result = generate(job)
                save_result(root, job, result, profile, plan["run_id"])
            except Exception as exc:
                consecutive_failures += 1
                state["failed"].append(dict(condition=job["settings"]["condition"],
                                            case_id=job["row"]["case_id"], error=repr(exc),
                                            traceback=traceback.format_exc()))
                print(f"FAILED {job['settings']['condition']} {job['row']['case_id']}: {exc}", flush=True)
                traceback.print_exc()
                if (isinstance(exc, AttributeError) and 'Synchronizer' in str(exc)) or consecutive_failures >= 3:
                    raise RuntimeError('Stopping lane after a synchronization error or 3 consecutive failures') from exc
            else:
                consecutive_failures = 0
                state["saved"] += 1
            state["updated_at"] = time.time()
            atomic_json(state_path, state)
            # Commit failures are fatal: never claim data is durable when it isn't.
            commit()
            if consecutive_failures == 0:
                print(f"COMMITTED {job['settings']['condition']} {job['row']['case_id']}", flush=True)
        state["status"] = "partial" if state["failed"] else "completed"
        state["updated_at"] = time.time()
        atomic_json(state_path, state)
        commit()
        return state
    except BaseException as exc:
        state.update(status="interrupted", error=repr(exc), updated_at=time.time())
        atomic_json(state_path, state)
        commit()
        raise
    finally:
        if release is not None:
            release()


def build_app(workers, gpu, timeout_seconds=86400):
    import modal
    from modal_inference import image as generation_image, hf_cache
    from shared.modal_runtime import HF_SECRET_NAME

    app = modal.App("orthogonalization-dreambench-durable-v1")
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    source_file = ROOT / "scripts" / "benchmark_modal.py"
    cpu_image = (modal.Image.debian_slim(python_version="3.11").pip_install("pillow==11.3.0")
                 .add_local_python_source("benchmark_remote_worker")
                 .add_local_file(source_file, "/root/scripts/benchmark_modal.py"))
    gpu_image = (generation_image.add_local_python_source("benchmark_remote_worker")
                 .add_local_file(source_file, "/root/scripts/benchmark_modal.py"))

    # Reuse the original inference code, unwrapping BEFORE serialization.
    original_generate = generation_function(workers, gpu)

    @app.function(image=cpu_image, serialized=True, volumes={str(REMOTE_ROOT): volume}, timeout=3600, memory=2048)
    def initialize(run_id):
        volume.reload()
        root = Path(REMOTE_ROOT) / "runs" / run_id
        if not (root / "ready.json").exists():
            extract_bundle(Path(REMOTE_ROOT) / "uploads" / f"{run_id}.zip", root)
        plan = legacy().read(root / "plan.json")
        if plan["run_id"] != run_id:
            raise ValueError("Run ID mismatch")
        counts = inspect_results(root, plan)
        if any(c["invalid"] for c in counts.values()):
            raise ValueError(f"Invalid uploaded results: {counts}")
        atomic_json(root / "ready.json", dict(run_id=run_id, initialized_at=time.time()))
        volume.commit()
        return counts

    @app.cls(image=gpu_image, gpu=gpu, serialized=True, max_containers=workers,
             min_containers=0, scaledown_window=60, timeout=timeout_seconds, startup_timeout=1800,
             volumes={str(REMOTE_ROOT): volume, "/cache": hf_cache},
             secrets=[modal.Secret.from_name(HF_SECRET_NAME)])
    class DurableGenerator:
        @modal.enter()
        def enter(self):
            self.model = None

        @modal.method()
        def generate_lane(self, run_id, lane, profile):
            locks = modal.Dict.from_name(LOCK_NAME, create_if_missing=True)
            volume.reload()
            root = Path(REMOTE_ROOT) / "runs" / run_id
            plan = legacy().read(root / "plan.json")
            key = f"{run_id}:{lane}"
            owner = modal.current_function_call_id()

            def claim():
                return locks.put(key, dict(call_id=owner, started_at=time.time()), skip_if_exists=True)

            def release():
                locks.pop(key, None)

            return run_lane(root, plan, lane, lambda job: original_generate(self, job),
                            volume.commit, profile, claim, release)

    return app, initialize, DurableGenerator
