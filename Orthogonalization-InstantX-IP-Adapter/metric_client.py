"""Local CMD client for Modal XVerse metric evaluation."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import time
from uuid import uuid4

import modal


EXPERIMENT_ROOT = Path(__file__).resolve().parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from orthogonalization_instantx_ip_adapter.config import (
    METRIC_APP_NAME,
    METRIC_CLASS_NAME,
    METRIC_RESULTS_VOLUME_NAME,
)
from orthogonalization_instantx_ip_adapter.evaluation_manifest import (
    DEFAULT_EVALUATION_DATASET_ROOT,
    EVALUATION_CONDITIONS,
    load_evaluation_manifest,
)
from orthogonalization_instantx_ip_adapter.implementation import (
    compute_metric_implementation_digest,
)
from orthogonalization_instantx_ip_adapter.evaluation_layout import (
    DEFAULT_EVALUATION_ID,
    EvaluationLayout,
)


METRICS = ("dpg", "ip", "aes")
BENCHMARKS = ("subset", "custom")
JOB_RECORD_ROOT = EXPERIMENT_ROOT / "run_history" / "metrics"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate downloaded XVerse generation batches on Modal GPUs."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    preview = commands.add_parser("preview")
    preview.add_argument("--run-id", required=True)
    preview.add_argument(
        "--dataset-root", type=Path, default=DEFAULT_EVALUATION_DATASET_ROOT
    )
    preview.add_argument("--evaluation-id", default=DEFAULT_EVALUATION_ID)
    preview.add_argument(
        "--generation-run-id",
        help="Generation batch ID component; defaults to --run-id.",
    )
    preview.set_defaults(handler=command_preview)

    check_access = commands.add_parser("check-access")
    check_access.set_defaults(handler=command_check_access)

    submit = commands.add_parser("submit")
    submit.add_argument("--run-id", required=True)
    submit.add_argument("--metric", choices=METRICS, required=True)
    submit.add_argument(
        "--dataset-root", type=Path, default=DEFAULT_EVALUATION_DATASET_ROOT
    )
    submit.add_argument("--evaluation-id", default=DEFAULT_EVALUATION_ID)
    submit.add_argument(
        "--generation-run-id",
        help="Generation batch ID component; defaults to --run-id.",
    )
    submit.add_argument("--yes", action="store_true")
    submit.set_defaults(handler=command_submit)

    status = commands.add_parser("status")
    status.add_argument("--run-id", required=True)
    status.add_argument("--metric", choices=METRICS, required=True)
    status.set_defaults(handler=command_status)

    download = commands.add_parser("download")
    download.add_argument("--run-id", required=True)
    download.add_argument(
        "--dataset-root", type=Path, default=DEFAULT_EVALUATION_DATASET_ROOT
    )
    download.add_argument("--evaluation-id", default=DEFAULT_EVALUATION_ID)
    download.add_argument("--allow-partial", action="store_true")
    download.add_argument("--force", action="store_true")
    download.add_argument("--scores-only", action="store_true")
    download.add_argument("--retries", type=int, default=5)
    download.add_argument("--workers", type=int, default=12)
    download.set_defaults(handler=command_download)

    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--run-id", required=True)
    aggregate.add_argument(
        "--dataset-root", type=Path, default=DEFAULT_EVALUATION_DATASET_ROOT
    )
    aggregate.add_argument("--evaluation-id", default=DEFAULT_EVALUATION_ID)
    aggregate.set_defaults(handler=command_aggregate)
    return parser


def command_preview(args: argparse.Namespace) -> int:
    profile = _active_profile()
    _reject_environment_token_override(profile)
    run_id = _validate_component(args.run_id, "run-id")
    generation_run_id = _validate_component(
        args.generation_run_id or run_id, "generation-run-id"
    )
    batches = _generation_batches(generation_run_id)
    print(f"Active Modal profile: {profile}")
    print(f"Metric run-id: {run_id}")
    print(f"Generation run-id: {generation_run_id}")
    for benchmark in BENCHMARKS:
        manifest = _validate_local_assembly(
            args.dataset_root, args.evaluation_id, benchmark
        )
        print(
            f"  {benchmark}: {len(manifest.jobs)} images/condition, "
            f"{len(EVALUATION_CONDITIONS)} conditions, local assembly OK"
        )
        for condition, batch_id in batches[benchmark].items():
            print(f"    {condition}: {batch_id}")
    print("Metrics: DPG, IP-Sim, AES (subset and custom in each job)")
    print("No GPU job was submitted.")
    return 0


def command_check_access(args: argparse.Namespace) -> int:
    profile = _active_profile()
    _reject_environment_token_override(profile)
    print(f"Checking deployed metric app under Modal profile: {profile}")
    function = modal.Function.from_name(METRIC_APP_NAME, "check_access")
    result = function.remote()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("access") == "OK" else 1


def command_submit(args: argparse.Namespace) -> int:
    profile = _active_profile()
    _reject_environment_token_override(profile)
    run_id = _validate_component(args.run_id, "run-id")
    for benchmark in BENCHMARKS:
        _validate_local_assembly(
            args.dataset_root, args.evaluation_id, benchmark
        )
    generation_run_id = _validate_component(
        args.generation_run_id or run_id, "generation-run-id"
    )
    batches = _generation_batches(generation_run_id)
    implementation_digest = compute_metric_implementation_digest(EXPERIMENT_ROOT)
    print("Metric submission")
    print(f"  profile: {profile}")
    print(f"  run-id: {run_id}")
    print(f"  generation-run-id: {generation_run_id}")
    print(f"  metric: {args.metric}")
    print("  benchmarks: subset, custom")
    print("  generated images: reused from the evaluation result Volume")
    if not args.yes:
        print("GPU submission not started. Add --yes after checking the settings.")
        return 2

    remote_class = modal.Cls.from_name(METRIC_APP_NAME, METRIC_CLASS_NAME)
    call = remote_class().evaluate.spawn(
        args.metric,
        run_id,
        batches,
        implementation_digest,
    )
    record = {
        "profile": profile,
        "run_id": run_id,
        "generation_run_id": generation_run_id,
        "metric": args.metric,
        "call_id": call.object_id,
        "dashboard_url": call.get_dashboard_url(),
        "submitted_at": datetime.now().astimezone().isoformat(),
        "generation_batches": batches,
        "implementation_digest": implementation_digest,
    }
    _write_job_record(profile, run_id, args.metric, record)
    print("Submitted successfully")
    print(f"  Function Call ID: {call.object_id}")
    print(f"  Dashboard: {record['dashboard_url']}")
    print("The job continues after VS Code or the local PC is closed.")
    return 0


def command_status(args: argparse.Namespace) -> int:
    profile = _active_profile()
    _reject_environment_token_override(profile)
    run_id = _validate_component(args.run_id, "run-id")
    status = _read_remote_status(run_id, args.metric)
    print("Remote Volume status:")
    print(json.dumps(status, ensure_ascii=False, indent=2) if status else "  pending")
    record = _read_job_record(profile, run_id, args.metric)
    if not record:
        return 0 if status else 1
    try:
        result = modal.FunctionCall.from_id(record["call_id"]).get(timeout=0)
    except TimeoutError:
        print("Modal call state: pending or running")
        return 0
    except modal.exception.OutputExpiredError:
        print("Modal call result expired; persisted Volume status is authoritative.")
        return 0
    except Exception as error:
        print(f"Modal call state: failed ({type(error).__name__}: {error})")
        return 1
    print("Modal call state: completed")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def command_download(args: argparse.Namespace) -> int:
    profile = _active_profile()
    _reject_environment_token_override(profile)
    run_id = _validate_component(args.run_id, "run-id")
    volume = modal.Volume.from_name(METRIC_RESULTS_VOLUME_NAME)
    statuses = {
        metric: _read_remote_status(run_id, metric, volume=volume)
        for metric in METRICS
    }
    incomplete = [
        metric
        for metric, status in statuses.items()
        if not status or status.get("state") != "completed"
    ]
    if incomplete and not args.allow_partial:
        print(
            "Metrics are not all complete: " + ", ".join(incomplete) + ". "
            "Wait or use --allow-partial intentionally."
        )
        return 1
    try:
        entries = volume.listdir(run_id, recursive=True)
    except (FileNotFoundError, modal.exception.NotFoundError):
        print(f"No remote metric results found for run {run_id!r}.")
        return 1
    files = [entry for entry in entries if entry.type.name == "FILE"]
    if args.scores_only:
        files = [
            entry
            for entry in files
            if "/a/" not in f"/{entry.path}/" and "/r/crops/" not in f"/{entry.path}/"
            and "/r/masks/" not in f"/{entry.path}/"
        ]
    evaluation_id = _validate_component(args.evaluation_id, "evaluation-id")
    layout = EvaluationLayout(Path(args.dataset_root), evaluation_id)
    destination_root = layout.evaluation_root
    print(f"Downloading {len(files)} metric files to {destination_root}")
    skipped = 0
    pending = []
    for entry in files:
        relative = _relative_remote_path(entry.path, run_id)
        destination = layout.metric_download_path(relative)
        if destination.is_file() and not args.force:
            skipped += 1
            continue
        pending.append((entry.path, destination))
    workers = max(1, min(int(args.workers), 32))
    completed_count = skipped
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _download_file_with_retries,
                volume,
                remote_path,
                destination,
                retries=args.retries,
            ): remote_path
            for remote_path, destination in pending
        }
        for future in as_completed(futures):
            future.result()
            completed_count += 1
            if completed_count % 100 == 0 or completed_count == len(files):
                print(f"  {completed_count}/{len(files)}")
    print(f"Download complete ({skipped} existing files reused)")
    return 0


def command_aggregate(args: argparse.Namespace) -> int:
    _validate_component(args.run_id, "run-id")
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    evaluation_id = _validate_component(args.evaluation_id, "evaluation-id")
    layout = EvaluationLayout(dataset_root, evaluation_id)
    script = dataset_root / "metrics" / "aggregate_scores.py"
    for benchmark in BENCHMARKS:
        score_root = layout.score_root(benchmark)
        inputs = {metric: score_root / f"{metric}.json" for metric in METRICS}
        missing = [str(path) for path in inputs.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Download all three metric JSON files first: " + ", ".join(missing)
            )
        output = layout.summary_root(benchmark)
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--benchmark-root",
                str(dataset_root),
                "--benchmark",
                benchmark,
                "--dpg",
                str(inputs["dpg"]),
                "--ip",
                str(inputs["ip"]),
                "--aes",
                str(inputs["aes"]),
                "--output-dir",
                str(output),
            ],
            check=True,
        )
    print(f"Aggregation complete -> {layout.evaluation_root / 'summaries'}")
    return 0


def _validate_local_assembly(
    dataset_root: Path, evaluation_id: str, benchmark: str
):
    root = Path(dataset_root).expanduser().resolve()
    benchmark_manifest = load_evaluation_manifest(root, benchmark)
    evaluation_id = _validate_component(evaluation_id, "evaluation-id")
    path = EvaluationLayout(root, evaluation_id).generated_manifest(benchmark)
    if not path.is_file():
        raise FileNotFoundError(f"Missing assembled generated manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("benchmark_id") != benchmark_manifest.benchmark_id:
        raise ValueError(f"Benchmark ID mismatch in {path}")
    if set(payload.get("conditions", [])) != set(EVALUATION_CONDITIONS):
        raise ValueError(f"Condition set mismatch in {path}")
    expected = {(job.record_id, job.seed) for job in benchmark_manifest.jobs}
    actual = {condition: set() for condition in EVALUATION_CONDITIONS}
    for row in payload.get("images", []):
        condition = str(row["condition"])
        if condition not in actual:
            raise ValueError(f"Unexpected condition in {path}: {condition}")
        key = (str(row["record_id"]), int(row["seed"]))
        if key in actual[condition]:
            raise ValueError(f"Duplicate local row: {condition}, {key}")
        actual[condition].add(key)
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = path.parent / image_path
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing local generated image: {image_path}")
    for condition, keys in actual.items():
        if keys != expected:
            raise ValueError(
                f"Incomplete local assembly for {benchmark}/{condition}: "
                f"missing={len(expected - keys)}, extra={len(keys - expected)}"
            )
    return benchmark_manifest


def _generation_batches(run_id: str) -> dict[str, dict[str, str]]:
    return {
        benchmark: {
            condition: f"xverse-{run_id}-{benchmark}-{condition}"
            for condition in EVALUATION_CONDITIONS
        }
        for benchmark in BENCHMARKS
    }


def _active_profile() -> str:
    executable = shutil.which("modal")
    if not executable:
        raise RuntimeError("The modal CLI is not available on PATH")
    completed = subprocess.run(
        [executable, "profile", "current"],
        capture_output=True,
        text=True,
        check=False,
    )
    value = completed.stdout.strip()
    if completed.returncode != 0 or not value:
        raise RuntimeError(
            "Could not determine the active Modal profile: "
            f"{completed.stderr.strip()}"
        )
    return value


def _reject_environment_token_override(profile: str) -> None:
    if os.environ.get("MODAL_TOKEN_ID") or os.environ.get("MODAL_TOKEN_SECRET"):
        raise RuntimeError(
            "MODAL_TOKEN_ID/MODAL_TOKEN_SECRET is set and can override the "
            f"active profile ({profile}). Clear both variables in this CMD session."
        )


def _read_remote_status(run_id: str, metric: str, *, volume=None):
    volume = volume or modal.Volume.from_name(METRIC_RESULTS_VOLUME_NAME)
    remote_path = f"{run_id}/_metadata/status_{metric}.json"
    try:
        payload = b"".join(volume.read_file(remote_path))
    except (FileNotFoundError, modal.exception.NotFoundError):
        return None
    return json.loads(payload.decode("utf-8"))


def _download_file_with_retries(
    volume, remote_path: str, destination: Path, *, retries: int
) -> None:
    for attempt in range(retries + 1):
        temporary = _download_temporary_path(destination)
        temporary.parent.mkdir(parents=True, exist_ok=True)
        try:
            with temporary.open("wb") as stream:
                for chunk in volume.read_file(remote_path):
                    stream.write(chunk)
            os.replace(temporary, destination)
            return
        except (ConnectionError, OSError, modal.exception.Error) as error:
            temporary.unlink(missing_ok=True)
            if attempt >= retries:
                raise
            delay = min(2**attempt, 16)
            print(
                f"  retry {attempt + 1}/{retries} for {remote_path} in {delay}s "
                f"({type(error).__name__})"
            )
            time.sleep(delay)


def _download_temporary_path(destination: Path) -> Path:
    return destination.with_name(f".d-{uuid4().hex[:12]}.tmp")


def _relative_remote_path(remote_path: str, run_id: str) -> PurePosixPath:
    remote = PurePosixPath(remote_path)
    prefix = PurePosixPath(run_id)
    try:
        return remote.relative_to(prefix)
    except ValueError as error:
        raise ValueError(f"Remote path escapes run root: {remote_path}") from error


def _safe_local_child(root: Path, relative: str) -> Path:
    candidate = (root / Path(*PurePosixPath(relative).parts)).resolve()
    resolved_root = root.resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ValueError(f"Path escapes local result root: {relative!r}")
    return candidate


def _safe_component(value: object) -> str:
    component = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return component or "unknown"


def _validate_component(value: str, label: str) -> str:
    if not value or value in {".", ".."} or _safe_component(value) != value:
        raise ValueError(f"Unsafe {label}: {value!r}")
    return value


def _record_path(profile: str, run_id: str, metric: str) -> Path:
    return JOB_RECORD_ROOT / run_id / f"{metric}.json"


def _write_job_record(profile: str, run_id: str, metric: str, record: dict) -> None:
    path = _record_path(profile, run_id, metric)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".w-{uuid4().hex[:12]}.tmp")
    temporary.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_job_record(profile: str, run_id: str, metric: str):
    path = _record_path(profile, run_id, metric)
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    return record if record.get("profile") == profile else None


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
