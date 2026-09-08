"""Local CLI for XVerse evaluation generation on Modal."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from datetime import datetime
from io import BytesIO
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
PROJECT_ROOT = EXPERIMENT_ROOT.parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from orthogonalization_instantx_ip_adapter.batch_config import (
    BatchGenerationConfig,
)
from orthogonalization_instantx_ip_adapter.config import (
    EVAL_BATCH_APP_NAME,
    EVAL_BATCH_CLASS_NAME,
    EVAL_BATCH_INPUTS_VOLUME_NAME,
    EVAL_BATCH_RESULTS_VOLUME_NAME,
)
from orthogonalization_instantx_ip_adapter.evaluation_manifest import (
    DEFAULT_EVALUATION_DATASET_ROOT,
    DEFAULT_EVALUATION_SOURCE_CAPTIONS,
    EVALUATION_CONDITIONS,
    load_evaluation_manifest,
    load_evaluation_source_captions,
)
from orthogonalization_instantx_ip_adapter.implementation import (
    compute_evaluation_implementation_digest,
)


SOURCE_CONDITIONS = frozenset(
    {
        "source_projection_ssg",
        "st_matched_projection_ssg",
        "st_all_projection_ssg",
    }
)
JOB_RECORD_ROOT = EXPERIMENT_ROOT / ".modal_eval_jobs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate XVerse subset/custom evaluation images on Modal."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    preview = commands.add_parser("preview")
    preview.add_argument("--benchmark", choices=("subset", "custom"), required=True)
    preview.add_argument(
        "--dataset-root", type=Path, default=DEFAULT_EVALUATION_DATASET_ROOT
    )
    preview.add_argument(
        "--source-captions", type=Path, default=DEFAULT_EVALUATION_SOURCE_CAPTIONS
    )
    preview.set_defaults(handler=command_preview)

    check_access = commands.add_parser("check-access")
    check_access.set_defaults(handler=command_check_access)

    submit = commands.add_parser("submit")
    submit.add_argument("--benchmark", choices=("subset", "custom"), required=True)
    submit.add_argument("--condition", choices=EVALUATION_CONDITIONS, required=True)
    submit.add_argument("--batch-id")
    submit.add_argument(
        "--dataset-root", type=Path, default=DEFAULT_EVALUATION_DATASET_ROOT
    )
    submit.add_argument(
        "--source-captions", type=Path, default=DEFAULT_EVALUATION_SOURCE_CAPTIONS
    )
    submit.add_argument("--orthogonalization-strength", type=float, default=1.0)
    submit.add_argument("--ip-adapter-scale", type=float, default=0.7)
    submit.add_argument("--guidance-scale", type=float, default=3.5)
    submit.add_argument("--steps", type=int, default=24)
    submit.add_argument("--width", type=int, default=1024)
    submit.add_argument("--height", type=int, default=1024)
    submit.add_argument("--overwrite", action="store_true")
    submit.add_argument("--yes", action="store_true")
    submit.set_defaults(handler=command_submit)

    status = commands.add_parser("status")
    status_target = status.add_mutually_exclusive_group(required=True)
    status_target.add_argument("--batch-id")
    status_target.add_argument("--call-id")
    status.set_defaults(handler=command_status)

    download = commands.add_parser("download")
    download.add_argument("--batch-id", required=True)
    download.add_argument(
        "--dataset-root", type=Path, default=DEFAULT_EVALUATION_DATASET_ROOT
    )
    download.add_argument("--force", action="store_true")
    download.add_argument("--allow-partial", action="store_true")
    download.add_argument("--retries", type=int, default=5)
    download.set_defaults(handler=command_download)

    assemble = commands.add_parser("assemble")
    assemble.add_argument("--benchmark", choices=("subset", "custom"), required=True)
    assemble.add_argument("--batch-id", action="append", required=True)
    assemble.add_argument(
        "--dataset-root", type=Path, default=DEFAULT_EVALUATION_DATASET_ROOT
    )
    assemble.add_argument("--output", type=Path)
    assemble.add_argument("--allow-subset-conditions", action="store_true")
    assemble.set_defaults(handler=command_assemble)
    return parser


def command_preview(args: argparse.Namespace) -> int:
    manifest = load_evaluation_manifest(args.dataset_root, args.benchmark)
    source = load_evaluation_source_captions(args.source_captions, manifest)
    _print_manifest(manifest)
    print(f"Source captions: {len(source.records)} validated")
    print(f"Source-caption digest: {source.semantic_digest}")
    return 0


def command_check_access(args: argparse.Namespace) -> int:
    profile = _active_profile()
    _reject_environment_token_override(profile)
    print(f"Checking deployed evaluation app under Modal profile: {profile}")
    function = modal.Function.from_name(EVAL_BATCH_APP_NAME, "check_access")
    result = function.remote()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("access") == "OK" else 1


def command_submit(args: argparse.Namespace) -> int:
    profile = _active_profile()
    _reject_environment_token_override(profile)
    manifest = load_evaluation_manifest(args.dataset_root, args.benchmark)
    source = None
    if args.condition in SOURCE_CONDITIONS:
        source = load_evaluation_source_captions(args.source_captions, manifest)
    batch_id = args.batch_id or _default_batch_id(
        profile, args.benchmark, args.condition
    )
    implementation_digest = compute_evaluation_implementation_digest(
        EXPERIMENT_ROOT
    )
    config = BatchGenerationConfig(
        batch_id=batch_id,
        condition=args.condition,
        orthogonalization_strength=args.orthogonalization_strength,
        ip_adapter_scale=args.ip_adapter_scale,
        guidance_scale=args.guidance_scale,
        steps=args.steps,
        seed=42,
        width=args.width,
        height=args.height,
        overwrite=args.overwrite,
        expected_manifest_digest=manifest.semantic_digest,
        implementation_digest=implementation_digest,
        source_caption_digest=source.semantic_digest if source else None,
    )
    _print_manifest(manifest)
    print("\nSubmission")
    print(f"  active Modal profile: {profile}")
    print(f"  batch-id: {batch_id}")
    print(f"  condition: {config.condition}")
    print(f"  projection lambda: {config.orthogonalization_strength:g}")
    print(
        f"  generation: {config.width}x{config.height}, "
        f"steps={config.steps}, IP scale={config.ip_adapter_scale:g}, "
        f"guidance={config.guidance_scale:g}"
    )
    if not args.yes:
        print("\nGPU submission not started. Add --yes after checking the settings.")
        return 2

    input_volume = modal.Volume.from_name(
        EVAL_BATCH_INPUTS_VOLUME_NAME, create_if_missing=True
    )
    remote_root = f"/{manifest.semantic_digest}"
    with input_volume.batch_upload(force=True) as upload:
        for reference in manifest.references:
            upload.put_file(
                reference.path,
                f"{remote_root}/references/{reference.filename}",
            )
        upload.put_file(
            BytesIO(
                json.dumps(
                    manifest.to_upload_dict(), ensure_ascii=False, indent=2
                ).encode("utf-8")
            ),
            f"{remote_root}/evaluation_manifest.json",
        )

    remote_class = modal.Cls.from_name(
        EVAL_BATCH_APP_NAME, EVAL_BATCH_CLASS_NAME
    )
    call = remote_class().generate_batch.spawn(
        config.to_dict(),
        manifest.to_upload_dict(),
        source.to_upload_dict() if source else None,
    )
    record = {
        "profile": profile,
        "batch_id": batch_id,
        "call_id": call.object_id,
        "dashboard_url": call.get_dashboard_url(),
        "submitted_at": datetime.now().astimezone().isoformat(),
        "benchmark": manifest.benchmark,
        "benchmark_id": manifest.benchmark_id,
        "condition": config.condition,
        "manifest_digest": manifest.semantic_digest,
        "source_caption_digest": source.semantic_digest if source else None,
        "config": config.to_dict(),
    }
    _write_job_record(profile, record)
    print("\nSubmitted successfully")
    print(f"  Batch ID: {batch_id}")
    print(f"  Function Call ID: {call.object_id}")
    print(f"  Dashboard: {record['dashboard_url']}")
    print("The job continues after VS Code or the local PC is closed.")
    return 0


def command_status(args: argparse.Namespace) -> int:
    profile = _active_profile()
    _reject_environment_token_override(profile)
    record = None
    batch_id = args.batch_id
    call_id = args.call_id
    if batch_id:
        _validate_batch_id(batch_id)
        record = _read_job_record(profile, batch_id)
        if record:
            call_id = record.get("call_id")
        remote_status = _read_remote_metadata(batch_id, "status.json")
        print("Remote Volume status:")
        print(
            json.dumps(remote_status, ensure_ascii=False, indent=2)
            if remote_status
            else "  not created yet"
        )
    elif call_id:
        record = _find_record_by_call_id(profile, call_id)
        if record:
            batch_id = record["batch_id"]

    if not call_id:
        print("No Function Call ID was found for the active Modal profile.")
        return 1
    call = modal.FunctionCall.from_id(call_id)
    try:
        result = call.get(timeout=0)
    except TimeoutError:
        print("Modal call state: pending or running")
        return 0
    except modal.exception.OutputExpiredError:
        print("Modal call result expired; use the persisted Volume status.")
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
    _validate_batch_id(args.batch_id)
    volume = modal.Volume.from_name(EVAL_BATCH_RESULTS_VOLUME_NAME)
    status = _read_remote_metadata(args.batch_id, "status.json", volume=volume)
    config = _read_remote_metadata(args.batch_id, "config.json", volume=volume)
    eval_manifest = _read_remote_metadata(
        args.batch_id, "evaluation_manifest.json", volume=volume
    )
    if not status or not config or not eval_manifest:
        print("Remote batch metadata is missing in the active Modal profile.")
        return 1
    expected = int(status["total_count"])
    try:
        entries = volume.listdir(args.batch_id, recursive=True)
    except (FileNotFoundError, modal.exception.NotFoundError):
        print(f"No remote results found for batch {args.batch_id!r}.")
        return 1
    files = [entry for entry in entries if entry.type.name == "FILE"]
    png_files = [entry for entry in files if entry.path.casefold().endswith(".png")]
    if len(png_files) != expected and not args.allow_partial:
        print(
            f"Remote PNG count is {len(png_files)}/{expected}. "
            "Wait for completion or use --allow-partial intentionally."
        )
        return 1

    output_root = Path(args.dataset_root).expanduser().resolve() / "evaluation_outputs"
    destination_root = _safe_local_child(
        output_root,
        f"{_safe_component(profile)}/{args.batch_id}",
    )
    print(
        f"Downloading {len(files)} files (PNG {len(png_files)}) to "
        f"{destination_root}"
    )
    skipped = 0
    for index, entry in enumerate(files, start=1):
        relative = _relative_remote_path(entry.path, args.batch_id)
        local_path = _safe_local_child(destination_root, str(relative))
        is_png = local_path.suffix.casefold() == ".png"
        if local_path.exists() and is_png and not args.force:
            skipped += 1
            continue
        local_path.parent.mkdir(parents=True, exist_ok=True)
        _download_file_with_retries(
            volume, entry.path, local_path, retries=args.retries
        )
        if index % 20 == 0 or index == len(files):
            print(f"  {index}/{len(files)}")

    local_pngs = list(destination_root.rglob("*.png"))
    if len(local_pngs) != expected and not args.allow_partial:
        raise RuntimeError(
            f"Local PNG validation failed: {len(local_pngs)}/{expected}"
        )
    record = _read_job_record(profile, args.batch_id) or {
        "profile": profile,
        "batch_id": args.batch_id,
        "benchmark": eval_manifest["benchmark"],
        "benchmark_id": eval_manifest["benchmark_id"],
        "condition": config["condition"],
    }
    record["download_path"] = str(destination_root)
    record["downloaded_at"] = datetime.now().astimezone().isoformat()
    _write_job_record(profile, record)
    print(f"Download complete: {len(local_pngs)} PNGs ({skipped} reused)")
    return 0


def command_assemble(args: argparse.Namespace) -> int:
    profile = _active_profile()
    _reject_environment_token_override(profile)
    benchmark_manifest = load_evaluation_manifest(args.dataset_root, args.benchmark)
    merged_rows = []
    seen_conditions = set()
    batch_records = []
    for batch_id in args.batch_id:
        record = _read_job_record(profile, batch_id)
        if not record or not record.get("download_path"):
            raise ValueError(
                f"Batch {batch_id!r} has not been downloaded under profile {profile!r}"
            )
        batch_root = Path(record["download_path"]).resolve()
        generated_path = batch_root / "generated_images_manifest.json"
        generated = json.loads(generated_path.read_text(encoding="utf-8"))
        if generated.get("benchmark_id") != benchmark_manifest.benchmark_id:
            raise ValueError(f"Benchmark mismatch in {batch_id}")
        condition = generated.get("condition")
        if condition in seen_conditions:
            raise ValueError(f"Duplicate condition in assembled batches: {condition}")
        seen_conditions.add(condition)
        for row in generated.get("images", []):
            source = _safe_local_child(generated_path.parent, row["image_path"])
            if not source.is_file():
                raise FileNotFoundError(f"Missing downloaded image: {source}")
            merged = dict(row)
            merged["image_path"] = str(source)
            merged_rows.append(merged)
        batch_records.append(
            {"batch_id": batch_id, "condition": condition, "path": str(batch_root)}
        )

    if not args.allow_subset_conditions and seen_conditions != set(
        EVALUATION_CONDITIONS
    ):
        missing = sorted(set(EVALUATION_CONDITIONS) - seen_conditions)
        extra = sorted(seen_conditions - set(EVALUATION_CONDITIONS))
        raise ValueError(
            f"Six-condition assembly mismatch: missing={missing}, extra={extra}"
        )
    _validate_assembled_rows(benchmark_manifest, merged_rows, seen_conditions)
    output = (
        args.output.expanduser().resolve()
        if args.output
        else (
            Path(args.dataset_root).expanduser().resolve()
            / "evaluation_outputs"
            / _safe_component(profile)
            / args.benchmark
            / "generated_images_manifest.json"
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    for row in merged_rows:
        row["image_path"] = os.path.relpath(row["image_path"], output.parent)
    payload = {
        "schema_version": 1,
        "benchmark_id": benchmark_manifest.benchmark_id,
        "modal_profile": profile,
        "conditions": sorted(seen_conditions),
        "batches": batch_records,
        "images": merged_rows,
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Assembled {len(merged_rows)} images -> {output}")
    return 0


def _validate_assembled_rows(manifest, rows: list[dict], conditions: set[str]) -> None:
    expected = {(job.record_id, job.seed) for job in manifest.jobs}
    by_condition = {condition: set() for condition in conditions}
    for row in rows:
        condition = str(row["condition"])
        key = (str(row["record_id"]), int(row["seed"]))
        if key in by_condition[condition]:
            raise ValueError(f"Duplicate assembled image: {condition}, {key}")
        by_condition[condition].add(key)
    errors = []
    for condition, keys in by_condition.items():
        if keys != expected:
            errors.append(
                f"{condition}: missing={len(expected - keys)}, extra={len(keys - expected)}"
            )
    if errors:
        raise ValueError("Incomplete assembled manifest: " + "; ".join(errors))


def _print_manifest(manifest) -> None:
    seed_counts = sorted({job.seed for job in manifest.jobs})
    print("Evaluation manifest validated")
    print(f"  benchmark: {manifest.benchmark} ({manifest.benchmark_id})")
    print(f"  references: {len(manifest.references)}")
    print(f"  images per condition: {len(manifest.jobs)}")
    print(f"  unique seeds: {len(seed_counts)}")
    print(f"  semantic digest: {manifest.semantic_digest}")


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
            f"active profile ({profile}). Clear both variables in this CMD "
            "session before using the evaluation client."
        )


def _default_batch_id(profile: str, benchmark: str, condition: str) -> str:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    return (
        f"xverse-{_safe_component(profile)}-{benchmark}-{condition}-{timestamp}"
    )[:180]


def _safe_component(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return text or "unknown"


def _record_path(profile: str, batch_id: str) -> Path:
    _validate_batch_id(batch_id)
    return JOB_RECORD_ROOT / _safe_component(profile) / f"{batch_id}.json"


def _write_job_record(profile: str, record: dict) -> None:
    path = _record_path(profile, record["batch_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _read_job_record(profile: str, batch_id: str):
    path = _record_path(profile, batch_id)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _find_record_by_call_id(profile: str, call_id: str):
    root = JOB_RECORD_ROOT / _safe_component(profile)
    if not root.is_dir():
        return None
    for path in root.glob("*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("call_id") == call_id:
            return value
    return None


def _read_remote_metadata(batch_id: str, filename: str, *, volume=None):
    volume = volume or modal.Volume.from_name(EVAL_BATCH_RESULTS_VOLUME_NAME)
    try:
        encoded = b"".join(volume.read_file(f"{batch_id}/_metadata/{filename}"))
    except (FileNotFoundError, modal.exception.NotFoundError):
        return None
    return json.loads(encoded.decode("utf-8"))


def _download_file_with_retries(
    volume, remote_path: str, local_path: Path, *, retries: int
) -> None:
    if retries < 1:
        raise ValueError("--retries must be at least 1")
    for attempt in range(1, retries + 1):
        # Keep the temporary basename nearly as short as the final basename.
        # Deep metadata paths plus a long batch ID can otherwise exceed the
        # classic Windows MAX_PATH limit even though the final path itself is
        # valid. Twelve random hex digits are sufficient for this sequential
        # downloader while retaining same-directory atomic replacement.
        temporary = _download_temporary_path(local_path)
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with temporary.open("wb") as stream:
                for chunk in volume.read_file(remote_path):
                    stream.write(chunk)
            os.replace(temporary, local_path)
            return
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == retries:
                raise
            delay = min(2 ** (attempt - 1), 8)
            print(f"  retry {attempt}/{retries} for {remote_path} in {delay}s")
            time.sleep(delay)


def _download_temporary_path(local_path: Path) -> Path:
    """Return a short same-directory path for an atomic downloaded file."""

    return local_path.parent / f".d-{uuid4().hex[:12]}.tmp"


def _relative_remote_path(remote_path: str, batch_id: str) -> PurePosixPath:
    normalized = PurePosixPath(remote_path.lstrip("/"))
    try:
        return normalized.relative_to(PurePosixPath(batch_id))
    except ValueError as error:
        raise ValueError(f"Remote path escapes batch: {remote_path}") from error


def _safe_local_child(root: Path, relative_value: str) -> Path:
    relative = Path(relative_value)
    if relative.is_absolute():
        raise ValueError(f"Expected relative path: {relative_value}")
    root = root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Path escapes output root: {relative_value}") from error
    return candidate


def _validate_batch_id(batch_id: str) -> None:
    BatchGenerationConfig(batch_id=batch_id)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
