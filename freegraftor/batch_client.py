"""VS Code/CMD-friendly client for FreeGraftor Modal batch experiments."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path, PurePosixPath
import sys
import time
from uuid import uuid4

import modal


EXPERIMENT_ROOT = Path(__file__).resolve().parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from freegraftor_experiment.batch_config import BatchConfig
from freegraftor_experiment.batch_manifest import build_batch_manifest
from freegraftor_experiment.constants import (
    APP_NAME,
    CLASS_NAME,
    INPUTS_VOLUME_NAME,
    RESULTS_VOLUME_NAME,
)
from freegraftor_experiment.implementation import compute_implementation_digest
from freegraftor_experiment.prompt_catalog import PROMPT_CATALOG


DEFAULT_REFERENCE_ROOT = (
    EXPERIMENT_ROOT.parent
    / "dreambooth_reference_dataset"
    / "InstantX-IP-Adapter-DreamBench_filtered"
)
DEFAULT_OUTPUT_ROOT = EXPERIMENT_ROOT / "outputs"
JOB_RECORD_ROOT = EXPERIMENT_ROOT / ".modal_jobs"
EXPECTED_PNG_COUNT = 10 * len(PROMPT_CATALOG)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the 10-reference x 16-prompt FreeGraftor baseline."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    preview = commands.add_parser("preview", help="Validate and print the 160 jobs.")
    preview.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    preview.set_defaults(handler=command_preview)

    warmup = commands.add_parser(
        "warmup",
        help="Download/cache all checkpoints and verify the deployed model stack.",
    )
    warmup.add_argument("--yes", action="store_true")
    warmup.set_defaults(handler=command_warmup)

    access = commands.add_parser(
        "check-access",
        help="Check the Modal HF secret and gated FLUX access without a GPU.",
    )
    access.set_defaults(handler=command_check_access)

    submit = commands.add_parser(
        "submit",
        help="Upload references and asynchronously start one baseline batch.",
    )
    submit.add_argument("--batch-id")
    submit.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    submit.add_argument("--shards", type=int, choices=(1, 2), default=2)
    submit.add_argument("--seed", type=int, default=42)
    submit.add_argument("--width", type=int, default=512)
    submit.add_argument("--height", type=int, default=512)
    submit.add_argument("--steps", type=int, default=25)
    submit.add_argument("--guidance", type=float, default=3.0)
    submit.add_argument("--inject-match-dropout", type=float, default=0.5)
    submit.add_argument("--alignment", default="center")
    submit.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing outputs in an otherwise identical batch.",
    )
    submit.add_argument(
        "--resume",
        action="store_true",
        help="Allow resubmission of an existing identical batch ID.",
    )
    submit.add_argument("--yes", action="store_true")
    submit.set_defaults(handler=command_submit)

    status = commands.add_parser("status", help="Show all shard states for a batch.")
    status.add_argument("--batch-id", required=True)
    status.set_defaults(handler=command_status)

    download = commands.add_parser(
        "download",
        help="Resumably download results into outputs/<batch-id>.",
    )
    download.add_argument("--batch-id", required=True)
    download.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    download.add_argument("--force", action="store_true")
    download.add_argument("--allow-partial", action="store_true")
    download.set_defaults(handler=command_download)
    return parser


def command_preview(args: argparse.Namespace) -> int:
    manifest = build_batch_manifest(args.reference_root)
    print_manifest(manifest)
    return 0


def command_warmup(args: argparse.Namespace) -> int:
    if not args.yes:
        print("Warm-up uses an A100-80GB and may download large checkpoints.")
        print("Run the same command with --yes to continue.")
        return 2
    if not _check_hf_access():
        return 1
    print("Loading the deployed FreeGraftor stack; first run can take a while...")
    remote_class = modal.Cls.from_name(APP_NAME, CLASS_NAME)
    result = remote_class().warmup.remote()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def command_check_access(_args: argparse.Namespace) -> int:
    return 0 if _check_hf_access() else 1


def command_submit(args: argparse.Namespace) -> int:
    manifest = build_batch_manifest(args.reference_root)
    batch_id = args.batch_id or default_batch_id()
    implementation_digest = compute_implementation_digest(EXPERIMENT_ROOT)
    config = BatchConfig(
        batch_id=batch_id,
        seed=args.seed,
        width=args.width,
        height=args.height,
        num_steps=args.steps,
        guidance=args.guidance,
        inject_match_dropout=args.inject_match_dropout,
        alignment=args.alignment,
        requires_offload=True,
        shard_count=args.shards,
        overwrite=args.overwrite,
        expected_manifest_digest=manifest.semantic_digest,
        implementation_digest=implementation_digest,
    )
    previous = _read_job_record(batch_id)
    if previous is not None:
        if not args.resume:
            print(
                f"Batch ID {batch_id!r} already has a local record. "
                "Use --resume only after checking its current status."
            )
            return 1
        if previous.get("config") != config.to_dict():
            print("The existing batch ID has a different configuration; use a new ID.")
            return 1

    print_manifest(manifest)
    groups = partition_references(
        [reference.stem for reference in manifest.references],
        args.shards,
    )
    print("\nSubmission settings")
    print(f"  batch ID: {batch_id}")
    print(f"  images: {EXPECTED_PNG_COUNT}")
    print(f"  GPU shards: {len(groups)} (A100-80GB, up to two containers)")
    print(
        f"  paper settings: {config.width}x{config.height}, "
        f"steps={config.num_steps}, guidance={config.guidance}, "
        f"dropout={config.inject_match_dropout}, seed={config.seed}"
    )
    print(f"  alignment/offload: {config.alignment}/{config.requires_offload}")
    for index, stems in enumerate(groups):
        print(f"  shard {index}: {', '.join(stems)}")
    if not args.yes:
        print("\nNo GPU task was submitted. Add --yes after reviewing the settings.")
        return 2

    if not _check_hf_access():
        print("No A100 shard was submitted.")
        return 1

    input_volume = modal.Volume.from_name(INPUTS_VOLUME_NAME, create_if_missing=True)
    remote_root = f"/{manifest.semantic_digest}"
    print("\nUploading 10 references to the content-addressed input Volume...")
    with input_volume.batch_upload(force=True) as upload:
        for reference in manifest.references:
            upload.put_file(
                reference.reference_path,
                f"{remote_root}/references/{reference.filename}",
            )

    record = previous or {
        "batch_id": batch_id,
        "submitted_at": datetime.now().astimezone().isoformat(),
        "reference_root": str(Path(args.reference_root).resolve()),
        "manifest_digest": manifest.semantic_digest,
        "config": config.to_dict(),
        "calls": [],
    }
    remote_class = modal.Cls.from_name(APP_NAME, CLASS_NAME)
    remote_instance = remote_class()
    print("Submitting asynchronous shard calls...")
    for shard_index, stems in enumerate(groups):
        call = remote_instance.generate_shard.spawn(
            config.to_dict(),
            stems,
            shard_index,
            len(groups),
        )
        call_record = {
            "shard_index": shard_index,
            "shard_count": len(groups),
            "reference_stems": stems,
            "call_id": call.object_id,
            "dashboard_url": call.get_dashboard_url(),
            "submitted_at": datetime.now().astimezone().isoformat(),
        }
        record["calls"].append(call_record)
        _write_job_record(record)
        print(f"  shard {shard_index}: {call.object_id}")

    print("\nSubmission complete. The local terminal and PC may now be closed.")
    print("Status:")
    print(f"  python freegraftor\\batch_client.py status --batch-id {batch_id}")
    print("Logs:")
    print(f"  python -m modal app logs {APP_NAME}")
    print("Download after all 160 PNGs are complete:")
    print(f"  python freegraftor\\batch_client.py download --batch-id {batch_id}")
    return 0


def command_status(args: argparse.Namespace) -> int:
    BatchConfig(batch_id=args.batch_id)
    statuses = _read_shard_statuses(args.batch_id)
    if statuses:
        print("Remote shard status")
        for status in statuses:
            print(
                f"  shard {status.get('shard_index')}: {status.get('state')} | "
                f"processed {status.get('processed_count')}/{status.get('total_count')} | "
                f"generated {status.get('generated_count')}, "
                f"skipped {status.get('skipped_count')}, "
                f"failed {status.get('failed_count')}"
            )
        print(
            "  aggregate: "
            f"processed {sum(item.get('processed_count', 0) for item in statuses)}/"
            f"{sum(item.get('total_count', 0) for item in statuses)}, "
            f"failed {sum(item.get('failed_count', 0) for item in statuses)}"
        )
    else:
        print("Remote shard status has not been written yet.")

    record = _read_job_record(args.batch_id)
    if record is None:
        print("No local call record was found; Volume status above is still authoritative.")
        return 0 if statuses else 1
    had_failure = False
    print("\nModal calls")
    for item in record.get("calls", []):
        call_id = item["call_id"]
        try:
            result = modal.FunctionCall.from_id(call_id).get(timeout=0)
        except TimeoutError:
            state = "pending/running"
        except modal.exception.OutputExpiredError:
            state = "result expired (use Volume status)"
        except Exception as error:
            state = f"failed: {type(error).__name__}: {error}"
            had_failure = True
        else:
            state = f"completed: {result.get('state', 'unknown')}"
        print(f"  shard {item['shard_index']} {call_id}: {state}")
    return 1 if had_failure else 0


def command_download(args: argparse.Namespace) -> int:
    BatchConfig(batch_id=args.batch_id)
    volume = modal.Volume.from_name(RESULTS_VOLUME_NAME)
    try:
        entries = _retry(lambda: volume.listdir(args.batch_id, recursive=True))
    except (FileNotFoundError, modal.exception.NotFoundError):
        print(f"No remote results found for batch {args.batch_id!r}.")
        return 1
    files = [entry for entry in entries if entry.type.name == "FILE"]
    png_files = [entry for entry in files if entry.path.casefold().endswith(".png")]
    if len(png_files) != EXPECTED_PNG_COUNT and not args.allow_partial:
        print(
            f"Remote PNG count is {len(png_files)}/{EXPECTED_PNG_COUNT}. "
            "Wait for completion, or use --allow-partial intentionally."
        )
        return 1

    output_root = Path(args.output_root).expanduser().resolve()
    destination_root = _safe_local_child(output_root, args.batch_id)
    plan = []
    for entry in files:
        relative = _relative_remote_path(entry.path, args.batch_id)
        local_path = _safe_local_child(destination_root, str(relative))
        if (
            local_path.is_file()
            and not args.force
            and local_path.stat().st_size == entry.size
        ):
            continue
        plan.append((entry, local_path))

    print(
        f"Downloading {len(plan)} changed/missing files "
        f"(remote PNG {len(png_files)}) to {destination_root}..."
    )
    for index, (entry, local_path) in enumerate(plan, start=1):
        local_path.parent.mkdir(parents=True, exist_ok=True)
        _download_entry(volume, entry.path, local_path)
        if index % 20 == 0 or index == len(plan):
            print(f"  {index}/{len(plan)}")

    local_png_count = sum(
        path.is_file() for path in destination_root.rglob("*.png")
    )
    print(f"Download complete: {destination_root}")
    print(f"Local PNG count: {local_png_count}")
    if not args.allow_partial and local_png_count != EXPECTED_PNG_COUNT:
        return 1
    return 0


def partition_references(stems: list[str], shard_count: int) -> list[list[str]]:
    if shard_count <= 0 or shard_count > len(stems):
        raise ValueError("Invalid shard count")
    groups = [[] for _ in range(shard_count)]
    for index, stem in enumerate(stems):
        groups[index % shard_count].append(stem)
    return groups


def _check_hf_access() -> bool:
    print("Checking Modal secret and gated FLUX.1-dev access on CPU...")
    function = modal.Function.from_name(APP_NAME, "check_hf_access")
    result = function.remote()
    if result.get("flux_access"):
        print(
            "  access: OK"
            + (
                f" (account={result.get('account')}, key={result.get('token_key')})"
                if result.get("account")
                else ""
            )
        )
        return True
    print(f"  access: FAILED ({result.get('error_type')})")
    if result.get("authenticated"):
        print(f"  token authentication: OK (account={result.get('account')})")
    print(f"  {result.get('error')}")
    print(
        "Accept the FLUX.1-dev terms with the token's Hugging Face account, "
        "then replace huggingface-secret with a read-enabled token."
    )
    return False


def print_manifest(manifest) -> None:
    dogs = sum(item.species == "dog" for item in manifest.references)
    cats = sum(item.species == "cat" for item in manifest.references)
    print("Validated FreeGraftor baseline manifest")
    print(f"  references: {len(manifest.references)} (dog {dogs}, cat {cats})")
    print(f"  prompts/reference: {len(PROMPT_CATALOG)}")
    print(f"  total outputs: {len(manifest.jobs)}")
    print(f"  semantic digest: {manifest.semantic_digest}")
    print("  references:")
    for item in manifest.references:
        print(f"    - {item.filename} -> {item.species}")
    print("  prompts:")
    for item in PROMPT_CATALOG:
        print(f"    - [{item.composition}] {item.suffix}")


def default_batch_id() -> str:
    return f"freegraftor-baseline-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def _record_path(batch_id: str) -> Path:
    BatchConfig(batch_id=batch_id)
    return JOB_RECORD_ROOT / f"{batch_id}.json"


def _write_job_record(record: dict) -> None:
    path = _record_path(record["batch_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".record-{uuid4().hex}.tmp"
    try:
        temporary.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_job_record(batch_id: str) -> dict | None:
    path = _record_path(batch_id)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_shard_statuses(batch_id: str) -> list[dict]:
    volume = modal.Volume.from_name(RESULTS_VOLUME_NAME)
    path = f"{batch_id}/_metadata/shards"
    try:
        entries = _retry(lambda: volume.listdir(path))
    except (FileNotFoundError, modal.exception.NotFoundError):
        return []
    statuses = []
    for entry in sorted(entries, key=lambda item: item.path):
        if entry.type.name != "FILE" or not entry.path.endswith(".json"):
            continue
        data = b"".join(_retry(lambda: list(volume.read_file(entry.path))))
        statuses.append(json.loads(data.decode("utf-8")))
    return statuses


def _relative_remote_path(remote_path: str, batch_id: str) -> PurePosixPath:
    normalized = PurePosixPath(remote_path.lstrip("/"))
    try:
        return normalized.relative_to(PurePosixPath(batch_id))
    except ValueError as error:
        raise ValueError(f"Remote path escapes batch {batch_id!r}: {remote_path}") from error


def _safe_local_child(root: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute():
        raise ValueError(f"Expected a relative path: {relative!r}")
    resolved_root = root.resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"Path escapes configured output root: {relative!r}") from error
    return candidate


def _download_entry(volume, remote_path: str, local_path: Path) -> None:
    last_error = None
    for attempt in range(5):
        temporary = local_path.parent / f".download-{uuid4().hex}.tmp"
        try:
            with temporary.open("wb") as stream:
                for chunk in volume.read_file(remote_path):
                    stream.write(chunk)
            os.replace(temporary, local_path)
            return
        except (ConnectionError, modal.exception.ConnectionError, OSError) as error:
            last_error = error
            if attempt == 4:
                raise
            time.sleep(2**attempt)
        finally:
            temporary.unlink(missing_ok=True)
    raise RuntimeError(f"Download failed: {remote_path}") from last_error


def _retry(operation):
    last_error = None
    for attempt in range(5):
        try:
            return operation()
        except (ConnectionError, modal.exception.ConnectionError, OSError) as error:
            last_error = error
            if attempt == 4:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("Operation failed after retries") from last_error


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
