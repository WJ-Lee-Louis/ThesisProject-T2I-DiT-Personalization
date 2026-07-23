"""Local CLI for previewing, submitting, monitoring, and downloading a batch."""

from __future__ import annotations

import argparse
from datetime import datetime
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import sys
from uuid import uuid4

import modal


EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from orthogonalization_instantx_ip_adapter.batch_config import (
    BatchGenerationConfig,
)
from orthogonalization_instantx_ip_adapter.batch_manifest import (
    DEFAULT_REFERENCE_ROOT,
    EXPECTED_JOB_COUNT,
    PROMPT_SUFFIXES,
    build_batch_manifest,
)
from orthogonalization_instantx_ip_adapter.config import (
    BATCH_APP_NAME,
    BATCH_CLASS_NAME,
    BATCH_INPUTS_VOLUME_NAME,
    BATCH_RESULTS_VOLUME_NAME,
)
from orthogonalization_instantx_ip_adapter.implementation import (
    compute_batch_implementation_digest,
)
from orthogonalization_instantx_ip_adapter.source_captions import (
    DEFAULT_SOURCE_CAPTIONS_PATH,
    load_source_caption_manifest,
)


DEFAULT_OUTPUT_ROOT = EXPERIMENT_ROOT / "outputs"
JOB_RECORD_ROOT = EXPERIMENT_ROOT / ".modal_jobs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Submit and recover the 10-reference x 9-prompt InstantX "
            "orthogonalization experiment."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    preview = commands.add_parser(
        "preview",
        help="Validate the local references and print the 90-job manifest.",
    )
    preview.add_argument(
        "--reference-root",
        type=Path,
        default=DEFAULT_REFERENCE_ROOT,
    )
    preview.add_argument(
        "--source-captions",
        type=Path,
        help="Optionally validate a per-reference source caption JSON file.",
    )
    preview.set_defaults(handler=command_preview)

    submit = commands.add_parser(
        "submit",
        help="Upload the references and asynchronously submit one 90-image run.",
    )
    submit.add_argument(
        "--condition",
        required=True,
        choices=(
            "baseline",
            "projection",
            "gate",
            "combined",
            "sa_projection",
            "sa_combined",
        ),
    )
    submit.add_argument("--batch-id")
    submit.add_argument(
        "--reference-root",
        type=Path,
        default=DEFAULT_REFERENCE_ROOT,
    )
    submit.add_argument(
        "--source-captions",
        type=Path,
        default=DEFAULT_SOURCE_CAPTIONS_PATH,
        help=(
            "Reference caption JSON used by sa_projection/sa_combined "
            f"(default: {DEFAULT_SOURCE_CAPTIONS_PATH})"
        ),
    )
    submit.add_argument(
        "--orthogonalization-strength",
        type=float,
        default=0.5,
    )
    submit.add_argument("--ip-adapter-scale", type=float, default=0.7)
    submit.add_argument("--guidance-scale", type=float, default=3.5)
    submit.add_argument("--steps", type=int, default=24)
    submit.add_argument("--seed", type=int, default=42)
    submit.add_argument("--width", type=int, default=960)
    submit.add_argument("--height", type=int, default=1280)
    submit.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate all outputs in an otherwise identical batch.",
    )
    submit.add_argument(
        "--yes",
        action="store_true",
        help="Confirm the remote GPU submission and its associated cost.",
    )
    submit.set_defaults(handler=command_submit)

    status = commands.add_parser(
        "status",
        help="Poll a submitted call without waiting for it to finish.",
    )
    status_target = status.add_mutually_exclusive_group(required=True)
    status_target.add_argument("--batch-id")
    status_target.add_argument("--call-id")
    status.set_defaults(handler=command_status)

    download = commands.add_parser(
        "download",
        help="Download one batch into outputs/<batch-id>.",
    )
    download.add_argument("--batch-id", required=True)
    download.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    download.add_argument(
        "--force",
        action="store_true",
        help="Replace local files that already exist.",
    )
    download.add_argument(
        "--allow-partial",
        action="store_true",
        help="Download even if fewer than 90 remote PNGs are present.",
    )
    download.set_defaults(handler=command_download)
    return parser


def command_preview(args: argparse.Namespace) -> int:
    manifest = build_batch_manifest(args.reference_root)
    print_manifest_summary(manifest)
    if args.source_captions is not None:
        source_captions = load_source_caption_manifest(
            args.source_captions,
            manifest,
        )
        print_source_caption_summary(
            source_captions,
            source_path=args.source_captions,
        )
    return 0


def command_submit(args: argparse.Namespace) -> int:
    manifest = build_batch_manifest(args.reference_root)
    source_captions = None
    if args.condition in {"sa_projection", "sa_combined"}:
        source_captions = load_source_caption_manifest(
            args.source_captions,
            manifest,
        )
    batch_id = args.batch_id or default_batch_id(args.condition)
    implementation_digest = compute_batch_implementation_digest(EXPERIMENT_ROOT)
    config = BatchGenerationConfig(
        batch_id=batch_id,
        condition=args.condition,
        orthogonalization_strength=args.orthogonalization_strength,
        ip_adapter_scale=args.ip_adapter_scale,
        guidance_scale=args.guidance_scale,
        steps=args.steps,
        seed=args.seed,
        width=args.width,
        height=args.height,
        overwrite=args.overwrite,
        expected_manifest_digest=manifest.semantic_digest,
        implementation_digest=implementation_digest,
        source_caption_digest=(
            source_captions.semantic_digest
            if source_captions is not None
            else None
        ),
    )

    print_manifest_summary(manifest)
    print("\n제출 설정")
    print(f"  batch-id: {config.batch_id}")
    print(f"  condition: {config.condition}")
    print(f"  implementation digest: {config.implementation_digest}")
    print(
        "  projection / text-row gate: "
        f"{'on' if config.projection_enabled else 'off'} / "
        f"{'on' if config.mask_single_text_rows else 'off'}"
    )
    print(
        f"  generation: seed={config.seed}, steps={config.steps}, "
        f"size={config.width}x{config.height}"
    )
    if source_captions is not None:
        print_source_caption_summary(
            source_captions,
            source_path=args.source_captions,
        )
    if not args.yes:
        print(
            "\n원격 GPU 비용이 발생하는 제출은 하지 않았습니다. "
            "설정이 맞으면 같은 명령에 --yes를 추가하세요."
        )
        return 2

    print("\nReference 10장을 Modal 입력 Volume에 업로드합니다...")
    input_volume = modal.Volume.from_name(
        BATCH_INPUTS_VOLUME_NAME,
        create_if_missing=True,
    )
    remote_dataset_root = f"/{manifest.semantic_digest}"
    with input_volume.batch_upload(force=True) as upload:
        for reference in manifest.references:
            upload.put_file(
                reference.reference_path,
                (
                    f"{remote_dataset_root}/references/"
                    f"{Path(reference.reference_path).name}"
                ),
            )
        upload.put_file(
            BytesIO(manifest.to_json().encode("utf-8")),
            f"{remote_dataset_root}/manifest.json",
        )

    print("배포된 GPU worker에 비동기 batch를 제출합니다...")
    remote_class = modal.Cls.from_name(BATCH_APP_NAME, BATCH_CLASS_NAME)
    call = remote_class().generate_batch.spawn(
        config.to_dict(),
        (
            source_captions.to_upload_dict()
            if source_captions is not None
            else None
        ),
    )
    call_id = call.object_id
    dashboard_url = call.get_dashboard_url()
    record = {
        "batch_id": config.batch_id,
        "call_id": call_id,
        "dashboard_url": dashboard_url,
        "submitted_at": datetime.now().astimezone().isoformat(),
        "reference_root": str(Path(args.reference_root).resolve()),
        "manifest_digest": manifest.semantic_digest,
        "source_caption_path": (
            str(Path(args.source_captions).resolve())
            if source_captions is not None
            else None
        ),
        "source_caption_digest": (
            source_captions.semantic_digest
            if source_captions is not None
            else None
        ),
        "config": config.to_dict(),
    }
    _write_job_record(record)

    print("\n제출이 완료되었습니다.")
    print(f"  Batch ID: {config.batch_id}")
    print(f"  Function Call ID: {call_id}")
    print(f"  Dashboard: {dashboard_url}")
    print(
        "\n이제 터미널을 닫거나 PC를 종료해도 Modal 작업은 계속됩니다. "
        "tmux는 필요하지 않습니다."
    )
    print("\n나중에 상태 확인:")
    print(
        "  python Orthogonalization-InstantX-IP-Adapter\\batch_client.py "
        f"status --batch-id {config.batch_id}"
    )
    print("상세 로그:")
    print(
        f"  python -m modal app logs {BATCH_APP_NAME} "
        f"--function-call {call_id} --timestamps"
    )
    print("완료 후 다운로드:")
    print(
        "  python Orthogonalization-InstantX-IP-Adapter\\batch_client.py "
        f"download --batch-id {config.batch_id}"
    )
    return 0


def command_status(args: argparse.Namespace) -> int:
    record = None
    batch_id = args.batch_id
    call_id = args.call_id

    if batch_id is not None:
        _validate_batch_id(batch_id)
        record = _read_job_record(batch_id)
        if record is not None:
            call_id = record.get("call_id")
    elif call_id is not None:
        record = _find_record_by_call_id(call_id)
        if record is not None:
            batch_id = record.get("batch_id")

    if batch_id is not None:
        remote_status = _read_remote_status(batch_id)
        if remote_status is None:
            print(f"Remote status: 아직 생성되지 않음 ({batch_id})")
        else:
            print("Remote Volume status:")
            print(json.dumps(remote_status, ensure_ascii=False, indent=2))

    if not call_id:
        print(
            "로컬 job 기록에서 Function Call ID를 찾지 못했습니다. "
            "--call-id fc-...로 다시 확인하세요."
        )
        return 1

    print(f"\nFunction Call ID: {call_id}")
    call = modal.FunctionCall.from_id(call_id)
    try:
        result = call.get(timeout=0)
    except TimeoutError:
        print("Modal call state: pending 또는 running")
        return 0
    except modal.exception.OutputExpiredError:
        print(
            "Modal call result: 조회 보존기간 만료. "
            "이미지와 status는 결과 Volume에서 계속 확인할 수 있습니다."
        )
        return 0
    except Exception as error:
        print(f"Modal call state: failed ({type(error).__name__}: {error})")
        return 1

    print("Modal call state: completed")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def command_download(args: argparse.Namespace) -> int:
    _validate_batch_id(args.batch_id)
    output_root = Path(args.output_root).expanduser().resolve()
    destination_root = _safe_local_child(output_root, args.batch_id)

    volume = modal.Volume.from_name(BATCH_RESULTS_VOLUME_NAME)
    try:
        entries = volume.listdir(args.batch_id, recursive=True)
    except (FileNotFoundError, modal.exception.NotFoundError):
        print(
            f"원격 결과를 찾지 못했습니다: "
            f"{BATCH_RESULTS_VOLUME_NAME}/{args.batch_id}"
        )
        return 1

    files = [entry for entry in entries if entry.type.name == "FILE"]
    png_files = [
        entry for entry in files if entry.path.casefold().endswith(".png")
    ]
    if len(png_files) != EXPECTED_JOB_COUNT and not args.allow_partial:
        print(
            f"원격 PNG가 {len(png_files)}/{EXPECTED_JOB_COUNT}장입니다. "
            "배치 완료 후 다시 실행하세요. 진행 중 결과가 꼭 필요하면 "
            "--allow-partial을 사용하세요."
        )
        return 1

    download_plan = [
        (
            entry,
            _safe_local_child(
                destination_root,
                str(_relative_remote_path(entry.path, args.batch_id)),
            ),
        )
        for entry in files
    ]
    conflicts = [path for _, path in download_plan if path.exists()]
    if conflicts and not args.force:
        print(
            f"로컬 파일 {len(conflicts)}개가 이미 존재합니다. "
            "교체하려면 --force를 추가하세요."
        )
        print(f"첫 충돌 파일: {conflicts[0]}")
        return 1

    print(
        f"{len(files)}개 파일(PNG {len(png_files)}장)을 "
        f"{destination_root}에 다운로드합니다..."
    )
    for index, (entry, local_path) in enumerate(download_plan, start=1):
        local_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = local_path.with_name(
            f".{local_path.name}.{uuid4().hex}.tmp"
        )
        with temporary.open("wb") as stream:
            for chunk in volume.read_file(entry.path):
                stream.write(chunk)
        os.replace(temporary, local_path)
        if index % 20 == 0 or index == len(download_plan):
            print(f"  {index}/{len(download_plan)}")

    local_png_count = sum(
        1 for path in destination_root.rglob("*.png") if path.is_file()
    )
    print(f"다운로드 완료: {destination_root}")
    print(f"로컬 PNG 수: {local_png_count}")
    if not args.allow_partial and local_png_count != EXPECTED_JOB_COUNT:
        print(
            f"경고: 정확히 {EXPECTED_JOB_COUNT}장을 기대했지만 "
            f"{local_png_count}장이 확인되었습니다."
        )
        return 1
    return 0


def print_manifest_summary(manifest) -> None:
    species_counts = {"dog": 0, "cat": 0}
    for reference in manifest.references:
        species_counts[reference.species] += 1

    print("Batch manifest 검증 완료")
    print(
        f"  references: {len(manifest.references)} "
        f"(dog {species_counts['dog']}, cat {species_counts['cat']})"
    )
    print(f"  prompts/reference: {len(PROMPT_SUFFIXES)}")
    print(f"  total jobs: {len(manifest.jobs)}")
    print(f"  semantic digest: {manifest.semantic_digest}")
    print("  references:")
    for reference in manifest.references:
        print(
            f"    - {Path(reference.reference_path).name} "
            f"-> {reference.species}"
        )
    print("  prompt suffixes:")
    for suffix in PROMPT_SUFFIXES:
        print(f"    - {suffix}")
    print("  output examples:")
    for job in manifest.jobs[:3]:
        print(f"    - {job.output_relative_path}")


def print_source_caption_summary(
    source_captions,
    *,
    source_path: str | Path,
) -> None:
    print("\nSource caption 검증 완료")
    print(f"  path: {Path(source_path).resolve()}")
    print(f"  references: {len(source_captions.records)}")
    print(f"  semantic digest: {source_captions.semantic_digest}")
    for record in source_captions.records:
        print(
            f"    - {record.reference_stem}: "
            f"behavior={record.source_behavior!r}, "
            f"background={record.source_background!r}"
        )


def default_batch_id(condition: str) -> str:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    return f"dreambench-{condition}-{timestamp}"


def _validate_batch_id(batch_id: str) -> None:
    BatchGenerationConfig(batch_id=batch_id)


def _record_path(batch_id: str) -> Path:
    _validate_batch_id(batch_id)
    return JOB_RECORD_ROOT / f"{batch_id}.json"


def _write_job_record(record: dict) -> None:
    path = _record_path(record["batch_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = _read_job_record(record["batch_id"])
    previous_calls = []
    if previous is not None:
        previous_calls.extend(previous.get("previous_call_ids", []))
        previous_call = previous.get("call_id")
        if previous_call and previous_call != record["call_id"]:
            previous_calls.append(previous_call)
    record["previous_call_ids"] = list(dict.fromkeys(previous_calls))

    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_job_record(batch_id: str):
    path = _record_path(batch_id)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _find_record_by_call_id(call_id: str):
    if not JOB_RECORD_ROOT.is_dir():
        return None
    for path in sorted(JOB_RECORD_ROOT.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if value.get("call_id") == call_id or call_id in value.get(
            "previous_call_ids", []
        ):
            return value
    return None


def _read_remote_status(batch_id: str):
    volume = modal.Volume.from_name(BATCH_RESULTS_VOLUME_NAME)
    remote_path = f"{batch_id}/_metadata/status.json"
    try:
        encoded = b"".join(volume.read_file(remote_path))
    except (FileNotFoundError, modal.exception.NotFoundError):
        return None
    return json.loads(encoded.decode("utf-8"))


def _relative_remote_path(remote_path: str, batch_id: str) -> PurePosixPath:
    normalized = PurePosixPath(remote_path.lstrip("/"))
    prefix = PurePosixPath(batch_id)
    try:
        return normalized.relative_to(prefix)
    except ValueError as error:
        raise ValueError(
            f"Remote result path is outside batch {batch_id!r}: {remote_path!r}"
        ) from error


def _safe_local_child(root: Path, relative_value: str) -> Path:
    relative_path = Path(relative_value)
    if relative_path.is_absolute():
        raise ValueError(f"Expected a relative path, got: {relative_value!r}")
    root_resolved = root.resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as error:
        raise ValueError(
            f"Path escapes its configured root: {relative_value!r}"
        ) from error
    return candidate


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
