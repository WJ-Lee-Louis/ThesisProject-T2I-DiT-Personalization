"""Modal worker for XVerse subset/custom evaluation image generation."""

# ruff: noqa: E402

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import traceback
from uuid import uuid4

import modal


EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parent
for source_root in (EXPERIMENT_ROOT, PROJECT_ROOT):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from orthogonalization_instantx_ip_adapter.config import (
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    BATCH_GPU,
    BATCH_MAX_CONTAINERS,
    BATCH_SCALEDOWN_WINDOW_SECONDS,
    EVAL_BATCH_APP_NAME,
    EVAL_BATCH_INPUTS_VOLUME_NAME,
    EVAL_BATCH_RESULTS_VOLUME_NAME,
    IMAGE_ENCODER_ID,
    IMAGE_ENCODER_REVISION,
    IP_ADAPTER_REPO_ID,
    IP_ADAPTER_REVISION,
    MODAL_PIP_PACKAGES,
)
from orthogonalization_instantx_ip_adapter.implementation import (
    compute_evaluation_implementation_digest,
)
from shared.modal_runtime import HF_SECRET_NAME, build_hf_cache_volume


REMOTE_INPUT_ROOT = Path("/inputs")
REMOTE_RESULTS_ROOT = Path("/results")
IMPLEMENTATION_DIGEST = compute_evaluation_implementation_digest(EXPERIMENT_ROOT)

app = modal.App(EVAL_BATCH_APP_NAME)
hf_cache = build_hf_cache_volume()
inputs_volume = modal.Volume.from_name(
    EVAL_BATCH_INPUTS_VOLUME_NAME, create_if_missing=True
)
results_volume = modal.Volume.from_name(
    EVAL_BATCH_RESULTS_VOLUME_NAME, create_if_missing=True
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(*MODAL_PIP_PACKAGES)
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "ORTHOGONALIZATION_EVAL_IMPLEMENTATION_DIGEST": IMPLEMENTATION_DIGEST,
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_python_source("shared")
    .add_local_dir(
        EXPERIMENT_ROOT / "orthogonalization_instantx_ip_adapter",
        remote_path="/root/orthogonalization_instantx_ip_adapter",
    )
)


@app.function(
    image=image,
    timeout=15 * 60,
    volumes={"/cache": hf_cache},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
)
def check_access() -> dict:
    """Verify the active workspace secret can read all required repositories."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    repositories = (
        (BASE_MODEL_ID, BASE_MODEL_REVISION),
        (IP_ADAPTER_REPO_ID, IP_ADAPTER_REVISION),
        (IMAGE_ENCODER_ID, IMAGE_ENCODER_REVISION),
    )
    checked = []
    for repo_id, revision in repositories:
        info = api.model_info(repo_id=repo_id, revision=revision)
        checked.append({"repo_id": repo_id, "sha": info.sha})
    small_files = (
        (BASE_MODEL_ID, BASE_MODEL_REVISION, "model_index.json"),
        (IP_ADAPTER_REPO_ID, IP_ADAPTER_REVISION, "attention_processor.py"),
        (IMAGE_ENCODER_ID, IMAGE_ENCODER_REVISION, "config.json"),
    )
    for repo_id, revision, filename in small_files:
        hf_hub_download(
            repo_id=repo_id,
            revision=revision,
            filename=filename,
            token=os.environ.get("HF_TOKEN"),
        )
    return {"access": "OK", "repositories": checked}


@app.cls(
    image=image,
    gpu=BATCH_GPU,
    max_containers=BATCH_MAX_CONTAINERS,
    retries=modal.Retries(
        max_retries=2,
        initial_delay=5.0,
        backoff_coefficient=2.0,
    ),
    scaledown_window=BATCH_SCALEDOWN_WINDOW_SECONDS,
    startup_timeout=30 * 60,
    timeout=24 * 60 * 60,
    volumes={
        "/cache": hf_cache,
        REMOTE_INPUT_ROOT: inputs_volume.with_mount_options(read_only=True),
        REMOTE_RESULTS_ROOT: results_volume,
    },
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
)
class EvaluationBatchGenerator:
    """Load InstantX once and persist one benchmark/condition batch."""

    @modal.enter()
    def load_model(self) -> None:
        from orthogonalization_instantx_ip_adapter.model_loader import (
            load_ip_adapter_model,
        )

        self.ip_model = load_ip_adapter_model()
        from infer_flux_ipa_siglip import resize_img

        self.resize_reference = resize_img

    @modal.method()
    def generate_batch(
        self,
        config_payload: dict,
        manifest_payload: dict,
        source_caption_payload: dict | None = None,
    ) -> dict:
        from PIL import Image

        from orthogonalization_instantx_ip_adapter.batch_config import (
            BatchGenerationConfig,
        )
        from orthogonalization_instantx_ip_adapter.evaluation_conditioning import (
            build_evaluation_conditioning,
        )
        from orthogonalization_instantx_ip_adapter.evaluation_manifest import (
            EVALUATION_CONDITIONS,
            build_evaluation_manifest_from_upload,
            build_evaluation_source_captions_from_upload,
        )
        from orthogonalization_instantx_ip_adapter.inference import _runtime_versions

        config = BatchGenerationConfig.from_dict(config_payload)
        if config.condition not in EVALUATION_CONDITIONS:
            raise ValueError("Unsupported XVerse evaluation condition")
        if config.expected_manifest_digest is None:
            raise ValueError("expected_manifest_digest is required")
        deployed_digest = os.environ.get(
            "ORTHOGONALIZATION_EVAL_IMPLEMENTATION_DIGEST"
        )
        if config.implementation_digest != deployed_digest:
            raise RuntimeError(
                "Local evaluation code does not match the deployed worker. "
                "Redeploy eval_batch_modal.py and use a new batch_id."
            )

        reference_root = (
            REMOTE_INPUT_ROOT / config.expected_manifest_digest / "references"
        )
        inputs_volume.reload()
        results_volume.reload()
        manifest = build_evaluation_manifest_from_upload(
            manifest_payload, reference_root
        )
        if manifest.semantic_digest != config.expected_manifest_digest:
            raise RuntimeError("Uploaded evaluation manifest digest mismatch")

        source_captions = None
        if config.requires_source_captions:
            if source_caption_payload is None:
                raise ValueError("Source captions are required for this condition")
            source_captions = build_evaluation_source_captions_from_upload(
                source_caption_payload, manifest
            )
            if source_captions.semantic_digest != config.source_caption_digest:
                raise RuntimeError("Source-caption digest mismatch")
        elif source_caption_payload is not None:
            raise ValueError("Unexpected source captions for a non-source condition")

        batch_root = _safe_child(REMOTE_RESULTS_ROOT, config.batch_id)
        metadata_root = batch_root / "_metadata"
        image_metadata_root = metadata_root / "i"
        config_path = metadata_root / "config.json"
        manifest_path = metadata_root / "evaluation_manifest.json"
        source_path = metadata_root / "source_captions.json"
        execution_path = metadata_root / "execution.json"
        status_path = metadata_root / "status.json"
        generated_manifest_path = batch_root / "generated_images_manifest.json"

        metadata_root.mkdir(parents=True, exist_ok=True)
        _validate_existing_batch(
            config_path=config_path,
            manifest_path=manifest_path,
            config=config,
            manifest_digest=manifest.semantic_digest,
        )
        previous_status = _read_json_if_present(status_path)
        started_at = (
            previous_status.get("started_at")
            if isinstance(previous_status, dict)
            else None
        ) or _utc_now()
        _atomic_write_json(config_path, config.to_dict())
        _atomic_write_json(manifest_path, manifest.to_upload_dict())
        if source_captions is not None:
            _atomic_write_json(source_path, source_captions.to_upload_dict())
        _atomic_write_json(
            execution_path,
            {
                "requested_gpu": BATCH_GPU,
                "actual_gpu": _cuda_device_name(),
                "max_containers": BATCH_MAX_CONTAINERS,
                "implementation_digest": IMPLEMENTATION_DIGEST,
            },
        )
        status = {
            "batch_id": config.batch_id,
            "benchmark": manifest.benchmark,
            "benchmark_id": manifest.benchmark_id,
            "condition": config.condition,
            "state": "running",
            "total_count": len(manifest.jobs),
            "completed_count": 0,
            "generated_count": 0,
            "skipped_count": 0,
            "failed_count": 0,
            "current_job": None,
            "started_at": started_at,
            "updated_at": _utc_now(),
            "completed_at": None,
        }
        _atomic_write_json(status_path, status)
        results_volume.commit()

        generated_rows = []
        current_reference_path = None
        prepared_reference = None
        current_job = None
        try:
            for index, job in enumerate(manifest.jobs, start=1):
                current_job = job
                output_path = _safe_child(batch_root, job.output_relative_path)
                diagnostics_path = _safe_child(
                    image_metadata_root,
                    f"{job.record_id}/seed_{job.seed:04d}.json",
                )
                result_row = {
                    "record_id": job.record_id,
                    "condition": config.condition,
                    "seed": job.seed,
                    "image_path": job.output_relative_path,
                    "prompt_id": job.prompt_id,
                    "composition": job.composition,
                }
                status["current_job"] = {
                    "index": index,
                    "record_id": job.record_id,
                    "seed": job.seed,
                    "prompt": job.prompt,
                }

                if (
                    not config.overwrite
                    and output_path.is_file()
                    and diagnostics_path.is_file()
                ):
                    generated_rows.append(result_row)
                    status["skipped_count"] += 1
                    status["completed_count"] += 1
                    _persist_progress(
                        status_path,
                        generated_manifest_path,
                        status,
                        manifest,
                        generated_rows,
                        config.condition,
                    )
                    results_volume.commit()
                    print(
                        f"[{index}/{len(manifest.jobs)}] skip "
                        f"{job.output_relative_path}",
                        flush=True,
                    )
                    continue

                if current_reference_path != job.reference_path:
                    with Image.open(job.reference_path) as source_image:
                        rgb_reference = source_image.convert("RGB")
                    prepared_reference = self.resize_reference(rgb_reference)
                    current_reference_path = job.reference_path

                source_caption = (
                    source_captions.for_reference(job.reference_stem)
                    if source_captions is not None
                    else None
                )
                conditioning = build_evaluation_conditioning(
                    job, config, source_caption=source_caption
                )
                generated = self.ip_model.generate(
                    pil_image=prepared_reference,
                    prompt=job.prompt,
                    scale=config.ip_adapter_scale,
                    guidance_scale=config.guidance_scale,
                    num_inference_steps=config.steps,
                    width=config.width,
                    height=config.height,
                    seed=job.seed,
                    orthogonalization=conditioning.orthogonalization,
                    mask_single_text_rows=conditioning.mask_single_text_rows,
                )[0]
                diagnostics = {
                    "batch_id": config.batch_id,
                    "benchmark": manifest.benchmark,
                    "benchmark_id": manifest.benchmark_id,
                    "condition": config.condition,
                    "implementation_digest": config.implementation_digest,
                    "job_index": index,
                    "job": job.upload_dict(),
                    "generation": {
                        "ip_adapter_scale": config.ip_adapter_scale,
                        "guidance_scale": config.guidance_scale,
                        "num_inference_steps": config.steps,
                        "seed": job.seed,
                        "width": config.width,
                        "height": config.height,
                    },
                    "conditioning": {
                        "basis_origin": conditioning.basis_origin,
                        "target_factors": list(conditioning.target_factors),
                        "source_factors": list(conditioning.source_factors),
                        "source_caption": (
                            source_caption.upload_dict()
                            if source_caption is not None
                            else None
                        ),
                        "orthogonalization": _request_to_dict(
                            conditioning.orthogonalization
                        ),
                        "mask_single_text_rows": (
                            conditioning.mask_single_text_rows
                        ),
                    },
                    "models": {
                        "base_model": BASE_MODEL_ID,
                        "base_model_revision": BASE_MODEL_REVISION,
                        "ip_adapter": IP_ADAPTER_REPO_ID,
                        "ip_adapter_revision": IP_ADAPTER_REVISION,
                        "siglip": IMAGE_ENCODER_ID,
                        "siglip_revision": IMAGE_ENCODER_REVISION,
                    },
                    "runtime_versions": _runtime_versions(),
                    "checkpoint_load": self.ip_model.checkpoint_load_diagnostics,
                    "orthogonalization_result": (
                        self.ip_model.last_orthogonalization_diagnostics
                    ),
                    "flux_text_seq_len": (
                        self.ip_model.last_single_stream_text_seq_len
                    ),
                    "generated_at": _utc_now(),
                }
                _atomic_save_png(output_path, generated)
                _atomic_write_json(diagnostics_path, diagnostics)
                generated_rows.append(result_row)
                status["generated_count"] += 1
                status["completed_count"] += 1
                _persist_progress(
                    status_path,
                    generated_manifest_path,
                    status,
                    manifest,
                    generated_rows,
                    config.condition,
                )
                results_volume.commit()
                print(
                    f"[{index}/{len(manifest.jobs)}] saved "
                    f"{job.output_relative_path}",
                    flush=True,
                )

            status["state"] = "completed"
            status["current_job"] = None
            status["completed_at"] = _utc_now()
            status["updated_at"] = status["completed_at"]
            _persist_progress(
                status_path,
                generated_manifest_path,
                status,
                manifest,
                generated_rows,
                config.condition,
            )
            results_volume.commit()
            return {
                "batch_id": config.batch_id,
                "benchmark": manifest.benchmark,
                "condition": config.condition,
                "state": "completed",
                "total_count": len(manifest.jobs),
                "generated_count": status["generated_count"],
                "skipped_count": status["skipped_count"],
                "remote_result_path": f"/{config.batch_id}",
            }
        except Exception as error:
            status["state"] = "failed"
            status["failed_count"] = 1
            status["updated_at"] = _utc_now()
            status["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
            if current_job is not None:
                status["failed_job"] = current_job.upload_dict()
            _atomic_write_json(status_path, status)
            results_volume.commit()
            raise


def _persist_progress(
    status_path: Path,
    generated_manifest_path: Path,
    status: dict,
    manifest,
    generated_rows: list[dict],
    condition: str,
) -> None:
    status["updated_at"] = _utc_now()
    _atomic_write_json(status_path, status)
    _atomic_write_json(
        generated_manifest_path,
        {
            "schema_version": 1,
            "benchmark_id": manifest.benchmark_id,
            "benchmark": manifest.benchmark,
            "condition": condition,
            "images": generated_rows,
        },
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cuda_device_name() -> str | None:
    import torch

    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None


def _safe_child(root: Path, relative_value: str) -> Path:
    relative_path = Path(relative_value)
    if relative_path.is_absolute():
        raise ValueError(f"Expected a relative path: {relative_value!r}")
    root_resolved = root.resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as error:
        raise ValueError(f"Path escapes configured root: {relative_value!r}") from error
    return candidate


def _validate_existing_batch(
    *, config_path: Path, manifest_path: Path, config, manifest_digest: str
) -> None:
    existing_config = _read_json_if_present(config_path)
    if existing_config is not None:
        current_config = config.to_dict()
        existing_config.pop("overwrite", None)
        current_config.pop("overwrite", None)
        if existing_config != current_config:
            raise RuntimeError(
                "This batch_id already exists with different settings. "
                "Use a new batch_id."
            )
    existing_manifest = _read_json_if_present(manifest_path)
    if (
        existing_manifest is not None
        and existing_manifest.get("semantic_digest") != manifest_digest
    ):
        raise RuntimeError("This batch_id already belongs to another benchmark")


def _read_json_if_present(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_save_png(path: Path, image_value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    image_value.save(temporary, format="PNG")
    os.replace(temporary, path)


def _request_to_dict(request) -> dict:
    value = {
        "mode": request.mode,
        "strength": request.strength,
        "include_behavior": request.include_behavior,
        "include_background": request.include_background,
        "include_interaction": request.include_interaction,
        "restore_norm": request.restore_norm,
        "relative_tolerance": request.relative_tolerance,
        "primary_label": request.primary_label,
        "additional_direction_sets": [
            {
                "mode": direction_set.mode,
                "include_behavior": direction_set.include_behavior,
                "include_background": direction_set.include_background,
                "include_interaction": direction_set.include_interaction,
                "label": direction_set.label,
                "prompts": asdict(direction_set.prompts),
            }
            for direction_set in request.additional_direction_sets
        ],
        "prompts": asdict(request.prompts) if request.prompts is not None else None,
    }
    return value


__all__ = ["EvaluationBatchGenerator", "app", "check_access"]
