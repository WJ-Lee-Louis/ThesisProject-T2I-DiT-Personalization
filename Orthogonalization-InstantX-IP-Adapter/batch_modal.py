"""Deployed Modal worker for one 10-reference by 9-prompt experiment batch."""

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
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from orthogonalization_instantx_ip_adapter.config import (
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    BATCH_APP_NAME,
    BATCH_INPUTS_VOLUME_NAME,
    BATCH_RESULTS_VOLUME_NAME,
    IMAGE_ENCODER_ID,
    IMAGE_ENCODER_REVISION,
    IP_ADAPTER_REPO_ID,
    IP_ADAPTER_REVISION,
    MODAL_PIP_PACKAGES,
)
from orthogonalization_instantx_ip_adapter.implementation import (
    compute_batch_implementation_digest,
)
from shared.modal_runtime import (
    DEFAULT_GPU,
    HF_SECRET_NAME,
    build_hf_cache_volume,
)


REMOTE_INPUT_ROOT = Path("/inputs")
REMOTE_RESULTS_ROOT = Path("/results")
IMPLEMENTATION_DIGEST = compute_batch_implementation_digest(EXPERIMENT_ROOT)

app = modal.App(BATCH_APP_NAME)
hf_cache = build_hf_cache_volume()
inputs_volume = modal.Volume.from_name(
    BATCH_INPUTS_VOLUME_NAME,
    create_if_missing=True,
)
results_volume = modal.Volume.from_name(
    BATCH_RESULTS_VOLUME_NAME,
    create_if_missing=True,
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(*MODAL_PIP_PACKAGES)
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "ORTHOGONALIZATION_IMPLEMENTATION_DIGEST": IMPLEMENTATION_DIGEST,
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_python_source("shared")
    .add_local_dir(
        EXPERIMENT_ROOT / "orthogonalization_instantx_ip_adapter",
        remote_path="/root/orthogonalization_instantx_ip_adapter",
    )
)


@app.cls(
    image=image,
    gpu=DEFAULT_GPU,
    max_containers=1,
    retries=modal.Retries(
        max_retries=2,
        initial_delay=5.0,
        backoff_coefficient=2.0,
    ),
    scaledown_window=300,
    startup_timeout=30 * 60,
    timeout=24 * 60 * 60,
    volumes={
        "/cache": hf_cache,
        REMOTE_INPUT_ROOT: inputs_volume.with_mount_options(read_only=True),
        REMOTE_RESULTS_ROOT: results_volume,
    },
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
)
class BatchGenerator:
    """Load InstantX once and generate all jobs in one resumable call."""

    @modal.enter()
    def load_model(self) -> None:
        from orthogonalization_instantx_ip_adapter.model_loader import (
            load_ip_adapter_model,
        )

        self.ip_model = load_ip_adapter_model()

        # The pinned InstantX snapshot is added to sys.path by model_loader.
        from infer_flux_ipa_siglip import resize_img

        self.resize_reference = resize_img

    @modal.method()
    def generate_batch(
        self,
        config_payload: dict,
        source_caption_payload: dict | None = None,
    ) -> dict:
        """Generate one validated batch and persist every completed item."""

        from PIL import Image

        from orthogonalization_instantx_ip_adapter.batch_conditioning import (
            build_batch_conditioning,
        )
        from orthogonalization_instantx_ip_adapter.batch_config import (
            BatchGenerationConfig,
        )
        from orthogonalization_instantx_ip_adapter.batch_manifest import (
            EXPECTED_JOB_COUNT,
            build_batch_manifest,
        )
        from orthogonalization_instantx_ip_adapter.inference import (
            _runtime_versions,
        )
        from orthogonalization_instantx_ip_adapter.source_captions import (
            build_source_caption_manifest,
        )

        config = BatchGenerationConfig.from_dict(config_payload)
        if config.expected_manifest_digest is None:
            raise ValueError(
                "expected_manifest_digest is required for a remote batch"
            )
        deployed_implementation_digest = os.environ.get(
            "ORTHOGONALIZATION_IMPLEMENTATION_DIGEST"
        )
        if config.implementation_digest is None:
            raise ValueError(
                "implementation_digest is required for a remote batch"
            )
        if config.implementation_digest != deployed_implementation_digest:
            raise RuntimeError(
                "Local submission code does not match the deployed worker. "
                "Run `modal deploy .../batch_modal.py` again, then submit with "
                "a new batch_id."
            )

        # A digest-specific input directory prevents stale files from an older
        # upload from changing the discovered 7-dog/3-cat dataset.
        reference_root = (
            REMOTE_INPUT_ROOT
            / config.expected_manifest_digest
            / "references"
        )
        inputs_volume.reload()
        results_volume.reload()
        manifest = build_batch_manifest(reference_root)
        if manifest.semantic_digest != config.expected_manifest_digest:
            raise RuntimeError(
                "Uploaded reference images do not match the submitted manifest: "
                f"expected {config.expected_manifest_digest}, "
                f"got {manifest.semantic_digest}"
            )

        source_captions = None
        if config.source_aware_projection_enabled:
            if source_caption_payload is None:
                raise ValueError(
                    "source_caption_payload is required for source-aware "
                    "conditions"
                )
            source_captions = build_source_caption_manifest(
                source_caption_payload,
                manifest,
            )
            if source_captions.semantic_digest != config.source_caption_digest:
                raise RuntimeError(
                    "Source caption payload does not match its submitted "
                    "digest: "
                    f"expected {config.source_caption_digest}, "
                    f"got {source_captions.semantic_digest}"
                )
        elif source_caption_payload is not None:
            raise ValueError(
                "source_caption_payload is only valid for source-aware "
                "conditions"
            )

        batch_root = _safe_child(REMOTE_RESULTS_ROOT, config.batch_id)
        metadata_root = batch_root / "_metadata"
        image_metadata_root = metadata_root / "images"
        config_path = metadata_root / "config.json"
        manifest_path = metadata_root / "manifest.json"
        source_captions_path = metadata_root / "source_captions.json"
        status_path = metadata_root / "status.json"

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
        _atomic_write_json(manifest_path, manifest.to_dict())
        if source_captions is not None:
            _atomic_write_json(
                source_captions_path,
                source_captions.to_dict(),
            )
        status = {
            "batch_id": config.batch_id,
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

        current_reference_path: str | None = None
        prepared_reference = None
        current_job = None

        try:
            for index, job in enumerate(manifest.jobs, start=1):
                current_job = job
                output_path = _safe_child(
                    batch_root,
                    job.output_relative_path,
                )
                diagnostics_path = _safe_child(
                    image_metadata_root,
                    str(Path(job.output_relative_path).with_suffix(".json")),
                )

                status["current_job"] = {
                    "index": index,
                    "reference_stem": job.reference_stem,
                    "prompt": job.prompt,
                    "output_relative_path": job.output_relative_path,
                }
                status["updated_at"] = _utc_now()

                if (
                    not config.overwrite
                    and output_path.is_file()
                    and diagnostics_path.is_file()
                ):
                    status["skipped_count"] += 1
                    status["completed_count"] += 1
                    _atomic_write_json(status_path, status)
                    results_volume.commit()
                    print(
                        f"[{index}/{EXPECTED_JOB_COUNT}] skip "
                        f"{job.output_relative_path}",
                        flush=True,
                    )
                    continue

                if current_reference_path != job.reference_path:
                    with Image.open(job.reference_path) as source:
                        rgb_reference = source.convert("RGB")
                    prepared_reference = self.resize_reference(rgb_reference)
                    current_reference_path = job.reference_path

                source_caption = (
                    source_captions.for_reference(job.reference_stem)
                    if source_captions is not None
                    else None
                )
                conditioning = build_batch_conditioning(
                    job,
                    config,
                    source_caption=source_caption,
                )
                generated = self.ip_model.generate(
                    pil_image=prepared_reference,
                    prompt=job.prompt,
                    scale=config.ip_adapter_scale,
                    guidance_scale=config.guidance_scale,
                    num_inference_steps=config.steps,
                    width=config.width,
                    height=config.height,
                    seed=config.seed,
                    orthogonalization=conditioning.orthogonalization,
                    mask_single_text_rows=conditioning.mask_single_text_rows,
                )[0]

                diagnostics = {
                    "batch_id": config.batch_id,
                    "condition": config.condition,
                    "implementation_digest": config.implementation_digest,
                    "job_index": index,
                    "job": job.to_dict(),
                    "generation": {
                        "ip_adapter_scale": config.ip_adapter_scale,
                        "guidance_scale": config.guidance_scale,
                        "num_inference_steps": config.steps,
                        "seed": config.seed,
                        "width": config.width,
                        "height": config.height,
                    },
                    "conditioning": {
                        "basis_origin": conditioning.basis_origin,
                        "removed_factors": list(
                            conditioning.removed_factors
                        ),
                        "source_caption": (
                            source_caption.to_dict()
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
                    "checkpoint_load": (
                        self.ip_model.checkpoint_load_diagnostics
                    ),
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
                status["generated_count"] += 1
                status["completed_count"] += 1
                status["updated_at"] = _utc_now()
                _atomic_write_json(status_path, status)
                results_volume.commit()
                print(
                    f"[{index}/{EXPECTED_JOB_COUNT}] saved "
                    f"{job.output_relative_path}",
                    flush=True,
                )

            status["state"] = "completed"
            status["current_job"] = None
            status["completed_at"] = _utc_now()
            status["updated_at"] = status["completed_at"]
            _atomic_write_json(status_path, status)
            results_volume.commit()
            return {
                "batch_id": config.batch_id,
                "condition": config.condition,
                "state": status["state"],
                "total_count": status["total_count"],
                "completed_count": status["completed_count"],
                "generated_count": status["generated_count"],
                "skipped_count": status["skipped_count"],
                "manifest_digest": manifest.semantic_digest,
                "source_caption_digest": config.source_caption_digest,
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
                status["failed_job"] = current_job.to_dict()
            _atomic_write_json(status_path, status)
            results_volume.commit()
            raise


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_child(root: Path, relative_value: str) -> Path:
    """Resolve a relative path while preventing escape from its Volume root."""

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


def _validate_existing_batch(
    *,
    config_path: Path,
    manifest_path: Path,
    config,
    manifest_digest: str,
) -> None:
    existing_config = _read_json_if_present(config_path)
    if existing_config is not None:
        current_config = config.to_dict()
        # overwrite controls this invocation, not the scientific condition.
        existing_config.pop("overwrite", None)
        current_config.pop("overwrite", None)
        if existing_config != current_config:
            raise RuntimeError(
                "This batch_id already exists with different experiment "
                "settings. Use a new batch_id instead of mixing conditions."
            )

    existing_manifest = _read_json_if_present(manifest_path)
    if (
        existing_manifest is not None
        and existing_manifest.get("semantic_digest") != manifest_digest
    ):
        raise RuntimeError(
            "This batch_id already exists for a different reference manifest. "
            "Use a new batch_id."
        )


def _read_json_if_present(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
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
        "prompts": None,
    }
    if request.prompts is not None:
        value["prompts"] = asdict(request.prompts)
    return value


__all__ = ["BatchGenerator", "app"]
