"""Deployed Modal worker for the condition-aware 10-reference experiment."""

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
    BATCH_GPU,
    BATCH_INPUTS_VOLUME_NAME,
    BATCH_MAX_CONTAINERS,
    BATCH_RESULTS_VOLUME_NAME,
    BATCH_SCALEDOWN_WINDOW_SECONDS,
    IMAGE_ENCODER_ID,
    IMAGE_ENCODER_REVISION,
    IP_ADAPTER_REPO_ID,
    IP_ADAPTER_REVISION,
    MODAL_PIP_PACKAGES,
)
from orthogonalization_instantx_ip_adapter.implementation import (
    compute_batch_implementation_digest,
)
from orthogonalization_instantx_ip_adapter.prompt_catalog import (
    PROMPT_CATALOG,
    get_prompt_spec,
)
from shared.modal_runtime import (
    HF_SECRET_NAME,
    build_hf_cache_volume,
)


REMOTE_INPUT_ROOT = Path("/inputs")
REMOTE_RESULTS_ROOT = Path("/results")
IMPLEMENTATION_DIGEST = compute_batch_implementation_digest(EXPERIMENT_ROOT)


def _selected_prompt_suffixes(condition: str) -> tuple[str, ...]:
    if _is_st_all_condition(condition):
        return tuple(
            spec.suffix for spec in PROMPT_CATALOG if not spec.is_factorial
        )
    return tuple(spec.suffix for spec in PROMPT_CATALOG)


def _select_jobs_for_condition(jobs, condition: str):
    jobs = tuple(jobs)
    if _is_st_all_condition(condition):
        selected_items = []
        redundant_items = []
        for job in jobs:
            destination = (
                redundant_items
                if get_prompt_spec(job.prompt_suffix).is_factorial
                else selected_items
            )
            destination.append(job)
        return tuple(selected_items), tuple(redundant_items)
    return jobs, ()


def _job_selection_metadata(jobs, condition: str) -> dict[str, object]:
    jobs = tuple(jobs)
    selected, redundant = _select_jobs_for_condition(jobs, condition)
    selected_suffixes = _selected_prompt_suffixes(condition)
    selected_suffix_set = frozenset(selected_suffixes)
    redundant_suffixes = tuple(
        spec.suffix
        for spec in PROMPT_CATALOG
        if spec.suffix not in selected_suffix_set
    )
    return {
        "schema_version": 1,
        "condition": condition,
        "full_manifest_count": len(jobs),
        "selected_job_count": len(selected),
        "selected_prompt_suffixes": list(selected_suffixes),
        "skipped_as_redundant_job_count": len(redundant),
        "skipped_as_redundant_prompt_suffixes": list(redundant_suffixes),
        "redundancy_reason": (
            "O+A+B is identical to the corresponding st_matched condition "
            "because matched union already includes both source factors"
            if redundant
            else None
        ),
    }


def _is_st_all_condition(condition: str) -> bool:
    # New ST-all runs are self-contained because a matched batch is optional.
    return False

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
        selected_jobs, _redundant_jobs = _select_jobs_for_condition(
            manifest.jobs,
            config.condition,
        )
        selection_metadata = _job_selection_metadata(
            manifest.jobs,
            config.condition,
        )
        selected_job_count = len(selected_jobs)

        source_captions = None
        if config.requires_source_captions:
            if source_caption_payload is None:
                raise ValueError(
                    "source_caption_payload is required for caption-aware "
                    "projection conditions"
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
                "source_caption_payload is only valid for caption-aware "
                "projection conditions"
            )

        batch_root = _safe_child(REMOTE_RESULTS_ROOT, config.batch_id)
        metadata_root = batch_root / "_metadata"
        image_metadata_root = metadata_root / "images"
        config_path = metadata_root / "config.json"
        manifest_path = metadata_root / "manifest.json"
        source_captions_path = metadata_root / "source_captions.json"
        selection_path = metadata_root / "job_selection.json"
        execution_path = metadata_root / "execution.json"
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
        _atomic_write_json(selection_path, selection_metadata)
        _atomic_write_json(
            execution_path,
            {
                "requested_gpu": BATCH_GPU,
                "actual_gpu": _cuda_device_name(),
                "max_containers": BATCH_MAX_CONTAINERS,
                "scaledown_window_seconds": BATCH_SCALEDOWN_WINDOW_SECONDS,
                "implementation_digest": IMPLEMENTATION_DIGEST,
            },
        )
        if source_captions is not None:
            _atomic_write_json(
                source_captions_path,
                source_captions.to_dict(),
            )
        status = {
            "batch_id": config.batch_id,
            "condition": config.condition,
            "state": "running",
            "total_count": selected_job_count,
            "full_manifest_count": len(manifest.jobs),
            "selected_prompt_suffixes": selection_metadata[
                "selected_prompt_suffixes"
            ],
            "skipped_as_redundant_count": selection_metadata[
                "skipped_as_redundant_job_count"
            ],
            "skipped_as_redundant_prompt_suffixes": selection_metadata[
                "skipped_as_redundant_prompt_suffixes"
            ],
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
            for index, job in enumerate(selected_jobs, start=1):
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
                        f"[{index}/{selected_job_count}] skip "
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
                        "prompt_spec": get_prompt_spec(
                            job.prompt_suffix
                        ).to_dict(),
                        "basis_origin": conditioning.basis_origin,
                        "class_projection_mode": (
                            conditioning.class_projection_mode
                        ),
                        "class_projection_strength": (
                            config.class_projection_strength
                            if config.class_projection_enabled
                            else None
                        ),
                        "class_superclass_prompt": (
                            config.class_superclass_prompt
                            if config.class_superclass_projection_enabled
                            else None
                        ),
                        "removed_factors": list(
                            conditioning.removed_factors
                        ),
                        "target_factors": list(
                            conditioning.target_factors
                        ),
                        "source_factors": list(
                            conditioning.source_factors
                        ),
                        "source_factor_policy": (
                            "all"
                            if config.all_source_factors_enabled
                            else (
                                "matched"
                                if conditioning.source_factors
                                else "none"
                            )
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
                    f"[{index}/{selected_job_count}] saved "
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
                "full_manifest_count": status["full_manifest_count"],
                "skipped_as_redundant_count": status[
                    "skipped_as_redundant_count"
                ],
                "selected_prompt_suffixes": status[
                    "selected_prompt_suffixes"
                ],
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


def _cuda_device_name() -> str | None:
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_name(0)


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
        "class_strength": request.class_strength,
        "primary_class_label": request.primary_class_label,
        "class_absolute_prompt": request.class_absolute_prompt,
        "residualize_class_against_nuisance": (
            request.residualize_class_against_nuisance
        ),
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
        "prompts": None,
        "comparison_prompts": None,
        "class_contextual_pairs": [
            asdict(pair) for pair in request.class_contextual_pairs
        ],
        "additional_class_direction_sets": [
            {
                "label": direction_set.label,
                "pairs": [asdict(pair) for pair in direction_set.pairs],
            }
            for direction_set in request.additional_class_direction_sets
        ],
    }
    if request.prompts is not None:
        value["prompts"] = asdict(request.prompts)
    if request.comparison_prompts is not None:
        value["comparison_prompts"] = asdict(request.comparison_prompts)
    return value


__all__ = ["BatchGenerator", "app"]
