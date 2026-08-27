"""Modal worker for the 10-reference x 16-prompt FreeGraftor baseline."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from uuid import uuid4

import modal


EXPERIMENT_ROOT = Path(__file__).resolve().parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from freegraftor_experiment.constants import (
    APP_NAME,
    CLASS_NAME,
    FLUX_MODEL_ID,
    FLUX_MODEL_REVISION,
    GPU,
    GROUNDING_DINO_MODEL_ID,
    GROUNDING_DINO_MODEL_REVISION,
    HF_SECRET_NAME,
    INPUTS_VOLUME_NAME,
    LAMA_MODEL_ID,
    LAMA_MODEL_REVISION,
    MAX_CONTAINERS,
    MODEL_CACHE_VOLUME_NAME,
    RESULTS_VOLUME_NAME,
    SAM_FILENAME,
    SAM_SHA256,
    SAM_SIZE_BYTES,
    SAM_URL,
    UPSTREAM_COMMIT,
    UPSTREAM_REPOSITORY,
)
from freegraftor_experiment.implementation import compute_implementation_digest
from freegraftor_experiment.prompt_catalog import PROMPT_CATALOG


REMOTE_INPUT_MOUNT = "/inputs"
REMOTE_RESULTS_MOUNT = "/results"
# Keep the deployed tree identical to the local experiment tree.  Apart from
# giving the runner its upstream source root, this makes the implementation
# digest deterministic when Modal imports this module inside either image.
REMOTE_SOURCE_MOUNT = "/root/src"
REMOTE_INPUT_ROOT = Path(REMOTE_INPUT_MOUNT)
REMOTE_RESULTS_ROOT = Path(REMOTE_RESULTS_MOUNT)
REMOTE_SOURCE_ROOT = Path(REMOTE_SOURCE_MOUNT)
IMPLEMENTATION_DIGEST = compute_implementation_digest(EXPERIMENT_ROOT)

app = modal.App(APP_NAME)
model_cache = modal.Volume.from_name(MODEL_CACHE_VOLUME_NAME, create_if_missing=True)
inputs_volume = modal.Volume.from_name(INPUTS_VOLUME_NAME, create_if_missing=True)
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "numpy==1.24.4",
        "transformers==4.51.3",
        "diffusers==0.31.0",
        "Pillow==9.5.0",
        "opencv-python-headless==4.10.0.84",
        "segment-anything-py==1.0.1",
        "huggingface-hub==0.31.2",
        "modelscope==1.25.0",
        "invisible-watermark==0.2.0",
        "kornia==0.8.1",
        "datasets==2.21.0",
        "addict==2.4.0",
        "sortedcontainers==2.4.0",
        "simplejson==3.20.1",
        "sentencepiece==0.2.0",
        "einops==0.8.1",
        "safetensors==0.5.3",
        "tqdm==4.67.1",
        "accelerate==1.6.0",
        "hf-transfer==0.1.9",
        "filelock==3.18.0",
        "scipy==1.14.1",
        "PyYAML==6.0.2",
    )
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "MODELSCOPE_CACHE": "/cache/modelscope",
            "MODELSCOPE_HOME": "/cache/modelscope",
            "LAMA_MODEL_REVISION": LAMA_MODEL_REVISION,
            "KORNIA_LAZY_LOADER_INSTALL_MODE": "off",
            "FREEGRAFTOR_IMPLEMENTATION_DIGEST": IMPLEMENTATION_DIGEST,
            "PYTHONPATH": "/root:/root/src",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_dir(
        EXPERIMENT_ROOT / "freegraftor_experiment",
        remote_path="/root/freegraftor_experiment",
    )
    .add_local_dir(
        EXPERIMENT_ROOT / "src",
        remote_path=REMOTE_SOURCE_MOUNT,
    )
)

access_check_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("huggingface-hub==0.31.2")
    .env(
        {
            "PYTHONPATH": "/root:/root/src",
            "PYTHONUNBUFFERED": "1",
        }
    )
    # Modal imports this whole module before invoking check_hf_access.  The
    # lightweight image therefore still needs both local source trees even
    # though the function itself only contacts Hugging Face.
    .add_local_dir(
        EXPERIMENT_ROOT / "freegraftor_experiment",
        remote_path="/root/freegraftor_experiment",
    )
    .add_local_dir(
        EXPERIMENT_ROOT / "src",
        remote_path=REMOTE_SOURCE_MOUNT,
    )
)


@app.function(
    image=access_check_image,
    cpu=0.25,
    memory=512,
    max_containers=1,
    retries=0,
    timeout=3 * 60,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
)
def check_hf_access() -> dict[str, object]:
    """Validate the secret and gated FLUX artifact without allocating a GPU."""

    from huggingface_hub import HfApi, hf_hub_download

    token_key = None
    token = None
    for candidate in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        if os.environ.get(candidate):
            token_key = candidate
            token = os.environ[candidate]
            break
    if token is None:
        return {
            "authenticated": False,
            "flux_access": False,
            "error_type": "MissingToken",
            "error": (
                "huggingface-secret must contain HF_TOKEN or "
                "HUGGINGFACE_HUB_TOKEN"
            ),
        }
    try:
        identity = HfApi().whoami(token=token)
    except Exception as error:
        return {
            "authenticated": False,
            "flux_access": False,
            "token_key": token_key,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    try:
        hf_hub_download(
            repo_id=FLUX_MODEL_ID,
            filename=".gitattributes",
            revision=FLUX_MODEL_REVISION,
            token=token,
            cache_dir="/tmp/hf-access-check",
        )
    except Exception as error:
        return {
            "authenticated": True,
            "flux_access": False,
            "token_key": token_key,
            "account": identity.get("name"),
            "error_type": type(error).__name__,
            "error": str(error),
        }
    return {
        "authenticated": True,
        "flux_access": True,
        "token_key": token_key,
        "account": identity.get("name"),
        "model_id": FLUX_MODEL_ID,
        "revision": FLUX_MODEL_REVISION,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_save_png(path: Path, image_value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".image-{uuid4().hex}.tmp"
    try:
        image_value.save(temporary, format="PNG")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _runtime_versions() -> dict[str, object]:
    import torch

    packages = {}
    for name in (
        "torch",
        "torchvision",
        "transformers",
        "huggingface-hub",
        "modelscope",
        "segment-anything-py",
        "numpy",
        "Pillow",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "packages": packages,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "backend_flags": {
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        },
    }


def _write_consistent_metadata(path: Path, value: object) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != value:
            raise RuntimeError(
                f"Batch ID already exists with different metadata: {path}"
            )
        return
    _atomic_write_json(path, value)


@app.cls(
    image=image,
    gpu=GPU,
    cpu=4.0,
    memory=98304,
    max_containers=MAX_CONTAINERS,
    retries=modal.Retries(
        max_retries=1,
        initial_delay=10.0,
        backoff_coefficient=2.0,
    ),
    scaledown_window=10 * 60,
    startup_timeout=60 * 60,
    timeout=24 * 60 * 60,
    volumes={
        "/cache": model_cache,
        REMOTE_INPUT_MOUNT: inputs_volume.with_mount_options(read_only=True),
        REMOTE_RESULTS_MOUNT: results_volume,
    },
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
)
class FreeGraftorBatch:
    """Load the full stack once, then run prompt-major resumable shards."""

    @modal.enter()
    def load_models(self) -> None:
        from freegraftor_experiment.model_assets import prepare_model_paths
        from freegraftor_experiment.runner import FreeGraftorRunner

        self.model_paths = prepare_model_paths("/cache")
        # The upstream implementation resolves FLUX paths during import, so the
        # runner is intentionally imported/constructed only after the env setup.
        self.runner = FreeGraftorRunner(
            REMOTE_SOURCE_ROOT,
            "/tmp/freegraftor-runtime-cache",
        )
        model_cache.commit()
        self.runtime = _runtime_versions()

    @modal.method()
    def warmup(self) -> dict[str, object]:
        """Materialize checkpoints and validate that the pipeline loads."""

        return {
            "state": "ready",
            "implementation_digest": IMPLEMENTATION_DIGEST,
            "upstream_commit": UPSTREAM_COMMIT,
            "model_paths": self.model_paths,
            "runtime": self.runtime,
        }

    @modal.method()
    def generate_shard(
        self,
        config_payload: dict,
        reference_stems: list[str],
        shard_index: int,
        shard_count: int,
    ) -> dict[str, object]:
        from freegraftor_experiment.batch_config import BatchConfig
        from freegraftor_experiment.batch_manifest import build_batch_manifest

        config = BatchConfig.from_dict(config_payload)
        if not config.requires_offload:
            raise ValueError("This deployed baseline is pinned to requires_offload=True")
        if config.expected_manifest_digest is None:
            raise ValueError("expected_manifest_digest is required")
        if config.implementation_digest != IMPLEMENTATION_DIGEST:
            raise RuntimeError(
                "Local code differs from the deployed worker. Redeploy "
                "batch_modal.py before submitting a new batch."
            )
        if (
            isinstance(shard_index, bool)
            or isinstance(shard_count, bool)
            or not isinstance(shard_index, int)
            or not isinstance(shard_count, int)
            or shard_count <= 0
            or not 0 <= shard_index < shard_count
        ):
            raise ValueError("Invalid shard index/count")
        if config.shard_count != shard_count:
            raise ValueError(
                f"Config shard_count={config.shard_count} does not match "
                f"the remote call shard_count={shard_count}"
            )
        if not reference_stems or len(set(reference_stems)) != len(reference_stems):
            raise ValueError("reference_stems must be a non-empty unique list")

        inputs_volume.reload()
        results_volume.reload()
        reference_root = (
            REMOTE_INPUT_ROOT
            / config.expected_manifest_digest
            / "references"
        )
        manifest = build_batch_manifest(reference_root)
        if manifest.semantic_digest != config.expected_manifest_digest:
            raise RuntimeError(
                "Remote references differ from the submitted manifest: "
                f"{manifest.semantic_digest} != {config.expected_manifest_digest}"
            )

        reference_by_stem = {item.stem: item for item in manifest.references}
        unknown = set(reference_stems) - set(reference_by_stem)
        if unknown:
            raise ValueError(f"Unknown reference stems: {sorted(unknown)}")
        selected_stems = tuple(reference_stems)
        selected_set = frozenset(selected_stems)
        selected_jobs = tuple(
            job for job in manifest.jobs if job.reference_stem in selected_set
        )
        expected_count = len(selected_stems) * len(PROMPT_CATALOG)
        if len(selected_jobs) != expected_count:
            raise RuntimeError("Shard job count is inconsistent with the manifest")

        batch_root = REMOTE_RESULTS_ROOT / config.batch_id
        metadata_root = batch_root / "_metadata"
        config_metadata = {
            "schema_version": 1,
            "config": config.to_dict(),
            "algorithm": {
                "name": "FreeGraftor",
                "repository": UPSTREAM_REPOSITORY,
                "upstream_commit": UPSTREAM_COMMIT,
                "template_reuse": (
                    "One deterministic template per prompt and shard; equivalent "
                    "to regenerating it with the same seed for every reference."
                ),
            },
            "models": {
                "flux": {"id": FLUX_MODEL_ID, "revision": FLUX_MODEL_REVISION},
                "grounding_dino": {
                    "id": GROUNDING_DINO_MODEL_ID,
                    "revision": GROUNDING_DINO_MODEL_REVISION,
                },
                "sam": {
                    "filename": SAM_FILENAME,
                    "url": SAM_URL,
                    "size_bytes": SAM_SIZE_BYTES,
                    "sha256": SAM_SHA256,
                },
                "lama": {"id": LAMA_MODEL_ID, "revision": LAMA_MODEL_REVISION},
            },
            "runtime": self.runtime,
        }
        _write_consistent_metadata(metadata_root / "config.json", config_metadata)
        _write_consistent_metadata(metadata_root / "manifest.json", manifest.to_dict())
        results_volume.commit()

        shard_name = f"shard-{shard_index:02d}-of-{shard_count:02d}"
        status_path = metadata_root / "shards" / f"{shard_name}.json"
        started_at = _utc_now()
        status = {
            "schema_version": 1,
            "state": "running",
            "batch_id": config.batch_id,
            "shard_index": shard_index,
            "shard_count": shard_count,
            "reference_stems": list(selected_stems),
            "total_count": expected_count,
            "generated_count": 0,
            "skipped_count": 0,
            "failed_count": 0,
            "processed_count": 0,
            "started_at": started_at,
            "updated_at": started_at,
        }
        _atomic_write_json(status_path, status)
        results_volume.commit()

        jobs_by_key = {
            (job.reference_stem, job.prompt_slug): job for job in selected_jobs
        }
        template_root = Path("/tmp/freegraftor-templates") / config.batch_id / shard_name
        scratch_output = Path("/tmp/freegraftor-unused-output")

        def persist_status() -> None:
            status["processed_count"] = (
                status["generated_count"]
                + status["skipped_count"]
                + status["failed_count"]
            )
            status["updated_at"] = _utc_now()
            _atomic_write_json(status_path, status)
            results_volume.commit()

        for prompt_spec in PROMPT_CATALOG:
            prompt_jobs = [
                jobs_by_key[(stem, prompt_spec.slug)] for stem in selected_stems
            ]
            pending_jobs = []
            for job in prompt_jobs:
                output_path = batch_root / job.output_relative_path
                sidecar_path = (
                    metadata_root
                    / "images"
                    / job.reference_stem
                    / f"{job.prompt_slug}.json"
                )
                if not config.overwrite and output_path.is_file() and sidecar_path.is_file():
                    status["skipped_count"] += 1
                else:
                    pending_jobs.append(job)
            if not pending_jobs:
                persist_status()
                continue

            template_path = template_root / f"{prompt_spec.slug}.png"
            template_prompt = pending_jobs[0].full_prompt
            # Dog/cat prompts differ. A shard can contain both species, so use a
            # separate deterministic template for each species when necessary.
            pending_by_species = {
                species: [job for job in pending_jobs if job.species == species]
                for species in {job.species for job in pending_jobs}
            }
            for species, species_jobs in sorted(pending_by_species.items()):
                species_template = template_path.with_name(
                    f"{prompt_spec.slug}_{species}.png"
                )
                full_prompt = species_jobs[0].full_prompt
                try:
                    template_started = time.perf_counter()
                    _, post_template_rng_state = self.runner.generate_template(
                        full_prompt,
                        config,
                        species_template,
                    )
                    template_seconds = time.perf_counter() - template_started
                    template_sha256 = _sha256_file(species_template)
                except Exception as error:
                    error_traceback = traceback.format_exc()
                    for job in species_jobs:
                        error_path = (
                            metadata_root
                            / "errors"
                            / job.reference_stem
                            / f"{job.prompt_slug}.json"
                        )
                        _atomic_write_json(
                            error_path,
                            {
                                "stage": "template_generation",
                                "reference_stem": job.reference_stem,
                                "prompt": job.full_prompt,
                                "error_type": type(error).__name__,
                                "error": str(error),
                                "traceback": error_traceback,
                                "timestamp": _utc_now(),
                            },
                        )
                        status["failed_count"] += 1
                        persist_status()
                    continue

                for job in species_jobs:
                    reference = reference_by_stem[job.reference_stem]
                    reference_path = reference_root / reference.filename
                    output_path = batch_root / job.output_relative_path
                    sidecar_path = (
                        metadata_root
                        / "images"
                        / job.reference_stem
                        / f"{job.prompt_slug}.json"
                    )
                    error_path = (
                        metadata_root
                        / "errors"
                        / job.reference_stem
                        / f"{job.prompt_slug}.json"
                    )
                    try:
                        import torch

                        torch.cuda.reset_peak_memory_stats()
                        started = time.perf_counter()
                        generated = self.runner.generate(
                            reference_path=reference_path,
                            class_name=job.species,
                            prompt=job.full_prompt,
                            template_path=species_template,
                            config=config,
                            scratch_output=scratch_output,
                            post_template_rng_state=post_template_rng_state,
                        )
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter() - started
                        peak_mib = torch.cuda.max_memory_allocated() / 1024**2
                        _atomic_save_png(output_path, generated)
                        _atomic_write_json(
                            sidecar_path,
                            {
                                "schema_version": 1,
                                "status": "completed",
                                "batch_id": config.batch_id,
                                "reference_stem": job.reference_stem,
                                "reference_filename": job.reference_filename,
                                "reference_sha256": job.reference_sha256,
                                "class_name": job.species,
                                "prompt": job.full_prompt,
                                "prompt_suffix": job.prompt_suffix,
                                "prompt_slug": job.prompt_slug,
                                "prompt_category": job.prompt_category,
                                "prompt_composition": job.prompt_composition,
                                "output_relative_path": job.output_relative_path,
                                "output_sha256": _sha256_file(output_path),
                                "template_sha256": template_sha256,
                                "template_generation_seconds": template_seconds,
                                "generation_seconds_excluding_template": elapsed,
                                "peak_cuda_memory_mib": peak_mib,
                                "config": config.to_dict(),
                                "upstream_commit": UPSTREAM_COMMIT,
                                "runtime": self.runtime,
                                "completed_at": _utc_now(),
                            },
                        )
                        error_path.unlink(missing_ok=True)
                        status["generated_count"] += 1
                    except Exception as error:
                        _atomic_write_json(
                            error_path,
                            {
                                "stage": "collage_inversion_or_final_generation",
                                "reference_stem": job.reference_stem,
                                "reference_filename": job.reference_filename,
                                "prompt": job.full_prompt,
                                "error_type": type(error).__name__,
                                "error": str(error),
                                "traceback": traceback.format_exc(),
                                "timestamp": _utc_now(),
                            },
                        )
                        status["failed_count"] += 1
                    persist_status()

        status["state"] = (
            "completed_with_errors" if status["failed_count"] else "completed"
        )
        status["finished_at"] = _utc_now()
        persist_status()
        return dict(status)


if __name__ == "__main__":
    print(
        f"Deploy with: python -m modal deploy {Path(__file__).name} "
        f"(app={APP_NAME}, class={CLASS_NAME})"
    )
