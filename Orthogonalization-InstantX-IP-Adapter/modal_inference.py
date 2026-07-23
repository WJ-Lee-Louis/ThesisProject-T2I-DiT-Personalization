"""Modal CLI for the isolated InstantX orthogonalization experiments."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from urllib.parse import urlparse

import modal


EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from orthogonalization_instantx_ip_adapter.config import (
    APP_NAME,
    DEFAULT_MASK_SINGLE_TEXT_ROWS,
    DEFAULT_ORTHOGONALIZATION_MODE,
    DEFAULT_ORTHOGONALIZATION_STRENGTH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_REFERENCE_IMAGE_PATH,
    DEFAULT_REFERENCE_IMAGE_URL,
    MODAL_PIP_PACKAGES,
)
from shared.modal_runtime import (
    DEFAULT_GPU,
    HF_SECRET_NAME,
    build_hf_cache_volume,
)


app = modal.App(APP_NAME)
hf_cache = build_hf_cache_volume()
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(*MODAL_PIP_PACKAGES)
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_python_source("shared")
    .add_local_dir(
        EXPERIMENT_ROOT / "orthogonalization_instantx_ip_adapter",
        remote_path="/root/orthogonalization_instantx_ip_adapter",
    )
)


def clean_path_arg(value: str) -> str:
    return value.strip().strip("\"'")


def next_available_path(path: Path) -> Path:
    for index in range(1000):
        candidate = path.with_name(f"{path.stem}_{index:03d}{path.suffix}")
        if not candidate.exists() and not candidate.with_suffix(".json").exists():
            return candidate
    raise RuntimeError(f"Could not find an available output path for {path}")


def reference_stem(reference_image_path: str, reference_image_url: str) -> str:
    if reference_image_path:
        return Path(reference_image_path).stem
    stem = Path(urlparse(reference_image_url).path).stem
    return stem or "reference"


def resolve_output_path(
    output_path: str,
    reference_image_path: str,
    reference_image_url: str,
) -> Path:
    filename = (
        f"{reference_stem(reference_image_path, reference_image_url)}_generated.png"
    )
    if output_path:
        path = Path(clean_path_arg(output_path))
        return next_available_path(path if path.suffix else path / filename)
    return next_available_path(Path(DEFAULT_OUTPUT_DIR) / filename)


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    volumes={"/cache": hf_cache},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    timeout=1800,
)
def generate_experiment(
    prompt: str,
    reference_image_url: str,
    reference_image_bytes: bytes | None = None,
    ip_adapter_scale: float = 0.7,
    guidance_scale: float = 3.5,
    num_inference_steps: int = 24,
    seed: int = 42,
    width: int = 960,
    height: int = 1280,
    orthogonalization_mode: str = DEFAULT_ORTHOGONALIZATION_MODE,
    orthogonalization_strength: float = DEFAULT_ORTHOGONALIZATION_STRENGTH,
    object_prompt: str = "",
    object_behavior_prompt: str = "",
    object_background_prompt: str = "",
    object_behavior_background_prompt: str = "",
    remove_behavior: bool = True,
    remove_background: bool = True,
    include_interaction: bool = False,
    restore_visual_norm: bool = True,
    mask_single_text_rows: bool = DEFAULT_MASK_SINGLE_TEXT_ROWS,
) -> tuple[bytes, dict]:
    from orthogonalization_instantx_ip_adapter.inference import (
        generate_png_bytes,
    )

    return generate_png_bytes(
        prompt=prompt,
        reference_image_url=reference_image_url,
        reference_image_bytes=reference_image_bytes,
        ip_adapter_scale=ip_adapter_scale,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
        width=width,
        height=height,
        orthogonalization_mode=orthogonalization_mode,
        orthogonalization_strength=orthogonalization_strength,
        object_prompt=object_prompt,
        object_behavior_prompt=object_behavior_prompt,
        object_background_prompt=object_background_prompt,
        object_behavior_background_prompt=object_behavior_background_prompt,
        remove_behavior=remove_behavior,
        remove_background=remove_background,
        include_interaction=include_interaction,
        restore_visual_norm=restore_visual_norm,
        mask_single_text_rows=mask_single_text_rows,
    )


@app.local_entrypoint()
def main(
    prompt: str = "a dog running on a beach",
    reference_image_url: str = DEFAULT_REFERENCE_IMAGE_URL,
    reference_image_path: str = DEFAULT_REFERENCE_IMAGE_PATH,
    ip_adapter_scale: float = 0.7,
    guidance_scale: float = 3.5,
    num_inference_steps: int = 24,
    seed: int = 42,
    width: int = 960,
    height: int = 1280,
    orthogonalization_mode: str = DEFAULT_ORTHOGONALIZATION_MODE,
    orthogonalization_strength: float = DEFAULT_ORTHOGONALIZATION_STRENGTH,
    object_prompt: str = "",
    object_behavior_prompt: str = "",
    object_background_prompt: str = "",
    object_behavior_background_prompt: str = "",
    remove_behavior: bool = True,
    remove_background: bool = True,
    include_interaction: bool = False,
    restore_visual_norm: bool = True,
    mask_single_text_rows: bool = DEFAULT_MASK_SINGLE_TEXT_ROWS,
    output_path: str = DEFAULT_OUTPUT_PATH,
):
    reference_image_bytes = None
    if reference_image_path:
        reference_path = Path(clean_path_arg(reference_image_path))
        reference_image_bytes = reference_path.read_bytes()
        print(f"Using local reference image: {reference_path.resolve()}")
    else:
        print(f"Using reference image URL: {reference_image_url}")

    output_bytes, diagnostics = generate_experiment.remote(
        prompt=prompt,
        reference_image_url=reference_image_url,
        reference_image_bytes=reference_image_bytes,
        ip_adapter_scale=ip_adapter_scale,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
        width=width,
        height=height,
        orthogonalization_mode=orthogonalization_mode,
        orthogonalization_strength=orthogonalization_strength,
        object_prompt=object_prompt,
        object_behavior_prompt=object_behavior_prompt,
        object_background_prompt=object_background_prompt,
        object_behavior_background_prompt=object_behavior_background_prompt,
        remove_behavior=remove_behavior,
        remove_background=remove_background,
        include_interaction=include_interaction,
        restore_visual_norm=restore_visual_norm,
        mask_single_text_rows=mask_single_text_rows,
    )

    path = resolve_output_path(
        output_path=output_path,
        reference_image_path=reference_image_path,
        reference_image_url=reference_image_url,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(output_bytes)
    path.with_suffix(".json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved image to {path.resolve()}")
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
