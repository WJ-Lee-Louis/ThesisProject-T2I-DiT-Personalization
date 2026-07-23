"""End-to-end inference entry point for the isolated experiment package."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from io import BytesIO

from diffusers.utils import load_image
from PIL import Image

from .adapter import OrthogonalizationRequest
from .config import (
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    IMAGE_ENCODER_ID,
    IMAGE_ENCODER_REVISION,
    IP_ADAPTER_REPO_ID,
    IP_ADAPTER_REVISION,
)
from .model_loader import load_ip_adapter_model
from .orthogonalization import PromptSet


def load_reference_image(
    reference_image_url: str,
    reference_image_bytes: bytes | None,
) -> Image.Image:
    if reference_image_bytes is not None:
        return Image.open(BytesIO(reference_image_bytes))
    return load_image(reference_image_url)


def build_orthogonalization_request(
    *,
    mode: str,
    strength: float,
    object_prompt: str,
    object_behavior_prompt: str,
    object_background_prompt: str,
    object_behavior_background_prompt: str,
    remove_behavior: bool,
    remove_background: bool,
    include_interaction: bool,
    restore_visual_norm: bool,
) -> OrthogonalizationRequest:
    if mode == "off":
        return OrthogonalizationRequest()

    prompts = PromptSet(
        object_only=object_prompt,
        object_behavior=object_behavior_prompt or None,
        object_background=object_background_prompt or None,
        object_behavior_background=object_behavior_background_prompt or None,
    )
    return OrthogonalizationRequest(
        mode=mode,
        strength=strength,
        prompts=prompts,
        include_behavior=remove_behavior,
        include_background=remove_background,
        include_interaction=include_interaction,
        restore_norm=restore_visual_norm,
    )


def generate_png_bytes(
    *,
    prompt: str,
    reference_image_url: str,
    reference_image_bytes: bytes | None,
    ip_adapter_scale: float,
    guidance_scale: float,
    num_inference_steps: int,
    seed: int,
    width: int,
    height: int,
    orthogonalization_mode: str,
    orthogonalization_strength: float,
    object_prompt: str,
    object_behavior_prompt: str,
    object_background_prompt: str,
    object_behavior_background_prompt: str,
    remove_behavior: bool,
    remove_background: bool,
    include_interaction: bool,
    restore_visual_norm: bool,
    mask_single_text_rows: bool,
) -> tuple[bytes, dict]:
    ip_model = load_ip_adapter_model()

    # The pinned InstantX snapshot has already been added by model_loader.
    from infer_flux_ipa_siglip import resize_img

    reference_image = resize_img(
        load_reference_image(
            reference_image_url,
            reference_image_bytes,
        ).convert("RGB")
    )
    request = build_orthogonalization_request(
        mode=orthogonalization_mode,
        strength=orthogonalization_strength,
        object_prompt=object_prompt,
        object_behavior_prompt=object_behavior_prompt,
        object_background_prompt=object_background_prompt,
        object_behavior_background_prompt=object_behavior_background_prompt,
        remove_behavior=remove_behavior,
        remove_background=remove_background,
        include_interaction=include_interaction,
        restore_visual_norm=restore_visual_norm,
    )

    result = ip_model.generate(
        pil_image=reference_image,
        prompt=prompt,
        scale=ip_adapter_scale,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        seed=seed,
        orthogonalization=request,
        mask_single_text_rows=mask_single_text_rows,
    )[0]

    buffer = BytesIO()
    result.save(buffer, format="PNG")
    diagnostics = {
        "models": {
            "base_model": BASE_MODEL_ID,
            "base_model_revision": BASE_MODEL_REVISION,
            "ip_adapter": IP_ADAPTER_REPO_ID,
            "ip_adapter_revision": IP_ADAPTER_REVISION,
            "siglip": IMAGE_ENCODER_ID,
            "siglip_revision": IMAGE_ENCODER_REVISION,
        },
        "generation": {
            "prompt": prompt,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "width": width,
            "height": height,
            "seed": seed,
            "ip_adapter_scale": ip_adapter_scale,
        },
        "runtime_versions": _runtime_versions(),
        "basis_prompts": {
            "object_only": object_prompt,
            "object_behavior": object_behavior_prompt,
            "object_background": object_background_prompt,
            "object_behavior_background": object_behavior_background_prompt,
        },
        "checkpoint_load": ip_model.checkpoint_load_diagnostics,
        "orthogonalization": ip_model.last_orthogonalization_diagnostics,
        "mask_single_text_rows": mask_single_text_rows,
        "flux_text_seq_len": ip_model.last_single_stream_text_seq_len,
    }
    return buffer.getvalue(), diagnostics


def _runtime_versions() -> dict[str, str]:
    versions = {}
    for distribution in (
        "accelerate",
        "diffusers",
        "huggingface-hub",
        "torch",
        "transformers",
    ):
        try:
            versions[distribution] = version(distribution)
        except PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions
