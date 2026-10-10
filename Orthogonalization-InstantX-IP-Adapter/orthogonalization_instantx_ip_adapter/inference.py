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
from .semantic_components import SemanticConditioning, SemanticLLM, prepare_semantic_conditioning


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
    if mode == "class_contextual":
        raise ValueError(
            "Class projection modes require batch conditioning; use "
            "batch_client.py so species/source-caption prompts and metadata "
            "are constructed reproducibly."
        )

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
    component_analysis_payload: dict | None = None,
) -> tuple[bytes, dict]:
    if component_analysis_payload is not None:
        conditioning = SemanticConditioning.from_dict(component_analysis_payload)
        if conditioning.target.text != prompt:
            raise ValueError("component analysis target does not match generation prompt")
        if orthogonalization_mode != "components":
            raise ValueError("component payload requires components mode")
        if any((object_prompt, object_behavior_prompt, object_background_prompt,
                object_behavior_background_prompt, include_interaction)):
            raise ValueError("components mode cannot mix legacy prompt options")
        request = OrthogonalizationRequest(mode="components", strength=orthogonalization_strength,
                                          restore_norm=restore_visual_norm, semantic_conditioning=conditioning)
    else:
        request = build_orthogonalization_request(
            mode=orthogonalization_mode, strength=orthogonalization_strength,
            object_prompt=object_prompt, object_behavior_prompt=object_behavior_prompt,
            object_background_prompt=object_background_prompt,
            object_behavior_background_prompt=object_behavior_background_prompt,
            remove_behavior=remove_behavior, remove_background=remove_background,
            include_interaction=include_interaction, restore_visual_norm=restore_visual_norm,
        )
    ip_model = load_ip_adapter_model()

    # The pinned InstantX snapshot has already been added by model_loader.
    from infer_flux_ipa_siglip import resize_img

    reference_image = resize_img(
        load_reference_image(
            reference_image_url,
            reference_image_bytes,
        ).convert("RGB")
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
        "semantic_conditioning": component_analysis_payload,
        "checkpoint_load": ip_model.checkpoint_load_diagnostics,
        "orthogonalization": ip_model.last_orthogonalization_diagnostics,
        "mask_single_text_rows": mask_single_text_rows,
        "flux_text_seq_len": ip_model.last_single_stream_text_seq_len,
    }
    return buffer.getvalue(), diagnostics


def generate_semantic_png_bytes(
    *, identity_term: str, target_prompt: str, reference_image_bytes: bytes,
    llm: SemanticLLM, source_caption: str | None = None,
    origin: str = "source_target", strategy: str = "local", max_contexts: int = 256,
    strength: float = 1.0, restore_visual_norm: bool = True,
    mask_single_text_rows: bool = True, ip_adapter_scale: float = 0.7,
    guidance_scale: float = 3.5, num_inference_steps: int = 24, seed: int = 42,
    width: int = 1024, height: int = 1024,
) -> tuple[bytes, dict]:
    """Zero-shot/free-text Python entry point with injectable (or mock) LLM."""
    # Validate projection settings before paid analysis as well as GPU allocation.
    OrthogonalizationRequest(mode="off", strength=strength)
    conditioning = prepare_semantic_conditioning(
        llm=llm, identity_term=identity_term, target_prompt=target_prompt,
        reference_image_bytes=reference_image_bytes, source_caption=source_caption,
        origin=origin, strategy=strategy, max_contexts=max_contexts,
    )
    return generate_png_bytes(
        prompt=target_prompt, reference_image_url="", reference_image_bytes=reference_image_bytes,
        ip_adapter_scale=ip_adapter_scale, guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps, seed=seed, width=width, height=height,
        orthogonalization_mode="components", orthogonalization_strength=strength,
        object_prompt="", object_behavior_prompt="", object_background_prompt="",
        object_behavior_background_prompt="", remove_behavior=True, remove_background=True,
        include_interaction=False, restore_visual_norm=restore_visual_norm,
        mask_single_text_rows=mask_single_text_rows, component_analysis_payload=conditioning.to_dict(),
    )


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
