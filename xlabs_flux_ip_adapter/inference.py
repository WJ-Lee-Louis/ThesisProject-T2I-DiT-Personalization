from io import BytesIO

import torch
from diffusers.utils import load_image

from xlabs_flux_ip_adapter.conditioning import (
    apply_conditioning_controls,
    prepare_reference_image,
)
from xlabs_flux_ip_adapter.model_loader import load_pipeline


def generate_png_bytes(
    prompt: str,
    reference_image_url: str,
    ip_adapter_scale: float,
    guidance_scale: float,
    num_inference_steps: int,
    seed: int,
    width: int,
    height: int,
) -> bytes:
    pipe = load_pipeline()
    pipe = apply_conditioning_controls(pipe, ip_adapter_scale=ip_adapter_scale)

    reference_image = prepare_reference_image(
        load_image(reference_image_url),
        width=width,
        height=height,
    )
    generator = torch.Generator(device="cuda").manual_seed(seed)

    result = pipe(
        prompt=prompt,
        ip_adapter_image=reference_image,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    ).images[0]

    buffer = BytesIO()
    result.save(buffer, format="PNG")
    return buffer.getvalue()
