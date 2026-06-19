from io import BytesIO

from diffusers.utils import load_image
from PIL import Image

from instantx_flux_ip_adapter.conditioning import prepare_reference_image
from instantx_flux_ip_adapter.model_loader import load_ip_adapter_model


def load_reference_image(reference_image_url: str, reference_image_bytes: bytes | None):
    if reference_image_bytes is not None:
        return Image.open(BytesIO(reference_image_bytes))
    return load_image(reference_image_url)


def generate_png_bytes(
    prompt: str,
    reference_image_url: str,
    reference_image_bytes: bytes | None,
    ip_adapter_scale: float,
    guidance_scale: float,
    num_inference_steps: int,
    seed: int,
    width: int,
    height: int,
) -> bytes:
    ip_model = load_ip_adapter_model()
    reference_image = prepare_reference_image(
        ip_model,
        load_reference_image(reference_image_url, reference_image_bytes),
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
    )[0]

    buffer = BytesIO()
    result.save(buffer, format="PNG")
    return buffer.getvalue()
