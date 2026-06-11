from pathlib import Path

import modal

from shared.modal_runtime import (
    DEFAULT_GPU,
    HF_SECRET_NAME,
    build_diffusers_image,
    build_hf_cache_volume,
)
from xlabs_flux_ip_adapter.config import (
    APP_NAME,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_REFERENCE_IMAGE_URL,
)


app = modal.App(APP_NAME)
hf_cache = build_hf_cache_volume()
image = build_diffusers_image()


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    volumes={"/cache": hf_cache},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    timeout=1800,
)
def generate_with_xlabs_ip_adapter(
    prompt: str,
    reference_image_url: str,
    ip_adapter_scale: float = 1.0,
    guidance_scale: float = 4.0,
    num_inference_steps: int = 20,
    seed: int = 42,
    width: int = 1024,
    height: int = 1024,
) -> bytes:
    from xlabs_flux_ip_adapter.inference import generate_png_bytes

    return generate_png_bytes(
        prompt=prompt,
        reference_image_url=reference_image_url,
        ip_adapter_scale=ip_adapter_scale,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
        width=width,
        height=height,
    )


@app.local_entrypoint()
def main(
    prompt: str = "a portrait photo of the same person wearing a black jacket, studio lighting",
    reference_image_url: str = DEFAULT_REFERENCE_IMAGE_URL,
    ip_adapter_scale: float = 1.0,
    guidance_scale: float = 4.0,
    num_inference_steps: int = 20,
    seed: int = 42,
    width: int = 1024,
    height: int = 1024,
    output_path: str = DEFAULT_OUTPUT_PATH,
):
    output_bytes = generate_with_xlabs_ip_adapter.remote(
        prompt=prompt,
        reference_image_url=reference_image_url,
        ip_adapter_scale=ip_adapter_scale,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
        width=width,
        height=height,
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(output_bytes)
    print(f"Saved image to {path.resolve()}")
