import modal


HF_SECRET_NAME = "huggingface-secret"
HF_CACHE_VOLUME_NAME = "hf-model-cache"
DEFAULT_GPU = "A100-80GB"


def build_hf_cache_volume() -> modal.Volume:
    return modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)


def build_diffusers_image() -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git")
        .pip_install(
            "accelerate>=1.9.0",
            "diffusers>=0.35.0",
            "hf_transfer>=0.1.9",
            "huggingface_hub>=0.33.0",
            "pillow>=11.0.0",
            "protobuf>=5.0.0",
            "safetensors>=0.5.0",
            "sentencepiece>=0.2.0",
            "torch>=2.7.0",
            "transformers>=4.52.0",
        )
        .env(
            {
                "HF_HOME": "/cache/huggingface",
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
                "PYTHONUNBUFFERED": "1",
            }
        )
        .add_local_python_source(
            "shared",
            "xlabs_flux_ip_adapter",
            "flux_redux",
            "instantx_flux_ip_adapter",
        )
    )
