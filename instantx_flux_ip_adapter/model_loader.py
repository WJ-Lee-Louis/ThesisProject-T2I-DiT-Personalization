import os
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from instantx_flux_ip_adapter.config import (
    BASE_MODEL_ID,
    IMAGE_ENCODER_ID,
    IP_ADAPTER_REPO_ID,
    IP_ADAPTER_WEIGHT_NAME,
)


INSTANTX_ALLOW_PATTERNS = [
    "attention_processor.py",
    "infer_flux_ipa_siglip.py",
    "pipeline_flux_ipa.py",
    "transformer_flux.py",
    IP_ADAPTER_WEIGHT_NAME,
]


def get_hf_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(
            "Missing Hugging Face token. Register a Modal Secret named "
            "`huggingface-secret` with HF_TOKEN or HUGGINGFACE_HUB_TOKEN."
        )
    return token


def download_instantx_files() -> Path:
    local_dir = snapshot_download(
        IP_ADAPTER_REPO_ID,
        allow_patterns=INSTANTX_ALLOW_PATTERNS,
        token=get_hf_token(),
    )
    return Path(local_dir)


def load_ip_adapter_model():
    instantx_dir = download_instantx_files()
    sys.path.insert(0, str(instantx_dir))

    from infer_flux_ipa_siglip import IPAdapter
    from pipeline_flux_ipa import FluxPipeline
    from transformer_flux import FluxTransformer2DModel

    transformer = FluxTransformer2DModel.from_pretrained(
        BASE_MODEL_ID,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        token=get_hf_token(),
    )
    pipe = FluxPipeline.from_pretrained(
        BASE_MODEL_ID,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
        token=get_hf_token(),
    )

    return IPAdapter(
        pipe,
        IMAGE_ENCODER_ID,
        str(instantx_dir / IP_ADAPTER_WEIGHT_NAME),
        device="cuda",
        num_tokens=128,
    )
