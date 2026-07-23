"""Load pinned InstantX source files with the experiment adapter overlay."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from .adapter import OrthogonalizedIPAdapterMixin
from .config import (
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    IMAGE_ENCODER_ID,
    IMAGE_ENCODER_REVISION,
    INSTANTX_ALLOW_PATTERNS,
    IP_ADAPTER_REPO_ID,
    IP_ADAPTER_REVISION,
    IP_ADAPTER_WEIGHT_NAME,
)


def get_hf_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(
            "Missing Hugging Face token. Register a Modal Secret named "
            "`huggingface-secret` with HF_TOKEN or HUGGINGFACE_HUB_TOKEN."
        )
    return token


def download_instantx_files() -> Path:
    return Path(
        snapshot_download(
            IP_ADAPTER_REPO_ID,
            revision=IP_ADAPTER_REVISION,
            allow_patterns=INSTANTX_ALLOW_PATTERNS,
            token=get_hf_token(),
        )
    )


def download_siglip_checkpoint() -> Path:
    """Resolve one pinned checkpoint directory shared by both SigLIP towers."""

    return Path(
        snapshot_download(
            IMAGE_ENCODER_ID,
            revision=IMAGE_ENCODER_REVISION,
            token=get_hf_token(),
        )
    )


def _prepend_source_path(path: Path) -> None:
    source_path = str(path)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)


def load_ip_adapter_model(device: str = "cuda"):
    instantx_dir = download_instantx_files()
    siglip_dir = download_siglip_checkpoint()
    _prepend_source_path(instantx_dir)

    # InstantX's published files use top-level imports, so they must be imported
    # only after their pinned snapshot directory is on sys.path.
    from infer_flux_ipa_siglip import IPAdapter
    from pipeline_flux_ipa import FluxPipeline
    from transformer_flux import FluxTransformer2DModel

    class OrthogonalizedIPAdapter(OrthogonalizedIPAdapterMixin, IPAdapter):
        """Runtime composition keeps the upstream class and weights unchanged."""

    transformer = FluxTransformer2DModel.from_pretrained(
        BASE_MODEL_ID,
        revision=BASE_MODEL_REVISION,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        token=get_hf_token(),
    )
    pipe = FluxPipeline.from_pretrained(
        BASE_MODEL_ID,
        revision=BASE_MODEL_REVISION,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
        token=get_hf_token(),
    )

    return OrthogonalizedIPAdapter(
        pipe,
        str(siglip_dir),
        str(instantx_dir / IP_ADAPTER_WEIGHT_NAME),
        device=device,
        num_tokens=128,
    )
