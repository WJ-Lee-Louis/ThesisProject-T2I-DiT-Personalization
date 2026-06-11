import torch
from diffusers import FluxPipeline

from xlabs_flux_ip_adapter.config import (
    BASE_MODEL_ID,
    IMAGE_ENCODER_ID,
    IP_ADAPTER_ID,
    IP_ADAPTER_WEIGHT_NAME,
)


def load_pipeline():
    pipe = FluxPipeline.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    pipe.load_ip_adapter(
        IP_ADAPTER_ID,
        weight_name=IP_ADAPTER_WEIGHT_NAME,
        image_encoder_pretrained_model_name_or_path=IMAGE_ENCODER_ID,
    )
    return pipe
