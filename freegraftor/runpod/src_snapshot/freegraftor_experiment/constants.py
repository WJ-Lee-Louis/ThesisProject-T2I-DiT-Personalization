"""Pinned upstream/model identifiers and Modal resource names."""

from __future__ import annotations


UPSTREAM_REPOSITORY = "https://github.com/Nihukat/FreeGraftor"
UPSTREAM_COMMIT = "b8e0f93e91753265e46e9d0b364daa38e25f1afa"

FLUX_MODEL_ID = "black-forest-labs/FLUX.1-dev"
FLUX_MODEL_REVISION = "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
GROUNDING_DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
GROUNDING_DINO_MODEL_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
SAM_FILENAME = "sam_vit_h_4b8939.pth"
SAM_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything/"
    "sam_vit_h_4b8939.pth"
)
SAM_SIZE_BYTES = 2_564_550_879
SAM_SHA256 = "a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e"
LAMA_MODEL_ID = "iic/cv_fft_inpainting_lama"
LAMA_MODEL_REVISION = "v1.0.0"

HF_SECRET_NAME = "huggingface-secret"
MODEL_CACHE_VOLUME_NAME = "hf-model-cache"
INPUTS_VOLUME_NAME = "freegraftor-batch-inputs"
RESULTS_VOLUME_NAME = "freegraftor-batch-results"
APP_NAME = "freegraftor-dreambench-batch"
CLASS_NAME = "FreeGraftorBatch"
GPU = "A100-80GB"
MAX_CONTAINERS = 2

EXPECTED_REFERENCE_COUNT = 10
EXPECTED_DOG_COUNT = 7
EXPECTED_CAT_COUNT = 3
