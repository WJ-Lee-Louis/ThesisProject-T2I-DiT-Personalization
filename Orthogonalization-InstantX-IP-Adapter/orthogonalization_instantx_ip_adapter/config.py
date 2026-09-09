"""Model revisions and inference defaults for the isolated experiment app."""

APP_NAME = "orthogonalization-instantx-flux-ip-adapter"
BATCH_APP_NAME = "orthogonalization-instantx-flux-batch"
BATCH_CLASS_NAME = "BatchGenerator"
BATCH_INPUTS_VOLUME_NAME = "orthogonalization-instantx-batch-inputs"
BATCH_RESULTS_VOLUME_NAME = "orthogonalization-instantx-batch-results"
BATCH_GPU = "A100-80GB"
BATCH_MAX_CONTAINERS = 3
BATCH_SCALEDOWN_WINDOW_SECONDS = 300

EVAL_BATCH_APP_NAME = "orthogonalization-instantx-xverse-eval"
EVAL_BATCH_CLASS_NAME = "EvaluationBatchGenerator"
EVAL_BATCH_INPUTS_VOLUME_NAME = "orthogonalization-instantx-xverse-eval-inputs"
EVAL_BATCH_RESULTS_VOLUME_NAME = "orthogonalization-instantx-xverse-eval-results"

METRIC_APP_NAME = "orthogonalization-instantx-xverse-metrics"
METRIC_CLASS_NAME = "XVerseMetricEvaluator"
METRIC_RESULTS_VOLUME_NAME = "orthogonalization-instantx-xverse-metric-results"
METRIC_GPU = "A10G"
METRIC_MAX_CONTAINERS = 3
XVERSE_REPOSITORY = "https://github.com/bytedance/XVerse.git"
XVERSE_REVISION = "65e2581aac0faffe4607881a10cc486186f6499d"
SAM2_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
    "sam2.1_hiera_large.pt"
)
AES_PREDICTOR_REVISION = "c0e15567fa61252cbff43d70c49a8bd27202ca9a"
AES_PREDICTOR_URL = (
    "https://raw.githubusercontent.com/discus0434/"
    f"aesthetic-predictor-v2-5/{AES_PREDICTOR_REVISION}/models/"
    "aesthetic_predictor_v2_5.pth"
)

BASE_MODEL_ID = "black-forest-labs/FLUX.1-dev"
BASE_MODEL_REVISION = "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
IP_ADAPTER_REPO_ID = "InstantX/FLUX.1-dev-IP-Adapter"
IP_ADAPTER_REVISION = "e44c6d889c951cac03ac806991e8d46c9ce1ddba"
IP_ADAPTER_WEIGHT_NAME = "ip-adapter.bin"

IMAGE_ENCODER_ID = "google/siglip-so400m-patch14-384"
IMAGE_ENCODER_REVISION = "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"

INSTANTX_ALLOW_PATTERNS = [
    "attention_processor.py",
    "infer_flux_ipa_siglip.py",
    "pipeline_flux_ipa.py",
    "transformer_flux.py",
    IP_ADAPTER_WEIGHT_NAME,
]

DEFAULT_REFERENCE_IMAGE_URL = (
    "https://huggingface.co/InstantX/FLUX.1-dev-IP-Adapter/resolve/main/"
    "assets/images/2.png"
)
DEFAULT_REFERENCE_IMAGE_PATH = ""
DEFAULT_OUTPUT_DIR = "outputs/orthogonalization_instantx_ip_adapter"
DEFAULT_OUTPUT_PATH = ""

DEFAULT_ORTHOGONALIZATION_MODE = "off"
DEFAULT_ORTHOGONALIZATION_STRENGTH = 0.5
DEFAULT_MASK_SINGLE_TEXT_ROWS = False

MODAL_PIP_PACKAGES = (
    "accelerate==1.10.0",
    "diffusers==0.35.2",
    "hf_transfer==0.1.9",
    "huggingface_hub==0.34.3",
    "pillow==11.3.0",
    "protobuf==6.31.1",
    "safetensors==0.6.2",
    "sentencepiece==0.2.0",
    "torch==2.7.1",
    "transformers==4.54.0",
)
