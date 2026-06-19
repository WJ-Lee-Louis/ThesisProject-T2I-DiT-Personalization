APP_NAME = "instantx-flux-ip-adapter-inference"

BASE_MODEL_ID = "black-forest-labs/FLUX.1-dev"
IP_ADAPTER_REPO_ID = "InstantX/FLUX.1-dev-IP-Adapter"
IP_ADAPTER_WEIGHT_NAME = "ip-adapter.bin"
IMAGE_ENCODER_ID = "google/siglip-so400m-patch14-384"

DEFAULT_REFERENCE_IMAGE_URL = (
    "https://huggingface.co/InstantX/FLUX.1-dev-IP-Adapter/resolve/main/"
    "assets/images/2.png"
)
DEFAULT_REFERENCE_IMAGE_PATH = ""
DEFAULT_OUTPUT_DIR = "outputs/instantx_flux_ip_adapter"
DEFAULT_OUTPUT_PATH = ""
