#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export TORCH_HOME="$project_root/diptych/outputs/model-cache/torch"
export CLIP_DOWNLOAD_ROOT="$project_root/diptych/outputs/model-cache/clip"
exec "$project_root/diptych/.venv-local/bin/python" -u -m diptych.local "${@:-run}"
