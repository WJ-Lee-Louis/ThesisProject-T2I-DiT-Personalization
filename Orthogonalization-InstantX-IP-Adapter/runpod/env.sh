# Source this after every SSH login, including on a new Pod.
export RUNPOD_WORKDIR="${RUNPOD_WORKDIR:-/workspace/instantx-0807}"
export HF_HOME="$RUNPOD_WORKDIR/cache/huggingface"
# Login credentials stay outside the shared model cache (enter HF_TOKEN each session).
export HF_TOKEN_PATH=/root/.cache/instantx-0807/token
export TORCH_HOME="$RUNPOD_WORKDIR/cache/torch"
export CLIP_DOWNLOAD_ROOT="$RUNPOD_WORKDIR/cache/clip"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
source "$RUNPOD_WORKDIR/venvs/generation/bin/activate"
# The existing model loader requires HF_TOKEN; also accept the saved HF login.
if [ -z "${HF_TOKEN:-}" ]; then
    export HF_TOKEN="$(python -c 'from huggingface_hub import get_token; print(get_token() or "")')"
fi
