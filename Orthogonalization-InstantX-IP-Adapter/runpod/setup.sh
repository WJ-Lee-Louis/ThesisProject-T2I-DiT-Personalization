#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export RUNPOD_WORKDIR="${RUNPOD_WORKDIR:-/workspace/instantx-0807}"
command -v python3.11 >/dev/null || { echo 'Python 3.11 is required. Use a Python 3.11 template or install it before setup.'; exit 1; }
apt-get update
apt-get install -y git tmux
mkdir -p "$RUNPOD_WORKDIR/venvs" "$RUNPOD_WORKDIR/cache"
python3.11 -m venv "$RUNPOD_WORKDIR/venvs/generation" || {
    apt-get install -y python3.11-venv
    python3.11 -m venv "$RUNPOD_WORKDIR/venvs/generation"
}
python3.11 -m venv "$RUNPOD_WORKDIR/venvs/metric"
"$RUNPOD_WORKDIR/venvs/generation/bin/python" -m pip install --no-cache-dir -r runpod/requirements-generation.txt
"$RUNPOD_WORKDIR/venvs/metric/bin/python" -m pip install --no-cache-dir -r requirements-benchmark-metrics.txt
"$RUNPOD_WORKDIR/venvs/generation/bin/python" -m pip check
"$RUNPOD_WORKDIR/venvs/metric/bin/python" -m pip check
echo 'Setup complete. Next: source runpod/env.sh'
