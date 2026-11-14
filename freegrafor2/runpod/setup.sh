#!/usr/bin/env bash
set -euo pipefail
cd /workspace/freegraftor
source env.sh
runtime=/opt/freegraftor-runtime
mkdir -p "$runtime/flux-native" "$runtime/uv-cache-local"
if [ ! -x "$runtime/generation-local/bin/python" ]; then
    uv venv --python /usr/bin/python3.10 "$runtime/generation-local"
fi
export UV_CACHE_DIR="$runtime/uv-cache-local"
req=requirements-generation.txt
if [ -s requirements-generation.lock.txt ]; then req=requirements-generation.lock.txt; fi
uv pip install --python "$runtime/generation-local/bin/python" --link-mode hardlink --extra-index-url https://download.pytorch.org/whl/cu124 --index-strategy unsafe-best-match -r "$req"
uv pip check --python "$runtime/generation-local/bin/python"
python scripts/prepare_inputs.py
"$runtime/generation-local/bin/python" scripts/convert_cached_flux.py
python scripts/mirror_text_encoders.py
"$runtime/generation-local/bin/python" scripts/prepare_public_assets.py
echo 'Ready. Start using: bash start.sh'
