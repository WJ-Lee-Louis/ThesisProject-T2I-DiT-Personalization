#!/usr/bin/env bash
set -euo pipefail
cd /workspace/Ours/ThesisProject-T2I-DiT-Personalization/Orthogonalization-InstantX-IP-Adapter
base=/workspace/instantx-0807
trap 'rc=$?; echo "$rc" > "$base/logs/pipeline.exit"' EXIT
echo "Waiting for setup and model prefetch: $(date -Is)"
for stage in setup prefetch; do
    while [ ! -f "$base/logs/$stage.exit" ]; do sleep 10; done
    if [ "$(cat "$base/logs/$stage.exit")" != 0 ]; then
        echo "$stage failed; inspect $base/logs/$stage.log"
        exit 1
    fi
done
source runpod/env.sh
python scripts/benchmark_runpod.py plan --gpus 1 --output "$base/results/full"
echo "Pilot started: $(date -Is)"
python scripts/benchmark_runpod.py all --gpus 1 --limit 5 --output "$base/results/pilot"
python - <<'PY'
import json
from pathlib import Path
p = Path('/workspace/instantx-0807/results/pilot')
status = json.loads((p / 'status-all-0.json').read_text())
assert status['status'] == 'complete' and status['images'] == 20, status
assert (p / 'metrics/summary.csv').is_file()
print('Pilot generation and CLIP/DINO evaluation passed')
PY
echo "Full sequential experiments and evaluation started: $(date -Is)"
python scripts/benchmark_runpod.py all --gpus 1 --output "$base/results/full"
echo "Pipeline complete: $(date -Is)"
