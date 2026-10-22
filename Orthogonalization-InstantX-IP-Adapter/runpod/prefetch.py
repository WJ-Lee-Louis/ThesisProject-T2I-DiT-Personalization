"""Download pinned weights once before launching parallel GPU workers."""
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from huggingface_hub import snapshot_download, get_token
from orthogonalization_instantx_ip_adapter import config

token = os.environ.get('HF_TOKEN') or get_token()
if not token:
    raise SystemExit('Run hf auth login, then source runpod/env.sh')
for repo, revision, patterns in (
    (config.BASE_MODEL_ID, config.BASE_MODEL_REVISION,
     ['model_index.json', 'scheduler/*', 'tokenizer/*', 'tokenizer_2/*',
      'text_encoder/*', 'text_encoder_2/*', 'transformer/*', 'vae/*']),
    (config.IMAGE_ENCODER_ID, config.IMAGE_ENCODER_REVISION, None),
    (config.IP_ADAPTER_REPO_ID, config.IP_ADAPTER_REVISION, config.INSTANTX_ALLOW_PATTERNS),
):
    print('Downloading', repo, flush=True)
    snapshot_download(repo, revision=revision, allow_patterns=patterns, token=token)
print('Generation weights cached')
