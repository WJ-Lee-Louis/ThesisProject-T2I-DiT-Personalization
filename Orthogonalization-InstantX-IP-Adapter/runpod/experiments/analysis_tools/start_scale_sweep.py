import os,subprocess,time
from pathlib import Path
out=Path('/workspace/instantx-0807/results/scale_sweep_20261025');out.mkdir(parents=True,exist_ok=True)
if (out/'status.json').exists(): (out/'status.json').rename(out/f'status_previous_{int(time.time())}.json')
env=dict(os.environ,HF_HOME='/workspace/instantx-0807/cache/huggingface',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_TOKEN='offline-cache-only',TORCH_HOME='/workspace/instantx-0807/cache/torch',CLIP_DOWNLOAD_ROOT='/workspace/instantx-0807/cache/clip',OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
with (out/'controller.log').open('a') as log:
 p=subprocess.Popen(['/workspace/instantx-0807/venvs/generation/bin/python','/workspace/instantx-0807/analysis_tools/run_scale_sweep.py','run'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(out/'controller.pid').write_text(str(p.pid)+'\n')
print('Controller PID:',p.pid)
