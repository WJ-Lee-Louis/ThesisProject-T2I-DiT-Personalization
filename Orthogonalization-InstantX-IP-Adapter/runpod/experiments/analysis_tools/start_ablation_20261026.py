import os, subprocess, json, sys
from pathlib import Path
ROOT=Path('/workspace/instantx-0807')
OUTPUT=ROOT/'results/ablation_20261026'
METHODS=['baseline_0.8','baseline_0.9','source_only','target_only']
env=dict(os.environ,HF_HOME=str(ROOT/'cache/huggingface'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_TOKEN='offline-cache-only',TORCH_HOME=str(ROOT/'cache/torch'),CLIP_DOWNLOAD_ROOT=str(ROOT/'cache/clip'),OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
if __name__=='__main__':
    OUTPUT.mkdir(parents=True,exist_ok=True)
    for gpu,method in enumerate(METHODS):
        local=dict(env,EXPERIMENT_METHOD=method,EXPERIMENT_GPU=str(gpu))
        if '--check' in sys.argv:
            subprocess.run([str(ROOT/'venvs/generation/bin/python'),'-c',"import run_ablation_20261026 as a; from collections import Counter; jobs=a.jobs_for(); assert len(jobs)==2100; [a.generation_kwargs(j,(a.ROOT/j['row']['image_path']).read_bytes()) for j in a.jobs_for(1)]; print(a.METHOD,Counter(j['row']['benchmark'] for j in jobs))"],env=local,cwd=ROOT/'analysis_tools',check=True)
        else:
            dest=OUTPUT/method;dest.mkdir(exist_ok=True)
            if (dest/'controller.pid').exists():raise RuntimeError('Existing controller; inspect before resuming')
            with (dest/'controller.log').open('a') as log:
                child=subprocess.Popen([str(ROOT/'venvs/generation/bin/python'),str(ROOT/'analysis_tools/run_ablation_20261026.py'),'run'],env=local,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            (dest/'controller.pid').write_text(str(child.pid)+'\n')
            print(method,'GPU',gpu,'controller',child.pid,flush=True)
