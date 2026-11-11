"""Wait for the existing sweep's successful completion, then run SSG-only."""
from pathlib import Path
import fcntl,hashlib,json,os,subprocess,sys,time,traceback
BASE=Path('/workspace/instantx-0807')
DEPENDENCY=BASE/'results/scale_sweep_20261025'
OUTPUT=BASE/'results/ssg_only_scale_0.7_20261025'
SWEEP=BASE/'analysis_tools/run_scale_sweep.py'
SUCCESSOR=BASE/'analysis_tools/run_ssg_only.py'
PYTHON=str(BASE/'venvs/generation/bin/python')

def digest(path):return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()

def write(path,data):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(path)

def validate_dependency(directory,expected_started,expected_sha):
    run=json.loads((directory/'run.json').read_text())
    state=json.loads((directory/'status.json').read_text())
    if run['started_at']!=expected_started or run['script_sha256']!=expected_sha:raise ValueError('Dependency run changed')
    if state.get('status')!='complete' or state.get('images')!=6300 or state.get('scored')!=6300:
        raise ValueError('Dependency did not complete all generation and evaluation')
    if state['updated_at']<run['started_at']:raise ValueError('Stale completion marker')
    result=json.loads((directory/'metrics/scores.json').read_text())
    keys=[(s['condition'],s['benchmark'],s['case_id']) for s in result['scores']]
    from collections import Counter
    counts=Counter((s['condition'],s['benchmark']) for s in result['scores'])
    expected={(f'method_scale_{scale:.1f}',benchmark):n for scale in (.6,.9,1.) for benchmark,n in (('dreambench',750),('dreambench_plus',1350))}
    if len(keys)!=6300 or len(set(keys))!=6300 or counts!=expected:raise ValueError('Dependency score coverage invalid')
    for lane in (0,1):
        stage=json.loads((directory/f'state/controller_lane_{lane}.json').read_text())
        if stage.get('status')!='complete':raise ValueError('Dependency lane incomplete')
    return state

def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with (OUTPUT/'.queue.lock').open('a') as queue_lock:
        fcntl.flock(queue_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run=json.loads((DEPENDENCY/'run.json').read_text())
        queued=dict(status='waiting',condition='ssg',ip_adapter_scale=.7,projection=False,ssg=True,
            images=2100,per_gpu=1050,gpus=2,queued_at=time.time(),dependency=str(DEPENDENCY),
            dependency_started_at=run['started_at'],dependency_sha256=run['script_sha256'],
            successor_sha256=digest(SUCCESSOR),queue_sha256=digest(__file__),poll_seconds=30)
        try:
            while True:
                if digest(SWEEP)!=queued['dependency_sha256'] or digest(SUCCESSOR)!=queued['successor_sha256']:
                    raise ValueError('A scheduled implementation changed; inspect queue before resuming')
                with (DEPENDENCY/'.sweep.lock').open('a') as dependency_lock:
                    try:fcntl.flock(dependency_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:
                        queued.update(status='waiting',updated_at=time.time())
                        write(OUTPUT/'queue.json',queued)
                    else:
                        validate_dependency(DEPENDENCY,queued['dependency_started_at'],queued['dependency_sha256'])
                        queued.update(status='running',started_at=time.time(),updated_at=time.time());write(OUTPUT/'queue.json',queued)
                        print('Dependency complete; starting SSG-only generation and evaluation',flush=True)
                        env=dict(os.environ,HF_HOME=str(BASE/'cache/huggingface'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                            HF_TOKEN='offline-cache-only',TORCH_HOME=str(BASE/'cache/torch'),CLIP_DOWNLOAD_ROOT=str(BASE/'cache/clip'),
                            OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
                        with (OUTPUT/'controller.log').open('a') as log:
                            subprocess.run([PYTHON,str(SUCCESSOR),'run'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                        state=json.loads((OUTPUT/'status.json').read_text())
                        if state.get('status')!='complete' or state.get('scored')!=2100:raise ValueError('Successor final status invalid')
                        queued.update(status='complete',completed_at=time.time(),updated_at=time.time());write(OUTPUT/'queue.json',queued)
                        print('SSG-only generation and evaluation complete',flush=True);return
                time.sleep(30)
        except Exception as exc:
            queued.update(status='failed',error=str(exc),updated_at=time.time());write(OUTPUT/'queue.json',queued)
            traceback.print_exc();raise
if __name__=='__main__':main()
