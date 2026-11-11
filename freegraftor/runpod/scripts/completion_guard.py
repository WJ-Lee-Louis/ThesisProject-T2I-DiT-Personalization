"""Preserve live native workers, repair non-style failures, then evaluate fully."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import fcntl,json,os,signal,subprocess,time,traceback
import benchmark as base
import mask_recovery as recovery

ROOT=base.ROOT;OUTPUT=base.OUTPUT;REC=ROOT/'recovery'

class MissingSubjectMasks(RuntimeError):pass

def process_identity(pid):
    try:
        text=Path(f'/proc/{pid}/stat').read_text();rest=text.rsplit(')',1)[1].split()
        return dict(state=rest[0],start_ticks=rest[19])
    except FileNotFoundError:return None

def alive(pid,start_ticks=None):
    p=process_identity(pid)
    return bool(p and p['state']!='Z' and (start_ticks is None or p['start_ticks']==start_ticks))

def resume_abandoned_pause():
    p=REC/'immediate-status.json'
    if not p.exists():return
    status=base.read(p);pid=status.get('paused_worker')
    if not pid:return
    state=process_identity(pid)
    supervisor=REC/'immediate-controller.pid'
    controller=int(supervisor.read_text()) if supervisor.exists() else None
    if state and state['state']=='T' and (not controller or not alive(controller)):
        os.kill(pid,signal.SIGCONT)
        base.write(REC/'pause-watchdog.json',dict(worker_resumed=pid,reason='Immediate recovery controller exited',updated_at=time.time()))

def unattempted(lane):
    return [j for j in base.jobs_for(lane) if not all(p.exists() for p in base.paths(j)) and not base.failure_path(j).exists()]

def await_lane(lane,worker):
    if worker:
        while alive(worker['pid'],worker['start_ticks']):
            resume_abandoned_pause();time.sleep(10)
    # Normally no work remains. If a native worker exited early, finish its
    # untouched jobs before collecting the explicit failed subject masks.
    for attempt in range(4):
        missing=unattempted(lane)
        if not missing:return
        before=len(missing)
        try:base.launch(lane,False,'generate')
        except Exception:
            traceback.print_exc()
            if len(unattempted(lane))>=before:raise
    if unattempted(lane):raise RuntimeError(f'Lane {lane} still has untouched jobs')

def repair_lane(lane):
    if not recovery.active_failures(lane):return
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(lane))
    with (REC/f'post-generation-lane-{lane}.log').open('a') as log:
        subprocess.run([str(base.GEN),'-u',str(ROOT/'scripts/mask_recovery.py'),'repair','--lane',str(lane)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)

def enforce_non_style_complete():
    required=[j for j in base.jobs_for() if base.metrics.category(j['row'])!='style']
    assert len(required)==1920
    missing=[];repaired=[]
    for j in required:
        if not recovery.validate_repair(j):missing.append(j['row'])
        elif base.read(base.paths(j)[1]).get('mask_recovery'):repaired.append(j['row'])
    report=dict(expected=1920,complete=len(required)-len(missing),missing=len(missing),missing_cases=missing,reprocessed=len(repaired),reprocessed_cases=repaired,updated_at=time.time())
    base.write(REC/'non-style-coverage.json',report)
    if missing:raise MissingSubjectMasks(f'{len(missing)} non-style cases need a valid subject mask; final benchmark metrics are withheld')
    return report

def run():
    config=base.read(REC/'completion-guard.json')
    with (OUTPUT/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert base.read(OUTPUT/'run.json')['provenance']['implementation']==base.source_digest(),'Native implementation changed'
        try:
            base.write(OUTPUT/'status.json',dict(status='running',stage='generation_with_non_style_recovery_guard',non_style_complete_required=True,updated_at=time.time()))
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(await_lane,lane,config['native_workers'].get(str(lane))) for lane in range(2)]
                for future in futures:future.result()
            # An immediate rescue owns GPU0 only while its native worker is
            # stopped; wait for the supervisor to release the paused worker.
            while (REC/'immediate.lock').exists():resume_abandoned_pause();time.sleep(5)
            base.write(OUTPUT/'status.json',dict(status='running',stage='non_style_mask_reprocessing',non_style_complete_required=True,updated_at=time.time()))
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(repair_lane,lane) for lane in range(2)]
                for future in futures:future.result()
            coverage=enforce_non_style_complete()
            policy=base.evaluation_policy()
            policy.update(non_style_complete_required=True,non_style_cases_complete=coverage['complete'],subject_mask_reprocessed_cases=coverage['reprocessed'])
            base.write(OUTPUT/'evaluation-policy.json',policy)
            base.write(OUTPUT/'status.json',dict(status='running',stage='full_metrics',evaluation_policy=policy,updated_at=time.time()))
            base.phase('score')
            generated,scored,failed=base.merge()
            result=base.read(OUTPUT/'metrics/comparison.json')
            result['subject_mask_recovery']=dict(cases=coverage['reprocessed'],records='../../recovery/non-style-coverage.json',note='Failed non-style cases were reprocessed with audited equivalent detector noun phrases. Diffusion prompts, selected references, generation hyperparameters and native detection thresholds are unchanged; original failures and per-image detector/SAM masks are preserved.')
            base.write(OUTPUT/'metrics/comparison.json',result)
            base.write(OUTPUT/'status.json',dict(status='complete_with_style_exclusions' if failed else 'complete',stage='all',expected=2100,images=generated,scored=scored,failed=failed,non_style_missing=0,subject_mask_reprocessed_cases=coverage['reprocessed'],evaluation_policy=policy,updated_at=time.time()))
        except MissingSubjectMasks as e:
            base.write(OUTPUT/'status.json',dict(status='needs_mask_review',stage='non_style_mask_reprocessing',error=str(e),non_style_complete_required=True,updated_at=time.time()));raise
        except Exception as e:
            base.write(OUTPUT/'status.json',dict(status='failed',error=str(e),updated_at=time.time()));raise

if __name__=='__main__':run()
