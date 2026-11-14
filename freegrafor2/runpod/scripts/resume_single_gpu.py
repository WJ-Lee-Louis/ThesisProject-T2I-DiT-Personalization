"""Resume the guarded two-lane experiment serially on physical GPU 0.

Keep the frozen benchmark source and logical case assignments unchanged.
"""
import os
import subprocess

import benchmark as base
import completion_guard as guard
import mask_recovery as recovery


def launch(lane, pilot=False, stage='generate'):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0')
    tag = f'lane-{lane}' + ('-pilot' if pilot else '')
    python = base.GEN if stage == 'generate' else base.METRIC
    with (base.OUTPUT/'logs'/f'{tag}.log').open('a') as log:
        base.write(base.OUTPUT/'state'/f'controller-{tag}.json',
                   dict(stage=stage, status='running', physical_gpu=0, updated_at=base.now()))
        args = [str(python), '-u', str(base.ROOT/'scripts/benchmark.py'), stage,
                '--lane', str(lane)] + (['--pilot'] if pilot else [])
        subprocess.run(args, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    base.write(base.OUTPUT/'state'/f'controller-{tag}.json',
               dict(stage=stage, status='complete', physical_gpu=0, updated_at=base.now()))


def phase(stage, pilot=False):
    for lane in range(2):
        launch(lane, pilot, stage)


def repair_lane(lane):
    if not recovery.active_failures(lane):
        return
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0')
    with (guard.REC/f'post-generation-lane-{lane}.log').open('a') as log:
        subprocess.run([str(base.GEN), '-u', str(base.ROOT/'scripts/mask_recovery.py'),
                        'repair', '--lane', str(lane)], env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    # The two thread-pool tasks inside the original guard must never compete
    # for this single GPU. Its saved worker PIDs belong to the previous Pod.
    import threading
    lock = threading.Lock()

    def await_lane(lane, worker):
        with lock:
            if guard.unattempted(lane):
                launch(lane)
            if guard.unattempted(lane):
                raise RuntimeError(f'Lane {lane} still has untouched cases')

    def serial_repair(lane):
        with lock:
            repair_lane(lane)

    base.launch = launch
    base.phase = phase
    guard.await_lane = await_lane
    guard.repair_lane = serial_repair
    guard.run()
