"""Recover only the verified existing UNO 512 export; preserve generation and metrics."""
import json
from pathlib import Path
from freecus.modal_client import connect
from freecus.run import write_json
from uno.experiment_512 import STATE, RUNS, read


def main():
    modal = connect('third')
    receipt = read(STATE / 'receipt.json')
    plans = read(STATE / 'plans.json')
    plan = next(p for p in plans if p['benchmark'] == 'dreambench_plus')
    volume = modal.Volume.from_name(RUNS)
    app = modal.App('uno-512-export-recovery-third-20261113')
    image = modal.Image.debian_slim(python_version='3.11').add_local_dir(plan['source'], '/root')

    @app.function(name='recover', serialized=True, image=image, volumes={'/work': volume},
                  timeout=3600, cpu=(2, 2), memory=(4096, 4096), retries=0,
                  max_containers=1, scaledown_window=2)
    def recover(queue_id, run_id):
        import time
        from freecus.operations.artifact_bundle import pack, verify
        from freecus.run import write_json
        volume.reload()
        root = Path('/work')
        original = read(root / 'workflows' / (queue_id + '.json'))
        recovery_id = queue_id + '-export-recovered'
        result_path = root / 'workflows' / (recovery_id + '.json')
        if result_path.exists():
            return read(result_path)
        lanes = sorted((root / 'results' / run_id / 'A100-80GB').glob('lane-*'))
        infos = [read(p / 'run.json') for p in lanes]
        if len(infos) != 2:
            raise ValueError('Expected two generation lanes')
        allowed = {'NVIDIA A100 80GB PCIe', 'NVIDIA A100-SXM4-80GB'}
        if any(i['gpu'] not in allowed for i in infos):
            raise ValueError('Unexpected generation device')
        for key in ['fingerprint', 'config', 'manifest_sha256', 'code', 'versions', 'cuda']:
            if any(i[key] != infos[0][key] for i in infos):
                raise ValueError('Generation identity differs: ' + key)
        folder = root / 'metrics' / run_id
        coverage = read(folder / 'coverage.json')
        metrics = read(folder / 'metrics.json')
        if coverage['generated'] != 1350 or coverage['missing_case_ids'] or coverage['status'] != 'complete':
            raise ValueError('Evaluation coverage incomplete')
        if len(metrics['scores']) != 1350:
            raise ValueError('Incomplete metric rows')
        bundle = root / 'bundles' / run_id
        extras = {p.name: p for p in [bundle / 'models.lock.json', bundle / 'config.512.json',
                  bundle / 'source.zip', bundle / 'identity.json', folder / 'coverage.json']}
        for lane in lanes:
            for p in (lane / 'conditioning').glob('*.png'):
                extras[f'conditioning/{lane.name}/{p.name}'] = p
        audit = dict(reason='A100 PCIe/SXM names differ; all other checked generation fields match',
                     generation_devices=[i['gpu'] for i in infos], generation_changed=False,
                     metrics_recomputed=False, original_queue_id=queue_id)
        write_json(folder / 'export-recovery.json', audit)
        extras['export-recovery.json'] = folder / 'export-recovery.json'
        destination = root / 'exports' / run_id
        index = pack(lanes, bundle / 'manifest.jsonl', bundle / 'input', destination,
                     'uno_512', 'dreambench_plus', run_id, 1350, resolution=(512, 512),
                     allow_gpu_variants=True, provenance_files=extras)
        verify(destination, index)
        report = dict(coverage=coverage, summary=metrics['summary'], verified=True,
                      parts=index['parts'], recovery=audit)
        write_json(folder / 'summary.json', report)
        write_json(destination / 'verification.json', report)
        original.update(queue_id=recovery_id, status='complete', recovered_from=queue_id,
                        original_error=original.pop('error', None), updated_at=time.time())
        original['evaluations']['dreambench_plus'] = report
        write_json(result_path, original)
        volume.commit()
        return original

    submission = STATE / 'export-recovery-submission.json'
    if submission.exists():
        call = modal.FunctionCall.from_id(read(submission)['call_id'])
    else:
        with modal.enable_output():
            app.deploy()
        call = modal.Function.from_name(app.name, 'recover').spawn(receipt['queue_id'], plan['run_id'])
        write_json(submission, dict(call_id=call.object_id, profile='third', app=app.name))
    result = call.get()
    write_json(STATE / 'export-recovery.json', result)
    if result['status'] != 'complete':
        raise ValueError('Recovery incomplete')
    write_json(STATE / 'receipt-before-export-recovery.json', receipt)
    receipt.update(queue_id=result['queue_id'], recovered_from=receipt['queue_id'], status='complete',
                   recovery_call_id=call.object_id)
    write_json(STATE / 'receipt.json', receipt)
    for p in plans:
        write_json(Path(p['local']) / 'receipt.json', receipt)
        write_json(Path(p['local']) / 'workflow-status.json', result)
    print(json.dumps(dict(status=result['status'], call_id=call.object_id,
                         coverage=result['evaluations']['dreambench_plus']['coverage'],
                         parts=result['evaluations']['dreambench_plus']['parts']), indent=2))


if __name__ == '__main__':
    main()
