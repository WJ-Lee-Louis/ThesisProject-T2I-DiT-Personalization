"""Native-resolution artifact validation; immutable shared scoring code is reused."""
import json
from pathlib import Path
from freecus.evaluate import prepare,project_evaluator
from freecus.manifest import sha256
from freecus.run import artifact_paths,complete


def validate(plan,account=None):
    if account is not None:raise ValueError('Separate complete benchmark runs required')
    from PIL import Image
    run=Path(plan['freecus_run'])
    info=json.loads((run/'run.json').read_text())
    method=info['config']['method']
    if method not in {'mosaic'}:raise ValueError('Unexpected method')
    if info['config']['seeds']!=[42] or (info['config']['width'],info['config']['height'])!=(512,512):
        raise ValueError('Expected native 512 and seed42')
    if sha256(plan['manifest'])!=plan['manifest_sha256'] or info['manifest_sha256']!=plan['manifest_sha256']:
        raise ValueError('Manifest mismatch')
    expected=prepare(Path(plan['manifest']),Path(plan['data_root']),run,project_evaluator(),plan['selection']['name'])
    if plan.get('partition'):
        ids=plan['partition']['case_ids']
        if len(ids)!=len(set(ids)) or plan['partition']['global_cases']!=len(expected['records']):
            raise ValueError('Invalid partition')
        expected['records']=[r for r in expected['records'] if r['row']['case_id'] in set(ids)]
        if len(expected['records'])!=len(ids):raise ValueError('Unknown partition cases')
    for rec in expected['records']:rec['condition']=method
    if expected['records']!=plan['records'] or expected['selection']!=plan['selection']:
        raise ValueError('Canonical cases differ')
    for record in plan['records']:
        paths=artifact_paths(run,dict(id=record['freecus_id']))
        if not complete(paths):raise ValueError('Corrupt/missing artifact')
        meta=json.loads(paths['metadata'].read_text())
        row=record['row']
        if (meta['fingerprint']!=info['fingerprint'] or meta['seed']!=42 or
            meta['case_id']!=row['case_id'] or meta['prompt']!=row['target_prompt'] or
            meta['reference_id']!=row['reference_id']):raise ValueError('Generation identity mismatch')
        if sha256(record['reference'])!=record['reference_sha256']:raise ValueError('Reference mismatch')
        with Image.open(record['generated']) as image:
            image.verify()
            if image.size!=(512,512):raise ValueError('Unexpected native image size')
