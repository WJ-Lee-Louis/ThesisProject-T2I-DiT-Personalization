"""Merge validated, disjoint case scores, never average workspace averages."""
import json
import math
from pathlib import Path
from freecus.manifest import PROJECT,load,sha256
from freecus.evaluate import project_evaluator
from freecus.run import write_json
from .split_dbpp import STATE


def main():
    plan=json.loads((STATE/'plan.json').read_text())
    evaluator=project_evaluator()
    rows=load(PROJECT/'xverse/manifests/dreambench_plus_seed42.jsonl',PROJECT)
    by_id={r['case_id']:r for r in rows};scores=[];seen=set();env=None;provenance={}
    root=PROJECT/'xverse/results/dreambench_plus'/plan['bundle_id']
    for workspace,part in plan['plans'].items():
        path=root/workspace/'metrics.json';result=json.loads(path.read_text())
        if result['protocol']!=evaluator.PROTOCOL:raise ValueError('Metric protocol differs')
        current={k:result[k] for k in ['device','device_name','versions']}
        if env is not None and current!=env:raise ValueError('Metric environments differ')
        env=current
        if result.get('generation_resolution')!=[768,768]:raise ValueError('Resolution differs')
        assigned=set(part['assigned_case_ids'])
        if set(result['partition']['case_ids'])!=assigned:raise ValueError('Partition assignment differs')
        if len(result['scores'])!=len(assigned):raise ValueError('Incomplete partition')
        for score in result['scores']:
            cid=score['case_id'];row=by_id[cid]
            if cid not in assigned or cid in seen:raise ValueError('Duplicate or unassigned score')
            if (score['benchmark']!='dreambench_plus' or score['condition']!='xverse' or
                score['reference_id']!=row['reference_id'] or score['target_prompt']!=row['prompt'] or
                score['reference_sha256']!=row['reference_sha256'] or
                not all(math.isfinite(score[m]) for m in evaluator.METRICS)):
                raise ValueError('Score identity mismatch')
            scores.append(score);seen.add(cid)
        provenance[workspace]=dict(score_file_sha256=sha256(path),cases=len(assigned),
            call=json.loads((STATE/workspace/'receipt.json').read_text()))
    if seen!=set(by_id):raise ValueError('Missing global cases')
    scores.sort(key=lambda s:s['case_id'])
    result=dict(protocol=evaluator.PROTOCOL,**env,generation_resolution=[768,768],
        source_sha256=plan['code_sha256'],bundle_id=plan['bundle_id'],
        summary=evaluator.summarize(scores),scores=scores,partitions=provenance)
    output=root/'metrics_merged.json'
    if output.exists():raise FileExistsError(output)
    write_json(output,result);print(json.dumps(result['summary'],indent=2))


if __name__=='__main__':main()
