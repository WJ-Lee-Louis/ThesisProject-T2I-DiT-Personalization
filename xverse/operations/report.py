"""Audit score-to-archive hashes and retain JSON/CSV reports and portable inventory."""
import csv
import json
import math
import shutil
from pathlib import Path
from freecus.manifest import PROJECT
from freecus.evaluate import project_evaluator
from freecus.operations.artifact_bundle import digest,write
from .collect import settings


def main():
    evaluator=project_evaluator()
    reports=PROJECT/'xverse/results/reports';reports.mkdir(parents=True,exist_ok=True)
    summaries=[];inventory=[]
    for benchmark,owner in [('dreambench','louislee'),('dreambench_plus','woojin716')]:
        _,bid,_,_=settings(owner)
        completed=PROJECT/'xverse/outputs/dreambench_recovery/complete_bundle.json'
        if benchmark=='dreambench' and completed.exists():bid=json.loads(completed.read_text())['bundle_id']
        root=PROJECT/'xverse/results'/benchmark/bid
        index=json.loads((root/'bundle_index.json').read_text())
        metric=root/'metrics.json'
        if benchmark=='dreambench_plus' and not metric.exists():shutil.copyfile(root/'metrics_merged.json',metric)
        result=json.loads(metric.read_text())
        rows=[json.loads(line) for line in (root/'manifest.jsonl').read_text().splitlines()]
        canonical={r['case_id']:r for r in rows}
        seen=set()
        for score in result['scores']:
            row=canonical[score['case_id']]
            if score['case_id'] in seen:raise ValueError('Duplicate score')
            seen.add(score['case_id'])
            if (score['condition']!='xverse' or score['target_prompt']!=row['prompt'] or
                score['reference_sha256']!=row['reference_sha256'] or score['reference_id']!=row['reference_id']):
                raise ValueError('Score input differs')
            if score['generated_sha256']!=index['members']['tgt_image/'+row['id']+'.png']['sha256']:
                raise ValueError('Score refers to different generated image')
            if not all(math.isfinite(score[k]) for k in evaluator.METRICS):raise ValueError('Invalid metric')
        expected=set(canonical)-set(index['missing_case_ids'])
        if seen!=expected:raise ValueError('Metric coverage differs from saved outputs')
        if result['protocol']!=evaluator.PROTOCOL or result['generation_resolution']!=[768,768]:
            raise ValueError('Evaluation protocol differs')
        if evaluator.summarize(result['scores'])!=result['summary']:raise ValueError('Summary differs')
        summary=result['summary'][benchmark+'/xverse']
        summary_row=dict(method='xverse',benchmark=benchmark,cases=len(seen),references=summary['subjects'],
            canonical_cases=len(rows),missing_cases=len(index['missing_case_ids']),
            selection='available-generated' if index['missing_case_ids'] else 'all',
            **summary['subject_macro'])
        summaries.append(summary_row)
        shutil.copyfile(metric,reports/(benchmark+'_scores.json'))
        with (reports/(benchmark+'_per_image.csv')).open('w',encoding='utf-8',newline='') as f:
            columns=['case_id','reference_id','category','target_prompt','generated_sha256','CLIP-I','CLIP-T','DINO-I']
            writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(result['scores'])
        provenance=json.loads((root/'provenance.json').read_text())
        provenance.update(metrics_sha256=digest(metric),metric_cases=len(seen),metric_versions=result['versions'],
            metric_device=result['device_name'],metric_protocol=result['protocol'],
            generation_fingerprint=index['generation_fingerprint'])
        write(root/'provenance.json',provenance)
        verification=json.loads((root/'verification.json').read_text())
        verification.update(metric_cases=len(seen),score_image_sha256_verified=True,
            coverage='partial' if index['missing_case_ids'] else 'complete',missing_case_ids=index['missing_case_ids'])
        write(root/'verification.json',verification)
        receipt=PROJECT/'xverse/outputs/saved_complete_evaluation.json'
        if benchmark=='dreambench' and receipt.exists():shutil.copyfile(receipt,root/'evaluation_receipt.json')
        inventory.append(dict(method='xverse',benchmark=benchmark,bundle_id=bid,**verification,
            archives=[dict(**p,path=(Path('xverse/artifacts')/benchmark/bid/p['file']).as_posix()) for p in index['parts']]))
    with (reports/'metrics_summary.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    write(reports/'metrics_audit.json',dict(status='verified',evaluations=summaries,
        metric_definition='Selected conditioning reference; subject macro of raw similarities',
        generation_resolution=[768,768],louislee_remote_compute_launched=False,
        notes=['DreamBench originally had 733 complete cases; 17 cases were recovered on louislee-85468 with the original generation source.',
               'DreamBench++ includes all 180 style cases; non_style aggregate is in scores JSON.',
               'Prior methods generated at 1024x1024; XVerse uses official native 768x768.'],
        scoring_source_sha256=digest(PROJECT/'freecus/project_metrics/benchmark_metrics.py'),
        saved_evaluation_source_sha256=digest(PROJECT/'xverse/operations/evaluate_saved.py'),
        validation_source_sha256=digest(PROJECT/'comparison_candidates/evaluation.py')))
    historical=[]
    complete_marker=PROJECT/'xverse/outputs/dreambench_recovery/complete_bundle.json'
    if complete_marker.exists():
        _,old_bid,_,_=settings('louislee')
        old=PROJECT/'xverse/results/dreambench'/old_bid
        old_index=json.loads((old/'bundle_index.json').read_text())
        historical.append(dict(benchmark='dreambench',bundle_id=old_bid,status='superseded-partial',
            superseded_by=json.loads(complete_marker.read_text())['bundle_id'],
            generated_images=old_index['generated_images'],missing_case_ids=old_index['missing_case_ids'],
            archives=old_index['parts'],incomplete_artifacts=json.loads((old/'incomplete_artifacts.json').read_text())))
    write(PROJECT/'xverse/results/transfer_inventory.json',dict(bundles=inventory,historical_bundles=historical))
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
