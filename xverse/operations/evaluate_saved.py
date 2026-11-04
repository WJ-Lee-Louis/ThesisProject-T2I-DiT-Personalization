"""Evaluate only verified saved DreamBench outputs on the credited woojin716 workspace."""
import argparse
import json
from pathlib import Path
from freecus.manifest import PROJECT
from freecus.operations.artifact_bundle import write,digest
from .collect import settings


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--status',action='store_true')
    parser.add_argument('--complete',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--submit-prepared',action='store_true')
    args=parser.parse_args()
    _,bid,_,_=settings('louislee')
    original_bid=bid
    recovered=args.complete
    result_bid=bid+'-complete-v1' if recovered else bid
    folder=PROJECT/'xverse/results/dreambench'/result_bid
    input_folder=PROJECT/'xverse/results/dreambench'/original_bid
    state=PROJECT/'xverse/outputs'/('saved_complete_evaluation.json' if recovered else 'saved_evaluation.json')
    preparation=state.with_name(state.stem+'_preparation.json')
    from freecus.modal_client import connect
    workspace='louislee-85468' if recovered else 'woojin716'
    modal=connect('method-comparison' if recovered else 'woojin716')
    runs=modal.Volume.from_name('xverse-runs')
    remote='saved_dreambench/'+result_bid
    app_name='xverse-complete-dreambench-metrics' if recovered else 'xverse-saved-dreambench-metrics'
    if args.status:
        receipt=json.loads(state.read_text())
        try: result=modal.FunctionCall.from_id(receipt['call_id']).get(timeout=0)
        except TimeoutError:
            print('running',receipt['call_id']);return
        data=b''.join(runs.read_file(result['scores']))
        target=folder/'metrics.json'
        if target.exists() and target.read_bytes()!=data:raise ValueError('Existing metrics differ')
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        print(json.dumps(result,indent=2));return
    if state.exists():raise ValueError('Already submitted; use --status')
    info=json.loads((input_folder/'bundle_index.json').read_text())
    def save_receipt(call):
        write(state,dict(call_id=call.object_id,workspace=workspace,gpu='T4',timeout_seconds=1800,
            bundle_id=result_bid,cases=750 if recovered else info['generated_images'],app=app_name))
        print(state.read_text())
    if args.submit_prepared:
        if not preparation.exists():raise ValueError('Prepare first')
        if recovered:
            status=json.loads((PROJECT/'xverse/outputs/dreambench_recovery/status.json').read_text())
            if status['status']!='complete':raise ValueError('Recovery is not complete')
        save_receipt(modal.Function.from_name(app_name,'evaluate').spawn(remote));return
    archives=PROJECT/'xverse/artifacts/dreambench'/bid
    with runs.batch_upload() as upload:
        upload.put_file(input_folder/'bundle_index.json','/'+remote+'/bundle_index.json')
        for part in info['parts']:
            if digest(archives/part['file'])!=part['sha256']:raise ValueError('Archive changed')
            upload.put_file(archives/part['file'],'/'+remote+'/'+part['file'])
    image=(modal.Image.debian_slim(python_version='3.11').apt_install('git')
        .pip_install('torch==2.7.1','torchvision==0.22.1','numpy==1.26.4','Pillow==11.3.0',
                     'ftfy==6.3.1','regex','tqdm','packaging')
        .pip_install('git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6')
        .env(dict(TORCH_HOME='/cache/eval-torch',CLIP_DOWNLOAD_ROOT='/cache/eval-clip',PYTHONUNBUFFERED='1')))
    for name in ['freecus/__init__.py','freecus/manifest.py','freecus/run.py','freecus/evaluate.py',
                 'freecus/operations/artifact_bundle.py','freecus/project_metrics/benchmark_metrics.py',
                 'freecus/project_metrics/SOURCE.json','comparison_candidates/__init__.py',
                 'comparison_candidates/evaluation.py']:
        image=image.add_local_file(PROJECT/name,'/root/'+name)
    cache=modal.Volume.from_name('freecus-method-comparison-models' if recovered else 'uno-models')
    app=modal.App(app_name)

    @app.function(name='evaluate',serialized=True,image=image,gpu='T4',cpu=(2,2),memory=(8192,8192),
        volumes={'/work':runs,'/cache':cache},timeout=1800,retries=0,max_containers=1,scaledown_window=2,
        single_use_containers=True)
    def evaluate(remote):
        from freecus.operations.artifact_bundle import restore,write
        from freecus.evaluate import prepare,project_evaluator
        from comparison_candidates.evaluation import validate
        runs.reload();cache.reload()
        folder=Path('/work')/remote
        index=json.loads((folder/'bundle_index.json').read_text())
        restored=folder/'restored'
        if not (restored/'restored_bundle.json').exists():restore(folder,index,restored)
        if recovered:
            import shutil,os
            from freecus.run import artifact_paths,complete
            from freecus.manifest import load
            recovery=json.loads((Path('/work/dreambench_recovery')/original_bid/'state.json').read_text())
            if recovery['status']!='complete':raise ValueError('Recovery incomplete')
            lane=Path('/work/results')/original_bid/'A100-80GB/lane-0'
            original_info=json.loads((restored/'run.json').read_text())
            recovery_info=json.loads((lane/'run.json').read_text())
            for key in ['fingerprint','config','manifest_sha256','code','versions','cuda']:
                if original_info[key]!=recovery_info[key]:raise ValueError('Recovered generation differs: '+key)
            for row in load(restored/'manifest.jsonl',restored/'input'):
                if row['case_id'] not in set(index['missing_case_ids']):continue
                source=artifact_paths(lane,row)
                if not complete(source):raise ValueError('Incomplete recovered artifact')
                for kind,target in artifact_paths(restored,row).items():
                    target.parent.mkdir(parents=True,exist_ok=True)
                    if not target.exists():
                        try:os.link(source[kind],target)
                        except OSError:shutil.copy2(source[kind],target)
            index['missing_case_ids']=[]
        evaluator=project_evaluator()
        plan=prepare(restored/'manifest.jsonl',restored/'input',restored,evaluator,'all')
        available={json.loads(p.read_text())['case_id'] for p in (restored/'metadata').rglob('*.json')}
        if recovered and len(available)!=750:raise ValueError('Expected all 750 cases')
        plan['records']=[r for r in plan['records'] if r['row']['case_id'] in available]
        for rec in plan['records']:rec['condition']='xverse'
        plan['partition']=dict(case_ids=sorted(available),global_cases=index['manifest_cases'])
        write(folder/'plan.json',plan)
        result=evaluator.evaluate(plan,folder/'scores.json','cuda',32,validator=validate,
            checkpoint=folder/'checkpoint.json',commit=lambda value:runs.commit())
        result.update(generation_resolution=[768,768],bundle_id=result_bid,
            generation_fingerprint=index['generation_fingerprint'],partition=plan['partition'],
            evaluation_selection=dict(name='all' if recovered else 'available-generated',included_cases=len(available),
                excluded_case_ids=index['missing_case_ids']),
            coverage=dict(status='partial' if index['missing_case_ids'] else 'complete',
                generated_cases=len(available),canonical_cases=index['manifest_cases'],
                missing_case_ids=index['missing_case_ids']),metric_workspace=workspace)
        write(folder/'scores.json',result);cache.commit();runs.commit()
        return dict(scores=(folder/'scores.json').relative_to('/work').as_posix(),
            cases=len(result['scores']),summary=result['summary'],coverage=result['coverage'])

    with modal.enable_output():app.deploy()
    write(preparation,dict(workspace=workspace,app=app_name,remote=remote,bundle_id=result_bid))
    if args.prepare_only:
        print('Prepared; submit after recovery completes:',preparation);return
    call=evaluate.spawn(remote)
    save_receipt(call)


if __name__=='__main__':main()
