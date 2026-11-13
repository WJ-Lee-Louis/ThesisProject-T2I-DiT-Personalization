"""Isolated, detached 512 workflow using the existing staging/worker/bundle machinery."""
from pathlib import Path

APP = 'freecus-512-paper-seed42-20261112'
CACHE = 'hf-model-cache'
RUNS = 'freecus-512-paper-seed42-20261112-runs'
SECRET = 'huggingface-secret'


def build(source):
    import modal
    from .modal_remote import build_app
    app, stage, gpu_lane = build_app(source, app_name=APP, cache_name=CACHE, runs_name=RUNS,
                                   secret_name=SECRET, worker_module='freecus.worker_512')
    runs = modal.Volume.from_name(RUNS)
    cache = modal.Volume.from_name(CACHE)
    cpu = (modal.Image.debian_slim(python_version='3.11').pip_install('huggingface-hub==0.29.3')
           .add_local_dir(source, '/root/freecus'))
    metric = (modal.Image.debian_slim(python_version='3.11').apt_install('git')
              .pip_install('torch==2.7.1','torchvision==0.22.1','numpy==1.26.4','Pillow==11.3.0',
                           'ftfy==6.3.1','regex','tqdm','packaging')
              .pip_install('git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6')
              .env(dict(TORCH_HOME='/cache/eval-torch',CLIP_DOWNLOAD_ROOT='/cache/eval-clip',PYTHONUNBUFFERED='1'))
              .add_local_dir(source,'/root/freecus'))

    @app.function(name='evaluate_512', serialized=True, image=metric, gpu='T4',
                  volumes={'/work':runs,'/cache':cache}, cpu=(2,2), memory=(8192,8192),
                  timeout=7200, max_containers=1, retries=0, scaledown_window=2)
    def evaluate_512(run_id):
        import csv
        import json
        import shutil
        import time
        from freecus.manifest import load
        from freecus.run import artifact_paths, complete, write_json
        from freecus.evaluate import prepare, project_evaluator, validate
        start = time.monotonic()
        runs.reload(); cache.reload()
        bundle = Path('/work/bundles')/run_id
        source = Path('/work/results')/run_id
        canonical = load(bundle/'manifest.jsonl',bundle/'input')
        rows = [r for r in canonical if r['benchmark']=='dreambench' or r['category']!='style']
        excluded = [r['case_id'] for r in canonical if r not in rows]
        destination = Path('/work/evaluation_inputs')/run_id
        destination.mkdir(parents=True,exist_ok=True)
        lanes = sorted(source.glob('lane-*'))
        infos = [json.loads((p/'run.json').read_text()) for p in lanes]
        if len(lanes)!=2 or any(i['fingerprint']!=infos[0]['fingerprint'] for i in infos):
            raise ValueError('Expected two matching lanes')
        (destination/'run.json').write_bytes((lanes[0]/'run.json').read_bytes())
        available, missing = [], []
        for row in rows:
            candidates = [(p,artifact_paths(p,row)) for p in lanes]
            found = [paths for p,paths in candidates if complete(paths)]
            if len(found)>1:
                raise ValueError('Duplicate case '+row['case_id'])
            if not found:
                failures = [p/'failures'/f"{row['case_id']}.json" for p in lanes]
                failure = next((json.loads(p.read_text()) for p in failures if p.exists()),None)
                missing.append(dict(case_id=row['case_id'], reference_id=row['reference_id'],
                                    category=row['category'], reason=failure or 'worker unfinished or invalid artifacts'))
                continue
            for kind, src in found[0].items():
                dst=artifact_paths(destination,row)[kind]
                dst.parent.mkdir(parents=True,exist_ok=True)
                if dst.exists():
                    if dst.read_bytes()!=src.read_bytes(): raise ValueError('Assembled output differs')
                else:
                    try: dst.hardlink_to(src)
                    except OSError: shutil.copy2(src,dst)
            available.append(row['case_id'])
        folder=Path('/work/metrics')/run_id
        coverage=dict(canonical_cases=len(canonical),excluded_case_ids=excluded,
                      expected=len(rows),generated=len(available),missing=missing,
                      status='complete' if not missing else 'incomplete',resolution=[512,512],
                      available_case_ids=available)
        write_json(folder/'coverage.json',coverage)
        evaluator=project_evaluator()
        cohorts=['dbpp-no-style'] if rows[0]['benchmark']=='dreambench_plus' else ['all']
        reports={}
        for cohort in cohorts:
            plan=prepare(bundle/'manifest.jsonl',bundle/'input',destination,evaluator,cohort,available)
            write_json(folder/f'{cohort}.plan.json',plan)
            scores=folder/f'{cohort}.scores.json'
            if not plan['records']:
                reports[cohort]=dict(status='no_valid_images',selection=plan['selection'],summary=None)
                continue
            if scores.exists():
                result=json.loads(scores.read_text())
            else:
                result=evaluator.evaluate(plan,scores,'cuda',32,validator=validate,
                          checkpoint=folder/f'{cohort}.checkpoint.json',commit=lambda _:runs.commit())
                result['evaluation_selection']=plan['selection']
                result['generation_resolution']=[512,512]
                result['complete']=plan['selection']['complete']
                result['generation_fingerprint']=infos[0]['fingerprint']
                write_json(scores,result)
            with (folder/f'{cohort}.scores.csv').open('w',newline='',encoding='utf-8-sig') as stream:
                fields=['case_id','reference_id','benchmark','category','condition','CLIP-I','CLIP-T','DINO-I']
                writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
                writer.writeheader();writer.writerows(result['scores'])
            reports[cohort]=dict(status='complete' if plan['selection']['complete'] else 'partial_not_full_benchmark',
                                selection=plan['selection'],summary=result['summary'])
        result=dict(coverage=coverage,cohorts=reports,elapsed_seconds=time.monotonic()-start)
        result['estimated_compute_usd']=(result['elapsed_seconds']+2)*(.000164+2*.0000131+8*.00000222)
        write_json(folder/'summary.json',result)
        cache.commit();runs.commit()
        return result

    @app.function(name='export_512', serialized=True,image=cpu,volumes={'/work':runs},
                  cpu=(1,1),memory=(2048,2048),timeout=7200,retries=0,scaledown_window=2)
    def export_512(run_id,benchmark):
        import json
        import zipfile
        from freecus.operations.artifact_bundle import pack,verify
        from freecus.run import write_json
        runs.reload()
        source=Path('/work/results')/run_id
        bundle=Path('/work/bundles')/run_id
        coverage=json.loads((Path('/work/metrics')/run_id/'coverage.json').read_text())
        archive=Path('/work/exports')/run_id
        extras={'models.lock.json':bundle/'models.lock.json','config.512.json':bundle/'config.json',
                'identity.json':bundle/'identity.json','source.zip':bundle/'source.zip',
                'coverage.json':Path('/work/metrics')/run_id/'coverage.json'}
        for lane in sorted(source.glob('lane-*')):
            for path in (lane/'failures').glob('*.json'):
                extras[f'failures/{lane.name}/{path.name}']=path
        index=pack(sorted(source.glob('lane-*')),bundle/'manifest.jsonl',bundle/'input',archive,
                   'freecus_512',benchmark,run_id,coverage['generated'],resolution=(512,512),provenance_files=extras)
        verify(archive,index)
        excluded=set(coverage['excluded_case_ids'])
        result=dict(verified=True,parts=index['parts'],generated=index['generated_images'],
                    excluded_case_ids=sorted(excluded),
                    missing_case_ids=sorted(set(index['missing_case_ids'])-excluded),
                    resolution=[512,512],max_part_bytes=1024**3)
        write_json(archive/'verification.json',result)
        runs.commit()
        return result

    @app.function(name='workflow_512',serialized=True,image=cpu,volumes={'/work':runs},
                  cpu=(.125,.125),memory=(512,512),timeout=86400,max_containers=1,retries=0,scaledown_window=2)
    def workflow_512(workflow_id,plans):
        import json
        import statistics
        import time
        import uuid
        from freecus.modal_cost import rate
        from freecus.run import write_json
        report=Path('/work/workflows')/f'{workflow_id}.json'
        runs.reload()
        if report.exists(): raise ValueError('Duplicate workflow')
        state=dict(workflow_id=workflow_id,status='starting',resolution=[512,512],plans=plans,
                   budget_policy='user-managed Modal workspace limit; no historical budget reused',
                   estimated_compute_usd=0,phases=[],started_at=time.time())
        def save():
            state['updated_at']=time.time()
            write_json(report,state);runs.commit()
        def phase(plan,kind,lanes):
            invocation=uuid.uuid4().hex
            record=dict(benchmark=plan['benchmark'],run_id=plan['run_id'],kind=kind,calls=[])
            state['phases'].append(record);state['status']=kind+'_'+plan['benchmark'];save()
            calls=[]
            for lane,ids in enumerate(lanes):
                call=gpu_lane.spawn(plan['run_id'],invocation,lane,ids,13800)
                calls.append(call);record['calls'].append(dict(call_id=call.object_id,lane=lane,cases=len(ids)))
                save()
            for i,call in enumerate(calls):
                try: result=call.get()
                except Exception as exc: result=dict(status='failed',error=repr(exc))
                record['calls'][i]['result']=result
                state['estimated_compute_usd']+=result.get('estimated_compute_usd',0)
                save()
            return record
        save()
        try:
            for plan in plans:
                run_id=plan['run_id']
                state['status']='staging_'+plan['benchmark'];save()
                call=stage.spawn(run_id)
                state.setdefault('stage_calls',[]).append(dict(call_id=call.object_id,run_id=run_id));save()
                staged=call.get();runs.reload()
                bundle=Path('/work/bundles')/run_id
                all_rows=[json.loads(s) for s in (bundle/'manifest.jsonl').read_text().splitlines()]
                selected=set(case for lane in plan['lanes'] for case in lane)
                rows=[r for r in all_rows if r['case_id'] in selected]
                pilot=[]
                for lane_ids in plan['lanes']:
                    lane_rows=[r for r in rows if r['case_id'] in set(lane_ids)]
                    categories=sorted({r['category'] for r in lane_rows})
                    refs=[]
                    for category in (categories[0],categories[-1]):
                        ref=next(r['reference_id'] for r in lane_rows if r['category']==category)
                        if ref not in refs: refs.append(ref)
                    indices={0,10,20} if plan['benchmark']=='dreambench' else {0,4,8}
                    pilot.append([r['case_id'] for r in lane_rows if r['reference_id'] in refs and r['prompt_index'] in indices])
                result=phase(plan,'pilot',pilot)
                events=[e for c in result['calls'] for e in c.get('result',{}).get('events',[])]
                generated=[e for e in events if e['phase']=='generated']
                if not generated or any(c['result']['status']!='complete' for c in result['calls']):
                    state['status']='pilot_failed';save();return state
                timings=[];overhead=0
                for c in result['calls']:
                    worker=c['result'];done=[e['seconds'] for e in worker['events'] if e['phase']=='generated']
                    timings.extend(done[1:] or done)
                    overhead+=max(0,worker['elapsed_seconds']-sum(done[1:]))
                caption=[e['seconds'] for e in events if e['phase']=='caption']
                per_image=statistics.mean(timings)
                caption_extra=max(0,len({r['reference_id'] for r in rows})-len(caption))*(max(caption) if caption else 0)
                estimate=(per_image*len(rows)+overhead+caption_extra)*rate()
                measurement=dict(images=len(generated),seconds_per_image=per_image,expected_images=len(rows),
                                 actual_resolutions=sorted({tuple(e['actual_resolution']) for e in generated}),
                                 peak_gpu_allocated_gib=max(e.get('peak_allocated_gib',0) for e in events),
                                 peak_gpu_reserved_gib=max(e.get('peak_reserved_gib',0) for e in events),
                                 host_peak_gib=max(e.get('host_peak_gib',0) for e in events),
                                 estimated_generation_usd=estimate,estimated_generation_usd_25pct_margin=estimate*1.25,
                                 estimate_excludes_build_stage_evaluation_export_storage=True,
                                 price_source='https://modal.com/pricing',price_checked='2026-11-12')
                state.setdefault('pilot_measurements',{})[plan['benchmark']]=measurement;save()
                phase(plan,'generation',plan['lanes'])
                state['status']='evaluating_'+plan['benchmark'];save()
                call=evaluate_512.spawn(run_id)
                state.setdefault('evaluation_calls',[]).append(dict(run_id=run_id,call_id=call.object_id));save()
                evaluation=call.get();state.setdefault('evaluations',{})[plan['benchmark']]=evaluation
                state['estimated_compute_usd']+=evaluation['estimated_compute_usd'];save()
                state['status']='exporting_'+plan['benchmark'];save()
                call=export_512.spawn(run_id,plan['benchmark'])
                state.setdefault('export_calls',[]).append(dict(run_id=run_id,call_id=call.object_id));save()
                state.setdefault('exports',{})[plan['benchmark']]=call.get();save()
            complete=all(x['coverage']['status']=='complete' for x in state['evaluations'].values())
            state['status']='complete' if complete else 'finished_with_missing_cases'
            save();return state
        except BaseException as exc:
            state['status']='failed';state['error']=repr(exc);save();raise

    @app.function(name='inspect_512',serialized=True,image=cpu,volumes={'/work':runs},
                  cpu=(.125,.125),memory=(512,512),timeout=300,retries=0,scaledown_window=2)
    def inspect_512(workflow_id):
        import json
        runs.reload()
        report=Path('/work/workflows')/f'{workflow_id}.json'
        if not report.exists(): return dict(status='queued',workflow_id=workflow_id)
        value=json.loads(report.read_text())
        value['progress']={}
        value['staging']={}
        for plan in value['plans']:
            stage_file=Path('/work/bundles')/plan['run_id']/'stage.json'
            if stage_file.exists(): value['staging'][plan['benchmark']]=json.loads(stage_file.read_text())
            folder=Path('/work/results')/plan['run_id']
            value['progress'][plan['benchmark']]=dict(expected=sum(map(len,plan['lanes'])),
                generated=sum(1 for p in folder.glob('lane-*/metadata/**/*.json')),
                failures=sum(1 for p in folder.glob('lane-*/failures/*.json')))
        for phase in value['phases']:
            for call in phase['calls']:
                if 'result' in call: call['result'].pop('events',None)
        return value
    return app,workflow_512
