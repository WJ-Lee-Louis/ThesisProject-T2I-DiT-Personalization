"""Deployed workflows survive local shutdown; four independent GPU workers."""
import json
from pathlib import Path

CACHE = 'freecus-method-comparison-models'
SECRET = 'dreamo-huggingface'


def build(source, method, gpu_kind='A100-80GB', *, app_name=None, cache_name=CACHE,
          secret_name=SECRET, lane_count=4, host_memory_mib=65536):
    import modal
    cache = modal.Volume.from_name(cache_name)
    runs = modal.Volume.from_name(method+'-runs',create_if_missing=True,version=2)
    secret = modal.Secret.from_name(secret_name,required_keys=['HF_TOKEN'])
    env = dict(HF_HOME='/cache/huggingface',HF_HUB_ENABLE_HF_TRANSFER='1',PYTHONUNBUFFERED='1',
               PYTHONPATH='/root',TOKENIZERS_PARALLELISM='false')
    cpu = (modal.Image.debian_slim(python_version='3.11')
        .pip_install('huggingface-hub==0.29.3','hf_transfer==0.1.9','Pillow==11.1.0')
        .env(env).add_local_dir(source,'/root'))
    gpu = (modal.Image.debian_slim(python_version='3.11').apt_install('libgl1','libglib2.0-0')
        .add_local_file(Path(source)/method/'requirements-gpu.txt','/tmp/requirements-gpu.txt',copy=True)
        .run_commands('pip install --no-cache-dir -r /tmp/requirements-gpu.txt').env(env)
        .add_local_dir(source,'/root'))
    metric = (modal.Image.debian_slim(python_version='3.11').apt_install('git')
        .pip_install('torch==2.7.1','torchvision==0.22.1','numpy==1.26.4','Pillow==11.3.0',
                     'ftfy==6.3.1','regex','tqdm','packaging')
        .pip_install('git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6')
        .env(dict(**env,TORCH_HOME='/cache/eval-torch',CLIP_DOWNLOAD_ROOT='/cache/eval-clip'))
        .add_local_dir(source,'/root'))
    app = modal.App(app_name or method+'-method-comparison')

    @app.function(name='stage',serialized=True,image=cpu,secrets=[secret],
        volumes={'/cache':cache,'/work':runs},cpu=(2,2),memory=(4096,8192),timeout=5400,
        max_containers=1,retries=0,scaledown_window=2)
    def stage(plans):
        import zipfile
        from huggingface_hub import snapshot_download
        from freecus.manifest import load,sha256
        from freecus.run import write_json
        runs.reload();cache.reload()
        lock = json.loads((Path('/root')/method/'models.lock.json').read_text())
        paths={}
        for key,item in lock.items():
            allow = ['model_index.json','scheduler/*','tokenizer/*','tokenizer_2/*','text_encoder/*',
                     'text_encoder_2/*','transformer/*','vae/*'] if key=='flux' else None
            paths[key]=snapshot_download(item['repo'],revision=item['revision'],allow_patterns=allow,
                ignore_patterns=['*.bin','*.onnx','*.h5','*.msgpack','*.tflite','*.ot'])
            cache.commit()
        for plan in plans:
            folder=Path('/work/bundles')/plan['bundle_id']
            folder.mkdir(parents=True,exist_ok=True)
            with zipfile.ZipFile(Path('/work/uploads')/(plan['bundle_id']+'.zip')) as z:
                for name in z.namelist():
                    if not (folder/name).resolve().is_relative_to(folder.resolve()):
                        raise ValueError('Unsafe archive member')
                z.extractall(folder)
            files=json.loads((folder/'bundle.json').read_text())['files']
            for name,digest in files.items():
                if sha256(folder/name)!=digest:raise ValueError('Input archive mismatch')
            rows=load(folder/'manifest.jsonl',folder/'input')
            if len(rows)!=plan['cases']:raise ValueError('Manifest size mismatch')
            write_json(folder/'resolved_models.json',paths)
            write_json(folder/'stage.json',dict(status='ready'))
        runs.commit()
        return dict(status='ready',models=lock)

    @app.function(name='lane',serialized=True,image=gpu,secrets=[secret],
        volumes={'/cache':cache,'/work':runs},gpu=gpu_kind,cpu=(2,2),memory=(host_memory_mib,65536),
        timeout=21600,max_containers=lane_count,retries=0,scaledown_window=2,single_use_containers=True)
    def lane(bundle_id,lane_id,cases,seconds_cap,invocation):
        import subprocess,sys,time
        from freecus.run import write_json
        runs.reload();cache.reload()
        bundle=Path('/work/bundles')/bundle_id
        output=Path('/work/results')/bundle_id/gpu_kind/f'lane-{lane_id}'
        journal=Path('/work/invocations')/invocation/f'lane-{lane_id}'
        journal.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-m','comparison_candidates.worker','--method',method,
                 '--bundle',str(bundle),'--output',str(output),'--cases',*cases]
        start=time.monotonic()
        with (journal/'worker.log').open('wb') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        capped=False
        try:
            while child.poll() is None:
                elapsed=time.monotonic()-start
                write_json(journal/'state.json',dict(status='running',elapsed_seconds=elapsed))
                runs.commit()
                if elapsed>=seconds_cap:
                    child.kill();child.wait();capped=True;break
                time.sleep(15)
        finally:
            if child.poll() is None:child.kill();child.wait()
        elapsed=time.monotonic()-start
        records=[json.loads(p.read_text()) for p in (output/'metadata').rglob('*.json')]
        result=dict(status='budget_paused' if capped else 'complete' if child.returncode==0 else 'failed',
            exit_code=child.returncode,elapsed_seconds=elapsed,generated_total=len(records),
            timings=[r['timing'] for r in records],requested_cases=len(cases),
            initialization=json.loads((output/'initialization.json').read_text()) if (output/'initialization.json').exists() else None,
            error_tail=(journal/'worker.log').read_text(errors='replace')[-7000:] if child.returncode else None)
        write_json(journal/'state.json',result);runs.commit()
        return result

    @app.function(name='evaluate',serialized=True,image=metric,
        volumes={'/cache':cache,'/work':runs},gpu='T4',cpu=(2,2),memory=(8192,8192),
        timeout=5400,max_containers=1,retries=0,scaledown_window=2,single_use_containers=True)
    def evaluate(plan):
        import os,shutil
        from freecus.run import write_json,artifact_paths,complete
        from freecus.evaluate import prepare,project_evaluator
        from comparison_candidates.evaluation import validate
        from freecus.manifest import load,sha256
        runs.reload();cache.reload()
        bundle=Path('/work/bundles')/plan['bundle_id']
        lanes=sorted((Path('/work/results')/plan['bundle_id']/gpu_kind).glob('lane-*'))
        if len(lanes)!=lane_count:raise ValueError('Unexpected lane count')
        infos=[json.loads((p/'run.json').read_text()) for p in lanes]
        if any(i!=infos[0] for i in infos[1:]):raise ValueError('Lane identities differ')
        output=Path('/work/metrics')/plan['bundle_id']
        output.mkdir(parents=True,exist_ok=True)
        assembled=output/'assembled'
        assembled.mkdir(exist_ok=True)
        write_json(assembled/'run.json',infos[0])
        assigned=set(plan.get('assigned_case_ids',[]))
        for row in load(bundle/'manifest.jsonl',bundle/'input'):
            if assigned and row['case_id'] not in assigned:continue
            found=[artifact_paths(p,row) for p in lanes if artifact_paths(p,row)['metadata'].exists()]
            if len(found)!=1 or not complete(found[0]):raise ValueError('Missing/duplicate/corrupt output: '+row['case_id'])
            for kind,target in artifact_paths(assembled,row).items():
                target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():
                    try:os.link(found[0][kind],target)
                    except OSError:shutil.copy2(found[0][kind],target)
                if sha256(target)!=sha256(found[0][kind]):raise ValueError('Assembly mismatch')
        evaluator=project_evaluator()
        scoring=prepare(bundle/'manifest.jsonl',bundle/'input',assembled,evaluator,'all')
        if assigned:
            scoring['records']=[r for r in scoring['records'] if r['row']['case_id'] in assigned]
            scoring['partition']=dict(case_ids=sorted(assigned),global_cases=1350)
        for rec in scoring['records']:rec['condition']=method
        write_json(output/'plan.json',scoring)
        result=evaluator.evaluate(scoring,output/'scores.json','cuda',32,validator=validate,
                                  checkpoint=output/'checkpoint.json')
        result['evaluation_selection']=scoring['selection']
        result['generation_resolution']=[768,768]
        if assigned:result['partition']=scoring['partition']
        write_json(output/'scores.json',result);cache.commit();runs.commit()
        return dict(status='complete',cases=len(result['scores']),summary=result['summary'],
                    scores=str((output/'scores.json').relative_to('/work')))

    @app.function(name='workflow',serialized=True,image=cpu,volumes={'/work':runs},
        cpu=(0.25,0.25),memory=(512,512),timeout=86400,max_containers=1,retries=0,scaledown_window=2)
    def workflow(plans,workflow_id,budget_usd):
        import time,uuid
        from freecus.run import write_json,artifact_paths,complete
        from freecus.manifest import load
        if not 0<budget_usd<=40:raise ValueError('Workspace compute cap is $40')
        # Reserve $3 for staging, supervisor and metric evaluation. GPU+RAM+CPU
        # lane cost accounted from allocated runtime, including publication margin.
        rate={'A100-80GB':0.000694,'H100':0.001097}[gpu_kind]+2*0.0000131+64*0.00000222
        state=dict(status='staging',method=method,gpu=gpu_kind,workflow_id=workflow_id,budget_usd=budget_usd,
                   generation_compute_usd=0.,benchmarks=[],started_at=time.time())
        path=Path('/work/workflows')/(workflow_id+'.json')
        runs.reload()
        if path.exists():
            state=json.loads(path.read_text())
            if state['status'] in {'complete','budget_review_required','pilot_failed','generation_incomplete','failed'}:
                return state
        def save():
            state['updated_at']=time.time();write_json(path,state);runs.commit()
        def execute(plan,selections,cap,label):
            phases=state['benchmarks'][-1]['phases']
            phase=next((p for p in phases if p['label']==label),None)
            if phase is None:
                phase=dict(label=label,invocation=uuid.uuid4().hex,calls=[])
                phases.append(phase);save()
            invocation=phase['invocation']
            handles=[]
            for i,cases in enumerate(selections):
                prior=next((c for c in phase['calls'] if c['lane']==i),None)
                if prior is not None:
                    handles.append((prior,modal.FunctionCall.from_id(prior['id'])))
                elif cases:
                    call=lane.spawn(plan['bundle_id'],i,cases,cap,invocation)
                    prior=dict(id=call.object_id,lane=i)
                    phase['calls'].append(prior);handles.append((prior,call));save()
            for entry,handle in handles:
                if 'result' in entry:continue
                entry['result']=handle.get()
                state['generation_compute_usd']+=(entry['result']['elapsed_seconds']+30)*rate
                save()
            return phase
        try:
            if state.get('stage_call'):
                call=modal.FunctionCall.from_id(state['stage_call'])
            else:
                call=stage.spawn(plans);state['stage_call']=call.object_id;save()
            call.get()
            for plan in plans:
                item=next((b for b in state['benchmarks'] if b['bundle_id']==plan['bundle_id']),None)
                if item is None:
                    item=dict(benchmark=plan['benchmark'],bundle_id=plan['bundle_id'],status='pilot',phases=[])
                    state['benchmarks'].append(item)
                if item['status']=='complete':continue
                state['status']='pilot';save()
                runs.reload()
                folder=Path('/work/bundles')/plan['bundle_id']
                rows=load(folder/'manifest.jsonl',folder/'input')
                if budget_usd-3-state['generation_compute_usd']<2:
                    item['status']='budget_review_required';state['status']='budget_review_required';save();return state
                # Pilot covers two ordinary references and DB++ human/style, while
                # keeping lane ownership fixed when the same cases resume later.
                pilot=[[] for _ in range(4)]
                categories={}
                for row in rows:
                    if row['category'] not in categories:categories[row['category']]=row['case_id']
                chosen=set(categories.values())
                for i,cases in enumerate(plan['lanes']):pilot[i]=[c for c in cases if c in chosen]
                phase=execute(plan,pilot,min(900,(budget_usd-3-state['generation_compute_usd'])/max(1,sum(bool(p) for p in pilot))/rate),'pilot')
                results=[c['result'] for c in phase['calls']]
                if not all(r['status']=='complete' for r in results):
                    item['status']='pilot_failed';state['status']='pilot_failed';save();return state
                timings=[t['seconds'] for r in results for t in r['timings']]
                speed=sum(timings)/len(timings)
                init=max(r['initialization']['seconds'] for r in results if r['initialization'])
                item['pilot_seconds_per_image']=speed
                item['pilot_peak_vram_gib']=max(t['peak_reserved_gib'] for r in results for t in r['timings'])
                remaining=[]
                source=Path('/work/results')/plan['bundle_id']/gpu_kind
                by_id={r['case_id']:r for r in rows}
                for i,cases in enumerate(plan['lanes']):
                    remaining.append([c for c in cases if not complete(artifact_paths(source/f'lane-{i}',by_id[c]))])
                needed=(sum(map(len,remaining))*speed+4*(init+60))*rate*1.20
                item['projected_remaining_usd']=needed
                left=budget_usd-3-state['generation_compute_usd']
                if needed>left:
                    item['status']='budget_review_required';state['status']='budget_review_required';save();return state
                state['status']='generating';item['status']='generating';save()
                cap=min(21000,left/4/rate-30)
                phase=execute(plan,remaining,cap,'full')
                if not all(c['result']['status']=='complete' for c in phase['calls']):
                    item['status']='generation_incomplete';state['status']='generation_incomplete';save();return state
                item['status']='evaluating';state['status']='evaluating';save()
                if item.get('evaluation_call'):
                    score=modal.FunctionCall.from_id(item['evaluation_call'])
                else:
                    score=evaluate.spawn(plan);item['evaluation_call']=score.object_id;save()
                item['evaluation']=score.get();item['status']='complete';save()
            state['status']='complete';save();return state
        except BaseException as exc:
            state['status']='failed';state['error']=repr(exc);save();raise
    return app
