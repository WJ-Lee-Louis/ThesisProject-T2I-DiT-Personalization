"""Audit immutable scores and render original or class-balanced paper tables.

Run from any directory: python tables_for_paper/build_method_tables.py
Requires matplotlib and Pillow; does not launch generation, metrics or remote jobs.
"""
import csv
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import sys
import tempfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
from freecus.evaluate import project_evaluator

METRICS=['CLIP-I','CLIP-T','DINO-I']
SOURCES={
    'freecus':dict(dreambench='freecus512-dreambench-s42-f86cadfd519ce417',dreambench_plus='freecus512-dreambench_plus-s42-a63b054d17b9ab87'),
    'dreamo':dict(dreambench='80f168f66963c39103d3a528',dreambench_plus='0d879ac6888dff2965d416b6'),
    'uno':dict(dreambench='uno512-dreambench-s42-50396742e580597d',dreambench_plus='uno512-dreambench_plus-s42-ca39a70a1f5ebec3'),
    'xverse':dict(dreambench='2e73e10891f643f354a846f4-complete-v1',dreambench_plus='f811b3fa80a9700e30dd8115'),
    'diptych':dict.fromkeys(['dreambench','dreambench_plus'],'local-gpu0-seed42-bf16-v1'),
    'mosaic':dict.fromkeys(['dreambench','dreambench_plus'],'local-gpu0-seed42-bf16-v1'),
    'uso':dict(dreambench='uso1024-dreambench-s42-59e9e8cf62e1b2c8',dreambench_plus='uso1024-dreambench_plus-s42-736623e31bbbb95d'),
    'freegrafor2':dict.fromkeys(['dreambench','dreambench_plus'],'seed42_512'),
    'xlabs_application':dict.fromkeys(['dreambench','dreambench_plus'],'598a7ef43356a8eb05e1314b'),
}
LABELS={'freecus':'FreeCus','dreamo':'DreamO','uno':'UNO','xverse':'XVerse',
        'uso':'USO','freegrafor2':'FreeGraftor','diptych':'Diptych','mosaic':'MOSAIC','xlabs_application':'XLabs-IP-Adapter'}
RESOLUTIONS=dict(freecus=512,dreamo=1024,uno=512,uso=1024,freegrafor2=512,xverse=768,diptych=768,mosaic=512,xlabs_application=1024)
COUNTS={'dreambench':dict(animal=(225,9),object=(525,21)),
        'dreambench_plus':dict(animal=(405,45),human=(180,20),object=(585,65),style=(180,20),non_style=(1170,130))}


NON_STYLE_METHODS={'freecus','freegrafor2'}
RECENT_METHODS={'freecus','uno','uso','freegrafor2'}

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def save_png(fig,target):
    # Replace a complete file, so an open image viewer cannot interrupt an in-place rewrite.
    with io.BytesIO() as output:
        fig.savefig(output,format='png',dpi=200,facecolor='white')
        if target.exists():
            from PIL import Image, ImageChops
            with Image.open(target) as previous, Image.open(io.BytesIO(output.getvalue())) as current:
                if previous.size==current.size and ImageChops.difference(previous.convert('RGB'),current.convert('RGB')).getbbox() is None:return
        with tempfile.NamedTemporaryFile(dir=target.parent,prefix='.table-',suffix='.tmp',delete=False) as temporary:
            temporary.write(output.getvalue())
        temporary_path=Path(temporary.name)
        try:temporary_path.replace(target)
        finally:
            if temporary_path.exists():temporary_path.unlink()


def same(a,b):
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same(v,b[k]) for k,v in a.items())
    if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    if isinstance(a,(int,float)) and not isinstance(a,bool):return isinstance(b,(int,float)) and math.isclose(a,b,rel_tol=0,abs_tol=1e-12)
    return a==b


def aggregate(scores):
    groups={}
    for s in scores:groups.setdefault(s['reference_id'],[]).append(s)
    return {m:statistics.fmean(statistics.fmean(s[m] for s in rows) for rows in groups.values()) for m in METRICS}


def source_scores(method,bench):
    source='xlabs_application' if method=='xlabs_projection' else method
    result=read(score_path(source,bench))
    condition=('source_target' if method=='xlabs_projection' else 'baseline') if source=='xlabs_application' else 'uno_512' if method=='uno' else 'freegraftor' if method=='freegrafor2' else method
    return [s for s in result['scores'] if s['condition']==condition and s['benchmark']==bench]


def score_path(method,bench):
    if method=='freegrafor2':return ROOT/'freegrafor2/results/runpod/seed42_512/metrics/scores.json'
    base=ROOT/method/('results_512' if method in {'freecus','uno'} else 'results')/bench/SOURCES[method][bench]
    if method=='freecus':return base/'metrics'/('all.scores.json' if bench=='dreambench' else 'dbpp-no-style.scores.json')
    return base/'metrics.json'


def audit_recent(method,bench,evaluator):
    path=score_path(method,bench);result=read(path);scores=source_scores(method,bench)
    if result['protocol']!=evaluator.PROTOCOL:raise ValueError('Metric protocol differs: '+str(path))
    if not same(evaluator.summarize(result['scores']),result['summary']):raise ValueError('Stored summary differs: '+str(path))
    shared_path=ROOT/'freecus/manifests'/(bench+'_project_seed42.jsonl')
    shared={r['case_id']:r for r in map(json.loads,shared_path.read_text().splitlines())}
    no_style=method in NON_STYLE_METHODS and bench=='dreambench_plus'
    expected={cid:r for cid,r in shared.items() if not(no_style and r['category']=='style')}
    extra={}
    if method=='freegrafor2':
        folder=path.parent.parent
        config=read(folder/'run.json')['provenance']['settings']
        index_path=ROOT/'freegrafor2/results/runpod/downloads/export_manifest.json'
        entries={(r['benchmark'],r['case_id']):r for r in read(index_path)['files']}
        if [config['width'],config['height'],config['seed']]!=[512,512,42]:raise ValueError('FreeGraftor settings differ')
    else:
        folder=path.parent.parent if method=='freecus' else path.parent
        index_path=folder/'bundle_index.json';index=read(index_path)
        if index['generation_resolution']!=[RESOLUTIONS[method]]*2:raise ValueError('Resolution differs')
        if digest(folder/'manifest.jsonl')!=digest(shared_path) or index['manifest_sha256']!=digest(shared_path):raise ValueError('Manifest differs')
        config=read(folder/('config.512.json' if method in {'freecus','uno'} else 'provenance.json'))
        if method=='uso':config=config['config']
        if [config['width'],config['height']]!=[RESOLUTIONS[method]]*2 or config.get('seeds',[config.get('seed')])!=[42]:raise ValueError('Generation settings differ')
        extra['generation_fingerprint']=index['generation_fingerprint']
    seen=set()
    for score in scores:
        cid=score['case_id']
        if cid not in expected or cid in seen:raise ValueError('Unknown or duplicate score: '+cid)
        seen.add(cid);row=expected[cid]
        if any(score[k]!=row[k] for k in ['reference_id','reference_sha256','category']) or score['target_prompt']!=row['prompt'] or row['seed']!=42 or row['evaluation_references']!=[row['reference']]:raise ValueError('Case identity differs: '+cid)
        if not all(math.isfinite(score[m]) for m in METRICS):raise ValueError('Nonfinite score')
        if method=='freegrafor2':
            entry=entries[bench,cid];metadata=folder/'execution_records'/bench/(cid+'.json');meta=read(metadata)
            if digest(metadata)!=entry['metadata_sha256'] or meta['png_sha256']!=score['generated_sha256'] or entry['png_sha256']!=score['generated_sha256']:raise ValueError('FreeGraftor artifact hash differs')
            if meta['settings']!=config or meta['row']['target_prompt']!=row['prompt'] or meta['reference_sha256']!=row['reference_sha256']:raise ValueError('FreeGraftor metadata differs')
        elif index['members']['tgt_image/'+row['id']+'.png']['sha256']!=score['generated_sha256']:raise ValueError('Artifact hash differs')
    if seen!=set(expected):raise ValueError('Incomplete cohort')
    groups={'overall':scores,**{g:[s for s in scores if (s['category']!='style' if g=='non_style' else s['category']==g)] for g in COUNTS[bench]}}
    records={}
    for group,selected in groups.items():
        refs={s['reference_id'] for s in selected}
        expected_count=(1170,130) if group=='overall' and no_style else ((750,30) if bench=='dreambench' else (1350,150)) if group=='overall' else COUNTS[bench][group]
        if selected and ((len(selected),len(refs))!=expected_count or {sum(s['reference_id']==r for s in selected) for r in refs}!={25 if bench=='dreambench' else 9}):raise ValueError('Coverage differs')
        if not selected and not(no_style and group=='style'):raise ValueError('Unexpected missing class')
        values=aggregate(selected) if selected else {m:None for m in METRICS}
        records[group]=dict(method=method,benchmark=bench,group=group,cases=len(selected),references=len(refs),resolution=RESOLUTIONS[method],selection='non-style only' if no_style else 'all',status='evaluated' if selected else 'excluded',**values)
    audit=dict(method=method,condition=scores[0]['condition'],benchmark=bench,source=path.relative_to(ROOT).as_posix(),sha256=digest(path),bundle_index_sha256=digest(index_path),shared_manifest_sha256=digest(shared_path),evaluated_cases=len(scores),references=len({s['reference_id'] for s in scores}),excluded_cases=sorted(set(shared)-seen),generation_resolution=[RESOLUTIONS[method]]*2,summary_recomputed=True,independent_subject_macro_verified=True,score_image_hashes_match_bundle_index=True,**extra)
    return records,audit,{k:result[k] for k in ['device','device_name','versions']}


def generation_record(folder,index,provenance,result,method):
    """Use retained metadata; image ZIPs were deliberately moved out of Git."""
    resolution=RESOLUTIONS[method]
    run_path=folder/'run.json'
    if method=='xlabs_application':
        run=read(run_path)
        if run['settings']!=result['generation_provenance']['settings'] or run['code']!=index['generation_fingerprint']:
            raise ValueError('XLabs generation provenance differs')
        config=run['settings']
        checked='retained run.json'
    elif run_path.exists():
        if digest(run_path)!=index['members']['run.json']['sha256']:raise ValueError('Run hash differs')
        run=read(run_path);config=run['config'];checked='retained run.json'
        if run['fingerprint']!=index['generation_fingerprint']:raise ValueError('Generation fingerprint differs')
    else:
        # Original run.json bytes remain in external image archives. Do not claim to have reread them.
        if provenance['generation_resolution']!=[resolution,resolution]:raise ValueError('Generation resolution differs')
        if provenance['generation_fingerprint']!=index['generation_fingerprint']:raise ValueError('Generation fingerprint differs')
        if index.get('generation_resolution',[resolution,resolution])!=[resolution,resolution]:raise ValueError('Index resolution differs')
        if 'generation_resolution' in result and result['generation_resolution']!=[resolution,resolution]:raise ValueError('Score resolution differs')
        return 'retained provenance and bundle index (external run.json not reread)'
    if [config['width'],config['height']]!=[resolution,resolution] or config.get('seeds',[config.get('seed')])!=[42]:
        raise ValueError('Generation resolution/seed differs')
    return checked


def audit_sources():
    evaluator=project_evaluator();audits=[];data={};exports=[];environments={}
    for method,bundles in SOURCES.items():
        for bench,bid in bundles.items():
            if method in RECENT_METHODS:
                records,audit,env=audit_recent(method,bench,evaluator)
                data[method,bench]=records;exports.extend(records.values());audits.append(audit);environments[method+'/'+bench]=env
                continue
            folder=ROOT/method/'results'/bench/bid
            path=folder/'metrics.json';result=read(path);index=read(folder/'bundle_index.json')
            manifest=folder/'manifest.jsonl';provenance=read(folder/'provenance.json')
            canonical_rows=list(map(json.loads,manifest.read_text(encoding='utf-8').splitlines()))
            canonical={r['case_id']:r for r in canonical_rows}
            if len(canonical)!=len(canonical_rows) or len(canonical)!=(750 if bench=='dreambench' else 1350):
                raise ValueError('Canonical manifest size or uniqueness differs')
            if digest(manifest)!=index['manifest_sha256'] or digest(manifest)!=provenance['manifest_sha256']:
                raise ValueError('Manifest hash differs')
            shared_manifest=ROOT/'freecus/manifests'/(bench+'_project_seed42.jsonl')
            shared={r['case_id']:r for r in map(json.loads,shared_manifest.read_text(encoding='utf-8').splitlines())}
            if set(shared)!=set(canonical):raise ValueError('Benchmark case IDs differ')
            if method!='xlabs_application' and digest(manifest)!=digest(shared_manifest):
                raise ValueError('Canonical manifest bytes differ')
            for cid,row in canonical.items():
                ref=shared[cid]
                if (row['reference_id']!=ref['reference_id'] or row['reference_sha256']!=ref['reference_sha256']
                    or row.get('prompt',row.get('target_prompt'))!=ref['prompt']):
                    raise ValueError('Canonical reference or prompt differs: '+cid)
                if ref['seed']!=42 or ref['reference_policy']!='selected_one' or ref['evaluation_references']!=[ref['reference']]:
                    raise ValueError('Shared seed/reference selection differs')
                if method!='xlabs_application' and (row['seed']!=42 or row['reference_policy']!='selected_one'
                    or row['evaluation_references']!=[row['reference']]):raise ValueError('Method reference selection differs')
            if result['protocol']!=evaluator.PROTOCOL or provenance.get('metric_protocol',result['protocol'])!=evaluator.PROTOCOL:
                raise ValueError('Metric protocol differs: '+str(path))
            current={k:result[k] for k in ['device','device_name','versions']}
            environments[method+'/'+bench]=current
            if provenance.get('metrics_sha256')!=digest(path):raise ValueError('Score provenance hash differs')
            generation_check=generation_record(folder,index,provenance,result,method)
            if not same(evaluator.summarize(result['scores']),result['summary']):raise ValueError('Stored summary differs from recomputation')
            report_name='dreambench_plus_no_style_scores.json' if method in NON_STYLE_METHODS and bench=='dreambench_plus' else bench+'_scores.json'
            report=ROOT/method/'results/reports'/report_name
            if not report.exists() or digest(report)!=digest(path):raise ValueError('Published score copy differs')
            members=index.get('members',{})
            lane_hashes={}
            for lane in index.get('lanes',[]):
                lane_path=folder/lane['index'];lane_index=read(lane_path)
                lane_hashes[lane['index']]=digest(lane_path)
                # Lane-specific run/conditioning records can share names; compare result images only.
                lane_members={k:v for k,v in lane_index['members'].items() if k.startswith('results/') and k.endswith('.png')}
                if any(members[k]!=lane_members[k] for k in set(members)&set(lane_members)):
                    raise ValueError('Conflicting lane member')
                members={**members,**lane_members}
            conditions=['baseline','source_target'] if method=='xlabs_application' else [method]
            if {s['condition'] for s in result['scores']}!=set(conditions):raise ValueError('Unexpected scored condition')
            no_style=method in NON_STYLE_METHODS and bench=='dreambench_plus'
            expected={cid:r for cid,r in shared.items() if not(no_style and r['category']=='style')}
            for condition in conditions:
                key='xlabs_projection' if condition=='source_target' else method
                scores=[s for s in result['scores'] if s['condition']==condition];seen=set()
                for score in scores:
                    cid=score['case_id']
                    if cid not in expected or cid in seen:raise ValueError('Unknown, excluded or duplicate score')
                    row=expected[cid];seen.add(cid)
                    if (score['benchmark']!=bench or score['reference_id']!=row['reference_id']
                        or score['reference_sha256']!=row['reference_sha256'] or score['target_prompt']!=row['prompt']
                        or score['category']!=row['category'] or score['category']!=evaluator.category(score)):
                        raise ValueError('Score identity differs: '+cid)
                    if not all(math.isfinite(score[m]) for m in METRICS):raise ValueError('Nonfinite score')
                    member=f'results/{condition}/{bench}/{cid}.png' if method=='xlabs_application' else 'tgt_image/'+row['id']+'.png'
                    if members[member]['sha256']!=score['generated_sha256']:raise ValueError('Score image hash differs from artifact index')
                if seen!=set(expected):raise ValueError('Incomplete evaluated cohort')
                groups={'overall':scores}
                for group in COUNTS[bench]:
                    groups[group]=[s for s in scores if (s['category']!='style' if group=='non_style' else s['category']==group)]
                data[key,bench]={}
                for group,selected in groups.items():
                    refs={s['reference_id'] for s in selected}
                    if selected:
                        expected_count=(1170,130) if group=='overall' and no_style else ((750,30) if bench=='dreambench' else (1350,150)) if group=='overall' else COUNTS[bench][group]
                        if (len(selected),len(refs))!=expected_count:raise ValueError('Class coverage differs')
                        if {sum(s['reference_id']==r for s in selected) for r in refs}!={25 if bench=='dreambench' else 9}:
                            raise ValueError('Uneven reference coverage')
                    values=aggregate(selected) if selected else {m:None for m in METRICS}
                    stored=result['summary'][bench+'/'+condition]
                    stored=stored['subject_macro'] if group=='overall' else stored['non_style']['subject_macro'] if group=='non_style' else stored.get('by_class',{}).get(group,{}).get('subject_macro')
                    if selected and not same(values,stored):raise ValueError('Independent macro check failed')
                    rec=dict(method=key,benchmark=bench,group=group,cases=len(selected),references=len(refs),
                        resolution=RESOLUTIONS[method],selection='non-style only' if no_style else 'all',
                        status='evaluated' if selected else 'excluded',**values)
                    data[key,bench][group]=rec;exports.append(rec)
                audits.append(dict(method=key,condition=condition,benchmark=bench,source=path.relative_to(ROOT).as_posix(),
                    sha256=digest(path),bundle_index_sha256=digest(folder/'bundle_index.json'),lane_index_sha256=lane_hashes,
                    shared_manifest_sha256=digest(shared_manifest),method_manifest_sha256=digest(manifest),
                    evaluated_cases=len(seen),references=len({s['reference_id'] for s in scores}),
                    excluded_cases=sorted(set(shared)-seen),generation_resolution=[RESOLUTIONS[method]]*2,
                    generation_fingerprint=index['generation_fingerprint'],generation_record_check=generation_check,
                    summary_recomputed=True,independent_subject_macro_verified=True,score_image_hashes_match_bundle_index=True))
    return data,exports,dict(date='2026-11-14',status='verified',environments=environments,protocol=evaluator.PROTOCOL,
        environment_note='Same metric protocol; stored hardware and CUDA builds differ. See environments for each method and benchmark. No cross-device rescoring performed.',
        evaluations=audits,total_scored_cases=sum(a['evaluated_cases'] for a in audits),
        scope='Recomputed stored per-image aggregates and checked canonical case/reference/prompt identity, coverage, provenance and score-to-index image hashes. Image archives were moved outside Git: archive bytes and missing run.json files were not reread. Generation resolution is checked against retained run metadata or provenance/index. Models were not rerun.')


def table_rows(data,historical,bench,group,applicability=False):
    if applicability:
        before=data['xlabs_application',bench][group];after=data['xlabs_projection',bench][group]
        return [dict(method='XLabs-IP-Adapter (scale 1.0)',cases=before['cases'],values=[before[m] for m in METRICS]),
                dict(method='+ Our Projection (source + target)',cases=after['cases'],values=[after[m] for m in METRICS]),
                dict(method='Change after projection (after − before)',cases=None,values=[after[m]-before[m] for m in METRICS])]
    count=(750 if bench=='dreambench' else 1350) if group=='overall' else COUNTS[bench][group][0]
    rows=[]
    for method,values in zip(historical['methods'],historical[bench][group]):
        rows.append(dict(method=method,cases=count,values=values,source='historical'))
    for method in SOURCES:
        r=data[method,bench][group]
        dagger=' †' if method in NON_STYLE_METHODS and bench=='dreambench_plus' else ''
        rows.append(dict(method=f"{LABELS[method]} ({r['resolution']} × {r['resolution']}){dagger}",
            cases=r['cases'],values=[r[m] for m in METRICS],source='recomputed'))
    return rows


def render(data,historical,bench,classes,applicability=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Serif','pdf.fonttype':42})
    groups=list(COUNTS[bench]) if classes else ['overall']
    notes=['Subject macro; one selected reference; seed 42; higher is better.',
        'Resolution shown per row. FreeCus / UNO / MOSAIC / FreeGraftor: 512; Diptych / XVerse: 768; others: 1024.',
        'Baseline / Ours: preserved historical PNG values (4 decimals); nine comparison methods: audited local scores.',
        '* Ours best uses provisional scale 0.8; this is not an optimized scale.']
    if bench=='dreambench_plus':notes += ['† FreeCus / FreeGraftor: 1,170 images / 130 references only; excludes all 20 style references (180 images).',
        'FreeCus / FreeGraftor overall is non-style only; compare the Non-Style block for a shared 1,170-image cohort.' if classes else 'FreeCus / FreeGraftor overall uses a different cohort from the 1,350-image rows; matched non-style results are in the classes table.']
    else:notes += ['USO and FreeGraftor: local per-image scores; external USO and 1024 FreeCus / UNO rows removed.']
    if applicability:
        notes=['Subject macro; one selected reference; original target prompt; higher is better.',
            'Both conditions: 1024 × 1024; seed 42; 24 steps; guidance 3.5; IP-Adapter scale 1.0; same cases and references.',
            'Our Projection: source + target; strength 1.0; norm restoration ON; single-stream gate OFF.',
            'Change = after − before (unrounded scores); positive means increase.',
            'CLIP-T increases on both benchmarks; identity metrics do not improve consistently.']
    if classes and bench=='dreambench_plus':notes += ['Non-Style pools animal + human + object; it is not another disjoint class.']
    row_h=74;group_h=92;top=360
    total=top+sum(group_h+row_h*len(table_rows(data,historical,bench,g,applicability)) for g in groups)+80+43*len(notes)+70
    fig=plt.figure(figsize=(15,total/200),dpi=200,facecolor='white');ax=fig.add_axes([0,0,1,1])
    ax.set_xlim(0,3000);ax.set_ylim(total,0);ax.axis('off')
    texts=[]
    def text(x,y,value,size=15,color='black',align='left'):
        texts.append(ax.text(x,y,value,fontsize=size,color=color,ha=align,va='center'))
    def rule(y,lw):ax.plot([130,2870],[y,y],color='black',lw=lw)
    text(132,82,'DreamBench' if bench=='dreambench' else 'DreamBench++',25)
    subtitle='XLabs-IP-Adapter — projection applicability' if applicability else 'Method comparison'
    text(132,161,subtitle+(' — classes' if classes else ''),15,'#4a4a4a')
    rule(240,1.7);text(160,295,'Method',15)
    columns=[1570,1940,2340,2740]
    for x,label in zip(columns,['N',*[m+' ↑' for m in METRICS]]):text(x,295,label,15,align='center')
    rule(350,.85);y=350
    for group in groups:
        if y!=350:rule(y,.7)
        label=group.replace('_','-').title()
        if group=='overall':
            label='Overall (750 images / method)' if bench=='dreambench' else 'Overall (1,350 images / method)' if applicability else 'Overall (1,350 images; FreeCus / FreeGraftor † 1,170)'
        else:label+=f' ({COUNTS[bench][group][0]:,} images)'
        text(160,y+49,label,14);y+=group_h
        for row in table_rows(data,historical,bench,group,applicability):
            text(160,y+row_h/2,row['method'],14)
            text(columns[0],y+row_h/2,'—' if row['cases'] is None else f"{row['cases']:,}",14,align='center')
            for x,value in zip(columns[1:],row['values']):
                formatted='—' if value is None else f'{value:+.4f}' if applicability and row['cases'] is None else f'{value:.4f}'
                text(x,y+row_h/2,formatted,15,align='center')
            y+=row_h
    rule(y+8,1.7)
    for i,note in enumerate(notes):text(160,y+70+43*i,note,10.2,'#4a4a4a')
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    bounds=fig.bbox
    for t in texts:
        box=t.get_window_extent(renderer)
        if box.x0<0 or box.x1>bounds.x1 or box.y0<0 or box.y1>bounds.y1:raise ValueError('Text clipped: '+t.get_text())
    target=HERE/(bench+'_method_comparison'+('_classes' if classes else '')+('_applicabiltiy' if applicability else '')+'.png')
    save_png(fig,target);plt.close(fig)
    return dict(path=target.name,sha256=digest(target),width=3000,height=int(total),notes=notes)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--balanced-only',action='store_true',help='Create the two additional class-balanced DreamBench++ PNGs')
    args=parser.parse_args()
    if args.balanced_only:
        build_balanced_tables();return
    untouched={p.name:digest(p) for p in HERE.glob('*.png') if 'method_comparison' not in p.name}
    data,rows,audit=audit_sources()
    historical=read(HERE/'historical_rows.json')
    audit['historical_rows_source']=dict(path='historical_rows.json',sha256=digest(HERE/'historical_rows.json'),
        historical_vs_new_measurements='Baseline/Ours are historical; all nine comparison methods and XLabs projection are recomputed from stored case scores.')
    audit['figures']=[render(data,historical,b,c) for b in ['dreambench','dreambench_plus'] for c in [False,True]]
    audit['figures'] += [render(data,historical,b,c,True) for b in ['dreambench','dreambench_plus'] for c in [False,True]]
    applicability=[]
    for bench in ['dreambench','dreambench_plus']:
        for group in ['overall',*COUNTS[bench]]:
            before=data['xlabs_application',bench][group];after=data['xlabs_projection',bench][group]
            applicability.append(dict(benchmark=bench,group=group,cases=before['cases'],references=before['references'],
                before={m:before[m] for m in METRICS},after={m:after[m] for m in METRICS},
                delta={m:after[m]-before[m] for m in METRICS}))
    write(HERE/'applicability_metrics.json',dict(aggregation='subject_macro',
        conditions=['baseline','source_target'],settings=read(ROOT/'xlabs_application/results/dreambench'/SOURCES['xlabs_application']['dreambench']/'run.json')['settings'],
        rows=applicability))
    for name,sha in untouched.items():
        if digest(HERE/name)!=sha:raise ValueError('Unrelated table changed')
    audit['unchanged_other_pngs']=untouched
    with (HERE/'method_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)
    write(HERE/'method_metrics.json',dict(aggregation='subject_macro',rows=rows))
    write(HERE/'aggregation_audit.json',audit)
    build_balanced_tables(data,dict(audit))
    print(json.dumps(dict(status='verified',evaluations=len(audit['evaluations']),scored_cases=audit['total_scored_cases'],
        figures=[p['path'] for p in audit['figures']],overall=[r for r in rows if r['group']=='overall']),indent=2))


def balanced_records(data,historical):
    bench='dreambench_plus';methods=[]
    for i,label in enumerate(historical['methods']):
        methods.append(dict(key='historical_'+str(i),label=label,resolution=None,historical=True,
            groups={g:{m:v for m,v in zip(METRICS,historical[bench][g][i])} for g in ['overall',*COUNTS[bench]]}))
    for key in SOURCES:
        methods.append(dict(key=key,label=f"{LABELS[key]} ({data[key,bench]['overall']['resolution']} × {data[key,bench]['overall']['resolution']})"+(' †' if key in NON_STYLE_METHODS else ''),
            resolution=data[key,bench]['overall']['resolution'],historical=False,
            groups={g:{m:r[m] for m in METRICS} for g,r in data[key,bench].items()}))
    records=[]
    for cohort,classes in [('four_class',['animal','human','object','style']),('non_style',['animal','human','object'])]:
        k=len(classes)
        baseline={m:statistics.fmean(methods[0]['groups'][c][m] for c in classes) for m in METRICS}
        for method in methods:
            eligible=not(cohort=='four_class' and method['key'] in NON_STYLE_METHODS)
            values={m:statistics.fmean(method['groups'][c][m] for c in classes) if eligible else None for m in METRICS}
            original_group='overall' if cohort=='four_class' else 'non_style'
            original=method['groups'][original_group]
            # Independent per-image weighting check for measured methods: each class sums to 1/K.
            verified=False
            if eligible and not method['historical']:
                scores=source_scores(method['key'],bench)
                selected=[s for s in scores if s['category'] in classes]
                for metric in METRICS:
                    direct=math.fsum(s[metric]/(k*COUNTS[bench][s['category']][0]) for s in selected)
                    if not math.isclose(direct,values[metric],abs_tol=1e-12,rel_tol=0):raise ValueError('Balanced per-image weighting check failed')
                if not math.isclose(math.fsum(1/(k*COUNTS[bench][s['category']][0]) for s in selected),1,abs_tol=1e-12):raise ValueError('Weights do not sum to one')
                verified=True
            records.append(dict(method=method['key'],label=method['label'],cohort=cohort,
                cases=1170 if method['key'] in NON_STYLE_METHODS or cohort=='non_style' else 1350,
                eligible=eligible,classes=classes,weights={c:1/k for c in classes},
                original_cohort='non_style' if method['key'] in NON_STYLE_METHODS else original_group,
                original=original,balanced=values,
                delta_vs_baseline={m:values[m]-baseline[m] if eligible else None for m in METRICS},
                class_means={c:method['groups'][c] for c in classes},
                weighted_contributions={c:{m:method['groups'][c][m]/k if eligible else None for m in METRICS} for c in classes},
                per_image_weighting_verified=verified,
                precision='historical rounded class means (4 decimals)' if method['historical'] else 'full precision measured class means',
                delta_baseline_precision='historical rounded class means (4 decimals)'))
    return methods,records


def render_balanced(methods,records,classes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Serif'})
    width=4200 if classes else 4600
    row_h=72;group_h=115;header=390
    groups=['animal','human','object','style','non_style'] if classes else ['four_class','non_style']
    notes=['Class-balanced: animal / human / object / style each receive 25%; Non-Style Balanced: first three each receive 1/3.',
        'Class means retain the existing subject-macro definition. Higher is better. N = evaluated images.',
        '† FreeCus / FreeGraftor exclude style (20 references / 180 images): no 4-class Balanced or delta; 3-class Non-Style Balanced is available.',
        'Baseline / Ours use historical class means rounded to 4 decimals; their balanced values and all baseline deltas are approximate.',
        '* Ours best is provisional scale 0.8. 512: FreeCus / UNO / MOSAIC / FreeGraftor; 768: Diptych / XVerse; 1024: other measured rows.']
    notes += ['Class score stays unchanged. Contribution = 0.25 × class score; sum four contributions for the Balanced Overall.',
        'FreeCus / FreeGraftor contributions to the 4-class metric are unavailable. The Non-Style block uses all three valid classes.'] if classes else [
        'Original = existing reference-weighted aggregate; Balanced = mean of class means; Δ = Balanced(method) − Balanced(baseline).',
        'FreeCus / FreeGraftor Original in the full-cohort block is their 1,170-image non-style scores, not a 1,350-image score.']
    height=header+len(groups)*(group_h+row_h*len(methods))+80+43*len(notes)+70
    fig=plt.figure(figsize=(width/200,height/200),dpi=200,facecolor='white');ax=fig.add_axes([0,0,1,1])
    ax.set_xlim(0,width);ax.set_ylim(height,0);ax.axis('off');texts=[]
    def text(x,y,value,size=14,color='black',align='left'):
        texts.append(ax.text(x,y,value,fontsize=size,color=color,ha=align,va='center'))
    def rule(y,lw=1):ax.plot([130,width-130],[y,y],color='black',lw=lw)
    text(140,80,'DreamBench++',25)
    text(140,158,'Method comparison — class-balanced contributions' if classes else 'Method comparison — class-balanced metrics',17,'#444444')
    rule(235,1.7);text(160,306,'Method',15);text(1530,306,'N',15,align='center')
    cols=[[1900+900*i+300*j for j in range(3)] for i in range(3)] if not classes else [[1900+850*i+330*j for j in range(2)] for i in range(3)]
    for metric,xs in zip(METRICS,cols):text(statistics.fmean(xs),274,metric+' ↑',16,align='center')
    # The classes table gives each panel its own subheading for raw vs weighted scores.
    if not classes:
        for xs in cols:
            for x,label in zip(xs,['Original','Balanced','Δ baseline']):text(x,335,label,12,align='center')
    rule(380,.9);y=380
    for group in groups:
        if y!=380:rule(y,.8)
        if group=='four_class':label='4-class comparison — equal class weights (25% each)'
        elif group=='non_style':label='Non-Style comparison — equal class weights (1/3 each; 1,170 images)'
        else:label=f"{group.title()} — 25% weight ({COUNTS['dreambench_plus'][group][0]:,} images)"
        text(160,y+44,label,15)
        if classes:
            for xs in cols:
                for x,sub in zip(xs,['Original','Balanced'] if group=='non_style' else ['Class score','Contribution']):text(x,y+84,sub,12,align='center')
        y+=group_h
        for method in methods:
            cohort='non_style' if group=='non_style' else 'four_class'
            r=next(r for r in records if r['method']==method['key'] and r['cohort']==cohort)
            n=r['cases'] if group in ['four_class','non_style'] else (0 if group=='style' and method['key'] in NON_STYLE_METHODS else COUNTS['dreambench_plus'][group][0])
            text(160,y+row_h/2,method['label'],14);text(1530,y+row_h/2,f'{n:,}',14,align='center')
            for metric,xs in zip(METRICS,cols):
                if not classes:values=[r['original'][metric],r['balanced'][metric],r['delta_vs_baseline'][metric]]
                elif group=='non_style':values=[r['original'][metric],r['balanced'][metric]]
                else:values=[method['groups'][group][metric],r['weighted_contributions'][group][metric]]
                for j,(x,value) in enumerate(zip(xs,values)):
                    formatted='—' if value is None else f'{value:+.4f}' if not classes and j==2 else f'{value:.4f}'
                    text(x,y+row_h/2,formatted,14,align='center')
            y+=row_h
    rule(y+8,1.7)
    for i,note in enumerate(notes):text(160,y+75+43*i,note,11,'#444444')
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for t in texts:
        box=t.get_window_extent(renderer)
        if box.x0<0 or box.x1>fig.bbox.x1 or box.y0<0 or box.y1>fig.bbox.y1:raise ValueError('Balanced text clipped: '+t.get_text())
    name='dreambench_plus_method_comparison'+('_classes' if classes else '')+'_balanced.png'
    target=HERE/name;save_png(fig,target);plt.close(fig)
    return dict(path=name,sha256=digest(target),width=width,height=height,notes=notes)


def build_balanced_tables(data=None,audit=None):
    # Existing tables stay available with their original aggregation.
    unchanged={p.name:digest(p) for p in HERE.glob('*.png') if not p.stem.endswith('_balanced')}
    if data is None:data,_,audit=audit_sources()
    historical=read(HERE/'historical_rows.json')
    methods,records=balanced_records(data,historical)
    audit.update(aggregation='equal-class macro after within-class subject macro',
        historical_source_sha256=digest(HERE/'historical_rows.json'),
        class_weights=dict(four_class={c:.25 for c in ['animal','human','object','style']},non_style={c:1/3 for c in ['animal','human','object']}),
        baseline='historical_0',non_style_policy='FreeCus/FreeGraftor: no four-class balanced metric; equal-weight three-class Non-Style only',
        figures=[render_balanced(methods,records,c) for c in [False,True]])
    for name,value in unchanged.items():
        if digest(HERE/name)!=value:raise ValueError('Existing PNG changed')
    audit['unchanged_existing_pngs']=unchanged
    write(HERE/'class_balanced_audit.json',audit)
    write(HERE/'class_balanced_metrics.json',dict(records=records))
    with (HERE/'class_balanced_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['method','cohort','cases','eligible','metric','original','balanced','delta_vs_baseline','precision']
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");writer.writeheader()
        for r in records:
            for metric in METRICS:writer.writerow(dict(method=r['method'],cohort=r['cohort'],cases=r['cases'],eligible=r['eligible'],
                metric=metric,original=r['original'][metric],balanced=r['balanced'][metric],delta_vs_baseline=r['delta_vs_baseline'][metric],precision=r['precision']))
    print(json.dumps(dict(status='verified',balanced_figures=[f['path'] for f in audit['figures']],methods=len(methods)),indent=2))


if __name__=='__main__':main()
