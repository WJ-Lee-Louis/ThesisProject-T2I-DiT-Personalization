"""Audit immutable benchmark scores and render the four paper comparison tables.

Run from any directory: python tables_for_paper/build_method_tables.py
Requires matplotlib and Pillow; does not launch generation, metrics or remote jobs.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
from freecus.evaluate import project_evaluator

METRICS=['CLIP-I','CLIP-T','DINO-I']
SOURCES={
    'freecus':dict(dreambench='477fc2c07d3451850b37b400',dreambench_plus='0fa5460f44b4b8672480a187'),
    'dreamo':dict(dreambench='80f168f66963c39103d3a528',dreambench_plus='0d879ac6888dff2965d416b6'),
    'uno':dict(dreambench='98bf99baf41eb1b758fceb23',dreambench_plus='2ff93b396bb6c6eda2ee2215'),
    'xverse':dict(dreambench='2e73e10891f643f354a846f4-complete-v1',dreambench_plus='f811b3fa80a9700e30dd8115'),
}
LABELS={'freecus':'FreeCus','dreamo':'DreamO','uno':'UNO','xverse':'XVerse'}
COUNTS={'dreambench':dict(animal=(225,9),object=(525,21)),
        'dreambench_plus':dict(animal=(405,45),human=(180,20),object=(585,65),style=(180,20),non_style=(1170,130))}


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def same(a,b):
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same(v,b[k]) for k,v in a.items())
    if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    if isinstance(a,(int,float)) and not isinstance(a,bool):return isinstance(b,(int,float)) and math.isclose(a,b,rel_tol=0,abs_tol=1e-12)
    return a==b


def aggregate(scores):
    groups={}
    for s in scores:groups.setdefault(s['reference_id'],[]).append(s)
    return {m:statistics.fmean(statistics.fmean(s[m] for s in rows) for rows in groups.values()) for m in METRICS}


def audit_sources():
    evaluator=project_evaluator();env=None;audits=[];data={};exports=[]
    for method,bundles in SOURCES.items():
        for bench,bid in bundles.items():
            folder=ROOT/method/'results'/bench/bid
            path=folder/'metrics.json';result=read(path);index=read(folder/'bundle_index.json')
            manifest=folder/'manifest.jsonl'
            canonical={r['case_id']:r for r in map(json.loads,manifest.read_text(encoding='utf-8').splitlines())}
            no_style=method=='freecus' and bench=='dreambench_plus'
            expected={cid:r for cid,r in canonical.items() if not(no_style and r['category']=='style')}
            if len(canonical)!=(750 if bench=='dreambench' else 1350):raise ValueError('Canonical manifest size differs')
            if digest(manifest)!=index['manifest_sha256']:raise ValueError('Manifest hash differs')
            shared_manifest=ROOT/'freecus/manifests'/(bench+'_project_seed42.jsonl')
            if digest(manifest)!=digest(shared_manifest):raise ValueError('Methods do not use the same canonical benchmark manifest')
            if result['protocol']!=evaluator.PROTOCOL:raise ValueError('Metric protocol differs: '+str(path))
            current={k:result[k] for k in ['device','device_name','versions']}
            if env is not None and current!=env:raise ValueError('Metric environments differ')
            env=current
            provenance=read(folder/'provenance.json')
            if provenance.get('metrics_sha256',digest(path))!=digest(path):raise ValueError('Score provenance hash differs')
            run=None
            for part in index['parts']:
                archive=ROOT/method/'artifacts'/bench/bid/part['file']
                if archive.stat().st_size!=part['size_bytes']:raise ValueError('Archive size differs')
                with zipfile.ZipFile(archive) as z:
                    if 'run.json' in z.namelist():
                        payload=z.read('run.json')
                        if hashlib.sha256(payload).hexdigest()!=index['members']['run.json']['sha256']:raise ValueError('Run hash differs')
                        run=json.loads(payload);break
            if run is None:raise ValueError('No generation run record')
            resolution=768 if method=='xverse' else 1024
            if [run['config']['width'],run['config']['height']]!=[resolution,resolution] or run['config']['seeds']!=[42]:
                raise ValueError('Generation resolution/seed differs')
            if run['fingerprint']!=index['generation_fingerprint']:raise ValueError('Generation fingerprint differs')
            seen=set()
            for s in result['scores']:
                cid=s['case_id']
                if cid not in expected or cid in seen:raise ValueError('Unknown, excluded or duplicate score')
                row=expected[cid];seen.add(cid)
                if (s['condition']!=method or s['benchmark']!=bench or s['reference_id']!=row['reference_id']
                    or s['reference_sha256']!=row['reference_sha256'] or s['target_prompt']!=row['prompt']
                    or s['category']!=evaluator.category(s) or row['seed']!=42
                    or row['reference_policy']!='selected_one' or row['evaluation_references']!=[row['reference']]):
                    raise ValueError('Score identity differs: '+cid)
                if not all(math.isfinite(s[m]) for m in METRICS):raise ValueError('Nonfinite score')
                if index['members']['tgt_image/'+row['id']+'.png']['sha256']!=s['generated_sha256']:
                    raise ValueError('Score image hash differs from artifact index')
            if seen!=set(expected):raise ValueError('Incomplete evaluated cohort')
            if no_style and (len(seen)!=1170 or len({r['reference_id'] for r in canonical.values() if r['category']=='style'})!=20):
                raise ValueError('Expected 1170 non-style cases; 20 style references excluded')
            if not same(evaluator.summarize(result['scores']),result['summary']):raise ValueError('Stored summary differs from recomputation')
            groups={'overall':result['scores']}
            for group in COUNTS[bench]:
                groups[group]=[s for s in result['scores'] if (s['category']!='style' if group=='non_style' else s['category']==group)]
            data[method,bench]={}
            for group,scores in groups.items():
                refs={s['reference_id'] for s in scores}
                if scores:
                    expected_count=(1170,130) if group=='overall' and no_style else ((750,30) if bench=='dreambench' else (1350,150)) if group=='overall' else COUNTS[bench][group]
                    if (len(scores),len(refs))!=expected_count:raise ValueError('Class coverage differs')
                    per_ref={r:sum(s['reference_id']==r for s in scores) for r in refs}
                    if set(per_ref.values())!={25 if bench=='dreambench' else 9}:raise ValueError('Uneven reference coverage')
                values=aggregate(scores) if scores else {m:None for m in METRICS}
                stored=result['summary'][bench+'/'+method]
                stored=stored['subject_macro'] if group=='overall' else stored['non_style']['subject_macro'] if group=='non_style' else stored.get('by_class',{}).get(group,{}).get('subject_macro')
                if scores and not same(values,stored):raise ValueError('Independent macro check failed')
                rec=dict(method=method,benchmark=bench,group=group,cases=len(scores),references=len(refs),
                    resolution=resolution,selection='non-style only' if no_style else 'all',status='evaluated' if scores else 'excluded',**values)
                data[method,bench][group]=rec;exports.append(rec)
            report_name='dreambench_plus_no_style_scores.json' if no_style else bench+'_scores.json'
            report=ROOT/method/'results/reports'/report_name
            if not report.exists() or digest(report)!=digest(path):raise ValueError('Published score copy differs')
            audits.append(dict(method=method,benchmark=bench,source=path.relative_to(ROOT).as_posix(),
                sha256=digest(path),bundle_index_sha256=digest(folder/'bundle_index.json'),
                shared_manifest_sha256=digest(shared_manifest),
                evaluated_cases=len(seen),references=len({s['reference_id'] for s in result['scores']}),
                excluded_cases=sorted(set(canonical)-seen),generation_resolution=[resolution,resolution],
                generation_fingerprint=run['fingerprint'],summary_recomputed=True,independent_subject_macro_verified=True,
                score_image_hashes_match_bundle_index=True))
    return data,exports,dict(date='2026-11-04',status='verified',environment=env,protocol=evaluator.PROTOCOL,
        evaluations=audits,total_scored_cases=sum(a['evaluated_cases'] for a in audits),
        scope='Recomputed stored per-image metric aggregates; checked canonical identity, exclusions, generation run, score provenance and score-to-bundle-index image hashes. Existing image ZIPs were not fully rehashed and models were not rerun.')


def table_rows(data,historical,bench,group):
    count=(750 if bench=='dreambench' else 1350) if group=='overall' else COUNTS[bench][group][0]
    rows=[]
    for method,values in zip(historical['methods'],historical[bench][group]):
        rows.append(dict(method=method,cases=count,values=values,source='historical'))
    for method in SOURCES:
        r=data[method,bench][group]
        dagger=' †' if method=='freecus' and bench=='dreambench_plus' else ''
        rows.append(dict(method=f"{LABELS[method]} ({r['resolution']} × {r['resolution']}){dagger}",
            cases=r['cases'],values=[r[m] for m in METRICS],source='recomputed'))
    if bench=='dreambench':
        rows.append(dict(method='USO (external)',cases=750 if group=='overall' else None,
            values=historical[bench]['uso_external'] if group=='overall' else [None]*3,source='historical'))
    return rows


def render(data,historical,bench,classes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Serif','pdf.fonttype':42})
    groups=list(COUNTS[bench]) if classes else ['overall']
    notes=['Subject macro; one selected reference; seed 42; higher is better.',
        'FreeCus / DreamO / UNO: 1024 × 1024. XVerse: native 768 × 768. N = evaluated images.',
        'Baseline / Ours: preserved historical PNG values (4 decimals); four comparison methods: audited local scores.',
        '* Ours best uses provisional scale 0.8; this is not an optimized scale.']
    if bench=='dreambench_plus':notes += ['† FreeCus: 1,170 images / 130 references only; excludes all 20 style references (180 images).',
        'FreeCus overall is non-style only; compare the Non-Style block for a shared 1,170-image cohort.' if classes else 'FreeCus overall uses a different cohort from the 1,350-image rows; matched non-style results are in the classes table.']
    else:notes += ['USO: historical external overall value; class-level results unavailable. Old external UNO values were replaced.']
    if classes and bench=='dreambench_plus':notes += ['Non-Style pools animal + human + object; it is not another disjoint class.']
    row_h=74;group_h=92;top=360
    total=top+sum(group_h+row_h*len(table_rows(data,historical,bench,g)) for g in groups)+80+43*len(notes)+70
    fig=plt.figure(figsize=(15,total/200),dpi=200,facecolor='white');ax=fig.add_axes([0,0,1,1])
    ax.set_xlim(0,3000);ax.set_ylim(total,0);ax.axis('off')
    texts=[]
    def text(x,y,value,size=15,color='black',align='left'):
        texts.append(ax.text(x,y,value,fontsize=size,color=color,ha=align,va='center'))
    def rule(y,lw):ax.plot([130,2870],[y,y],color='black',lw=lw)
    text(132,82,'DreamBench' if bench=='dreambench' else 'DreamBench++',25)
    text(132,161,'Method comparison — classes' if classes else 'Method comparison',15,'#4a4a4a')
    rule(240,1.7);text(160,295,'Method',15)
    columns=[1570,1940,2340,2740]
    for x,label in zip(columns,['N',*[m+' ↑' for m in METRICS]]):text(x,295,label,15,align='center')
    rule(350,.85);y=350
    for group in groups:
        if y!=350:rule(y,.7)
        label=group.replace('_','-').title()
        if group=='overall':
            label='Overall (750 images / method)' if bench=='dreambench' else 'Overall (1,350 images; FreeCus † 1,170)'
        else:label+=f' ({COUNTS[bench][group][0]:,} images)'
        text(160,y+49,label,14);y+=group_h
        for row in table_rows(data,historical,bench,group):
            text(160,y+row_h/2,row['method'],14)
            text(columns[0],y+row_h/2,'—' if row['cases'] is None else f"{row['cases']:,}",14,align='center')
            for x,value in zip(columns[1:],row['values']):text(x,y+row_h/2,'—' if value is None else f'{value:.4f}',15,align='center')
            y+=row_h
    rule(y+8,1.7)
    for i,note in enumerate(notes):text(160,y+70+43*i,note,10.2,'#4a4a4a')
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    bounds=fig.bbox
    for t in texts:
        box=t.get_window_extent(renderer)
        if box.x0<0 or box.x1>bounds.x1 or box.y0<0 or box.y1>bounds.y1:raise ValueError('Text clipped: '+t.get_text())
    target=HERE/(bench+'_method_comparison'+('_classes' if classes else '')+'.png')
    fig.savefig(target,dpi=200,facecolor='white');plt.close(fig)
    return dict(path=target.name,sha256=digest(target),width=3000,height=int(total),notes=notes)


def main():
    untouched={p.name:digest(p) for p in HERE.glob('*.png') if 'method_comparison' not in p.name}
    data,rows,audit=audit_sources()
    historical=read(HERE/'historical_rows.json')
    audit['historical_rows_source']=dict(path='historical_rows.json',sha256=digest(HERE/'historical_rows.json'),
        historical_vs_new_measurements='Baseline/Ours/USO are historical; FreeCus/DreamO/UNO/XVerse are recomputed from their stored case scores.')
    audit['figures']=[render(data,historical,b,c) for b in ['dreambench','dreambench_plus'] for c in [False,True]]
    for name,sha in untouched.items():
        if digest(HERE/name)!=sha:raise ValueError('Unrelated table changed')
    audit['unchanged_other_pngs']=untouched
    with (HERE/'method_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(HERE/'method_metrics.json',dict(aggregation='subject_macro',rows=rows))
    write(HERE/'aggregation_audit.json',audit)
    print(json.dumps(dict(status='verified',evaluations=len(audit['evaluations']),scored_cases=audit['total_scored_cases'],
        figures=[p['path'] for p in audit['figures']],overall=[r for r in rows if r['group']=='overall']),indent=2))


if __name__=='__main__':main()
