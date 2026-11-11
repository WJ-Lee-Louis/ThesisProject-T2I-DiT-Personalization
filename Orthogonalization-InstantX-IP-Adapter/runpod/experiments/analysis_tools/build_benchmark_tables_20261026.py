"""Render benchmark tables from recorded subject-macro summaries."""
from pathlib import Path
import csv
import json
import zipfile
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

BASE = Path('/workspace/instantx-0807')
UPDATED_TABLES = '--updated-tables' in sys.argv
OUT = BASE / ('reports/tables' if UPDATED_TABLES else 'reports/benchmark_tables_20261026')
OUT.mkdir(parents=True, exist_ok=True)
OLD = BASE / 'reports/benchmark_comparison_20261025'
HISTORY = json.loads((OLD / 'historical_screenshot_transcription.json').read_text())['values']
SOURCES = {
    'full': BASE / 'results/full/metrics/scores.json',
    'sweep': BASE / 'results/scale_sweep_20261025/metrics/scores.json',
    'ssg': BASE / 'results/ssg_only_scale_0.7_20261025/metrics/scores.json',
}
if UPDATED_TABLES:
    SOURCES.update({method: BASE / f'results/ablation_20261026/{method}/metrics/scores.json'
                    for method in ('baseline_0.8', 'baseline_0.9', 'source_only', 'target_only')})
DATA = {k: json.loads(p.read_text()) for k, p in SOURCES.items()}
M = ('CLIP-I', 'CLIP-T', 'DINO-I')
BEST_SCALE = 0.8  # Explicit user selection, not an optimized result.
GROUPS = {'dreambench': ['animal', 'object'],
          'dreambench_plus': ['animal', 'human', 'object', 'style', 'non_style']}
N = {'dreambench': dict(overall=750, animal=225, object=525),
     'dreambench_plus': dict(overall=1350, animal=405, human=180, object=585, style=180, non_style=1170)}
EXTERNAL = {'FreeCus': (.8058, .3004, .6708), 'UNO': (.8356, .3039, .7604), 'USO': (.8301, .3178, .7686)}


def values(benchmark, group, method):
    if method in ('baseline', 'ours_0.7'):
        raw = HISTORY[benchmark][group][0 if method == 'baseline' else 1]
        return dict(zip(('CLIP-T', 'CLIP-I', 'DINO-I'), raw)), 'historical screenshot (4 decimals)'
    if method in EXTERNAL:
        return (dict(zip(M, EXTERNAL[method])) if group == 'overall' else dict.fromkeys(M)), 'provided comparison PNG; class scores unavailable'
    if method in ('baseline_0.8', 'baseline_0.9', 'source_only', 'target_only'):
        source, condition = method, method
    elif method == 'proj':
        source, condition = 'full', 'source_target'
    elif method == 'ssg':
        source, condition = 'ssg', 'ssg'
    elif method == 'ours_0.8':
        source, condition = 'full', 'source_target_ssg'
    else:
        source, condition = 'sweep', 'method_scale_' + method.split('_')[1]
    s = DATA[source]['summary'][f'{benchmark}/{condition}']
    s = s if group == 'overall' else s['non_style'] if group == 'non_style' else s['by_class'][group]
    assert s['cases'] == N[benchmark][group]
    return s['subject_macro'], str(SOURCES[source])


def methods(kind, benchmark):
    if kind == 'method_comparison':
        rows = [('baseline', 'InstantX-IP-Adapter (baseline)'), ('ours_0.7', 'Ours (scale 0.7)'),
                ('ours_0.8', 'Ours (scale best*)')]
        if benchmark == 'dreambench':
            rows += [(k, k + (' [T]' if k in ('UNO', 'USO') else '')) for k in EXTERNAL]
        return rows
    if kind == 'scale_sensitivity':
        return [(f'ours_{s:.1f}', f'Ours (scale {s:.1f})') for s in (.6, .7, .8, .9, 1.)]
    if UPDATED_TABLES and kind == 'method_ablation':
        return [('baseline', 'InstantX-IP-Adapter (scale 0.7)'),
                ('source_only', '+ Source Proj. (scale 0.7)'),
                ('target_only', '+ Target Proj. (scale 0.7)'),
                ('proj', '+ Source + Target Proj. (scale 0.7)'),
                ('ssg', '+ SSG (scale 0.7)'), ('ours_0.7', 'Ours (scale 0.7)')]
    return [('baseline', 'InstantX-IP-Adapter (baseline)'), ('proj', '+ Proj. (scale 0.7)'),
            ('ssg', '+ SSG (scale 0.7)'), ('ours_0.7', 'Ours (scale 0.7)')]


plt.rcParams.update({'font.family': 'DejaVu Serif', 'pdf.fonttype': 42})
TITLES = {'method_comparison': 'Method comparison', 'scale_sensitivity': 'Scale hyperparameter sensitivity',
          'method_ablation': 'Method ablation (scale 0.7)'}
records = []
artifacts = []


def render(benchmark, kind, groups, filename, pdf, append_baseline=False):
    title = 'DreamBench' if benchmark == 'dreambench' else 'DreamBench++'
    rows = methods(kind, benchmark)
    if append_baseline:
        if UPDATED_TABLES:
            rows += [('baseline', 'InstantX-IP-Adapter (scale 0.7)'),
                     ('baseline_0.8', 'InstantX-IP-Adapter (scale 0.8)'),
                     ('baseline_0.9', 'InstantX-IP-Adapter (scale 0.9)')]
        else:
            rows.append(('baseline', 'InstantX-IP-Adapter (baseline)'))
    notes = ['Subject macro average; single selected reference; higher is better.']
    if any(k in ('baseline', 'ours_0.7') for k, _ in rows):
        notes.append('Baseline and Ours scale 0.7: historical screenshot values (4-decimal precision).')
    if kind == 'method_comparison':
        notes.append('* best row uses scale 0.8 as requested; this is a provisional choice, not an optimized scale.')
        if benchmark == 'dreambench':
            notes.append('[T] tuning-based (UNO, USO). External values copied from the supplied comparison PNG.')
            if groups != ['overall']:
                notes.append('—: external class-level results are unavailable in the supplied PNG.')
    if kind == 'method_ablation':
        notes.append('Proj.: projection ON / SSG OFF; SSG: projection OFF / SSG ON; Ours: both ON.')
        notes.append('Provisional scale 0.7 ablation; rerun all variants at the final selected scale is pending.')
        if UPDATED_TABLES:
            notes.append('Source / Target Proj.: selected-side projection only; SSG OFF. Baseline scale 0.7 confirmed by user.')
        else:
            notes.append('Baseline scale is unavailable in the historical screenshot.')
    if 'non_style' in groups:
        notes.append('Non-style is a pooled subset (animal + human + object), not an additional disjoint class.')
    height = 1.95 + len(groups) * (0.50 + len(rows) * .34) + .22 * len(notes)
    fig, ax = plt.subplots(figsize=(12.5, height))
    fig.subplots_adjust(left=.035, right=.965, top=.97, bottom=.02)
    ax.set_xlim(0, 1); ax.set_ylim(height, 0); ax.axis('off')
    ax.text(.01, .05, title, fontsize=19, fontweight='normal', va='top')
    ax.text(.01, .44, TITLES[kind], fontsize=12, va='top', color='#444')
    xs = (.02, .60, .77, .94)
    y = .92
    ax.plot([.01, .99], [y, y], color='black', lw=1.5)
    y += .25
    for x, label in zip(xs, ('Method', 'CLIP-I ↑', 'CLIP-T ↑', 'DINO-I ↑')):
        ax.text(x, y, label, fontsize=12, fontweight='normal', va='center', ha='left' if x == xs[0] else 'center')
    y += .23
    ax.plot([.01, .99], [y, y], color='black', lw=.8)
    for group in groups:
        y += .27
        label = 'Overall' if group == 'overall' else group.replace('_', '-').title()
        ax.text(.02, y, f'{label} ({N[benchmark][group]:,} images / method)', fontsize=11, fontweight='normal', va='center')
        items = [(key, label, *values(benchmark, group, key)) for key, label in rows]
        for key, label, v, source in items:
            y += .34
            ax.text(xs[0], y, label, fontsize=12, va='center')
            for x, m in zip(xs[1:], M):
                text = '—' if v[m] is None else f'{v[m]:.4f}'
                ax.text(x, y, text, fontsize=12, va='center', ha='center', fontweight='normal')
        y += .22
        ax.plot([.01, .99], [y, y], color='black', lw=1.5 if group == groups[-1] else .6)
    y += .25
    for note in notes:
        ax.text(.02, y, note, fontsize=9, va='top', color='#444'); y += .22
    assert y < height, (filename, y, height)
    fig.savefig(OUT / f'{filename}.png', dpi=240, facecolor='white')
    if pdf is not None:
        pdf.savefig(fig, facecolor='white')
    plt.close(fig)
    artifacts.append(filename + '.png')


import sys
if UPDATED_TABLES:
    for benchmark in GROUPS:
        render(benchmark, 'scale_sensitivity', ['overall'],
               f'{benchmark}_method_verification', None, append_baseline=True)
        render(benchmark, 'method_ablation', ['overall'],
               f'{benchmark}_method_ablation', None)
        render(benchmark, 'method_ablation', GROUPS[benchmark],
               f'{benchmark}_method_ablation_classes', None)
    print(f'{len(artifacts)} updated PNG tables written to {OUT}')
    sys.exit(0)

if '--verification-only' in sys.argv:
    for benchmark in GROUPS:
        render(benchmark, 'scale_sensitivity', ['overall'],
               f'{benchmark}_method_verification', None, append_baseline=True)
    print('Created DreamBench and DreamBench++ method verification PNGs.')
    sys.exit(0)

with PdfPages(OUT / 'benchmark_tables.pdf') as pdf:
    for benchmark in GROUPS:
        for kind in TITLES:
            for group in ['overall', *GROUPS[benchmark]]:
                for key, label in methods(kind, benchmark):
                    v, source = values(benchmark, group, key)
                    records.append(dict(benchmark=benchmark, table=kind, group=group, method=key, label=label,
                                        images=N[benchmark][group], source=source, **{m: v[m] for m in M}))
            render(benchmark, kind, ['overall'], f'{benchmark}_{kind}', pdf)
            render(benchmark, kind, GROUPS[benchmark], f'{benchmark}_{kind}_classes', pdf)

with (OUT / 'table_values.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
(OUT / 'table_values.json').write_text(json.dumps(dict(best_scale=BEST_SCALE, best_scale_basis='Explicit user selection; provisional',
    aggregation='subject_macro', external_source=str(OLD / 'dreambench_method_comparison.png'), rows=records), indent=2))
readme = ['# Benchmark tables — 2026-10-26', '',
          'Best* 행은 사용자 지정에 따라 scale 0.8로 고정했습니다. 최적화로 선정한 값이 아닙니다.',
          '집계는 기존 평가와 동일한 subject macro average이며, 열 순서는 CLIP-I / CLIP-T / DINO-I입니다.',
          'baseline 및 Ours scale 0.7은 기존 스크린샷 전사값입니다. 나머지 로컬 실험은 scores.json의 집계를 사용합니다.',
          'FreeCus / UNO / USO는 지정 PNG의 DreamBench 전체 수치만 사용합니다. class별 수치는 없어 —로 표시했습니다.',
          'DreamBench++에는 FreeCus / UNO / USO를 넣지 않았습니다.',
          'Non-style은 animal + human + object의 통합 부분집합입니다.',
          'Ablation은 scale 0.7 결과이며 최종 scale 기준 재실험은 아직 수행하지 않았습니다.', '',
          '## PNG tables', '', '| Table | DreamBench | DreamBench++ |', '|---|---|---|']
for kind, title in TITLES.items():
    readme.append(f'| {title} | [Overall](dreambench_{kind}.png) · [Classes](dreambench_{kind}_classes.png) | [Overall](dreambench_plus_{kind}.png) · [Classes](dreambench_plus_{kind}_classes.png) |')
(OUT / 'README.md').write_text('\n'.join(readme) + '\n')
with zipfile.ZipFile(OUT / 'benchmark_tables_bundle.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(OUT.iterdir()):
        if path.suffix != '.zip': archive.write(path, path.name)
print(f'{len(artifacts)} PNG tables written to {OUT}')
