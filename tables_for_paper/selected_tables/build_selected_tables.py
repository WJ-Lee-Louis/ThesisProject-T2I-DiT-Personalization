"""Render four compact booktabs-style PNG tables; no PDF files are written.

Run: python tables_for_paper/selected_tables/build_selected_tables.py
Dependencies: Pillow, reportlab, PyMuPDF. Fonts: macOS Times New Roman.
"""
from pathlib import Path
import hashlib
import io
import json

import fitz
from PIL import Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

HERE = Path(__file__).resolve().parent
TABLES = HERE.parent
ROOT = TABLES.parent
RUNS = ROOT / 'Orthogonalization-InstantX-IP-Adapter/results/runpod'
METRICS = ['CLIP-I', 'DINO-I', 'CLIP-T']
FONTS = '/System/Library/Fonts/Supplemental/'
for name, filename in [('TNR', 'Times New Roman.ttf'), ('TNR-Bold', 'Times New Roman Bold.ttf')]:
    pdfmetrics.registerFont(TTFont(name, FONTS + filename))
SOURCES = {}


def read(path):
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


HISTORY = read(TABLES / 'historical_rows.json')
MEASURED = {(r['method'], r['benchmark'], r['group']): r
            for r in read(TABLES / 'method_metrics.json')['rows']}
FULL = read(RUNS / 'full/metrics/summary.json')
SWEEP = read(RUNS / 'scale_sweep_20261025/metrics/summary.json')
SSG = read(RUNS / 'ssg_only_scale_0.7_20261025/metrics/summary.json')
ABLATIONS = {key: read(RUNS / f'ablation_20261026/{key}/metrics/summary.json')
             for key in ['source_only', 'target_only', 'baseline_0.8', 'baseline_0.9']}
BENCHES = ['dreambench', 'dreambench_plus']


def original(bench, group, index):
    return dict(zip(HISTORY['metric_order'], HISTORY[bench][group][index]))


def summary(data, bench, condition, group='overall'):
    r = data[f'{bench}/{condition}']
    assert r['cases'] == (750 if bench == 'dreambench' else 1350)
    if group != 'overall':
        r = r['non_style'] if group == 'non_style' else r['by_class'][group]
    return r['subject_macro']


def values(key, bench, group='overall'):
    if key == 'baseline':
        return original(bench, group, 0)
    if key == 'ours_0.7':
        return original(bench, group, 1)
    if key == 'ours_0.8':
        return summary(FULL, bench, 'source_target_ssg', group)
    return MEASURED[key, bench, group]


METHODS = [('IP-Adapter', 'baseline'), ('UNO', 'uno'), ('DreamO', 'dreamo'),
           ('XVerse', 'xverse'), ('USO', 'uso'), ('MOSAIC', 'mosaic'),
           ('Diptych Prompting', 'diptych'), ('FreeCus', 'freecus'),
           ('FreeGraftor', 'freegrafor2'), ('VIP-Adapter (0.7)', 'ours_0.7'),
           ('VIP-Adapter', 'ours_0.8')]


def cells(record):
    return ['-' if record[m] is None else f'{record[m]:.4f}' for m in METRICS]


def checkmark(c, x, y, enabled):
    # Vector marks avoid substitution of the Times New Roman text font.
    c.saveState()
    c.setFillColor('#0000FF' if enabled else '#FF0000')
    c.setStrokeColor('#0000FF' if enabled else '#FF0000')
    c.setLineWidth(0.95)
    if enabled:
        # Filled, tapered check: rounded heavy short arm and rising long arm.
        path = c.beginPath()
        path.moveTo(x-4.8, y)
        path.curveTo(x-5.2, y+1.1, x-3.6, y+2.4, x-2.8, y+1.3)
        path.lineTo(x-1.3, y-1.9)
        path.curveTo(x+0.7, y+1.6, x+3.3, y+4.9, x+5.8, y+7.1)
        path.curveTo(x+6.4, y+7.7, x+7.0, y+7.2, x+6.4, y+6.5)
        path.curveTo(x+3.1, y+2.3, x+0.9, y-1.6, x-1.0, y-5)
        path.curveTo(x-1.4, y-5.8, x-2.9, y-5.7, x-3.3, y-4.6)
        path.close()
        c.drawPath(path, stroke=0, fill=1)
    else:
        c.setLineWidth(1.8)
        c.setLineCap(1)
        c.line(x-2.7, y-3.8, x+2.7, y+4.0)
        c.line(x-2.5, y+3.7, x+2.8, y-3.5)
    c.restoreState()


def render(filename, left_headers, left_widths, groups, rows, separators=(), cell_ranks=None,
           merged_labels=(), method_column=None, method_indent=4):
    """Two header tiers, compact body, booktabs horizontal rules, no grid boxes."""
    numeric_width = 58
    pad, row_height = 9, 17
    widths = left_widths + [numeric_width] * (3*len(groups))
    if method_column is None and len(left_headers) == 1:
        method_column = 0
    w = sum(widths) + 2*pad
    h = 47 + len(rows)*row_height + 13
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=(w, h), initialFontName='TNR')
    positions = [pad]
    for width in widths:
        positions.append(positions[-1]+width)

    def label(s, x, y, bold=False, center=True, size=11):
        c.setFont('TNR-Bold' if bold else 'TNR', size)
        if center:
            c.drawCentredString(x, y, s)
        else:
            c.drawString(x, y, s)

    c.setLineWidth(1.0)
    c.line(pad, h-7, w-pad, h-7)
    for i, title in enumerate(left_headers):
        parts = title.split('\n')
        for j, part in enumerate(parts):
            label(part, positions[i]+method_indent if i == method_column else (positions[i]+positions[i+1])/2,
                  h-27 + (len(parts)-1)*5.5-j*11, center=i != method_column)
    start = len(left_headers)
    for g, title in enumerate(groups):
        lo, hi = positions[start+3*g], positions[start+3*g+3]
        label(title, (lo+hi)/2, h-20)
        c.setLineWidth(.45)
        c.line(lo+4, h-25, hi-4, h-25)
        for k, metric in enumerate(METRICS):
            i = start+3*g+k
            label(metric+' ↑', (positions[i]+positions[i+1])/2, h-38)
    c.setLineWidth(.65)
    c.line(pad, h-45, w-pad, h-45)
    for index, row in enumerate(rows):
        y = h-59-index*row_height
        if index in separators:
            c.setLineWidth(.35)
            c.line(pad, y+12, w-pad, y+12)
        for i, value in enumerate(row):
            if isinstance(value, bool):
                checkmark(c, (positions[i]+positions[i+1])/2, y+3.5, value)
            else:
                rank = cell_ranks[index][i] if cell_ranks is not None else None
                bold = (i == method_column and str(value).startswith('VIP-Adapter')) or rank == 1
                center_x = (positions[i]+positions[i+1])/2
                label(value, positions[i]+method_indent if i == method_column
                      else center_x, y, bold=bold,
                      center=i != method_column)
                if rank == 2:
                    half_width = pdfmetrics.stringWidth(value, 'TNR', 11)/2
                    c.setLineWidth(.5)
                    c.line(center_x-half_width, y-1.7, center_x+half_width, y-1.7)
    for col, first_row, last_row, title in merged_labels:
        label(title, (positions[col]+positions[col+1])/2,
              h-59-(first_row+last_row)*row_height/2)
    c.setLineWidth(1.0)
    c.line(pad, 8, w-pad, 8)
    c.save()
    doc = fitz.open(stream=buffer.getvalue(), filetype='pdf')
    page = doc[0]
    assert all('TimesNewRoman' in f[3] for f in page.get_fonts())
    if cell_ranks is not None:
        numeric_spans = [s for b in page.get_text('dict')['blocks'] if 'lines' in b
                         for line in b['lines'] for s in line['spans']
                         if s['text'] == '-' or s['text'].replace('.', '').isdigit()]
        expected = [(value, cell_ranks[r][col]) for r, row in enumerate(rows)
                    for col, value in enumerate(row) if col >= start]
        assert len(numeric_spans) == len(expected)
        for span, (value, rank) in zip(numeric_spans, expected):
            assert span['text'] == value
            assert ('Bold' in span['font']) == (rank == 1)
    for block in page.get_text('blocks'):
        assert 0 <= block[0] <= block[2] <= w and 0 <= block[1] <= block[3] <= h
    pix = page.get_pixmap(matrix=fitz.Matrix(4, 4), alpha=False)
    pix.set_dpi(288, 288)
    path = HERE / filename
    pix.save(path)
    doc.close()
    with Image.open(path) as im:
        assert im.size == (int(w*4), int(h*4))
        im.verify()
    return {'filename': filename, 'headers': left_headers, 'groups': groups,
            'metric_order': METRICS, 'rows': rows,
            'png_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    artifacts = []
    rows = [[label] + cells(values(key, 'dreambench_plus', 'non_style'))
            + cells(values(key, 'dreambench_plus', 'style')) for label, key in METHODS]
    assert all(rows[i][-3:] == ['-']*3 for i in [7, 8])
    artifacts.append(render('01_dreambench_plus_comparison.png', ['Method'], [127],
                            ['DreamBench++ Subject', 'DreamBench++ Style'], rows, [9]))
    rows = [[label] + cells(values(key, 'dreambench')) for label, key in METHODS]
    artifacts.append(render('02_dreambench_comparison.png', ['Method'], [127],
                            ['DreamBench'], rows, [9]))
    configs = [(False, False, False, 'baseline'), (True, False, False, 'source_only'),
               (False, True, False, 'target_only'), (True, True, False, 'source_target'),
               (False, False, True, 'ssg'), (True, True, True, 'ours_0.7')]
    rows = []
    for source, target, gate, key in configs:
        row = [source, target, gate]
        for bench in BENCHES:
            if key in ['baseline', 'ours_0.7']:
                record = values(key, bench)
            else:
                data = FULL if key == 'source_target' else SSG if key == 'ssg' else ABLATIONS[key]
                record = summary(data, bench, key)
            row += cells(record)
        rows.append(row)
    artifacts.append(render('03_ablation.png', ['source-caption\nprojection', 'target-prompt\nprojection', 'single-stream\ngate'],
                            [80, 80, 78], ['DreamBench', 'DreamBench++'], rows))
    rows = []
    for kind, scales in [('VIP-Adapter', ['0.6', '0.7', '0.8', '0.9', '1.0']),
                         ('IP-Adapter', ['0.7', '0.8', '0.9'])]:
        for scale in scales:
            row = [f'{kind} (scale {scale})']
            for bench in BENCHES:
                if kind == 'IP-Adapter':
                    record = values('baseline', bench) if scale == '0.7' else summary(
                        ABLATIONS['baseline_'+scale], bench, 'baseline_'+scale)
                else:
                    record = values('ours_'+scale, bench) if scale in ['0.7', '0.8'] else summary(
                        SWEEP, bench, 'method_scale_'+scale)
                row += cells(record)
            rows.append(row)
    artifacts.append(render('04_scale_sensitivity.png', ['Method'], [140],
                            ['DreamBench', 'DreamBench++'], rows, [5]))
    (HERE/'table_audit.json').write_text(json.dumps({
        'aggregation': 'subject_macro; non-style pooled across 130 references, not equal class weights',
        'ablation_scale': 0.7, 'main_vip_adapter_scale': 0.8, 'ip_adapter_comparison_scale': 0.7,
        'precision_note': 'Baseline 0.7 and VIP-Adapter 0.7 use retained historical four-decimal values; other rows use stored subject-macro measurements.',
        'source_sha256': SOURCES, 'artifacts': artifacts
    }, ensure_ascii=False, indent=2)+'\n')
    print('Created and verified 4 PNG tables:', *(a['filename'] for a in artifacts), sep='\n')


if __name__ == '__main__':
    main()
