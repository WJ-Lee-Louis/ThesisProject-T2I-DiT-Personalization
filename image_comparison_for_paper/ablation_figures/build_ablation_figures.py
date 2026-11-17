"""Render two one-row ablation grids, with seven images and an italic prompt."""
from pathlib import Path
import hashlib
import io
import json
import sys

import fitz
from PIL import Image
from reportlab.pdfgen import canvas

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import build_selected_figures as layout

ROOT = HERE.parents[2]
W, H = 2638, 700
layout.W, layout.H = W, H
CASES = [
    ('175-06', '0.7', [('A ', layout.BLACK), ('pair of worn leather boots', layout.RED),
                     (', illustrated in a ', layout.BLACK),
                     ('monochromatic ink sketch', layout.BLUE), (' style', layout.BLACK)]),
    ('175-05', '0.6', [('A ', layout.BLACK), ('teapot steaming gently', layout.RED),
                     (' on an ', layout.BLACK), ('old wooden table', layout.RED),
                     (', in a ', layout.BLACK), ('monochromatic ink sketch', layout.BLUE),
                     (' style', layout.BLACK)]),
]
METHODS = [
    ('Reference', 'Reference'),
    ('IP-Adapter', 'baseline_scale0.7'),
    ('w/Source Proj.', 'ablation_source_only_scale0.7'),
    ('w/Target Proj.', 'ablation_target_only_scale0.7'),
    ('w/Source+Target\nProj.', 'ablation_source_target_scale0.7'),
    ('w/SSG', 'ablation_ssg_only_scale0.7'),
    ('Ours', None),
]


def main():
    mapping = {r['case_id']:r for r in json.loads((HERE.parent/'case_mapping.json').read_text())
               if r['benchmark'] == 'dreambench_plus'}
    artifacts = []
    for case, scale, prompt_spans in CASES:
        record = mapping[case]
        assert ''.join(s for s,_ in prompt_spans) == record['prompt']
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=(W/2,H/2), initialFontName=layout.FONT)
        c.setTitle(case+' ablation comparison')
        c.scale(.5,.5)
        sources = []
        for index, (label, method) in enumerate(METHODS):
            method = method or 'Ours_scale'+scale
            path = Path(record['sources'][method])
            assert path.is_file(), path
            x = 25+index*(344+30)
            dimensions = layout.tile(c, path, x, 140)
            for line_index, line in enumerate(label.splitlines()):
                layout.centered(c, line, x+172, 25+line_index*53, 44,
                                font=layout.BOLD_FONT if label=='Ours' else layout.FONT)
            entry = {'label':label, 'method':method, 'path':str(path),
                     'dimensions':dimensions, 'sha256':layout.sha256(path)}
            if method != 'Reference':
                meta_base = ROOT/method/'execution_records'
                paths = [meta_base/'dreambench_plus'/f'{case}.json',
                         meta_base/'images/dreambench_plus'/f'{case}.json']
                metadata = next((p for p in paths if p.is_file()), None)
                if metadata:
                    data = json.loads(metadata.read_text())
                    assert data['png_sha256'] == entry['sha256']
                    assert data['row']['target_prompt'] == record['prompt']
                    assert data['row']['identity_term'] == 'monochromatic ink sketch'
                    entry.update({'metadata':str(metadata), 'identity_term':data['row']['identity_term'],
                                  'scale':data['settings']['ip_adapter_scale'],
                                  'condition':data['settings']['condition']})
            sources.append(entry)
        # Align the larger two-line identity with the prompt band.
        layout.rich_center(c, [('“',layout.BLACK), ('monochromatic',layout.BLUE)],
                           197, 561, 44, 344, 53)
        layout.rich_center(c, [('ink sketch',layout.BLUE), ('”',layout.BLACK)],
                           197, 614, 44, 344, 53)
        c.setStrokeColor('#246C87')
        c.setLineWidth(1.7)
        c.line(399,H-535,W-25,H-535)
        quoted = [('“',layout.BLACK)]+prompt_spans+[('”',layout.BLACK)]
        prefix_width = layout.width('Prompt:',48)
        gap = 18
        available = W-25-399
        lines = layout.rich_lines(quoted,48,available-prefix_width-gap,layout.ITALIC_FONT)
        assert len(lines)<=2
        body_width = max(layout.width(''.join(ch for ch,_ in line),48,layout.ITALIC_FONT)
                         for line in lines)
        group_width = prefix_width+gap+body_width
        group_left = 399+(available-group_width)/2
        top = 590-(len(lines)-1)*29
        layout.text(c,'Prompt:',group_left,590,48)
        layout.rich_center(c,quoted,group_left+prefix_width+gap+body_width/2,
                           top,48,available-prefix_width-gap,58,layout.ITALIC_FONT)
        c.save()
        doc = fitz.open(stream=buffer.getvalue(),filetype='pdf')
        page=doc[0]
        assert len(page.get_images())==7
        spans=[s for b in page.get_text('dict')['blocks'] if 'lines' in b
               for line in b['lines'] for s in line['spans']]
        assert all('TimesNewRoman' in font[3] for font in page.get_fonts())
        assert [s['text'] for s in spans if 'Bold' in s['font']]==['Ours']
        italic=''.join(''.join(s['text'].split()) for s in spans if 'Italic' in s['font'])
        assert italic==''.join(('“'+record['prompt']+'”').split())
        for block in page.get_text('blocks'):
            assert 0<=block[0]<=block[2]<=W/2 and 0<=block[1]<=block[3]<=H/2
        pix=page.get_pixmap(matrix=fitz.Matrix(4,4),alpha=False)
        pix.set_dpi(288,288)
        output=HERE/f'{case}.png'
        pix.save(output)
        doc.close()
        with Image.open(output) as image:
            assert image.size==(5276,1400)
            image.verify()
        artifacts.append({'case_id':case,'benchmark':'dreambench_plus','ours_scale':scale,
                          'comparison_scale':'0.7','prompt':record['prompt'],
                          'identity':'monochromatic ink sketch','prompt_spans':prompt_spans,
                          'sources':sources,'png':output.name,'png_sha256':layout.sha256(output)})
        print('Created and verified',output.name,flush=True)
    (HERE/'figure_manifest.json').write_text(json.dumps({
        'layout':'Method headers above one horizontal image row; large two-line reference identity aligned with prompt band; divider starts at column two; Prompt label and sentence centered as one group over columns two through seven.',
        'dimensions':[5276,1400],'dpi':288,'font':layout.FONT_PATH,
        'prompt_font':layout.ITALIC_FONT_PATH,'ours_font':layout.BOLD_FONT_PATH,
        'note':'175-05 uses Ours 0.6 against existing 0.7 comparison variants; it is not a same-scale factorial ablation.',
        'artifacts':artifacts},ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    main()
