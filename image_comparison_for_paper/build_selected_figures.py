"""Reproduce the requested PNG paper figures with Times New Roman.

Run with Python containing Pillow, reportlab and pymupdf. All outputs are
written beside this script under selected_figures/. Source images are unchanged.
"""
from pathlib import Path
import hashlib
import io
import itertools
import json
import re

import fitz
from PIL import Image, ImageOps
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = HERE / 'selected_figures'
FONT_PATH = '/System/Library/Fonts/Supplemental/Times New Roman.ttf'
FONT = 'TimesNewRomanEmbedded'
pdfmetrics.registerFont(TTFont(FONT, FONT_PATH))
BOLD_FONT_PATH = '/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf'
BOLD_FONT = 'TimesNewRomanBoldEmbedded'
pdfmetrics.registerFont(TTFont(BOLD_FONT, BOLD_FONT_PATH))
ITALIC_FONT_PATH = '/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf'
ITALIC_FONT = 'TimesNewRomanItalicEmbedded'
pdfmetrics.registerFont(TTFont(ITALIC_FONT, ITALIC_FONT_PATH))
W, H = 2376, 1280
SIDE = 344
XS = [481, 860, 1238, 1620, 2001]
BLUE, RED, BLACK = '#0070C0', '#FF0000', '#000000'
# Spans concatenate verbatim to the original benchmark prompt.
CASES = [
    ('dreambench', '016-21', '0.8', 'dog', [('a ', BLACK), ('purple', RED), (' ', BLACK), ('dog', BLUE)]),
    ('dreambench', '015-21', '0.7', 'dog', [('a ', BLACK), ('purple', RED), (' ', BLACK), ('dog', BLUE)]),
    ('dreambench', '014-21', '0.8', 'dog', [('a ', BLACK), ('purple', RED), (' ', BLACK), ('dog', BLUE)]),
    ('dreambench', '013-21', '0.8', 'dog', [('a ', BLACK), ('purple', RED), (' ', BLACK), ('dog', BLUE)]),
    ('dreambench_plus', '177-08', '0.7', 'cosmic nebula', [
        ('A ', BLACK), ('ancient sailboat navigating through a sea of stars and nebulae', RED),
        (' in ', BLACK), ('cosmic nebula', BLUE), (' style', BLACK)]),
    ('dreambench_plus', '166-05', '0.7', 'volcanic landscape', [
        ('A ', BLACK), ('forest recovering after a volcanic eruption', RED), (', with ', BLACK),
        ('new growth sprouting through ash', RED), (', in the style of a ', BLACK), ('volcanic landscape', BLUE)]),
    ('dreambench_plus', '164-03', '0.7', 'Impressionist swirls', [
        ('A ', BLACK), ('serene lakeside at sunset', RED), (', depicted with ', BLACK),
        ('vibrant', RED), (' ', BLACK), ('Impressionist swirls', BLUE)]),
    ('dreambench_plus', '103-02', '0.6', 'tree', [
        ('A detailed photograph of a ', BLACK), ('cherry', RED), (' ', BLACK), ('tree', BLUE),
        (' in ', BLACK), ('full blossom', RED), (' against a ', BLACK), ('clear blue sky', RED)]),
    ('dreambench_plus', '043-04', '0.7', 'polar bear', [
        ('An ', BLACK), ('anime-style illustration', RED), (' of a ', BLACK), ('polar bear', BLUE),
        (' ', BLACK), ('wearing a scarf', RED), (' and ', BLACK), ('sipping hot chocolate', RED)]),
]
METHODS = [
    ('UNO', 'UNO_512'), ('DreamO', 'DreamO_1024'), ('XVerse', 'XVerse_768'),
    ('USO', 'USO_1024'), ('MOSAIC', 'MOSAIC_512'),
    ('VIP-Adapter\n(Ours)', None), ('IP-Adapter', 'baseline_scale0.7'),
    ('Diptych\nPrompting', 'Diptych_768'), ('FreeCus', 'FreeCus_512'),
    ('FreeGraftor', 'FreeGraftor_512'),
]


def width(s, size, font=FONT):
    return pdfmetrics.stringWidth(s, font, size)


def text(c, s, x, top, size=52, color=BLACK, font=FONT):
    c.setFont(font, size)
    c.setFillColor(color)
    c.drawString(x, H - top - size * 0.82, s)


def centered(c, s, cx, top, size=52, color=BLACK, font=FONT):
    text(c, s, cx - pdfmetrics.stringWidth(s, font, size) / 2, top, size, color, font)


def rich_lines(spans, size, max_width, font=FONT):
    # Keep punctuation attached to words while retaining exact spaces and colors.
    chars = [(ch, color) for span, color in spans for ch in span]
    s = ''.join(ch for ch, _ in chars)
    tokens = []
    for m in re.finditer(r'\S+\s*', s):
        tokens.append(chars[m.start():m.end()])
    lines, line = [], []
    for token in tokens:
        candidate = line + token
        if line and width(''.join(ch for ch, _ in candidate).rstrip(), size, font) > max_width:
            while line and line[-1][0].isspace():
                line.pop()
            lines.append(line)
            line = token
        else:
            line = candidate
    if line:
        while line and line[-1][0].isspace():
            line.pop()
        lines.append(line)
    # Balance wrapped lines so the final line never contains a lone trailing word.
    if len(lines) > 1:
        best, best_score = lines, float('inf')
        for breaks in itertools.combinations(range(1, len(tokens)), len(lines)-1):
            bounds = (0,) + breaks + (len(tokens),)
            candidate_lines = []
            widths = []
            for start, end in zip(bounds, bounds[1:]):
                candidate = [char for token in tokens[start:end] for char in token]
                while candidate and candidate[-1][0].isspace():
                    candidate.pop()
                candidate_lines.append(candidate)
                widths.append(width(''.join(ch for ch, _ in candidate), size, font))
            if max(widths) <= max_width:
                score = sum((w-sum(widths)/len(widths))**2 for w in widths)
                if score < best_score:
                    best, best_score = candidate_lines, score
        lines = best
    return lines


def rich_center(c, spans, cx, top, size, max_width, leading, font=FONT):
    lines = rich_lines(spans, size, max_width, font)
    for i, line in enumerate(lines):
        x = cx - width(''.join(ch for ch, _ in line), size, font) / 2
        run, color = '', None
        for ch, new_color in line + [('', None)]:
            if new_color != color:
                if run:
                    text(c, run, x, top + i * leading, size, color, font)
                    x += width(run, size, font)
                run, color = ch, new_color
            else:
                run += ch
    return len(lines)


def tile(c, path, x, y):
    with Image.open(path) as src:
        img = ImageOps.exif_transpose(src).convert('RGB')
        iw, ih = img.size
        factor = min(SIDE / iw, SIDE / ih)
        dw, dh = iw * factor, ih * factor
        c.drawImage(ImageReader(img), x + (SIDE-dw)/2,
                    H-y-(SIDE+dh)/2, dw, dh)
        return [iw, ih]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    mapping = {(r['benchmark'], r['case_id']): r for r in json.loads((HERE/'case_mapping.json').read_text())}
    report = []
    for bench, case, scale, identity, spans in CASES:
        record = mapping[bench, case]
        assert ''.join(s for s, _ in spans) == record['prompt'], case
        folder = OUT / ('dreamBench' if bench == 'dreambench' else 'dreambench_plus')
        folder.mkdir(parents=True, exist_ok=True)
        # Use a vector canvas in memory; only the final PNG is written to disk.
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=(W/2, H/2), pageCompression=1, initialFontName=FONT)
        c.setTitle(f'{case} method comparison')
        c.scale(0.5, 0.5)
        centered(c, 'Reference:', 197, 199)
        ref = Path(record['reference'])
        tile(c, ref, 25, 270)
        ref_spans = [('“', BLACK), (identity, BLUE), ('”', BLACK)]
        n = rich_center(c, ref_spans, 197, 637, 49, 395, 58)
        centered(c, '('+record['category']+')', 197, 637+n*58+8)
        sources = []
        for index, (label, method) in enumerate(METHODS):
            method = method or 'Ours_scale'+scale
            row, col = divmod(index, 5)
            x, y = XS[col], [35, 501][row]
            source = record['sources'][method]
            missing = source == '(impossible)'
            if missing:
                assert record['category'] == 'style' and method in ('FreeCus_512', 'FreeGraftor_512')
                centered(c, '(impossible)', x+SIDE/2, y+SIDE/2-24, 46)
                dimensions = None
            else:
                path = Path(source)
                assert path.is_file(), path
                dimensions = tile(c, path, x, y)
            for j, label_line in enumerate(label.splitlines()):
                centered(c, label_line, x+SIDE/2, y+SIDE+19+j*65,
                         font=BOLD_FONT if index == 5 else FONT)
            sources.append({'label': label, 'method': method, 'path': source,
                            'dimensions': dimensions, 'sha256': None if missing else sha256(path)})
        c.setStrokeColor('#246C87')
        c.setLineWidth(1.7)
        c.line(481, H-1017, 2345, H-1017)
        prompt_spans = [('“', BLACK)] + spans + [('”', BLACK)]
        lines = rich_lines(prompt_spans, 50, 1510, ITALIC_FONT)
        assert len(lines) <= 3
        prompt_top = 1135 - len(lines)*32
        centered(c, 'Prompt:', 650, 1110)
        rich_center(c, prompt_spans, 1565, prompt_top, 50, 1510, 64, ITALIC_FONT)
        c.save()
        doc = fitz.open(stream=buffer.getvalue(), filetype='pdf')
        page = doc[0]
        pixmap = page.get_pixmap(matrix=fitz.Matrix(4, 4), alpha=False)
        pixmap.set_dpi(288, 288)
        pixmap.save(folder/f'{case}.png')
        assert all('TimesNewRoman' in font[3] for font in page.get_fonts()), page.get_fonts()
        text_spans = [s for b in page.get_text('dict')['blocks'] if 'lines' in b
                      for line in b['lines'] for s in line['spans']]
        assert [s['text'] for s in text_spans if 'Bold' in s['font']] == ['VIP-Adapter', '(Ours)']
        italic_spans = [s for s in text_spans if 'Italic' in s['font']]
        assert ''.join(''.join(s['text'].split()) for s in italic_spans) == ''.join(
            ('“' + record['prompt'] + '”').split())
        assert '(impossible)' in page.get_text() if record['category']=='style' else '(impossible)' not in page.get_text()
        assert len(page.get_images()) == (9 if record['category']=='style' else 11)
        doc.close()
        report.append({'benchmark': bench, 'case_id': case, 'ours_scale': scale,
                       'baseline_scale': '0.7', 'identity': identity, 'category': record['category'],
                       'prompt': record['prompt'], 'prompt_spans': spans, 'reference': str(ref),
                       'reference_sha256': sha256(ref), 'sources': sources,
                       'png': str((folder/f'{case}.png').relative_to(HERE)), 'png_dimensions': [4752, 2560]})
        print(f'Created and verified {bench}/{case}', flush=True)
    (OUT/'figure_manifest.json').write_text(json.dumps({
        'font': FONT_PATH, 'bold_font': BOLD_FONT_PATH,
        'bold_labels': ['VIP-Adapter', '(Ours)'],
        'italic_font': ITALIC_FONT_PATH, 'prompt_italic': True,
        'colors': {'identity': BLUE, 'requested_change': RED, 'general': BLACK},
        'png_dpi': 288, 'cases': report
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
