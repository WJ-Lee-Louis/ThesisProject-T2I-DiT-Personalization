"""Compile actual LaTeX and export two tightly cropped, 400-DPI PNG figures.

Requires Tectonic (or XeLaTeX), Pillow and PyMuPDF. No PDFs are kept in this folder.
Example: python build_algorithm_figures.py --tectonic /path/to/tectonic
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import fitz
from PIL import Image, ImageChops

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DPI = 400


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tectonic', default=shutil.which('tectonic'))
    args = parser.parse_args()
    engine = args.tectonic or shutil.which('xelatex')
    if not engine:
        parser.error('Install Tectonic or XeLaTeX, or provide --tectonic PATH.')
    report = []
    with tempfile.TemporaryDirectory(prefix='vip-algorithm-') as folder:
        temp = Path(folder)
        for stem in ['vip_adapter_algorithm', 'vip_adapter_preliminaries']:
            if args.tectonic:
                command = [engine, stem+'.tex', '--outdir', str(temp), '--keep-logs']
            else:
                command = [engine, '-interaction=nonstopmode', '-halt-on-error',
                           '-output-directory='+str(temp), stem+'.tex']
            process = subprocess.run(command, cwd=HERE, text=True, capture_output=True)
            log = (temp/(stem+'.log')).read_text(errors='replace') if (temp/(stem+'.log')).exists() else ''
            if process.returncode:
                raise RuntimeError(process.stdout+'\n'+process.stderr+'\n'+log[-6000:])
            assert 'Overfull \\hbox' not in log, stem+': line overflows; inspect compilation log.'
            assert 'Overfull \\vbox' not in log, stem+': page overflows.'
            assert 'Missing character:' not in log, stem+': missing font glyph.'
            with fitz.open(temp/(stem+'.pdf')) as doc:
                assert len(doc) == 1, (stem, len(doc))
                page = doc[0]
                text = page.get_text()
                assert '\ufffd' not in text
                fonts = sorted({f[3] for f in page.get_fonts()})
                if Path('/System/Library/Fonts/Supplemental/Times New Roman.ttf').exists():
                    assert any('TimesNewRoman' in f for f in fonts), fonts
                pix = page.get_pixmap(matrix=fitz.Matrix(DPI/72, DPI/72), alpha=False)
                image = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
            bbox = ImageChops.difference(image, Image.new('RGB', image.size, 'white')).getbbox()
            assert bbox
            margin = 30
            crop = image.crop((max(0,bbox[0]-margin), max(0,bbox[1]-margin),
                               min(image.width,bbox[2]+margin), min(image.height,bbox[3]+margin)))
            output = HERE/(stem+'.png')
            crop.save(output, dpi=(DPI,DPI))
            report.append({'png':output.name, 'dimensions':list(crop.size), 'dpi':DPI,
                           'fonts':fonts, 'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
                           'single_page':True, 'latex_overflow_checks_passed':True})
            print('Created', output.name, crop.size, flush=True)
    sources = [
        'Orthogonalization-InstantX-IP-Adapter/orthogonalization_instantx_ip_adapter/contrast_directions.py',
        'Orthogonalization-InstantX-IP-Adapter/orthogonalization_instantx_ip_adapter/orthogonalization.py',
        'Orthogonalization-InstantX-IP-Adapter/orthogonalization_instantx_ip_adapter/adapter.py',
        'Orthogonalization-InstantX-IP-Adapter/orthogonalization_instantx_ip_adapter/attention_processor.py',
        'Orthogonalization-InstantX-IP-Adapter/orthogonalization_instantx_ip_adapter/semantic_contrasts.py',
        'Orthogonalization-InstantX-IP-Adapter/docs/prompts/module_a_caption_system_prompt.md',
        'Orthogonalization-InstantX-IP-Adapter/docs/prompts/module_b_contrast_system_prompt.md',
    ]
    (HERE/'figure_audit.json').write_text(json.dumps({
        'engine':str(engine), 'figures':report,
        'latex_sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.tex')},
        'implementation_sources':{s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},
        'scope':'semantic_local source+target conditioning with single-stream gate; frozen pretrained models',
    }, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
