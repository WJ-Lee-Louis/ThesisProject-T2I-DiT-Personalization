"""Render the saved real monitoring run; missing images are explicitly marked."""
import argparse
import html
import json
from pathlib import Path
import shutil

from PIL import Image, ImageDraw, ImageFont, ImageOps


def read(path, default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def main(root):
    selection = read(root / 'selection.json', [])
    attempts = [json.loads(line) for line in (root / 'llm_attempts.jsonl').read_text(encoding='utf-8').splitlines()]
    summary = []
    cards = []
    esc = html.escape
    for case in selection:
        cid = case['case_id']
        directory = root / cid
        directory.mkdir(exist_ok=True)
        shutil.copyfile(case['reference_image'], directory / 'reference.jpg')
        status = read(directory / 'status.json', {})
        source = read(directory / 'source_analysis.json', {})
        target = read(directory / 'target_analysis.json', {})
        row = dict(case_id=cid, identity_term=case['identity_term'], target_prompt=case['target_prompt'],
            source_caption=source.get('text'), status=status.get('status'),
            source_component_count=len(source.get('components', [])), target_component_count=len(target.get('components', [])),
            retries=sum(bool(a['feedback']) for a in attempts if a['case_id']==cid), strategies={})
        pics = [f'<figure><img src="{cid}/reference.jpg"><figcaption>Reference</figcaption></figure>']
        details = []
        for strategy in ('local', 'factorial'):
            audit = read(directory / f'audit_{strategy}.json', {})
            gpu = read(directory / f'gpu_{strategy}.json', {})
            orth = gpu.get('diagnostics', {}).get('orthogonalization', {})
            row['strategies'][strategy] = dict(
                contexts=audit.get('unique_text_context_count'), pairs=audit.get('contrast_pair_count'),
                fixed_contexts=[a['all_removed_prompt'] for a in audit.get('analyses', [])],
                generated=gpu.get('status')=='generated', error=gpu.get('error'),
                basis_rank=orth.get('basis_rank'), visual_cosine_to_input=orth.get('cosine_to_input'),
                max_abs_basis_dot=orth.get('max_abs_basis_dot'), original_norm=orth.get('original_norm'),
                projection_component_norm=orth.get('projection_component_norm'),
                fixed_context_projection_fraction=gpu.get('text_geometry',{}).get('fixed_context_projection_fraction'))
            png = directory / f'{strategy}.png'
            if png.exists():
                pics.append(f'<figure><a href="{cid}/{strategy}.png"><img src="{cid}/{strategy}.png"></a><figcaption>{strategy}</figcaption></figure>')
            else:
                pics.append(f'<figure><div class="missing">No generated image</div><figcaption>{strategy}</figcaption></figure>')
            pairs = ''.join(f'<tr><td>{esc(c["direction"])}</td><td>{esc(p)}</td><td>{esc(n)}</td></tr>'
                for c in audit.get('contrasts', []) for p,n in c['pairs'])
            details.append(f'<details><summary>{strategy}: {audit.get("unique_text_context_count",0)} contexts, {audit.get("contrast_pair_count",0)} pairs</summary>'
                f'<a href="{cid}/audit_{strategy}.json">Full audit JSON</a> | <a href="{cid}/gpu_{strategy}.json">GPU diagnostics JSON</a>'
                f'<table><tr><th>Direction</th><th>Positive</th><th>Negative</th></tr>{pairs}</table></details>')
        summary.append(row)
        formatted = esc(json.dumps(dict(source=source,target=target),ensure_ascii=False,indent=2))
        metrics = esc(json.dumps(row['strategies'],ensure_ascii=False,indent=2))
        cards.append(f'<article id="{cid}"><h2>{cid}: {esc(case["identity_term"])}</h2><p>{esc(case["selection_reason"])}</p>'
            f'<p><b>Target:</b> {esc(case["target_prompt"])}</p><p><b>Source caption:</b> {esc(source.get("text","Unavailable"))}</p>'
            f'<div class="images">{"".join(pics)}</div><details><summary>LLM output (source / target)</summary><pre>{formatted}</pre></details>'
            + f'<details><summary>Numerical monitoring</summary><pre>{metrics}</pre></details>' + ''.join(details) + '</article>')
    (root / 'monitoring_summary.json').write_text(json.dumps(dict(cases=summary,
        recorded_api_attempts=len(attempts), probe_caption_calls=1 if (root/'probe_caption.txt').exists() else 0,
        retries=sum(bool(a['feedback']) for a in attempts)),ensure_ascii=False,indent=2),encoding='utf-8')
    css='body{font:16px system-ui;max-width:1250px;margin:auto;padding:24px;background:#f5f5f5}article{background:white;padding:20px;margin:22px 0;border-radius:12px}.images{display:flex;gap:14px}figure{flex:1;margin:0}img{width:100%;height:330px;object-fit:contain;background:#eee}figcaption{text-align:center;padding:8px}summary{cursor:pointer;padding:12px}pre{white-space:pre-wrap;background:#f3f5f8;padding:14px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #ddd;padding:8px;text-align:left}.missing{height:330px;background:#eee;display:grid;place-items:center}nav a{padding:8px;display:inline-block}'
    nav=''.join(f'<a href="#{c["case_id"]}">{c["case_id"]}</a>' for c in selection)
    intro='<h1>DreamBench++ semantic pipeline: 10 real cases</h1><p>Source + target projection; local vs factorial; seed 42, 24 steps, 1024x1024, strength 1.0, IP scale 0.7, guidance 3.5, single-stream gate ON.</p><p>Span validation is not proof of identity preservation. Fixed-text overlap with the nuisance basis is a geometric diagnostic, not a style-quality score. No unmodified IP-Adapter baseline or quantitative benchmark score was run.</p>'
    (root/'index.html').write_text(f'<!doctype html><html><meta charset="utf-8"><title>Semantic pipeline monitoring</title><style>{css}</style>{intro}<nav>{nav}</nav>'+''.join(cards)+'</html>',encoding='utf-8')
    try:
        font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    except OSError:
        font=ImageFont.load_default()
    for start in (0,5):
        group=selection[start:start+5]
        sheet=Image.new('RGB',(1200,len(group)*390),'white')
        draw=ImageDraw.Draw(sheet)
        for index,case in enumerate(group):
            y=index*390
            draw.text((8,y+4),f'{case["case_id"]}  {case["identity_term"]}',font=font,fill='black')
            for column,name in enumerate(('reference','local','factorial')):
                p=root/case['case_id']/(name+('.jpg' if name=='reference' else '.png'))
                draw.text((column*400+8,y+31),name,font=font,fill='black')
                if p.exists():
                    image=ImageOps.contain(Image.open(p).convert('RGB'),(390,325))
                    sheet.paste(image,(column*400+(400-image.width)//2,y+58))
                else:
                    draw.text((column*400+8,y+110),'No generated image',font=font,fill='red')
        sheet.save(root/f'comparison_{start+1:02d}_{start+len(group):02d}.jpg',quality=95)
    print('Saved index.html, monitoring_summary.json and two comparison sheets')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('output_dir',type=Path)
    main(parser.parse_args().output_dir)
