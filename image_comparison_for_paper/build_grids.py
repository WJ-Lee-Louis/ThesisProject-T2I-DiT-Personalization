from pathlib import Path
import json,hashlib,math,textwrap,os
from concurrent.futures import ThreadPoolExecutor
from PIL import Image,ImageDraw,ImageFont,ImageOps
OUT=Path(__file__).resolve().parent
ROOT=OUT.parent.parent
REPO=OUT.parent
METHODS=['FreeCus_512','FreeGraftor_512','DreamO_1024','UNO_512','USO_1024','XVerse_768','Diptych_768','MOSAIC_512','XLabs_baseline_scale_1.0','XLabs_Ours_scale_1.0','baseline_scale0.7','baseline_scale0.8','baseline_scale0.9','ablation_source_only_scale0.7','ablation_target_only_scale0.7','ablation_source_target_scale0.7','ablation_ssg_only_scale0.7','Ours_scale0.6','Ours_scale0.7','Ours_scale0.8','Ours_scale0.9','Ours_scale1.0']
FONT='/System/Library/Fonts/Supplemental/Times New Roman.ttf'
SIDE=384;GAP=24;PAD=28;LABEL=72;TOP=130;COLS=6
W=PAD*2+COLS*SIDE+(COLS-1)*GAP
H=TOP+4*(SIDE+LABEL)+3*GAP+PAD
font=ImageFont.truetype(FONT,28);titlefont=ImageFont.truetype(FONT,30);missingfont=ImageFont.truetype(FONT,36)
rows=[];mapping=[]
for b,folder in [('dreambench','dreamBench'),('dreambench_plus','dreambench_plus')]:
 manifest=REPO/'diptych/results'/b/'local-gpu0-seed42-bf16-v1/manifest.jsonl'
 cases=[json.loads(x) for x in manifest.read_text().splitlines() if x.strip()]
 expected={r['case_id']+'.png' for r in cases}
 assert len(expected)==(750 if b=='dreambench' else 1350)
 for method in METHODS:
  actual={p.name for p in (ROOT/method/b).glob('*.png')}
  excluded={r['case_id']+'.png' for r in cases if r['category']=='style'} if method in ['FreeCus_512','FreeGraftor_512'] else set()
  assert actual==expected-excluded,(method,b,'unexpected coverage',len(actual))
 for r in cases:
  ref=ROOT/'references'/(r['subject']+'.jpg');assert ref.is_file(),ref
  sources=[('Reference',ref)]
  for m in METHODS:
   p=ROOT/m/b/(r['case_id']+'.png')
   impossible=r['category']=='style' and m in ['FreeCus_512','FreeGraftor_512']
   assert impossible or p.is_file(),p
   sources.append((m,None if impossible else p))
  target=OUT/folder/(r['case_id']+'.png')
  rows.append((r,sources,target))
  mapping.append({'benchmark':b,'case_id':r['case_id'],'category':r['category'],'prompt':r['prompt'],'reference':str(ref),'output':str(target.relative_to(OUT)),'sources':{n:str(p) if p else '(impossible)' for n,p in sources}})
print('Preflight passed:',len(rows),'cases;',len(METHODS),'methods',flush=True)

def center(draw,txt,y,f,x,width):
 draw.text((x+width/2,y),txt,font=f,fill='black',anchor='mt')
def render(item):
 r,sources,target=item
 im=Image.new('RGB',(W,H),'white');d=ImageDraw.Draw(im)
 center(d,r['case_id']+' | '+r['benchmark']+' | '+r['category'],15,titlefont,0,W)
 lines=textwrap.wrap(r['prompt'],width=145)
 for i,line in enumerate(lines):center(d,line,53+i*33,titlefont,0,W)
 assert len(lines)<=2,'Prompt exceeds header'
 for idx,(name,path) in enumerate(sources):
  row,col=divmod(idx,COLS);x=PAD+col*(SIDE+GAP);y=TOP+row*(SIDE+LABEL+GAP)
  if path:
   with Image.open(path) as src:
    tile=ImageOps.contain(ImageOps.exif_transpose(src).convert('RGB'),(SIDE,SIDE),Image.Resampling.LANCZOS)
    im.paste(tile,(x+(SIDE-tile.width)//2,y+(SIDE-tile.height)//2))
  else:
   d.rectangle((x,y,x+SIDE-1,y+SIDE-1),outline='#b0b0b0',width=1)
   center(d,'(impossible)',y+SIDE/2-18,missingfont,x,SIDE)
  # Wrap names at underscores to keep every experiment identity readable.
  label=name.replace('_',' ')
  ls=textwrap.wrap(label,width=25)
  for j,line in enumerate(ls):center(d,line,y+SIDE+10+j*31,font,x,SIDE)
 target.parent.mkdir(parents=True,exist_ok=True)
 im.save(target,compress_level=3)
 return str(target)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--samples',action='store_true');args=p.parse_args()
 if args.samples:
  selected=[rows[0],next(x for x in rows if x[0]['category']=='style')]
  for x in selected:print(render(x),flush=True)
 else:
  with ThreadPoolExecutor(max_workers=4) as pool:
   for i,_ in enumerate(pool.map(render,rows),1):
    if i%50==0:print('Rendered',i,'/',len(rows),flush=True)
  for b,n in [('dreamBench',750),('dreambench_plus',1350)]:
   files=list((OUT/b).glob('*.png'));assert len(files)==n
   for file in files:
    with Image.open(file) as img:img.verify()
  (OUT/'case_mapping.json').write_text(json.dumps(mapping,ensure_ascii=False,indent=2))
  (OUT/'generation_report.json').write_text(json.dumps({'total_grids':len(rows),'methods':METHODS,'font':FONT,'dimensions':[W,H],'counts':{'dreamBench':750,'dreambench_plus':1350},'impossible_cells':360,'all_pngs_verified':True},indent=2))
  print('COMPLETE: 2100 grids verified',flush=True)
