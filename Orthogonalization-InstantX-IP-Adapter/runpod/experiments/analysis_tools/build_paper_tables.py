"""Paper tables: historical screenshot transcription + locally verified score JSON."""
from pathlib import Path
import csv,json,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
BASE=Path('/workspace/instantx-0807')
OUT=BASE/'reports/benchmark_comparison_20261025';OUT.mkdir(parents=True,exist_ok=True)
# Values are transcribed exactly at the four-decimal precision provided by the user.
HISTORY={
'dreambench':{
'overall':[(.2939,.8319,.6900),(.3123,.8022,.6548)],
'animal':[(.2917,.8901,.7872),(.3204,.8561,.7562)],
'object':[(.2949,.8070,.6483),(.3089,.7791,.6113)]},
'dreambench_plus':{
'overall':[(.2994,.8534,.6820),(.3280,.8115,.6154)],
'animal':[(.3117,.8902,.7609),(.3352,.8548,.7226)],
'human':[(.2610,.7941,.6424),(.3010,.7389,.5466)],
'object':[(.3042,.8492,.6584),(.3299,.8116,.6004)],
'style':[(.2945,.8436,.6207),(.3325,.7865,.4916)],
'non_style':[(.3001,.8549,.6914),(.3273,.8153,.6345)]}}
N={'dreambench':{'overall':750,'animal':225,'object':525},'dreambench_plus':{'overall':1350,'animal':405,'human':180,'object':585,'style':180,'non_style':1170}}
METHODS=[('baseline','InstantX-IP-Adapter','—','OFF','screenshot'),('method_0.7','Ours','0.7','ON','screenshot'),('method_0.8','Ours','0.8','ON','local_json'),('no_ssg_0.7','Ours w/o SSG','0.7','OFF','local_json')]
M=('CLIP-T','CLIP-I','DINO-I')
local=json.loads((BASE/'results/full/metrics/scores.json').read_text())
summary=local['summary']
rows=[]
for b in HISTORY:
 for group in HISTORY[b]:
  for i,(key,label,scale,ssg,source) in enumerate(METHODS):
   if i<2:values=HISTORY[b][group][i]
   else:
    v=summary[b+('/source_target_ssg' if i==2 else '/source_target')]
    a=v if group=='overall' else v['non_style'] if group=='non_style' else v['by_class'][group]
    values=tuple(a['subject_macro'][m] for m in M)
   rows.append(dict(benchmark=b,group=group,method=key,label=label,scale=scale,ssg=ssg,images=N[b][group],
                    source=source,**dict(zip(M,values))))
(OUT/'historical_screenshot_transcription.json').write_text(json.dumps(dict(precision_decimals=4,source='User-provided screenshot, 2026-10-25; individual scores unavailable',values=HISTORY),indent=2))
(OUT/'combined_metrics.json').write_text(json.dumps(dict(aggregation='subject_macro',protocol=local['protocol'],local_environment={k:local[k] for k in ('device_name','versions')},rows=rows),indent=2))
with (OUT/'combined_metrics.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
plt.rcParams.update({'font.family':'DejaVu Serif','pdf.fonttype':42,'svg.fonttype':'none'})

def table_figure(title,groups):
 # Each group is (name, four method rows).
 height=2.4+len(groups)*1.42
 fig,ax=plt.subplots(figsize=(13,height));fig.subplots_adjust(left=.025,right=.985,top=.97,bottom=.03)
 ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
 ax.text(.01,.96,title,fontsize=18,fontweight='bold',va='top')
 ax.text(.01,.96-.38/height,'Single selected reference · Subject macro average · Higher is better',fontsize=10,color='#555',va='top')
 xs=[.02,.36,.46,.60,.76,.92]
 y=.96-.90/height
 ax.plot([.01,.99],[y,y],color='black',lw=1.5)
 y-=.27/height
 for x,t in zip(xs,['Method','IP scale','SSG','CLIP-T ↑','CLIP-I ↑','DINO-I ↑']):ax.text(x,y,t,fontsize=11,fontweight='bold',ha='left' if x==xs[0] else 'center',va='center')
 y-=.22/height;ax.plot([.01,.99],[y,y],color='black',lw=.8)
 for name,rs in groups:
  y-=.28/height
  ax.text(.02,y,name,fontsize=11,fontweight='bold',va='center')
  maxima={m:max(r[m] for r in rs) for m in M}
  for r in rs:
   y-=.245/height
   vals=[r['label']+(' †' if r['source']=='screenshot' else ''),r['scale'],r['ssg']]+[f'{r[m]:.4f}' for m in M]
   for j,(x,t) in enumerate(zip(xs,vals)):
    bold=j>=3 and r[M[j-3]]==maxima[M[j-3]]
    ax.text(x,y,t,fontsize=11,ha='left' if j==0 else 'center',va='center',fontweight='bold' if bold else 'normal')
  y-=.18/height;ax.plot([.01,.99],[y,y],color='#777',lw=.55)
 ax.text(.01,y-.22/height,'† Historical screenshot (4-decimal precision); other rows: verified local score JSON.',fontsize=9,va='top',color='#444')
 ax.text(.01,y-.43/height,'Baseline scale is unavailable in the screenshot. Scores are cosine values (no ×100). Bold: best per group.',fontsize=9,va='top',color='#444')
 return fig

def group(b,g):return [r for r in rows if r['benchmark']==b and r['group']==g]
sections=[('overall','Table 1. Subject fidelity and text alignment',[(f'DreamBench (750 images / method)',group('dreambench','overall')),(f'DreamBench++ (1,350 images / method)',group('dreambench_plus','overall'))]),
 ('dreambench_classes','Table 2. DreamBench results by reference class',[(f'{g.title()} ({N["dreambench"][g]:,} images / method)',group('dreambench',g)) for g in ('animal','object')]),
 ('dreambench_plus_classes','Table 3. DreamBench++ results by reference class',[(f'{g.replace("_","-").title()} ({N["dreambench_plus"][g]:,} images / method)',group('dreambench_plus',g)) for g in ('animal','human','object','style','non_style')])]
with PdfPages(OUT/'benchmark_tables.pdf') as pdf:
 for name,title,groups in sections:
  fig=table_figure(title,groups)
  fig.savefig(OUT/f'{name}.png',dpi=220,facecolor='white')
  fig.savefig(OUT/f'{name}.pdf',facecolor='white')
  fig.savefig(OUT/f'{name}.svg',facecolor='white')
  pdf.savefig(fig,facecolor='white');plt.close(fig)
# Booktabs LaTeX with grouped benchmark columns, suitable for a two-column table*.
tex=[r'\begin{table*}[t]',r'\centering',r'\caption{Text alignment and subject fidelity (subject macro average; higher is better). Historical values marked $\dagger$ are transcribed from a four-decimal screenshot.}',r'\label{tab:scale_ablation}',r'\begin{tabular}{lccrrrrrr}',r'\toprule',r'& & & \multicolumn{3}{c}{DreamBench (750)} & \multicolumn{3}{c}{DreamBench++ (1350)} \\',r'\cmidrule(lr){4-6}\cmidrule(lr){7-9}',r'Method & IP scale & SSG & CLIP-T & CLIP-I & DINO-I & CLIP-T & CLIP-I & DINO-I \\',r'\midrule']
for i,(key,label,scale,ssg,source) in enumerate(METHODS):
 cells=[label+(r'$\dagger$' if source=='screenshot' else ''),r'---' if scale=='—' else scale,ssg]
 for b in ('dreambench','dreambench_plus'):
  rs=group(b,'overall');r=rs[i]
  for m in M:
   value=f'{r[m]:.4f}'
   cells.append(r'\textbf{'+value+'}' if r[m]==max(q[m] for q in rs) else value)
 tex.append(' & '.join(cells)+r' \\')
tex.extend([r'\bottomrule',r'\end{tabular}',r'\end{table*}'])
(OUT/'overall.tex').write_text('\n'.join(tex)+'\n')
md=['# Benchmark comparison','', 'Four versions, with separate benchmark and class summaries.','', '| Benchmark | Method | Scale | SSG | Images | CLIP-T ↑ | CLIP-I ↑ | DINO-I ↑ |','|---|---|---|---|---:|---:|---:|---:|']
for b in ('dreambench','dreambench_plus'):
 for r in group(b,'overall'):md.append('| '+' | '.join([b,r['label']+(' †' if r['source']=='screenshot' else ''),r['scale'],r['ssg'],str(r['images']),*[f'{r[m]:.4f}' for m in M]])+' |')
md.extend(['','† User-provided historical screenshot: four-decimal rounded values; individual scores and runtime metadata are unavailable. Baseline scale is not shown in that screenshot.','', 'Local source: `/workspace/instantx-0807/results/full/metrics/scores.json`; A100-SXM4-80GB. Protocol: `uno-selected-reference-v1` (CLIP ViT-B/32; DINO v1 ViT-S/16; one selected reference). Historical screenshot states the same reference policy and aggregation, but full protocol/runtime equivalence cannot be independently checked without its original JSON. No significance tests are possible for screenshot-only rows.','', 'Ours scale 0.7 has the highest CLIP-T among the four versions on both benchmarks. Raising the gated method scale from 0.7 to 0.8 lowers CLIP-T and raises CLIP-I/DINO-I. Baseline retains the highest CLIP-I and DINO-I. The no-SSG row at scale 0.7 allows a same-scale comparison with gated scale 0.7; comparing scale 0.8 with no-SSG scale 0.7 changes both scale and gate.','', 'Artifacts: `benchmark_tables.pdf` (three pages), `overall.png`, `overall.pdf`, `overall.svg`, `overall.tex` (requires booktabs), class tables, and combined CSV/JSON.','', 'New scale sweep: `/workspace/instantx-0807/results/scale_sweep_20261025`, scales 0.6/0.9/1.0 with SSG ON, 2,100 images per scale, two GPUs. These pending results are not included in the completed comparison.'])
(OUT/'README.md').write_text('\n'.join(md)+'\n')
print(OUT)
