from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path('/workspace/instantx-0807/reports/benchmark_comparison_20261025')
data=json.loads((out/'combined_metrics.json').read_text())
previous={r['method']:r for r in data['rows'] if r['benchmark']=='dreambench' and r['group']=='overall'}
metrics=('CLIP-I','CLIP-T','DINO-I')
rows=[]
for key,label in [('baseline','InstantX-IP-Adapter (baseline)'),('method_0.7','Ours - scale 0.7'),('no_ssg_0.7','Ours w/o SSG - scale 0.7'),('method_0.8','Ours - scale 0.8')]:
    rows.append((label,tuple(previous[key][m] for m in metrics)))
rows.extend([('FreeCus',(.8058,.3004,.6708)),('UNO*',(.8356,.3039,.7604)),('USO*',(.8301,.3178,.7686))])
assert len(rows)==7
plt.rcParams.update({'font.family':'DejaVu Serif','mathtext.fontset':'dejavuserif'})
fig,ax=plt.subplots(figsize=(12.5,5.2))
fig.subplots_adjust(left=.035,right=.965,top=.95,bottom=.04)
ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
ax.text(.01,.98,'DreamBench',fontsize=19,fontweight='bold',va='top')
xs=(.02,.60,.77,.94)
ax.plot([.01,.99],[.855,.855],color='black',lw=1.5)
for x,label in zip(xs,('Method','CLIP-I ↑','CLIP-T ↑','DINO-I ↑')):
    ax.text(x,.805,label,ha='left' if x==xs[0] else 'center',va='center',fontsize=12,fontweight='bold')
ax.plot([.01,.99],[.757,.757],color='black',lw=.8)
maxima=tuple(max(r[1][i] for r in rows) for i in range(3))
for index,(label,values) in enumerate(rows):
    y=.695-index*.082
    ax.text(xs[0],y,label,ha='left',va='center',fontsize=12)
    for i,value in enumerate(values):
        ax.text(xs[i+1],y,f'{value:.4f}',ha='center',va='center',fontsize=12,
                fontweight='normal')
ax.plot([.01,.99],[.145,.145],color='black',lw=1.5)
ax.text(.02,.085,'* tuning-based',fontsize=11,va='center')
path=out/'dreambench_method_comparison.png'
fig.savefig(path,dpi=300,facecolor='white')
plt.close(fig)
print(path)
print(rows)
