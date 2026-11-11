from pathlib import Path
from collections import Counter
import json

r=Path('/workspace/freegraftor/results/seed42_512')
def read(p):
    try:return json.loads(p.read_text())
    except FileNotFoundError:return None
images={b:sum(1 for _ in (r/'images'/b).glob('*.png')) for b in ['dreambench','dreambench_plus']}
errors=Counter()
for p in (r/'errors').glob('*/*.json'):
    x=read(p)
    if x:errors[x['category']]+=1
print(json.dumps(dict(controller=read(r/'status.json'),pilot=read(r/'pilot-passed.json'),images=images,errors_by_category=dict(errors),lanes={p.name:read(p) for p in (r/'state').glob('*.json')}),ensure_ascii=False,indent=2))
