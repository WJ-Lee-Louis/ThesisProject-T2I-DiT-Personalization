from pathlib import Path
from collections import Counter, defaultdict
import hashlib, json, shutil

ROOT=Path('/workspace/freegraftor')
PROJECT=Path('/workspace/Ours/ThesisProject-T2I-DiT-Personalization/Orthogonalization-InstantX-IP-Adapter')
SOURCE=PROJECT/'outputs/semantic_benchmark_gpt56_sol/generation_manifest.json'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    rows=json.loads(SOURCE.read_text())
    assert Counter(r['benchmark'] for r in rows)=={'dreambench':750,'dreambench_plus':1350}
    assert len({r['case_id'] for r in rows})==2100
    refs={};groups=defaultdict(list)
    for row in rows:
        assert row['status']=='READY'
        groups[row['reference_id']].append(row)
        if row['reference_id'] not in refs:
            p=PROJECT/row['image_path'];target=ROOT/'inputs/references'/row['reference_id']/p.name
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists() or sha(target)!=sha(p):shutil.copyfile(p,target)
            assert sha(target)==sha(p)
            refs[row['reference_id']]=dict(path=str(target),sha256=sha(target),original_path=str(p),identity_term=row['identity_term'])
    lanes=[[],[]]
    for benchmark in ['dreambench','dreambench_plus']:
        for i,ref in enumerate(sorted(k for k,v in groups.items() if v[0]['benchmark']==benchmark)):
            lanes[i%2].extend(r['case_id'] for r in groups[ref])
    assert [len(l) for l in lanes]==[1050,1050]
    dest=ROOT/'inputs'
    for name,obj in [('manifest.json',rows),('references.json',refs),('assignment.json',lanes),('provenance.json',dict(source=str(SOURCE),source_sha256=sha(SOURCE),cases=2100,references=180,per_lane=1050,policy='same original target prompts, selected reference bytes and seed42 as historical InstantX; assign whole reference groups alternately per benchmark'))]:
        (dest/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2))
    print('Verified/copy inputs: 750 DreamBench + 1350 DreamBench++, 180 references; 1050 cases/GPU')

if __name__=='__main__':main()
