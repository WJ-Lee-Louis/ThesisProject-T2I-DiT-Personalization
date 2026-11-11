"""Package existing outputs; count/name/ZIP-directory checks only, as requested."""
from package_experiments import BASE, OUT, MANIFEST, SPECS, save
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json, shutil, zipfile

ROWS=json.loads(MANIFEST.read_text())

def package(spec):
    name, relative, run_relative, condition, scale = spec
    root=BASE/'results'/relative;run=BASE/'results'/run_relative
    final=OUT/(name+'.zip');part=OUT/(name+'.zip.part')
    assert not final.exists(), final
    members=[]
    for bench,count in [('dreambench',750),('dreambench_plus',1350)]:
        pngs=sorted((root/bench).glob('*.png'))
        metas=sorted((root/bench).glob('*.json'))
        expected={r['case_id'] for r in ROWS if r['benchmark']==bench}
        assert len(pngs)==len(metas)==count
        assert {p.stem for p in pngs}=={p.stem for p in metas}==expected
        members.extend((p,f'{bench}/{p.name}') for p in pngs)
        members.extend((p,f'execution_records/images/{bench}/{p.name}') for p in metas)
    members.append((MANIFEST,'execution_records/generation_manifest.json'))
    for p in sorted(run.iterdir()):
        if p.is_file() and p.suffix in {'.json','.log','.exit'}:
            members.append((p,'execution_records/run/'+p.name))
    members.extend((p,'execution_records/run/state/'+p.name) for p in sorted((run/'state').glob('*.json')))
    assert len({arc for p,arc in members})==len(members)
    print('PACKAGING',name,flush=True)
    with zipfile.ZipFile(part,'w',allowZip64=True) as z:
        for i,(p,arc) in enumerate(members,1):
            z.writestr(arc,p.read_bytes(),compress_type=zipfile.ZIP_STORED if p.suffix=='.png' else zipfile.ZIP_DEFLATED,compresslevel=1)
            if i%750==0:print(name,i,'/',len(members),'files',flush=True)
    with zipfile.ZipFile(part) as z:
        names=z.namelist()
        assert len(names)==len(members)==len(set(names))
        assert sum(n.startswith('dreambench/') for n in names)==750
        assert sum(n.startswith('dreambench_plus/') for n in names)==1350
        assert all(n.startswith(('dreambench/','dreambench_plus/','execution_records/')) for n in names)
    part.replace(final)
    result=dict(name=name,path=str(final),bytes=final.stat().st_size,dreambench=750,dreambench_plus=1350,
                execution_records=len(members)-2100,checks='counts, case-id filenames, ZIP central directory; full content verification skipped at user request')
    save(OUT/(name+'.verification.json'),result)
    print('COMPLETE',name,round(result['bytes']/1024**3,3),'GiB',flush=True)
    return result

def main():
    assert shutil.disk_usage(OUT).free>28*1024**3,'Require local disk headroom'
    save(OUT/'status.json',dict(status='running',mode='fast',conditions=10,images=21000))
    results=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(package,s) for s in SPECS]):
            results.append(future.result())
            save(OUT/'status.json',dict(status='running',mode='fast',complete=len(results),conditions=10))
    results.sort(key=lambda x:x['name'])
    save(OUT/'index.json',dict(status='complete',mode='fast',archives=results,total_images=21000,total_bytes=sum(r['bytes'] for r in results)))
    save(OUT/'status.json',dict(status='complete',mode='fast',conditions=10,images=21000))

if __name__=='__main__':
    try:main()
    except Exception as e:
        save(OUT/'status.json',dict(status='failed',error=str(e)))
        raise
