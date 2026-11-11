"""Export existing final PNGs and execution records without changing sources."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib, json, os, shutil, struct, time, zipfile

BASE = Path('/workspace/instantx-0807')
REPO = Path('/workspace/Ours/ThesisProject-T2I-DiT-Personalization')
PROJECT = REPO / 'Orthogonalization-InstantX-IP-Adapter'
OUT = Path('/opt/instantx-exports-20261111')
MANIFEST = PROJECT / 'outputs/semantic_benchmark_gpt56_sol/generation_manifest.json'
SPECS = [
    ('source_target_scale0.7', 'full/source_target', 'full', 'source_target', .7),
    ('source_target_ssg_scale0.8', 'full/source_target_ssg', 'full', 'source_target_ssg', .8),
    ('ssg_only_scale0.7', 'ssg_only_scale_0.7_20261025/ssg', 'ssg_only_scale_0.7_20261025', 'ssg', .7),
    *[(f'source_target_ssg_scale{s}', f'scale_sweep_20261025/scale_{s}/source_target_ssg', 'scale_sweep_20261025', f'method_scale_{s}', float(s)) for s in ['0.6','0.9','1.0']],
    *[(name.replace('_0.', '_scale0.') if name.startswith('baseline') else name+'_scale0.7', f'ablation_20261026/{name}/{name}', f'ablation_20261026/{name}', name, scale)
      for name, scale in [('baseline_0.8',.8),('baseline_0.9',.9),('source_only',.7),('target_only',.7)]]
]

def save(path, data):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    temp.replace(path)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def plan(spec):
    name, relative, run_relative, condition, scale = spec
    root, run = BASE/'results'/relative, BASE/'results'/run_relative
    rows = json.loads(MANIFEST.read_text())
    scores = json.loads((run/'metrics/scores.json').read_text())['scores']
    scores = {(x['benchmark'], x['case_id']): x for x in scores if x['condition'] == condition}
    expected = {(x['benchmark'], x['case_id']): x for x in rows}
    assert len(expected) == len(scores) == 2100, (name, len(scores))
    images, records = [], [(MANIFEST, 'execution_records/generation_manifest.json')]
    for bench, count in [('dreambench',750),('dreambench_plus',1350)]:
        pngs = sorted((root/bench).glob('*.png'))
        sidecars = sorted((root/bench).glob('*.json'))
        ids = {cid for b,cid in expected if b == bench}
        assert len(pngs) == len(sidecars) == count
        assert {p.stem for p in pngs} == {p.stem for p in sidecars} == ids
        for png in pngs:
            meta = png.with_suffix('.json')
            d = json.loads(meta.read_text()); row = d['row']; score = scores[(bench,png.stem)]
            assert row == expected[(bench,png.stem)], (name,png,'manifest mismatch')
            assert d['settings']['ip_adapter_scale'] == scale
            assert d['settings']['seed'] == 42
            assert score['generated_sha256'] == d['png_sha256']
            assert score['target_prompt'] == row['target_prompt']
            assert os.path.abspath(score['generated']) == os.path.abspath(png)
            images.append((png, f'{bench}/{png.name}', d['png_sha256']))
            records.append((meta, f'execution_records/images/{bench}/{meta.name}'))
    for p in sorted(run.iterdir()):
        if p.is_file() and p.suffix in {'.json','.log','.exit'}:
            records.append((p, 'execution_records/run/'+p.name))
    for p in sorted((run/'state').glob('*.json')):
        records.append((p, 'execution_records/run/state/'+p.name))
    assert len({arc for _,arc in records}) == len(records)
    return images, records

def package(spec, planned):
    name = spec[0]; final = OUT/(name+'.zip'); part = OUT/(name+'.zip.part')
    images, records = planned
    assert not final.exists(), final
    hashes = {}
    with zipfile.ZipFile(part, 'w', allowZip64=True) as z:
        for i, (p, arc, expected) in enumerate(images,1):
            before = p.stat(); h = hashlib.sha256()
            with p.open('rb') as src, z.open(arc, 'w', force_zip64=True) as dst:
                header = src.read(24)
                assert header[:8] == b'\x89PNG\r\n\x1a\n'
                assert struct.unpack('>II',header[16:24]) == (1024,1024)
                h.update(header); dst.write(header)
                for block in iter(lambda:src.read(1024*1024),b''):
                    h.update(block); dst.write(block)
            assert h.hexdigest() == expected, (p, 'SHA256 mismatch')
            after = p.stat()
            assert before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns
            hashes[arc] = expected
            if i % 300 == 0: print(name, i, '/2100', flush=True)
        for p, arc in records:
            data = p.read_bytes(); hashes[arc] = hashlib.sha256(data).hexdigest()
            z.writestr(arc, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=1)
    print(name, 'verifying all archived bytes', flush=True)
    with zipfile.ZipFile(part) as z:
        assert len(z.namelist()) == len(hashes) == len(set(z.namelist()))
        assert sum(n.startswith('dreambench/') for n in z.namelist()) == 750
        assert sum(n.startswith('dreambench_plus/') for n in z.namelist()) == 1350
        for n in z.namelist():
            assert n.startswith(('dreambench/','dreambench_plus/','execution_records/'))
            h = hashlib.sha256()
            with z.open(n) as f:
                for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
            assert h.hexdigest() == hashes[n], (name,n)
    checksum = sha(part); part.replace(final)
    result = dict(name=name, path=str(final), bytes=final.stat().st_size, sha256=checksum,
                  dreambench=750, dreambench_plus=1350, execution_records=len(records),
                  all_member_sha256_verified=True, metric_manifest_matched=True)
    save(OUT/(name+'.verification.json'), result)
    print('COMPLETE',name,round(result['bytes']/1024**3,3),'GiB',flush=True)
    return result

def main():
    plans = []
    for s in SPECS:
        plans.append(plan(s))
        print('MANIFEST_MATCHED',s[0],flush=True)
    size = sum(p.stat().st_size for images,records in plans for p,*_ in images+records)
    assert shutil.disk_usage(OUT).free > size+3*1024**3, 'Insufficient local disk'
    print('PREFLIGHT',len(SPECS),'conditions',2100*len(SPECS),'images',round(size/1024**3,3),'GiB input',flush=True)
    save(OUT/'status.json',dict(status='running',conditions=10,images=21000,source_bytes=size))
    results=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(package,s,p) for s,p in zip(SPECS,plans)]):
            results.append(f.result())
            save(OUT/'status.json',dict(status='running',complete=len(results),conditions=10))
    results.sort(key=lambda x:x['name'])
    save(OUT/'index.json',dict(status='complete',archives=results,total_images=21000,total_bytes=sum(r['bytes'] for r in results)))
    (OUT/'SHA256SUMS.txt').write_text(''.join(r['sha256']+'  '+r['name']+'.zip\n' for r in results))
    save(OUT/'status.json',dict(status='complete',conditions=10,images=21000))

if __name__ == '__main__':
    try: main()
    except Exception as e:
        save(OUT/'status.json',dict(status='failed',error=str(e)))
        raise
