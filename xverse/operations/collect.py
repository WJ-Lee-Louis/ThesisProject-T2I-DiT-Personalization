"""Read-only Modal collection; verified native-resolution portable artifacts."""
import argparse
import asyncio
import json
import shutil
import zipfile
from pathlib import Path
from freecus.manifest import PROJECT
from freecus.operations.artifact_bundle import pack, verify, write, digest


def settings(workspace):
    if workspace in {'louislee','louislee-85468-dreambench'}:
        plan = json.loads((PROJECT/'xverse/outputs/plan.json').read_text())
        part = next(p for p in plan['plans'] if p['benchmark']=='dreambench')
        return ('method-comparison' if workspace.endswith('-dreambench') else plan['profile']), part['bundle_id'], 'dreambench', plan['source']
    plan=json.loads((PROJECT/'xverse/outputs/dbpp_split/plan.json').read_text())
    return plan['plans'][workspace]['profile'],plan['bundle_id'],'dreambench_plus',plan['source']


async def download(workspace):
    from freecus.modal_client import connect
    profile,bid,benchmark,source=settings(workspace)
    modal=connect(profile)
    volume=modal.Volume.from_name('xverse-runs')
    destination=PROJECT/'xverse/outputs/collected'/workspace
    semaphore=asyncio.Semaphore(32)
    entries=[]
    async for entry in volume.iterdir.aio(f'results/{bid}/A100-80GB',recursive=True):
        if entry.type.name=='FILE': entries.append(entry)
    count=0
    def target_for(entry):
        relative=Path(entry.path).relative_to(f'results/{bid}/A100-80GB')
        return destination/(workspace+'-'+relative.parts[0])/Path(*relative.parts[1:])
    async def one(entry):
        nonlocal count
        target=target_for(entry)
        async with semaphore:
            if not target.exists() or target.stat().st_size!=entry.size:
                target.parent.mkdir(parents=True,exist_ok=True)
                temp=target.with_suffix(target.suffix+'.download')
                with temp.open('wb') as stream:
                    await volume.read_file_into_fileobj.aio(entry.path,stream)
                if temp.stat().st_size!=entry.size:raise ValueError('Download size differs')
                temp.replace(target)
            count+=1
            if count%400==0: print(workspace,count,'/',len(entries),'files',flush=True)
    print(workspace,'downloading',len(entries),'files; no remote compute',flush=True)
    # Fetch identities before binary data, then transfer identical src PNGs once.
    small=[e for e in entries if '/src_image/' not in e.path and '/tgt_image/' not in e.path]
    await asyncio.gather(*(one(e) for e in small))
    groups={}
    for e in entries:
        if '/src_image/' not in e.path:continue
        target=target_for(e)
        meta=target.parents[2]/'metadata'/target.parent.name/(target.stem+'.json')
        key=json.loads(meta.read_text())['artifact_sha256']['src_image'] if meta.exists() else e.path
        groups.setdefault(key,[]).append(e)
    async def source_group(key,group):
        existing=next((target_for(e) for e in group if target_for(e).exists()
            and digest(target_for(e))==key),None)
        if existing is None:
            await one(group[0]);existing=target_for(group[0])
            if len(key)==64 and digest(existing)!=key:raise ValueError('Source image changed')
        for e in group:
            target=target_for(e)
            if target==existing:continue
            if target.exists():
                if digest(target)!=key:raise ValueError('Local source differs')
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                target.hardlink_to(existing)
    await asyncio.gather(*(source_group(k,g) for k,g in groups.items()),
        *(one(e) for e in entries if '/tgt_image/' in e.path))
    write(destination/'download.json',dict(workspace=workspace,profile=profile,bundle_id=bid,
        files=len(entries),processed_downloads=count))
    print(workspace,'download complete',flush=True)


def bundle(benchmark):
    recovered=benchmark=='dreambench_complete'
    if recovered:benchmark='dreambench'
    owners=['louislee'] if benchmark=='dreambench' else ['wj-lee-louis','woojin716','louislee-85468']
    if recovered:owners.append('louislee-85468-dreambench')
    _,bid,_,source=settings(owners[0])
    if recovered:bid+='-complete-v1'
    lanes=[p for owner in owners for p in (PROJECT/'xverse/outputs/collected'/owner).glob(owner+'-lane-*')]
    expected=sum(len(list((p/'metadata').rglob('*.json'))) for p in lanes)
    archives=PROJECT/'xverse/artifacts'/benchmark/bid
    result=PROJECT/'xverse/results'/benchmark/bid
    manifest=PROJECT/'xverse/manifests'/(benchmark+'_seed42.jsonl')
    info=pack(lanes,manifest,PROJECT,archives,'xverse',benchmark,bid,expected,resolution=(768,768),
        allow_gpu_variants=benchmark=='dreambench_plus' or recovered)
    if recovered and (info['generated_images']!=750 or info['missing_case_ids']):raise ValueError('Recovery is incomplete')
    verify(archives,info)
    result.mkdir(parents=True,exist_ok=True)
    write(result/'bundle_index.json',info)
    shutil.copyfile(manifest,result/'manifest.jsonl')
    target=result/'generation_source.zip'
    if not target.exists():
        with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for p in sorted(Path(source).rglob('*')):
                if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(source).as_posix())
    write(result/'provenance.json',dict(method='xverse',benchmark=benchmark,bundle_id=bid,
        workspaces=[o.removesuffix('-dreambench') for o in owners],generation_resolution=[768,768],source_archive_sha256=digest(target),
        manifest_sha256=digest(manifest),generated_images=expected,missing_case_ids=info['missing_case_ids']))
    write(result/'verification.json',dict(status='verified',generated_images=expected,
        zip_files=len(info['parts']),total_archive_bytes=sum(p['size_bytes'] for p in info['parts']),
        all_member_sha256_verified=True,source_reference_paths_recoverable=True))
    completed={json.loads(p.read_text())['id'] for lane in lanes for p in (lane/'metadata').rglob('*.json')}
    orphaned=[(lane,p) for lane in lanes for kind in ['tgt_image','src_image','text']
        for p in (lane/kind).rglob('*') if p.is_file() and p.suffix!='.download'
        and p.relative_to(lane/kind).with_suffix('').as_posix() not in completed]
    if orphaned:
        target=result/'incomplete_artifacts.zip'
        if not target.exists():
            with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_STORED) as z:
                for lane,p in orphaned:z.write(p,lane.name+'/'+p.relative_to(lane).as_posix())
        write(result/'incomplete_artifacts.json',dict(status='excluded-from-metrics',archive_sha256=digest(target),
            files=[dict(path=lane.name+'/'+p.relative_to(lane).as_posix(),sha256=digest(p),bytes=p.stat().st_size)
                for lane,p in orphaned]))
    print(json.dumps(dict(benchmark=benchmark,generated=expected,missing=len(info['missing_case_ids']),
        zip_files=len(info['parts']),gib=sum(p['size_bytes'] for p in info['parts'])/2**30)),flush=True)
    if recovered:
        recovery=PROJECT/'xverse/outputs/dreambench_recovery'
        write(result/'recovery.json',dict(original_complete_cases=733,recovered_cases=17,
            receipt=json.loads((recovery/'receipt.json').read_text()),
            result=json.loads((recovery/'status.json').read_text())))
        write(recovery/'complete_bundle.json',dict(bundle_id=bid))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['download','bundle'])
    parser.add_argument('name')
    args=parser.parse_args()
    if args.command=='download':asyncio.run(download(args.name))
    else:bundle(args.name)
