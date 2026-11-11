"""Four independently verified, uncompressed ZIP downloads; no source writes."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json,os,shutil,stat,time,zipfile

BASE=Path('/workspace/instantx-0807')
OUT=Path('/opt/instantx-downloads')
META=BASE/'download_exports'
EXPORTS=[
    ('target_only_scale0.7','target_only',BASE/'results/ablation_20261026/target_only/target_only',.7),
    ('source_target_scale0.7','source_target',BASE/'results/full/source_target',.7),
    ('ssg_only_scale0.7','ssg_only',BASE/'results/ssg_only_scale_0.7_20261025/ssg',.7),
    ('source_target_ssg_scale0.8','source_target_ssg',BASE/'results/full/source_target_ssg',.8),
]

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));temp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def inputs(root):
    files=[];counts={}
    for bench,n in [('dreambench',750),('dreambench_plus',1350)]:
        entries=sorted(p for p in (root/bench).iterdir() if p.is_file() and p.suffix in ('.png','.json'))
        png=[p for p in entries if p.suffix=='.png'];meta=[p for p in entries if p.suffix=='.json']
        assert len(png)==len(meta)==n,(root,bench,len(png),len(meta))
        assert {p.stem for p in png}=={p.stem for p in meta}
        counts[bench]=dict(png=len(png),json=len(meta));files.extend(entries)
    return files,counts

def package(spec):
    name,label,root,scale=spec
    progress=META/(name+'.progress.json')
    final=OUT/(name+'.zip');part=OUT/(name+'.zip.part')
    record_path=META/(name+'.json')
    if final.exists() and record_path.exists():
        result=json.loads(record_path.read_text())
        assert sha(final)==result['sha256'],'Existing archive checksum changed'
        return result
    files,counts=inputs(root);size=sum(p.stat().st_size for p in files)
    write(progress,dict(phase='packaging',files=0,total_files=len(files),bytes=0,total_bytes=size))
    manifest=[];processed=0
    with zipfile.ZipFile(part,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as archive:
        for i,path in enumerate(files,1):
            before=path.stat();relative=Path(label)/path.relative_to(root)
            info=zipfile.ZipInfo(relative.as_posix(),time.localtime(before.st_mtime)[:6]);info.compress_type=zipfile.ZIP_STORED
            info.file_size=before.st_size;info.create_system=3;info.external_attr=(stat.S_IFREG|0o644)<<16
            h=hashlib.sha256();written=0
            with path.open('rb') as source,archive.open(info,'w') as destination:
                for block in iter(lambda:source.read(1024*1024),b''):
                    destination.write(block);h.update(block);written+=len(block)
            after=path.stat()
            assert written==before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns,'Source changed during export'
            manifest.append(dict(path=relative.as_posix(),bytes=written,sha256=h.hexdigest()))
            processed+=written
            if i%50==0 or i==len(files):write(progress,dict(phase='packaging',files=i,total_files=len(files),bytes=processed,total_bytes=size))
        metadata=dict(method=label,ip_adapter_scale=scale,source_root=str(root),counts=counts,compression='ZIP_STORED',original_png_and_json_bytes_preserved=True,files=manifest)
        archive.writestr(label+'/export_manifest.json',json.dumps(metadata,ensure_ascii=False,indent=2))
        archive.writestr(label+'/README.txt',f'{label}; IP-Adapter scale {scale}\nDreamBench: 750 PNG + JSON; DreamBench++: 1350 PNG + JSON.\nOriginal 1024x1024 PNG and generation JSON files are preserved without recompression.\nRead each original JSON for the exact effective generation settings and any fallback flags.\nexport_manifest.json contains every source member SHA256.\n')
    write(progress,dict(phase='verifying_zip_crc',files=len(files),total_files=len(files),bytes=processed,total_bytes=size))
    with zipfile.ZipFile(part) as archive:
        assert archive.testzip() is None,'ZIP CRC integrity check failed'
        assert len([i for i in archive.infolist() if i.filename.endswith('.png')])==2100
        assert len(archive.infolist())==4202
    checksum=sha(part);part.replace(final)
    result=dict(name=name,method=label,scale=scale,path=str(final),workspace_path=str(BASE/'downloads'/final.name),bytes=final.stat().st_size,GiB=round(final.stat().st_size/1024**3,3),sha256=checksum,images=2100,metadata_files=2100,counts=counts,zip_crc_verified=True,source_member_sha256=True,compression='ZIP_STORED',download_url=f'https://{os.environ["RUNPOD_POD_ID"]}-8888.proxy.runpod.net/files/opt/instantx-downloads/{final.name}?download=1')
    write(record_path,result);(OUT/(final.name+'.sha256')).write_text(checksum+'  '+final.name+'\n')
    write(progress,dict(phase='complete',**result));print('READY',final.name,result['GiB'],'GiB',flush=True)
    return result

def main():
    OUT.mkdir(parents=True,exist_ok=True);META.mkdir(parents=True,exist_ok=True)
    required=sum(p.stat().st_size for _,_,root,_ in EXPORTS for p in inputs(root)[0])
    assert shutil.disk_usage(OUT).free>required+2*1024**3,'Require at least 2GiB remaining local SSD headroom'
    write(META/'status.json',dict(status='running',archives=4,source_only_excluded=True,created_at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat()))
    results=[]
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(package,spec) for spec in EXPORTS]
            for future in as_completed(futures):results.append(future.result())
        results.sort(key=lambda r:r['name'])
        index=dict(status='complete',source_only_excluded=True,archives=results,total_images=8400,total_bytes=sum(r['bytes'] for r in results),local_ssd_storage=True,original_results_unchanged=True,completed_at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat())
        write(META/'index.json',index);write(OUT/'index.json',index)
        (OUT/'SHA256SUMS.txt').write_text(''.join(r['sha256']+'  '+Path(r['path']).name+'\n' for r in results))
        instructions='Download the four ZIP files after logging in to the existing RunPod Jupyter session.\nHTTPS download links and SHA256 values are in index.json.\nEach ZIP contains 2100 original PNG files, 2100 original JSON files, a per-member SHA256 manifest and README.\nZIP_STORED avoids expensive PNG recompression.\nThese ZIP copies live on this Pod local SSD and are temporary: download before stopping/replacing the Pod. Original result folders remain on /workspace.\nOptional local PowerShell: scp -r YOUR_EXISTING_SSH_HOST_ALIAS:/opt/instantx-downloads ./instantx_downloads\nUse your existing SSH host alias/key configuration; scp displays per-file transfer rate/progress.\n'
        (OUT/'DOWNLOAD.txt').write_text(instructions)
        write(META/'status.json',dict(status='complete',archives=4,total_images=8400,total_bytes=index['total_bytes']))
    except Exception as e:
        write(META/'status.json',dict(status='failed',error=str(e)));raise

if __name__=='__main__':main()
