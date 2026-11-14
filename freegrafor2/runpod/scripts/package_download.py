"""Export exactly the scored 512px cohort, separating PNGs and metadata."""
import argparse
from collections import Counter
from datetime import datetime
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib
import json
import os
import shutil
import zipfile

from PIL import Image


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('/workspace/freegraftor'))
    parser.add_argument('--destination', type=Path, default=Path('/workspace/FreeGraftor_512'))
    parser.add_argument('--archive', type=Path, default=Path('/opt/freegraftor-downloads/FreeGraftor_512.zip'))
    args = parser.parse_args()
    source, dest = args.source, args.destination
    run = source/'results/seed42_512'
    scores = read(run/'metrics/scores.json')['scores']
    assert len(scores) == len({(s['benchmark'], s['case_id']) for s in scores}) == 1920
    assert Counter(s['benchmark'] for s in scores) == {'dreambench':750, 'dreambench_plus':1170}
    assert all(s['category'] != 'style' for s in scores)
    assert not dest.exists(), f'Export destination already exists: {dest}'
    folders = {'dreambench':'dreamBench', 'dreambench_plus':'dreamBench_plus'}
    for folder in [*folders.values(), 'execution_records']:
        (dest/folder).mkdir(parents=True, exist_ok=True)
    members = []
    for index, score in enumerate(scores, 1):
        bench, case = score['benchmark'], score['case_id']
        original = run/'images'/bench/(case+'.png')
        meta_path = original.with_suffix('.json')
        meta_bytes = meta_path.read_bytes()
        meta = json.loads(meta_bytes)
        data = original.read_bytes()
        checksum = hashlib.sha256(data).hexdigest()
        assert checksum == meta['png_sha256'] == score['generated_sha256']
        assert meta['row']['case_id'] == case and meta['row']['benchmark'] == bench
        assert meta['reference_sha256'] == score['reference_sha256']
        with Image.open(BytesIO(data)) as image:
            assert image.format == 'PNG' and image.size == (512,512)
            image.verify()
        image_relative = Path(folders[bench])/(case+'.png')
        meta_relative = Path('execution_records/image_metadata')/folders[bench]/(case+'.json')
        (dest/image_relative).write_bytes(data)
        (dest/meta_relative).parent.mkdir(parents=True, exist_ok=True)
        (dest/meta_relative).write_bytes(meta_bytes)
        members.append(dict(benchmark=bench, case_id=case, image=image_relative.as_posix(),
                            metadata=meta_relative.as_posix(), png_sha256=checksum,
                            metadata_sha256=hashlib.sha256(meta_bytes).hexdigest(),
                            reference_id=score['reference_id'],
                            mask_reprocessed=bool(meta.get('mask_recovery'))))
        if index % 100 == 0:
            print('EXPORTED',index,'/ 1920',flush=True)
    for name in ['scores.json','summary.json','summary.csv','comparison.json']:
        target = dest/'execution_records/metrics'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run/'metrics'/name,target)
    for name in ['run.json','status.json','coverage.json','evaluation-policy.json']:
        shutil.copy2(run/name,dest/'execution_records'/name)
    for name in ['manifest.json','references.json','assignment.json']:
        target = dest/'execution_records/inputs'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/'inputs'/name,target)
    for rel in ['PROVENANCE.json','recovery/non-style-coverage.json',
                'recovery/detector_aliases.json','recovery/PROTOCOL.md',
                'cache/flux-native/conversion.json','recovery/resume-20261114/provenance.json']:
        target = dest/'execution_records/source_records'/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/rel,target)
    write(dest/'execution_records/export_manifest.json',dict(
        exported_at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
        counts={'dreamBench':750,'dreamBench_plus':1170}, image_count=1920,
        source_root=str(source), cohort='final scored non-style cases',
        original_image_and_metadata_bytes_preserved=True,
        original_absolute_paths_retained_in_metadata=True,
        mask_audit_note='Historical mask image paths in original metadata refer to the source run; mask images are not included in this image-and-metadata export.',
        style_images_included=False, mask_reprocessed_cases=sum(m['mask_reprocessed'] for m in members),
        files=members))
    (dest/'execution_records/README.txt').write_text(
        'FreeGraftor 512x512, seed 42.\n'
        'dreamBench/: 750 original PNG images only.\n'
        'dreamBench_plus/: 1170 original non-style PNG images only.\n'
        'execution_records/image_metadata/: original per-image JSON files, matched by case ID.\n'
        'execution_records/metrics/: final CLIP-T, CLIP-I, DINO-I and matched InstantX comparison.\n'
        'All style cases are excluded. 36 cases used audited detector-label recovery.\n'
        'Original JSON files preserve source-machine paths and signed audit data.\n'
        'export_manifest.json maps the portable image/metadata paths and checksums.\n')
    for bench, expected in [('dreamBench',750),('dreamBench_plus',1170)]:
        files=list((dest/bench).iterdir())
        assert len(files)==expected and all(p.suffix=='.png' and p.is_file() for p in files)
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    partial = args.archive.with_suffix('.zip.part')
    with zipfile.ZipFile(partial,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as archive:
        for path in sorted(dest.rglob('*')):
            if path.is_file():
                archive.write(path,(Path(dest.name)/path.relative_to(dest)).as_posix())
    print('VERIFYING ARCHIVE',flush=True)
    with zipfile.ZipFile(partial) as archive:
        assert archive.testzip() is None
        pngs=[n for n in archive.namelist() if n.endswith('.png')]
        assert len(pngs)==1920
        for m in members:
            for field, hash_field in [('image','png_sha256'),('metadata','metadata_sha256')]:
                assert hashlib.sha256(archive.read(dest.name+'/'+m[field])).hexdigest()==m[hash_field]
    checksum=sha_file(partial)
    partial.replace(args.archive)
    args.archive.with_suffix('.zip.sha256').write_text(checksum+'  '+args.archive.name+'\n')
    pod=os.environ.get('RUNPOD_POD_ID')
    result=dict(status='complete',directory=str(dest),archive=str(args.archive),
                bytes=args.archive.stat().st_size,sha256=checksum,images=1920,metadata_files=1920,
                zip_crc_verified=True,all_image_and_metadata_sha256_verified=True,
                download_url=(f'https://{pod}-8888.proxy.runpod.net/files{args.archive}?download=1' if pod else None))
    write(source/'download_exports/export.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
