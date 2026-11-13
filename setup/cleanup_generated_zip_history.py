"""Narrow image-ZIP-only history cleanup in an isolated bare repository."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'setup/image_zip_cleanup_20261113.json'
PATHS = ROOT / 'setup/image_zip_cleanup_paths_20261113.txt'
BARE = ROOT.parent / '_image_zip_history_cleanup_20261113_v2.git'
PATTERNS = [
    r'^(freecus|uno|dreamo|xverse)/artifacts/(dreambench|dreambench_plus)/[^/]+/images-\d+\.zip$',
    r'^Orthogonalization-InstantX-IP-Adapter/outputs/benchmark_images_v1/(baseline|source_target_ssg)/(dreambench|dreambench_plus)/images-\d+\.zip$',
]


def git(*args, cwd=ROOT):
    env = dict(os.environ, GIT_NO_LAZY_FETCH='1', GIT_LFS_SKIP_SMUDGE='1')
    return subprocess.check_output(['git', *args], cwd=cwd, env=env)


def selected(path):
    return any(re.fullmatch(pattern, path) for pattern in PATTERNS)


def plan():
    candidates = sorted(set(git('log', '--all', '--format=', '--name-only', '--', '*.zip').decode().splitlines()))
    paths = [p for p in candidates if selected(p)]
    if not paths:
        raise ValueError('No historical generated image ZIPs selected')
    keep = [p for p in candidates if p and p not in paths]
    assert all(p.endswith('.zip') and 'source.zip' not in p for p in paths)
    assert not any(p.startswith(('diptych/', 'mosaic/')) for p in paths)
    PATHS.write_text(''.join(p+'\n' for p in paths), encoding='utf-8')
    report = dict(scope='Only generated-image ZIPs for methods present in the shared Drive',
                  drive_folder='https://drive.google.com/drive/folders/1TGZGEdoiG_2uvCHQ9PfAmMor9qfMOTmT',
                  removed_paths=paths, preserved_zip_paths=keep, removed_path_count=len(paths),
                  local_modal_drive_originals_deleted=False,
                  validation_policy='Every rewritten commit must preserve every non-selected path, mode and blob ID')
    REPORT.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(selected=len(paths), preserved_zip_paths=len(keep)), indent=2))


def snapshot(commit, cwd):
    kept=[]
    for entry in git('ls-tree', '-r', '-z', commit, cwd=cwd).split(b'\0'):
        if not entry:
            continue
        _, name=entry.split(b'\t',1)
        if not selected(name.decode()):
            kept.append(entry)
    return hashlib.sha256(b'\0'.join(kept)).hexdigest()


def rewrite():
    if BARE.exists():
        raise ValueError('Cleanup workspace already exists; inspect rather than overwrite')
    paths=PATHS.read_text().splitlines()
    if paths!=json.loads(REPORT.read_text())['removed_paths'] or not all(selected(p) for p in paths):
        raise ValueError('Removal list differs from approved scope')
    source=git('rev-parse','HEAD').decode().strip()
    git('clone','--bare','--shared',str(ROOT),str(BARE))
    env=dict(os.environ,GIT_NO_LAZY_FETCH='1',GIT_LFS_SKIP_SMUDGE='1')
    def write_object(command,data):
        return subprocess.check_output(['git',*command],input=data,cwd=BARE,env=env).decode().strip()
    removed=set(paths)
    cache={}
    def tree(oid,prefix=''):
        key=(oid,prefix)
        if key in cache:return cache[key]
        original=git('ls-tree','-z',oid,cwd=BARE)
        entries=[]
        for entry in original.split(b'\0'):
            if not entry:continue
            info,name=entry.split(b'\t',1)
            mode,kind,child=info.split()
            full=prefix+name.decode()
            if full in removed:continue
            if kind==b'tree' and any(p.startswith(full+'/') for p in removed):
                updated=tree(child.decode(),full+'/')
                if updated=='4b825dc642cb6eb9a060e54bf8d69288fbee4904':continue
                info=b' '.join([mode,kind,updated.encode()])
            entries.append(info+b'\t'+name+b'\0')
        data=b''.join(entries)
        # --missing retains untouched promisor blobs without fetching their contents.
        result=oid if data==original else write_object(['mktree','-z','--missing'],data)
        cache[key]=result
        return result
    mappings=[]
    remap={}
    for old in git('rev-list','--reverse','--topo-order',source).decode().splitlines():
        raw=git('cat-file','commit',old)
        header,message=raw.split(b'\n\n',1)
        if b'\ngpgsig ' in header or b'\nmergetag ' in header:
            raise ValueError('Signed commit requires a separate explicit signature policy')
        lines=[]
        for line in header.split(b'\n'):
            if line.startswith(b'tree '):line=b'tree '+tree(line[5:].decode()).encode()
            elif line.startswith(b'parent '):line=b'parent '+remap[line[7:].decode()].encode()
            lines.append(line)
        updated=b'\n'.join(lines)+b'\n\n'+message
        new=old if updated==raw else write_object(['hash-object','-t','commit','-w','--stdin'],updated)
        remap[old]=new;mappings.append(old+' '+new)
    target=remap[source]
    git('update-ref','refs/heads/main',target,source,cwd=BARE)
    (BARE/'image-zip-commit-map.txt').write_text('old new\n'+'\n'.join(mappings)+'\n',encoding='ascii')
    validate()


def validate():
    paths=PATHS.read_text().splitlines()
    mappings=(BARE/'image-zip-commit-map.txt').read_text().splitlines()[1:]
    source=mappings[-1].split()[0]
    target=git('rev-parse','main',cwd=BARE).decode().strip()
    checked=0
    def metadata(commit,cwd):
        header,message=git('cat-file','commit',commit,cwd=cwd).split(b'\n\n',1)
        return b'\n'.join(line for line in header.split(b'\n') if not line.startswith((b'tree ',b'parent ')))+b'\n\n'+message
    for line in mappings:
        old,new=line.split()
        if set(new)=={'0'}:
            raise ValueError('A commit was unexpectedly removed')
        if snapshot(old,ROOT)!=snapshot(new,BARE):
            raise ValueError('Non-image contents changed at '+old)
        if metadata(old,ROOT)!=metadata(new,BARE):
            raise ValueError('Commit metadata changed at '+old)
        checked+=1
    remaining=git('log','main','--format=','--name-only','--','*.zip',cwd=BARE).decode().splitlines()
    if any(selected(p) for p in remaining):
        raise ValueError('Selected image ZIP remains in rewritten history')
    if git('rev-parse',source+'^{tree}')!=git('rev-parse',target+'^{tree}',cwd=BARE):
        raise ValueError('Latest tree changed; do not push or replace local refs')
    result=dict(source_head=source,cleaned_head=target,verified_commits=checked,
                non_image_paths_modes_blob_ids_unchanged=True,commit_metadata_unchanged=True,
                latest_tree_identical=True,selected_zip_paths_absent_from_history=True,
                cleanup_repository=str(BARE),removed_path_count=len(paths),phase='validated before push')
    dest=ROOT/'setup/image_zip_cleanup_validation_20261113.json'
    dest.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['plan','rewrite','validate'])
    args=parser.parse_args()
    {'plan':plan,'rewrite':rewrite,'validate':validate}[args.command]()
