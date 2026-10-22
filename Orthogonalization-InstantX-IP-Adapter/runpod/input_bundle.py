"""Build/restore immutable benchmark inputs separately from Git-managed code."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = 'outputs/semantic_benchmark_gpt56_sol/generation_manifest.json'
LOCK = ROOT / 'runpod/input_bundle.json'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def names_for(rows):
    if (Counter(r['benchmark'] for r in rows) != {'dreambench': 750, 'dreambench_plus': 1350}
            or len({r['case_id'] for r in rows}) != 2100 or any(r['status'] != 'READY' for r in rows)):
        raise ValueError('Expected 2100 unique READY cases (750 + 1350)')
    names = {MANIFEST}
    for row in rows:
        for field, prefix in [('image_path', 'datasets/'),
                              ('conditioning_path', 'outputs/semantic_benchmark_gpt56_sol/conditioning/')]:
            name = row[field]
            path = PurePosixPath(name)
            if (not name.startswith(prefix) or path.is_absolute() or '..' in path.parts
                    or '\\' in name or ':' in name or str(path) != name):
                raise ValueError(f'Invalid input path: {name}')
            names.add(name)
    return names


def build(root=ROOT, lock=LOCK):
    root = Path(root).resolve()
    names = names_for(json.loads((root / MANIFEST).read_text(encoding='utf-8')))
    contents = {}
    for name in sorted(names):
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Input escapes project')
        contents[name] = path.read_bytes()
    index = {name: sha(data) for name, data in contents.items()}
    target = root / 'outputs/runpod_delivery/runpod_inputs.tar.gz'
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix('.tmp')
    # Stable tar/gzip metadata makes repeated builds byte-identical.
    with temp.open('wb') as raw, gzip.GzipFile(fileobj=raw, filename='', mode='wb', mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as archive:
            for name, data in {**contents, 'input_index.json': json.dumps(index, sort_keys=True).encode()}.items():
                member = tarfile.TarInfo(name)
                member.size = len(data)
                member.mode = 0o644
                archive.addfile(member, io.BytesIO(data))
    temp.replace(target)
    digest = sha(target.read_bytes())
    metadata = dict(schema_version=1, filename=target.name, sha256=digest,
                    bytes=target.stat().st_size, files=len(index), cases=2100)
    Path(lock).write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    target.with_name(target.name + '.sha256').write_text(digest + '  ' + target.name + '\n', encoding='ascii')
    print(json.dumps(metadata, indent=2))
    return target


def restore(archive_path, root=ROOT, lock=LOCK):
    root = Path(root).resolve()
    metadata = json.loads(Path(lock).read_text(encoding='utf-8'))
    archive_path = Path(archive_path)
    if archive_path.stat().st_size != metadata['bytes'] or sha(archive_path.read_bytes()) != metadata['sha256']:
        raise ValueError('Input bundle SHA256/size mismatch; use the bundle matching this Git revision')
    with tarfile.open(archive_path, 'r:gz') as archive:
        members = archive.getmembers()
        by_name = {m.name: m for m in members}
        if len(by_name) != len(members) or any(not m.isfile() for m in members):
            raise ValueError('Duplicate or non-regular archive member')
        rows = json.load(archive.extractfile(by_name[MANIFEST]))
        expected = names_for(rows)
        index = json.load(archive.extractfile(by_name['input_index.json']))
        if set(index) != expected or set(by_name) != expected | {'input_index.json'}:
            raise ValueError('Archive contains unexpected or missing inputs')
        # Validate every file and existing destination BEFORE writing anything.
        data_by_name = {}
        for name in sorted(expected):
            data = archive.extractfile(by_name[name]).read()
            if sha(data) != index[name]:
                raise ValueError(f'Input hash mismatch: {name}')
            target = (root / name).resolve()
            if not target.is_relative_to(root):
                raise ValueError('Input target escapes project')
            if target.exists() and target.read_bytes() != data:
                raise ValueError(f'Existing input differs; preserve/review it first: {name}')
            data_by_name[name] = data
        for name, data in data_by_name.items():
            target = root / name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                temp = target.with_suffix(target.suffix + '.input-tmp')
                temp.write_bytes(data)
                temp.replace(target)
    print(f'Restored/verified {len(expected)} files for 2100 cases. Code was not overwritten.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['build', 'restore'])
    parser.add_argument('archive', nargs='?', type=Path)
    args = parser.parse_args()
    if args.command == 'build':
        build()
    else:
        if args.archive is None:
            parser.error('restore requires archive path')
        restore(args.archive)


if __name__ == '__main__':
    main()
