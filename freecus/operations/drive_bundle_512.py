"""Repackage verified Modal archives into the shared Drive layout without resizing."""
import hashlib
import json
from pathlib import Path
import struct
import zipfile

from freecus.manifest import PROJECT
from freecus.operations.artifact_bundle import digest, verify
from freecus.run import write_json


def build(method, state, expected):
    plans = json.loads(state.read_text(encoding='utf-8'))
    destination = PROJECT.parent / 'AI_PROJECT_images/drive_upload_512'
    destination.mkdir(parents=True, exist_ok=True)
    final = destination / (method + '.zip')
    proof = destination / (method + '.verification.json')
    if final.exists():
        if proof.exists():
            previous = json.loads(proof.read_text(encoding='utf-8'))
            if previous['sha256'] == digest(final) and previous['verified']:
                return previous
        raise ValueError('Existing upload ZIP must be preserved: ' + str(final))
    partial = final.with_suffix('.zip.partial')
    if partial.exists():
        raise ValueError('Unfinished packaging exists; inspect before retry')
    members, images, sources, coverage = {}, {}, [], {}
    with zipfile.ZipFile(partial, 'w', compression=zipfile.ZIP_STORED) as output:
        def add(name, data):
            if name in members:
                raise ValueError('Duplicate output member: ' + name)
            output.writestr(name, data)
            members[name] = dict(sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))

        for plan in plans:
            benchmark, run_id = plan['benchmark'], plan['run_id']
            folder = PROJECT.parent / 'AI_PROJECT_images' / method.lower() / benchmark / run_id
            index = json.loads((folder / 'bundle_index.json').read_text(encoding='utf-8'))
            verify(folder, index)
            prefix = f'execution_records/{benchmark}/{run_id}/'
            manifest = Path(plan['local']) / 'manifest.jsonl'
            rows = [json.loads(s) for s in manifest.read_text(encoding='utf-8').splitlines() if s.strip()]
            by_id = {r['id']: r for r in rows}
            excluded = set(plan['identity'].get('excluded_case_ids', []))
            wanted = {r['case_id'] for r in rows} - excluded
            found = set()
            for part in index['parts']:
                with zipfile.ZipFile(folder / part['file']) as archive:
                    for entry in archive.infolist():
                        name = entry.filename
                        if name.startswith('tgt_image/') and name.endswith('.png'):
                            row = by_id[name[len('tgt_image/'):-4]]
                            case = row['case_id']
                            if case in found or case not in wanted:
                                raise ValueError('Duplicate or excluded image: ' + case)
                            data = archive.read(entry)
                            if data[:8] != b'\x89PNG\r\n\x1a\n' or struct.unpack('>II', data[16:24]) != (512, 512):
                                raise ValueError('Not a native-size 512 PNG: ' + name)
                            target = f'{benchmark}/{case}.png'
                            add(target, data)
                            if members[target]['sha256'] != index['members'][name]['sha256']:
                                raise ValueError('Repackaging changed image bytes')
                            images[target] = dict(case_id=case, original_member=name, run_id=run_id,
                                                  reference_id=row['reference_id'], prompt=row['prompt'],
                                                  sha256=members[target]['sha256'])
                            found.add(case)
                        elif not name.startswith(('input/', 'references/', 'src_image/', 'provenance/conditioning/')):
                            # Reference pixels stay in the original preservation bundle; IDs/hashes stay here.
                            add(prefix + name, archive.read(entry))
            if found != wanted:
                raise ValueError('Image coverage mismatch: ' + benchmark)
            if len(found) != expected[benchmark]:
                raise ValueError('Unexpected benchmark image count')
            coverage[benchmark] = dict(generated=len(found), excluded_case_ids=sorted(excluded), missing=[])
            add(prefix + 'original_bundle_index.json', json.dumps(index, ensure_ascii=False, indent=2).encode())
            local = Path(plan['local'])
            for file in sorted(local.rglob('*')):
                if file.is_file() and file.suffix in {'.json', '.jsonl', '.csv', '.txt'}:
                    add(prefix + 'evaluation_records/' + file.relative_to(local).as_posix(), file.read_bytes())
            sources.append(dict(benchmark=benchmark, run_id=run_id, archive_folder=str(folder), parts=index['parts']))
            print(method, benchmark, len(found), 'original PNGs repackaged', flush=True)
        if len(images) != sum(expected.values()):
            raise ValueError('Unexpected combined image count')
        record = dict(method=method, resolution=[512, 512], images=images, coverage=coverage,
                      source_bundles=sources, resized=False, image_bytes_unchanged=True,
                      reference_pixels='Preserved in original Modal bundles; reference IDs/hashes in manifest and metadata')
        add('execution_records/upload_manifest.json', json.dumps(record, ensure_ascii=False, indent=2).encode())
    with zipfile.ZipFile(partial) as archive:
        if set(archive.namelist()) != set(members):
            raise ValueError('Output member list mismatch')
        for entry in archive.infolist():
            h = hashlib.sha256()
            with archive.open(entry) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(block)
            if h.hexdigest() != members[entry.filename]['sha256']:
                raise ValueError('Output member SHA mismatch')
    partial.replace(final)
    report = dict(verified=True, archive=str(final), size_bytes=final.stat().st_size,
                  sha256=digest(final), images=len(images), resolution=[512, 512], coverage=coverage,
                  original_png_bytes_unchanged=True, member_sha256_and_crc_verified=True)
    write_json(proof, report)
    write_json(destination / (method + '.bundle_index.json'), dict(report, members=members, source_bundles=sources))
    for plan in plans:
        write_json(Path(plan['local']) / 'drive_upload_verification.json', report)
    return report


def main():
    for method, state, expected in [
        ('FreeCus_512', PROJECT / 'freecus/outputs/modal512-20261112/plans.json',
         dict(dreambench=750, dreambench_plus=1170)),
        ('UNO_512', PROJECT / 'uno/outputs/modal512-20261112-third/plans.json',
         dict(dreambench=750, dreambench_plus=1350)),
    ]:
        print(json.dumps(build(method, state, expected), ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
