"""Validate completed USO records and inspect the public Drive ZIP without downloading images."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

from freecus.manifest import PROJECT
from freecus.run import write_json

FILE_ID = '1kKre7W5IRzoij9pIaozEFLQ9NwmqSIcQ'
URL = 'https://drive.usercontent.google.com/download?id=' + FILE_ID + '&export=download&confirm=t'
FOLDER = 'https://drive.google.com/drive/folders/1TGZGEdoiG_2uvCHQ9PfAmMor9qfMOTmT'


class RemoteArchive(io.RawIOBase):
    def __init__(self, size):
        self.size, self.position, self.transferred = size, 0, 0

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if not 0 <= position <= self.size:
            raise ValueError('Invalid remote ZIP seek')
        self.position = position
        return position

    def read(self, size=-1):
        size = min(self.size - self.position, size if size >= 0 else self.size - self.position)
        if size == 0:
            return b''
        if size > 16 * 1024**2:
            raise ValueError('Refusing large image/archive download during metadata verification')
        start = self.position
        request = urllib.request.Request(URL, headers={'Range': f'bytes={start}-{start + size - 1}'})
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{start + size - 1}/{self.size}':
                raise ValueError('Drive did not honor exact ZIP byte range/size')
            data = response.read(size + 1)
        if len(data) != size:
            raise ValueError('Incomplete Drive range response')
        self.position += size
        self.transferred += size
        return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checked-date', required=True)
    args = parser.parse_args()
    records = PROJECT / 'uso/results'
    upload = json.loads((records / 'reports/drive_upload_verification.json').read_text(encoding='utf-8'))
    index_path = PROJECT.parent / 'AI_PROJECT_images/drive_upload_1024/USO_1024.bundle_index.json'
    index = json.loads(index_path.read_text(encoding='utf-8'))
    if (not upload['local_sha256_verified'] or not upload['actual_png_dimensions_verified']
            or upload['sha256'] != index['sha256'] or upload['size_bytes'] != index['size_bytes']):
        raise ValueError('Local verified upload records differ')
    remote = RemoteArchive(index['size_bytes'])
    with zipfile.ZipFile(remote) as archive:
        if len(archive.infolist()) != len(index['members']) or set(archive.namelist()) != set(index['members']):
            raise ValueError('Drive ZIP entry list differs')
        for entry in archive.infolist():
            if entry.file_size != index['members'][entry.filename]['size_bytes']:
                raise ValueError('Drive ZIP entry size differs')
        raw = archive.read('execution_records/upload_manifest.json')
        if hashlib.sha256(raw).hexdigest() != index['members']['execution_records/upload_manifest.json']['sha256']:
            raise ValueError('Drive upload manifest SHA-256 differs')
        manifest = json.loads(raw)
        images = manifest['images']
        if len(images) != 2100 or manifest['resolution'] != [1024, 1024]:
            raise ValueError('Drive image count/resolution record differs')
        for name, image in images.items():
            if image['sha256'] != index['members'][name]['sha256']:
                raise ValueError('Drive image identity hash differs')
    proof = dict(folder=FOLDER, checked_date=args.checked_date,
                 archives=[dict(file='USO_1024.zip', file_id=FILE_ID,
                                url=f'https://drive.google.com/file/d/{FILE_ID}/view',
                                size_bytes=index['size_bytes'], images=2100, local_sha256=index['sha256'],
                                verified=True, remote_check='Exact byte length, complete ZIP entry names and sizes, '
                                'upload-manifest SHA-256 and all recorded image hashes match the verified local ZIP',
                                metadata_bytes_transferred=remote.transferred, full_remote_sha256_computed=False)])
    summaries, rows = {}, []
    for benchmark, expected in [('dreambench', 750), ('dreambench_plus', 1350)]:
        runs = list((records / benchmark).glob('uso1024-*-s42-*'))
        if len(runs) != 1:
            raise ValueError('Expected exactly one canonical completed USO run')
        run = runs[0]
        result = json.loads((run / 'metrics.json').read_text(encoding='utf-8'))
        coverage = json.loads((run / 'coverage.json').read_text(encoding='utf-8'))
        original = [json.loads(line) for line in (run / 'manifest.jsonl').read_text(encoding='utf-8').splitlines() if line]
        scored = result['scores']
        if (coverage['status'] != 'complete' or coverage['generated'] != expected
                or coverage['missing_case_ids'] or coverage['excluded_case_ids']
                or len(scored) != expected or len({r['case_id'] for r in scored}) != expected
                or {r['case_id'] for r in original} != {r['case_id'] for r in scored}):
            raise ValueError('Incomplete or mixed metric coverage: ' + benchmark)
        canonical = PROJECT / f'freecus/manifests/{benchmark}_project_seed42.jsonl'
        if canonical.read_bytes() != (run / 'manifest.jsonl').read_bytes():
            raise ValueError('Manifest is not the canonical seed-42 cohort')
        summary = result['summary'][benchmark + '/uso']
        summaries[benchmark] = dict(run_id=run.name, coverage=coverage, summary=summary)
        rows.append(dict(method='USO', benchmark=benchmark, scope='full', images=expected,
                         resolution=1024, **summary['subject_macro']))
        if benchmark == 'dreambench_plus':
            non_style = summary['non_style']
            rows.append(dict(method='USO', benchmark=benchmark, scope='non-style',
                             images=non_style['cases'], resolution=1024, **non_style['subject_macro']))
        write_json(run / 'google_drive_verification.json', proof)
    write_json(records / 'reports/google_drive_verification.json', proof)
    write_json(records / 'reports/summary.json', dict(method='USO', resolution=[1024, 1024],
               reference_policy='one selected canonical reference', target_prompt='original',
               benchmarks=summaries, drive_folder=FOLDER, archive_sha256=index['sha256']))
    with (records / 'reports/summary.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['method', 'benchmark', 'scope', 'images', 'resolution',
                                                    'CLIP-I', 'CLIP-T', 'DINO-I'])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(remote_verified=True, images=2100, metadata_bytes=remote.transferred,
                          metric_rows=rows), indent=2))


if __name__ == '__main__':
    main()
