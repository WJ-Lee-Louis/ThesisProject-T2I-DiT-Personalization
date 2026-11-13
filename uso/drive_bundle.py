"""Package existing USO outputs remotely, then download one lossless Drive ZIP."""
import argparse
import hashlib
import json
from pathlib import Path
import secrets
import shutil
import struct
import time
import urllib.request
import zipfile

from freecus.manifest import PROJECT
from freecus.modal_client import connect
from freecus.operations.artifact_bundle import digest
from freecus.run import write_json
from uso.experiment import STATE, PROFILE, RUNS, read

APP = 'uso1024-drive-export-s42-20261113'
REMOTE = '/work/drive_upload_1024'
DESTINATION = PROJECT.parent / 'AI_PROJECT_images/drive_upload_1024'
EXPECTED = dict(dreambench=750, dreambench_plus=1350)


def deploy(modal, token):
    volume = modal.Volume.from_name(RUNS)
    image = modal.Image.debian_slim(python_version='3.11').pip_install('fastapi==0.115.12')
    for name in ['__init__.py', 'manifest.py', 'run.py',
                 'operations/artifact_bundle.py', 'operations/drive_bundle_512.py']:
        image = image.add_local_file(PROJECT / 'freecus' / name, '/root/freecus/' + name)
    app = modal.App(APP)

    @app.function(name='package', serialized=True, image=image, volumes={'/work': volume},
                  cpu=2, memory=8192, timeout=7200, max_containers=1,
                  retries=0, scaledown_window=2)
    def package(plans, receipt):
        from freecus.operations.drive_bundle_512 import build
        from freecus.run import write_json
        def read(path):
            return json.loads(path.read_text(encoding='utf-8'))
        volume.reload()
        root = Path(REMOTE)
        prepared, sources = [], {}
        for plan in plans:
            run_id, benchmark = plan['run_id'], plan['benchmark']
            metric = Path('/work/metrics') / run_id
            coverage = read(metric / 'coverage.json')
            if (coverage['status'] != 'complete' or coverage['generated'] != EXPECTED[benchmark]
                    or coverage['missing_case_ids'] or coverage['excluded_case_ids']):
                raise ValueError('USO is not complete: ' + benchmark)
            source = Path('/work/exports') / run_id
            index = read(source / 'bundle_index.json')
            if (index['generation_resolution'] != [1024, 1024]
                    or index['generated_images'] != EXPECTED[benchmark]
                    or index['missing_case_ids']):
                raise ValueError('USO export identity differs')
            local = root / 'records' / benchmark / run_id
            local.mkdir(parents=True, exist_ok=True)
            for file in metric.iterdir():
                if file.is_file():
                    shutil.copy2(file, local / file.name)
            for name in ['manifest.jsonl', 'config.json', 'models.lock.json', 'identity.json']:
                shutil.copy2(Path('/work/bundles') / run_id / name, local / name)
            write_json(local / 'receipt.json', receipt)
            workflow = Path('/work/workflows') / (receipt['workflow_id'] + '.json')
            shutil.copy2(workflow, local / 'workflow-status.json')
            prepared.append(dict(plan, local=str(local)))
            sources[benchmark] = str(source)
        state = root / 'packaging_plans.json'
        write_json(state, prepared)
        result = build('USO_1024', state, EXPECTED, resolution=(1024, 1024),
                       destination=root, source_folders=sources)
        volume.commit()
        return result

    @app.function(name='download', serialized=True, image=image, volumes={'/work': volume},
                  secrets=[modal.Secret.from_dict({'DOWNLOAD_TOKEN': token})],
                  cpu=1, memory=1024, timeout=3600, max_containers=1, scaledown_window=2)
    @modal.asgi_app()
    def download():
        import hmac
        import os
        import re
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import StreamingResponse
        volume.reload()
        api = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

        @api.get('/files/{name}')
        def file(name: str, request: Request):
            if not hmac.compare_digest(request.headers.get('authorization', ''),
                                       'Bearer ' + os.environ['DOWNLOAD_TOKEN']):
                raise HTTPException(403)
            if name not in {'USO_1024.zip', 'USO_1024.verification.json', 'USO_1024.bundle_index.json'}:
                raise HTTPException(404)
            path = Path(REMOTE) / name
            if not path.is_file():
                raise HTTPException(404)
            size, offset = path.stat().st_size, 0
            end = size - 1
            ranged = False
            if requested := request.headers.get('range'):
                match = re.fullmatch(r'bytes=(\d+)-(\d*)', requested)
                if not match or int(match[1]) >= size:
                    raise HTTPException(416)
                offset = int(match[1])
                end = min(int(match[2]), end) if match[2] else end
                if end < offset:
                    raise HTTPException(416)
                ranged = True

            def chunks():
                with path.open('rb') as source:
                    source.seek(offset)
                    remaining = end - offset + 1
                    while remaining:
                        block = source.read(min(1024 * 1024, remaining))
                        if not block:
                            raise ValueError('Unexpected end of immutable archive')
                        remaining -= len(block)
                        yield block
            headers = {'Content-Length': str(end - offset + 1), 'Accept-Ranges': 'bytes'}
            if ranged:
                headers['Content-Range'] = f'bytes {offset}-{end}/{size}'
            return StreamingResponse(chunks(), status_code=206 if ranged else 200,
                                     media_type='application/octet-stream', headers=headers)
        return api

    with modal.enable_output():
        app.deploy()
    return modal.Function.from_name(APP, 'download').get_web_url()


def fetch(url, token, name, offset=0, end=None):
    headers = {'Authorization': 'Bearer ' + token}
    if offset or end is not None:
        headers['Range'] = f'bytes={offset}-' + (str(end) if end is not None else '')
    response = urllib.request.urlopen(urllib.request.Request(url + '/files/' + name, headers=headers), timeout=120)
    if (offset or end is not None) and response.status != 206:
        response.close()
        raise ValueError('Server did not honor resume')
    return response


def collect(url, token):
    DESTINATION.mkdir(parents=True, exist_ok=True)
    with fetch(url, token, 'USO_1024.bundle_index.json') as response:
        index = json.load(response)
    if not index['verified'] or index['images'] != 2100 or index['resolution'] != [1024, 1024]:
        raise ValueError('Unexpected remote upload bundle')
    final = DESTINATION / 'USO_1024.zip'
    partial = final.with_suffix('.zip.partial')
    if final.exists() and digest(final) != index['sha256']:
        raise ValueError('Existing final ZIP differs; refusing overwrite')
    if not final.exists():
        offset = partial.stat().st_size if partial.exists() else 0
        remaining = index['size_bytes'] - offset
        if remaining < 0 or shutil.disk_usage(DESTINATION).free < remaining + 256 * 1024**2:
            raise ValueError('Insufficient disk space for final ZIP plus safety margin')
        failures = 0
        while True:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset == index['size_bytes']:
                break
            end = min(offset + 64 * 1024**2, index['size_bytes']) - 1
            try:
                with fetch(url, token, final.name, offset, end) as response, partial.open('ab') as target:
                    checkpoint = offset // (100 * 1024**2)
                    while block := response.read(1024 * 1024):
                        target.write(block)
                        offset += len(block)
                        if offset // (100 * 1024**2) > checkpoint:
                            checkpoint = offset // (100 * 1024**2)
                            print(round(offset / 1024**2), 'MiB downloaded', flush=True)
                if offset != end + 1:
                    raise ValueError('Incomplete HTTP response')
                failures = 0
            except Exception as exc:
                failures += 1
                print('Transfer retry', failures, type(exc).__name__, flush=True)
                if failures == 12:
                    raise
                time.sleep(3)
        if digest(partial) != index['sha256']:
            raise ValueError('Downloaded ZIP SHA-256 mismatch')
        partial.replace(final)
    counts = {k: 0 for k in EXPECTED}
    plans = {p['benchmark']: p for p in read(STATE / 'plans.json')}
    with zipfile.ZipFile(final) as archive:
        if len(archive.infolist()) != len(index['members']) or set(archive.namelist()) != set(index['members']):
            raise ValueError('Archive membership differs')
        for entry in archive.infolist():
            h, header = hashlib.sha256(), b''
            with archive.open(entry) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    if not header:
                        header = block[:24]
                    h.update(block)
            expected = index['members'][entry.filename]
            if entry.file_size != expected['size_bytes'] or h.hexdigest() != expected['sha256']:
                raise ValueError('Archive member differs: ' + entry.filename)
            benchmark = entry.filename.split('/')[0]
            if benchmark in counts and entry.filename.endswith('.png'):
                if header[:8] != b'\x89PNG\r\n\x1a\n' or struct.unpack('>II', header[16:24]) != (1024, 1024):
                    raise ValueError('Not a native-size 1024 PNG')
                counts[benchmark] += 1
        if counts != EXPECTED:
            raise ValueError('Output counts differ')
        for benchmark, plan in plans.items():
            prefix = f"execution_records/{benchmark}/{plan['run_id']}/evaluation_records/"
            for name in archive.namelist():
                if name.startswith(prefix):
                    relative = Path(name[len(prefix):])
                    if relative.is_absolute() or '..' in relative.parts:
                        raise ValueError('Unsafe record path')
                    path = Path(plan['local']) / 'evaluation_records' / relative
                    data = archive.read(name)
                    if path.exists() and path.read_bytes() != data:
                        raise ValueError('Existing local record differs: ' + str(path))
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                    aliases = {'metrics.json', 'per_image.csv', 'coverage.json', 'summary.json',
                               'checkpoint.json', 'workflow-status.json', 'identity.json'}
                    if relative.as_posix() in aliases or relative.as_posix() == 'plan.json':
                        alias = Path(plan['local']) / ('evaluation_plan.json' if relative.as_posix() == 'plan.json'
                                                      else relative.name)
                        if alias.exists() and alias.read_bytes() != data:
                            raise ValueError('Existing local summary differs: ' + str(alias))
                        alias.write_bytes(data)
            original_index = json.loads(archive.read(
                f"execution_records/{benchmark}/{plan['run_id']}/original_bundle_index.json"))
            write_json(Path(plan['local']) / 'bundle_index.json', original_index)
    report = {k: v for k, v in index.items() if k not in {'members', 'source_bundles'}}
    report.update(archive=str(final), local_sha256_verified=True,
                  member_sha256_and_crc_verified=True, actual_png_dimensions_verified=True)
    write_json(DESTINATION / 'USO_1024.bundle_index.json', index)
    write_json(DESTINATION / 'USO_1024.verification.json', report)
    for plan in plans.values():
        write_json(Path(plan['local']) / 'drive_upload_verification.json', report)
    write_json(PROJECT / 'uso/results/reports/drive_upload_verification.json', report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['package', 'status', 'download', 'serve', 'stop'])
    args = parser.parse_args()
    modal = connect(PROFILE)
    auth_file = STATE / 'drive-download-auth.json'
    receipt_file = STATE / 'drive-packaging-receipt.json'
    if args.command == 'package':
        if receipt_file.exists():
            print(json.dumps(read(receipt_file), indent=2))
            return
        token = read(auth_file)['token'] if auth_file.exists() else secrets.token_urlsafe(32)
        write_json(auth_file, dict(token=token))
        url = deploy(modal, token)
        call = modal.Function.from_name(APP, 'package').spawn(read(STATE / 'plans.json'), read(STATE / 'receipt.json'))
        receipt = dict(call_id=call.object_id, app=APP, url=url)
        write_json(receipt_file, receipt)
        print(json.dumps(receipt, indent=2))
    elif args.command == 'status':
        receipt = read(receipt_file)
        try:
            result = modal.FunctionCall.from_id(receipt['call_id']).get(timeout=0)
        except TimeoutError:
            print(json.dumps(dict(status='packaging', call_id=receipt['call_id'])))
            return
        write_json(STATE / 'drive-packaging-status.json', result)
        print(json.dumps(result, indent=2))
    elif args.command == 'serve':
        receipt = read(receipt_file)
        receipt['url'] = deploy(modal, read(auth_file)['token'])
        write_json(receipt_file, receipt)
        print('Download endpoint ready; existing ZIP and packaging receipt preserved')
    elif args.command == 'download':
        collect(read(receipt_file)['url'], read(auth_file)['token'])
    else:
        receipt = read(receipt_file)
        app = modal.App.lookup(receipt['app'])
        app.stop()
        print('Temporary export app stopped')


if __name__ == '__main__':
    main()
