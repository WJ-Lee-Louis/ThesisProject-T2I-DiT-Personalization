"""Authenticated, resumable HTTP fallback for stalled Modal Volume downloads."""
import json
import secrets
import time
import urllib.request
from pathlib import Path

from freecus.modal_client import connect
from freecus.operations.artifact_bundle import digest, verify
from freecus.run import write_json
from freecus.manifest import PROJECT


def main():
    modal = connect('third')
    state = PROJECT / 'uno/outputs/modal512-20261112-third'
    plans = json.loads((state / 'plans.json').read_text(encoding='utf-8'))
    run_ids = [p['run_id'] for p in plans]
    auth_file = state / 'download-auth.json'
    if auth_file.exists():
        token = json.loads(auth_file.read_text())['token']
    else:
        token = secrets.token_urlsafe(32)
        write_json(auth_file, dict(token=token))
    volume = modal.Volume.from_name('uno-512-seed42-20261112-runs-third')
    app = modal.App('uno512-authenticated-download-third-20261113')
    image = modal.Image.debian_slim(python_version='3.11').pip_install('fastapi==0.115.12')

    @app.function(name='download', serialized=True, image=image, volumes={'/work': volume},
                  secrets=[modal.Secret.from_dict({'DOWNLOAD_TOKEN': token})],
                  timeout=3600, cpu=(1, 1), memory=(1024, 1024),
                  max_containers=1, scaledown_window=2)
    @modal.asgi_app()
    def download():
        import hmac
        import os
        import re
        from pathlib import Path
        from fastapi import FastAPI, Request, HTTPException
        from fastapi.responses import StreamingResponse
        api = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

        @api.get('/archive/{run_id}/{filename}')
        def archive(run_id: str, filename: str, request: Request):
            if not hmac.compare_digest(request.headers.get('authorization', ''), 'Bearer ' + os.environ['DOWNLOAD_TOKEN']):
                raise HTTPException(403)
            if run_id not in run_ids or not re.fullmatch(r'images-\d{3}\.zip', filename):
                raise HTTPException(404)
            volume.reload()
            path = Path('/work/exports') / run_id / filename
            if not path.is_file():
                raise HTTPException(404)
            size = path.stat().st_size
            offset = 0
            requested = request.headers.get('range')
            if requested:
                match = re.fullmatch(r'bytes=(\d+)-', requested)
                if not match:
                    raise HTTPException(416)
                offset = int(match[1])
                if offset >= size:
                    raise HTTPException(416)
            def stream():
                with path.open('rb') as source:
                    source.seek(offset)
                    for block in iter(lambda: source.read(1024 * 1024), b''):
                        yield block
            headers = {'Content-Length': str(size - offset), 'Accept-Ranges': 'bytes'}
            if offset:
                headers['Content-Range'] = f'bytes {offset}-{size-1}/{size}'
            return StreamingResponse(stream(), status_code=206 if offset else 200,
                                     media_type='application/zip', headers=headers)
        return api

    with modal.enable_output():
        app.deploy()
    url = modal.Function.from_name(app.name, 'download').get_web_url()
    if not url:
        raise ValueError('Download endpoint was not deployed')
    for plan in plans:
        root = PROJECT.parent / 'AI_PROJECT_images/uno_512' / plan['benchmark'] / plan['run_id']
        root.mkdir(parents=True, exist_ok=True)
        index_path = root / 'bundle_index.json'
        if index_path.exists():
            index = json.loads(index_path.read_text())
        else:
            index = json.loads(b''.join(volume.read_file(f"exports/{plan['run_id']}/bundle_index.json")))
            write_json(index_path, index)
        for part in index['parts']:
            final = root / part['file']
            if final.exists() and digest(final) == part['sha256']:
                continue
            partial = final.with_suffix('.http.partial')
            for attempt in range(5):
                offset = partial.stat().st_size if partial.exists() else 0
                if offset == part['size_bytes']:
                    break
                headers = {'Authorization': 'Bearer ' + token}
                if offset:
                    headers['Range'] = f'bytes={offset}-'
                req = urllib.request.Request(url + f"/archive/{plan['run_id']}/{part['file']}", headers=headers)
                try:
                    with urllib.request.urlopen(req, timeout=90) as response:
                        if offset and response.status != 206:
                            raise ValueError('Server did not honor resume request')
                        with partial.open('ab') as target:
                            last = offset // (50 * 1024**2)
                            while block := response.read(1024 * 1024):
                                target.write(block)
                                offset += len(block)
                                if offset // (50 * 1024**2) > last:
                                    last = offset // (50 * 1024**2)
                                    print(plan['benchmark'], round(offset / 1024**2), 'MiB downloaded', flush=True)
                    break
                except Exception as exc:
                    print('Transfer retry', attempt + 1, type(exc).__name__, flush=True)
                    if attempt == 4:
                        raise
                    time.sleep(3)
            if partial.stat().st_size != part['size_bytes'] or digest(partial) != part['sha256']:
                raise ValueError('Downloaded archive SHA-256 mismatch')
            partial.replace(final)
        verify(root, index)
        report = dict(verified=True, generated=index['generated_images'], parts=index['parts'],
                      archive_folder=str(root), member_hashes_and_crc_verified=True,
                      missing_case_ids=index['missing_case_ids'])
        write_json(root / 'verification.json', report)
        write_json(Path(plan['local']) / 'bundle_index.json', index)
        write_json(Path(plan['local']) / 'verification.json', report)
        print(plan['benchmark'], 'download and verification complete', flush=True)


if __name__ == '__main__':
    main()
