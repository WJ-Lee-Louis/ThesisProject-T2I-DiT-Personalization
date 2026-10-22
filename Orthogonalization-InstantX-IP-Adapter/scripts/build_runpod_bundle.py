"""Package only allowlisted code and manifest inputs, never credentials or old images."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
manifest = ROOT / 'outputs/semantic_benchmark_gpt56_sol/generation_manifest.json'


def files():
    selected = {manifest, ROOT / 'requirements-benchmark-metrics.txt'}
    selected.add(ROOT / 'runpod/input_bundle.json')
    for folder in (ROOT / 'orthogonalization_instantx_ip_adapter', BASE / 'shared'):
        selected.update(folder.glob('*.py'))
    selected.update(p for p in (ROOT / 'runpod').iterdir() if p.suffix in ('.sh', '.py', '.txt'))
    selected.update(ROOT / 'scripts' / p for p in ('benchmark_runpod.py', 'benchmark_modal.py', 'benchmark_metrics.py'))
    for row in json.loads(manifest.read_text(encoding='utf-8')):
        for key in ('image_path', 'conditioning_path'):
            path = (ROOT / row[key]).resolve()
            if not path.is_relative_to(ROOT) or not path.is_file():
                raise ValueError(f'Unsafe/missing manifest input: {path}')
            selected.add(path)
    return sorted(selected)


def main():
    output = ROOT / 'outputs/runpod_delivery'
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'runpod_bundle.tar.gz'
    entries = files()
    temporary = target.with_suffix('.tmp')
    with tarfile.open(temporary, 'w:gz') as archive:
        for path in entries:
            archive.add(path, arcname='AI_PROJECT/' + path.relative_to(BASE).as_posix(), recursive=False)
    temporary.replace(target)
    checksum = hashlib.sha256(target.read_bytes()).hexdigest()
    (output / 'runpod_bundle.tar.gz.sha256').write_text(checksum + '  runpod_bundle.tar.gz\n', encoding='ascii')
    (output / 'bundle_manifest.json').write_text(json.dumps(
        {p.relative_to(BASE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in entries}, indent=2), encoding='utf-8')
    print(json.dumps(dict(path=str(target), files=len(entries), bytes=target.stat().st_size, sha256=checksum), indent=2))


if __name__ == '__main__':
    main()
