"""Portable hashes, atomic publication and safe archive extraction."""
import hashlib
import json
from pathlib import Path
import zipfile
import time
import uuid


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Manual status and the background collector may publish concurrently on Windows.
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        for attempt in range(8):
            try:
                temp.replace(path)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(.025 * (attempt + 1))
    finally:
        temp.unlink(missing_ok=True)


def extract(archive, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            name = info.filename
            if ("\\" in name or ":" in name or name.startswith("/")
                    or ".." in Path(name).parts or (info.external_attr >> 16) & 0o170000 == 0o120000
                    or not (destination / name).resolve().is_relative_to(destination)):
                raise ValueError("Unsafe ZIP member")
        z.extractall(destination)
