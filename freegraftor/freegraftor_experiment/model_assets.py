"""Download pinned model assets into the persistent Modal cache."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import urllib.request
from uuid import uuid4

from .constants import (
    FLUX_MODEL_ID,
    FLUX_MODEL_REVISION,
    GROUNDING_DINO_MODEL_ID,
    GROUNDING_DINO_MODEL_REVISION,
    SAM_FILENAME,
    SAM_SHA256,
    SAM_SIZE_BYTES,
    SAM_URL,
)


def prepare_model_paths(cache_root: str | Path = "/cache") -> dict[str, str]:
    from filelock import FileLock
    from huggingface_hub import snapshot_download

    cache_root = Path(cache_root)
    hf_cache = cache_root / "huggingface"
    hf_cache.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGINGFACE_HUB_TOKEN"
    )
    if not token:
        raise RuntimeError(
            "The Modal secret must define HF_TOKEN or HUGGINGFACE_HUB_TOKEN "
            "for gated FLUX.1-dev access."
        )
    flux_path = snapshot_download(
        repo_id=FLUX_MODEL_ID,
        revision=FLUX_MODEL_REVISION,
        cache_dir=str(hf_cache),
        token=token,
    )
    dino_path = snapshot_download(
        repo_id=GROUNDING_DINO_MODEL_ID,
        revision=GROUNDING_DINO_MODEL_REVISION,
        cache_dir=str(hf_cache),
        token=token,
    )

    checkpoint_dir = cache_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sam_path = checkpoint_dir / SAM_FILENAME
    checksum_path = checkpoint_dir / f"{SAM_FILENAME}.sha256"
    lock_path = checkpoint_dir / f".{SAM_FILENAME}.lock"
    with FileLock(str(lock_path), timeout=60 * 60):
        if sam_path.is_file() and sam_path.stat().st_size == SAM_SIZE_BYTES:
            actual_digest = _sha256_file(sam_path)
        else:
            actual_digest = None
        if actual_digest != SAM_SHA256:
            sam_path.unlink(missing_ok=True)
            checksum_path.unlink(missing_ok=True)
            temporary = checkpoint_dir / f".{SAM_FILENAME}.{uuid4().hex}.tmp"
            try:
                urllib.request.urlretrieve(SAM_URL, temporary)
                if temporary.stat().st_size != SAM_SIZE_BYTES:
                    raise RuntimeError(
                        "Downloaded SAM checkpoint has an unexpected size: "
                        f"{temporary.stat().st_size} != {SAM_SIZE_BYTES}"
                    )
                if _sha256_file(temporary) != SAM_SHA256:
                    raise RuntimeError("SAM checkpoint SHA-256 verification failed")
                os.replace(temporary, sam_path)
            finally:
                temporary.unlink(missing_ok=True)
        marker = checkpoint_dir / f".{SAM_FILENAME}.{uuid4().hex}.sha256.tmp"
        try:
            marker.write_text(SAM_SHA256 + "\n", encoding="ascii")
            os.replace(marker, checksum_path)
        finally:
            marker.unlink(missing_ok=True)

    values = {
        "FLUX_DEV": str(Path(flux_path).resolve()),
        "GROUNDING_DINO": str(Path(dino_path).resolve()),
        "SAM": str(sam_path.resolve()),
    }
    os.environ.update(values)
    return values


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
