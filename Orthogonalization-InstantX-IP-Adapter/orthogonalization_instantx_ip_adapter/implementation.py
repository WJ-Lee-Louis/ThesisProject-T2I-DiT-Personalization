"""Deterministic source fingerprint for scientifically consistent resumes."""

from __future__ import annotations

import hashlib
from pathlib import Path


def compute_batch_implementation_digest(
    experiment_root: str | Path | None = None,
) -> str:
    """Hash all generation package sources plus the deployed batch worker."""

    root = (
        Path(experiment_root).resolve()
        if experiment_root is not None
        else Path(__file__).resolve().parents[1]
    )
    package_root = root / "orthogonalization_instantx_ip_adapter"
    worker_path = root / "batch_modal.py"
    paths = [worker_path, *sorted(package_root.rglob("*.py"))]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Cannot fingerprint the batch implementation; missing source: "
            + ", ".join(str(path) for path in missing)
        )

    digest = hashlib.sha256()
    for path in paths:
        relative_path = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative_path).to_bytes(8, "big"))
        digest.update(relative_path)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


__all__ = ["compute_batch_implementation_digest"]
