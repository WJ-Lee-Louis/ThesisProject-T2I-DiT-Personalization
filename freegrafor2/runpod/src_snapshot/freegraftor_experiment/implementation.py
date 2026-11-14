"""Content digest used to reject stale Modal deployments."""

from __future__ import annotations

import hashlib
from pathlib import Path


def compute_implementation_digest(experiment_root: str | Path) -> str:
    root = Path(experiment_root)
    candidates = [root / "batch_modal.py"]
    candidates.extend(sorted((root / "freegraftor_experiment").rglob("*.py")))
    candidates.extend(sorted((root / "src").rglob("*.py")))
    digest = hashlib.sha256()
    for path in candidates:
        if not path.is_file():
            raise FileNotFoundError(f"Implementation file missing: {path}")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()
