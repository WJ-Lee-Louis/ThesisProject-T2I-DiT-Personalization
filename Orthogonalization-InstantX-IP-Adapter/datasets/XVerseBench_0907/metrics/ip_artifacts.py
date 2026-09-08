"""Shared helpers for auditable XVerse IP-Sim foreground artifacts."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image


def as_rgb_image(image: Image.Image | np.ndarray) -> Image.Image:
    """Return detector output as an RGB PIL image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def mask_from_white_background(image: Image.Image) -> Image.Image:
    """Recover the binary foreground mask used by the released XVerse code.

    Florence2+SAM2 returns a segmented instance composited onto pure white.
    The released reference-preparation code treats every non-white pixel as
    foreground, so this helper deliberately preserves that exact convention.
    """
    rgb = np.asarray(image.convert("RGB"))
    foreground = (rgb[..., :3] != 255).any(axis=-1)
    return Image.fromarray(foreground.astype(np.uint8) * 255, mode="L")


def png_name(filename: str) -> str:
    return Path(filename).with_suffix(".png").name


def safe_path_component(value: object) -> str:
    component = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return component or "item"


def resolve_segmented_crop(root: Path, reference_filename: str) -> Path:
    """Resolve both the new crops/ layout and the previous flat layout."""
    root = root.resolve()
    png_filename = png_name(reference_filename)
    candidates = (
        root / "crops" / png_filename,
        root / "crops" / reference_filename,
        root / reference_filename,
        root / png_filename,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    expected = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Segmented reference is missing for {reference_filename}. Tried: {expected}"
    )


def resolve_segmented_mask(root: Path, reference_filename: str) -> Path | None:
    root = root.resolve()
    candidates = (
        root / "masks" / png_name(reference_filename),
        root / "masks" / reference_filename,
    )
    return next((path for path in candidates if path.is_file()), None)
