import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image


MODULE_PATH = (
    Path(__file__).parents[1]
    / "datasets"
    / "XVerseBench_0907"
    / "metrics"
    / "ip_artifacts.py"
)
SPEC = importlib.util.spec_from_file_location("xverse_ip_artifacts", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
ip_artifacts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ip_artifacts)


def test_mask_from_white_background_uses_official_binary_convention() -> None:
    image = Image.new("RGB", (3, 2), "white")
    image.putpixel((1, 0), (255, 254, 255))
    image.putpixel((2, 1), (0, 0, 0))

    mask = ip_artifacts.mask_from_white_background(image)

    assert mask.mode == "L"
    assert np.asarray(mask).tolist() == [[0, 255, 0], [0, 0, 255]]


def test_segmented_artifact_resolver_supports_new_and_legacy_layouts(
    tmp_path: Path,
) -> None:
    crop_root = tmp_path / "crops"
    mask_root = tmp_path / "masks"
    crop_root.mkdir()
    mask_root.mkdir()
    crop = crop_root / "07_dog.png"
    mask = mask_root / "07_dog.png"
    Image.new("RGB", (2, 2)).save(crop)
    Image.new("L", (2, 2)).save(mask)

    assert ip_artifacts.resolve_segmented_crop(tmp_path, "07_dog.jpg") == crop
    assert ip_artifacts.resolve_segmented_mask(tmp_path, "07_dog.jpg") == mask

    crop.unlink()
    legacy = tmp_path / "07_dog.jpg"
    Image.new("RGB", (2, 2)).save(legacy)
    assert ip_artifacts.resolve_segmented_crop(tmp_path, "07_dog.jpg") == legacy
