"""Create official-style foreground references for animal IP-Sim.

The raw reference copies remain untouched.  This script must run in the pinned
XVerse environment because it reuses the official Florence2+SAM2 stack.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from common import (
    import_official_xverse,
    load_benchmark,
    resolve_reference_path,
    write_json,
)
from ip_artifacts import as_rgb_image, mask_from_white_background, png_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--xverse-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = build_parser().parse_args()
    import_official_xverse(args.xverse_root)
    from eval.tools.florence_sam import ObjectDetector
    from src.utils.data_utils import cv2pil, merge_bboxes

    manifest, _ = load_benchmark(args.benchmark_root, "subset")
    detector = ObjectDetector(args.device)
    output_root = args.output_root.resolve()
    crop_root = output_root / "crops"
    mask_root = output_root / "masks"
    crop_root.mkdir(parents=True, exist_ok=True)
    mask_root.mkdir(parents=True, exist_ok=True)
    results = []
    for record in manifest["records"]:
        source = resolve_reference_path(args.benchmark_root, record)
        artifact_name = png_name(record["reference_filename"])
        crop_destination = crop_root / artifact_name
        mask_destination = mask_root / artifact_name
        if (
            crop_destination.exists()
            and mask_destination.exists()
            and not args.overwrite
        ):
            results.append(
                {
                    "reference_filename": record["reference_filename"],
                    "status": "reused",
                    "subject_label": record["subject_label"],
                    "foreground_crop": crop_destination.relative_to(output_root).as_posix(),
                    "binary_mask": mask_destination.relative_to(output_root).as_posix(),
                    "foreground_crop_sha256": _sha256(crop_destination),
                    "binary_mask_sha256": _sha256(mask_destination),
                }
            )
            continue
        with Image.open(source) as opened:
            image = opened.convert("RGB")
        instance_result = detector.get_multiple_instances(
            image,
            record["subject_label"],
            min_size=max(1, image.size[0] // 20),
        )
        bboxes = instance_result["instance_bboxes"]
        instance_images = instance_result["instance_images"]
        if not bboxes:
            raise RuntimeError(
                "Reference foreground detection failed for "
                f"{record['reference_filename']} ({record['subject_label']})"
            )
        canvas = Image.new("RGB", image.size, "white")
        mask_canvas = Image.new("L", image.size, 0)
        used_bboxes = []
        for bbox, instance in zip(bboxes, instance_images):
            bbox = np.asarray(bbox, dtype=int).tolist()
            used_bboxes.append(bbox)
            instance_image = as_rgb_image(cv2pil(instance))
            mask_image = mask_from_white_background(instance_image)
            canvas.paste(instance_image, (bbox[0], bbox[1]), mask_image)
            mask_canvas.paste(mask_image, (bbox[0], bbox[1]), mask_image)
        merged = tuple(int(value) for value in merge_bboxes(used_bboxes))
        foreground_crop = canvas.crop(merged)
        binary_mask = mask_canvas.crop(merged)
        foreground_crop.save(crop_destination, format="PNG")
        binary_mask.save(mask_destination, format="PNG")
        results.append(
            {
                "reference_filename": record["reference_filename"],
                "status": "created",
                "subject_label": record["subject_label"],
                "detected_instance_count": len(bboxes),
                "source_bbox_xyxy": list(merged),
                "foreground_crop": crop_destination.relative_to(output_root).as_posix(),
                "binary_mask": mask_destination.relative_to(output_root).as_posix(),
                "foreground_crop_sha256": _sha256(crop_destination),
                "binary_mask_sha256": _sha256(mask_destination),
            }
        )
    write_json(
        output_root / "segmentation_manifest.json",
        {
            "schema_version": 2,
            "method": "official XVerse Florence2+SAM2 reference segmentation",
            "mask_convention": "255=foreground, 0=background",
            "crop_root": str(crop_root),
            "mask_root": str(mask_root),
            "records": results,
        },
    )
    print(f"Prepared {len(results)} segmented references -> {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
