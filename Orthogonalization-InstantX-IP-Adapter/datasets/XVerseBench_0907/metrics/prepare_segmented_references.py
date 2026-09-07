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
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for record in manifest["records"]:
        source = resolve_reference_path(args.benchmark_root, record)
        destination = output_root / record["reference_filename"]
        if destination.exists() and not args.overwrite:
            results.append(
                {
                    "reference_filename": destination.name,
                    "status": "reused",
                    "sha256": _sha256(destination),
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
        used_bboxes = []
        for bbox, instance in zip(bboxes, instance_images):
            bbox = np.asarray(bbox, dtype=int).tolist()
            used_bboxes.append(bbox)
            instance_image = cv2pil(instance)
            mask = (np.asarray(instance_image)[..., :3] != 255).any(axis=-1)
            mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
            canvas.paste(instance_image, (bbox[0], bbox[1]), mask_image)
        merged = merge_bboxes(used_bboxes)
        canvas.crop(merged).save(destination)
        results.append(
            {
                "reference_filename": destination.name,
                "status": "created",
                "subject_label": record["subject_label"],
                "detected_instance_count": len(bboxes),
                "sha256": _sha256(destination),
            }
        )
    write_json(
        output_root / "segmentation_manifest.json",
        {
            "schema_version": 1,
            "method": "official XVerse Florence2+SAM2 reference segmentation",
            "records": results,
        },
    )
    print(f"Prepared {len(results)} segmented references -> {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
