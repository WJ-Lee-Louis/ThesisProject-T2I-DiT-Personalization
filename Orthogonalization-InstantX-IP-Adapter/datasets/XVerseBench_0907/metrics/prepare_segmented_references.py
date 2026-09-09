"""Create official-style foreground references for animal IP-Sim.

The raw reference copies remain untouched.  This script must run in the pinned
XVerse environment because it reuses the official Florence2+SAM2 stack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
from florence2_compat import (
    pin_florence2_checkpoint,
    repair_xverse_florence2_weight_tying,
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
    prepare_references(
        benchmark_root=args.benchmark_root,
        xverse_root=args.xverse_root,
        device=args.device,
        output_root=args.output_root,
        overwrite=args.overwrite,
    )
    return 0


def load_detector(xverse_root: Path, device: str):
    import_official_xverse(xverse_root)
    from eval.tools.florence_sam import ObjectDetector

    checkpoint = pin_florence2_checkpoint()
    detector = ObjectDetector(device)
    detector._florence2_checkpoint = checkpoint
    repair_xverse_florence2_weight_tying(detector)
    return detector


def segmentation_queries(subject_label: str) -> tuple[str, ...]:
    """Return precise Florence queries, broadening only to the head noun."""

    exact = " ".join(subject_label.strip().rstrip(".").split())
    if not exact:
        raise ValueError("subject_label must not be empty")
    head = exact.split()[-1]
    candidates = (exact, f"{exact}.", head, f"{head}.")
    return tuple(dict.fromkeys(candidates))


def detect_reference_subject(detector, image, subject_label: str, min_size: int):
    """Try exact label first and a semantically contained head-noun fallback."""

    attempts = []
    for query in segmentation_queries(subject_label):
        result = detector.get_multiple_instances(image, query, min_size=min_size)
        attempts.append(query)
        if result.get("instance_bboxes"):
            return result, query, attempts
    return None, None, attempts


def prepare_references(
    *,
    benchmark_root: Path,
    xverse_root: Path,
    device: str,
    output_root: Path,
    overwrite: bool = False,
    detector=None,
) -> None:
    import_official_xverse(xverse_root)
    from src.utils.data_utils import cv2pil, merge_bboxes

    manifest, _ = load_benchmark(benchmark_root, "subset")
    detector = detector or load_detector(xverse_root, device)
    output_root = output_root.resolve()
    crop_root = output_root / "crops"
    mask_root = output_root / "masks"
    crop_root.mkdir(parents=True, exist_ok=True)
    mask_root.mkdir(parents=True, exist_ok=True)
    previous_manifest_path = output_root / "segmentation_manifest.json"
    previous_rows = {}
    if previous_manifest_path.is_file():
        try:
            previous_payload = json.loads(
                previous_manifest_path.read_text(encoding="utf-8")
            )
            compatibility = previous_payload.get("florence2_compatibility") or {}
            repaired = compatibility.get("after") or {}
            previous_records = previous_payload.get("records", [])
            if (
                previous_payload.get("schema_version") == 4
                and previous_payload.get("fallback_method") is None
                and repaired.get("lm_head_tied_to_shared") is True
                and all(
                    row.get("segmentation_method")
                    == "florence2_open_vocabulary+sam2"
                    for row in previous_records
                )
            ):
                previous_rows = {
                    row["reference_filename"]: row for row in previous_records
                }
        except (KeyError, OSError, TypeError, ValueError):
            previous_rows = {}
    results = []
    for record in manifest["records"]:
        source = resolve_reference_path(benchmark_root, record)
        source_sha256 = _sha256(source)
        if source_sha256 != record["content_sha256"]:
            raise RuntimeError(
                f"Reference hash mismatch for {record['reference_filename']}"
            )
        artifact_name = png_name(record["reference_filename"])
        crop_destination = crop_root / artifact_name
        mask_destination = mask_root / artifact_name
        previous = previous_rows.get(record["reference_filename"])
        if (
            crop_destination.exists()
            and mask_destination.exists()
            and previous is not None
            and previous.get("requested_subject_label", previous.get("subject_label"))
            == record["subject_label"]
            and previous.get("segmentation_method")
            and previous.get("segmentation_query_attempts")
            and not overwrite
        ):
            reused = dict(previous)
            reused["status"] = "reused"
            reused["foreground_crop_sha256"] = _sha256(crop_destination)
            reused["binary_mask_sha256"] = _sha256(mask_destination)
            results.append(reused)
            continue
        with Image.open(source) as opened:
            image = opened.convert("RGB")
        instance_result, segmentation_query, query_attempts = detect_reference_subject(
            detector,
            image,
            record["subject_label"],
            min_size=max(1, image.size[0] // 20),
        )
        segmentation_method = "florence2_open_vocabulary+sam2"
        if instance_result is None:
            raise RuntimeError(
                "Reference foreground detection failed for "
                f"{record['reference_filename']} ({record['subject_label']}); "
                f"queries tried: {query_attempts}"
            )
        bboxes = instance_result["instance_bboxes"]
        instance_images = instance_result["instance_images"]
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
                "source_sha256": source_sha256,
                "status": "created",
                "subject_label": record["subject_label"],
                "requested_subject_label": record["subject_label"],
                "segmentation_method": segmentation_method,
                "segmentation_query": segmentation_query,
                "segmentation_query_attempts": query_attempts,
                "used_fallback_query": bool(
                    segmentation_query
                    and segmentation_query
                    != segmentation_queries(record["subject_label"])[0]
                ),
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
            "schema_version": 4,
            "method": "official XVerse Florence2+SAM2 reference segmentation",
            "fallback_method": None,
            "florence2_compatibility": getattr(
                detector, "_florence2_compat_report", None
            ),
            "mask_convention": "255=foreground, 0=background",
            "crop_root": str(crop_root),
            "mask_root": str(mask_root),
            "records": results,
        },
    )
    print(f"Prepared {len(results)} segmented references -> {output_root}")


if __name__ == "__main__":
    raise SystemExit(main())
