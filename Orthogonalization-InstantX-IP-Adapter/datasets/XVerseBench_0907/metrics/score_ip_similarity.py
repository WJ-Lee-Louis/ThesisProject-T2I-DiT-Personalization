"""Manifest-based animal IP-Sim adapted from the released XVerse code."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps

from common import (
    import_official_xverse,
    load_benchmark,
    load_generated_images,
    resolve_reference_path,
    write_json,
)
from ip_artifacts import (
    as_rgb_image,
    mask_from_white_background,
    resolve_segmented_crop,
    resolve_segmented_mask,
    safe_path_component,
)
from florence2_compat import (
    pin_florence2_checkpoint,
    repair_xverse_florence2_weight_tying,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--benchmark", choices=("subset", "custom"), required=True)
    parser.add_argument("--generated-manifest", type=Path, required=True)
    parser.add_argument("--xverse-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help=(
            "Directory for generated foreground crops and binary masks. "
            "Defaults to <output-stem>_artifacts next to the score JSON."
        ),
    )
    parser.add_argument(
        "--segmented-reference-root",
        type=Path,
        help=(
            "Directory created by prepare_segmented_references.py. The previous "
            "flat reference layout remains supported."
        ),
    )
    parser.add_argument(
        "--allow-raw-reference",
        action="store_true",
        help="Diagnostic only; raw-background references are not strict official IP-Sim.",
    )
    return parser


def _pad_to_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = max(width, height)
    left = (side - width) // 2
    top = (side - height) // 2
    return ImageOps.expand(
        image,
        border=(left, top, side - width - left, side - height - top),
        fill="white",
    )


def main() -> int:
    args = build_parser().parse_args()
    if args.segmented_reference_root is None and not args.allow_raw_reference:
        raise SystemExit(
            "IP-Sim requires --segmented-reference-root. Use "
            "--allow-raw-reference only for a non-official diagnostic run."
        )
    score_benchmark(
        benchmark_root=args.benchmark_root,
        benchmark=args.benchmark,
        generated_manifest=args.generated_manifest,
        xverse_root=args.xverse_root,
        device=args.device,
        output=args.output,
        artifact_root=args.artifact_root,
        segmented_reference_root=args.segmented_reference_root,
        allow_raw_reference=args.allow_raw_reference,
    )
    return 0


def load_ip_models(xverse_root: Path, device: str):
    import_official_xverse(xverse_root)
    from eval.tools.dino import DINOScore
    from eval.tools.florence_sam import ObjectDetector

    checkpoint = pin_florence2_checkpoint()
    detector = ObjectDetector(device)
    detector._florence2_checkpoint = checkpoint
    repair_xverse_florence2_weight_tying(detector)
    return detector, DINOScore(device)


def score_benchmark(
    *,
    benchmark_root: Path,
    benchmark: str,
    generated_manifest: Path,
    xverse_root: Path,
    device: str,
    output: Path,
    artifact_root: Path | None,
    segmented_reference_root: Path | None,
    allow_raw_reference: bool = False,
    detector=None,
    dino=None,
) -> None:
    if segmented_reference_root is None and not allow_raw_reference:
        raise ValueError("IP-Sim requires segmented references")
    if (detector is None) != (dino is None):
        raise ValueError("detector and dino must be supplied together")
    if detector is None:
        detector, dino = load_ip_models(xverse_root, device)
    manifest, _ = load_benchmark(benchmark_root, benchmark)
    rows = load_generated_images(generated_manifest, manifest)
    output_path = output.resolve()
    artifact_root = (
        artifact_root.resolve()
        if artifact_root is not None
        else output_path.parent / f"{output_path.stem}_artifacts"
    )
    generated_artifact_root = artifact_root / "generated"
    generated_artifact_root.mkdir(parents=True, exist_ok=True)
    scores = []
    detected_image_count = 0
    reference_cache: dict[str, Image.Image] = {}
    reference_artifacts: dict[str, dict[str, str | None]] = {}
    for row in rows:
        record = row["benchmark_record"]
        filename = record["reference_filename"]
        if filename not in reference_cache:
            if segmented_reference_root is not None:
                reference_path = resolve_segmented_crop(
                    segmented_reference_root, filename
                )
                reference_mask = resolve_segmented_mask(
                    segmented_reference_root, filename
                )
            else:
                reference_path = resolve_reference_path(benchmark_root, record)
                reference_mask = None
            with Image.open(reference_path) as opened:
                reference_cache[filename] = opened.convert("RGB")
            reference_artifacts[filename] = {
                "foreground_crop": str(reference_path),
                "binary_mask": str(reference_mask) if reference_mask else None,
            }

        with Image.open(row["image_path"]) as opened:
            generated = opened.convert("RGB")
        candidates = detector.get_instances(
            generated,
            record["subject_label"],
            min_size=max(1, generated.size[0] // 20),
        )[:3]
        detected_image_count += int(bool(candidates))
        item_root = (
            generated_artifact_root
            / safe_path_component(row["condition"])
            / safe_path_component(row["record_id"])
            / f"seed_{safe_path_component(row['seed'])}"
        )
        item_root.mkdir(parents=True, exist_ok=True)
        candidate_artifacts = []
        candidate_scores = []
        for candidate_index, candidate in enumerate(candidates):
            foreground_crop = as_rgb_image(candidate)
            binary_mask = mask_from_white_background(foreground_crop)
            crop_path = item_root / f"candidate_{candidate_index:02d}_crop.png"
            mask_path = item_root / f"candidate_{candidate_index:02d}_mask.png"
            foreground_crop.save(crop_path, format="PNG")
            binary_mask.save(mask_path, format="PNG")
            candidate_score = dino(
                reference_cache[filename], _pad_to_square(foreground_crop)
            )
            candidate_scores.append(candidate_score)
            candidate_artifacts.append(
                {
                    "candidate_index": candidate_index,
                    "ip_similarity": candidate_score,
                    "foreground_crop": crop_path.relative_to(artifact_root).as_posix(),
                    "binary_mask": mask_path.relative_to(artifact_root).as_posix(),
                }
            )
        selected_index = (
            max(range(len(candidate_scores)), key=candidate_scores.__getitem__)
            if candidate_scores
            else None
        )
        score = candidate_scores[selected_index] if selected_index is not None else 0.0
        selected_artifact = (
            candidate_artifacts[selected_index] if selected_index is not None else None
        )
        scores.append(
            {
                "record_id": row["record_id"],
                "condition": row["condition"],
                "seed": row["seed"],
                "image_path": row["image_path"],
                "reference_filename": filename,
                "detected_candidate_count": len(candidates),
                "selected_candidate_index": selected_index,
                "selected_foreground_crop": (
                    selected_artifact["foreground_crop"] if selected_artifact else None
                ),
                "selected_binary_mask": (
                    selected_artifact["binary_mask"] if selected_artifact else None
                ),
                "candidate_artifacts": candidate_artifacts,
                "ip_similarity": score,
            }
        )
    write_json(
        output_path,
        {
            "schema_version": 3,
            "metric": "IP-Sim",
            "benchmark_id": manifest["benchmark_id"],
            "detector": "official XVerse Florence2+SAM2",
            "embedding_model": "facebook/dino-vits16",
            "implementation_note": (
                "The released XVerse script uses facebook/dino-vits16 even "
                "though the paper/README describes the metric as DINOv2. "
                "The pinned XVerse Florence-2 implementation is repaired to "
                "share its checkpoint embedding with the otherwise missing "
                "language-model output head."
            ),
            "florence2_compatibility": getattr(
                detector, "_florence2_compat_report", None
            ),
            "reference_mode": (
                "segmented" if segmented_reference_root else "raw_diagnostic"
            ),
            "mask_convention": "255=foreground, 0=background",
            "quality_control": {
                "image_count": len(scores),
                "detected_image_count": detected_image_count,
                "detection_rate": (
                    detected_image_count / len(scores) if scores else 0.0
                ),
                "valid": bool(scores) and detected_image_count > 0,
                "failure_stage_when_zero": "florence2_detection_before_sam2",
            },
            "artifact_root": str(artifact_root),
            "reference_artifacts": reference_artifacts,
            "scores": scores,
        },
    )
    print(f"IP-Sim scoring complete: {len(scores)} images -> {output_path}")
    print(f"Foreground crops and masks -> {artifact_root}")
    if scores and detected_image_count == 0:
        raise RuntimeError(
            "IP-Sim invalid: Florence2 returned zero generated-image candidates "
            "for every image, so SAM2 and DINO were never reached."
        )


if __name__ == "__main__":
    raise SystemExit(main())
