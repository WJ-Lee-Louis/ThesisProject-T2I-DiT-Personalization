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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--benchmark", choices=("subset", "custom"), required=True)
    parser.add_argument("--generated-manifest", type=Path, required=True)
    parser.add_argument("--xverse-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--segmented-reference-root",
        type=Path,
        help=(
            "Directory containing official-style foreground-segmented references "
            "with the same filenames as the benchmark references."
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
    import_official_xverse(args.xverse_root)
    from eval.tools.dino import DINOScore
    from eval.tools.florence_sam import ObjectDetector

    manifest, _ = load_benchmark(args.benchmark_root, args.benchmark)
    rows = load_generated_images(args.generated_manifest, manifest)
    detector = ObjectDetector(args.device)
    dino = DINOScore(args.device)
    scores = []
    reference_cache: dict[str, Image.Image] = {}
    for row in rows:
        record = row["benchmark_record"]
        filename = record["reference_filename"]
        if filename not in reference_cache:
            if args.segmented_reference_root is not None:
                reference_path = args.segmented_reference_root.resolve() / filename
                if not reference_path.is_file():
                    raise FileNotFoundError(
                        f"Segmented reference is missing: {reference_path}"
                    )
            else:
                reference_path = resolve_reference_path(args.benchmark_root, record)
            with Image.open(reference_path) as opened:
                reference_cache[filename] = opened.convert("RGB")

        with Image.open(row["image_path"]) as opened:
            generated = opened.convert("RGB")
        candidates = detector.get_instances(
            generated,
            record["subject_label"],
            min_size=max(1, generated.size[0] // 20),
        )[:3]
        candidate_scores = [
            dino(reference_cache[filename], _pad_to_square(candidate))
            for candidate in candidates
        ]
        score = max(candidate_scores) if candidate_scores else 0.0
        scores.append(
            {
                "record_id": row["record_id"],
                "condition": row["condition"],
                "seed": row["seed"],
                "image_path": row["image_path"],
                "reference_filename": filename,
                "detected_candidate_count": len(candidates),
                "ip_similarity": score,
            }
        )
    write_json(
        args.output,
        {
            "schema_version": 1,
            "metric": "IP-Sim",
            "benchmark_id": manifest["benchmark_id"],
            "detector": "official XVerse Florence2+SAM2",
            "embedding_model": "facebook/dino-vits16",
            "implementation_note": (
                "The released XVerse script uses facebook/dino-vits16 even "
                "though the paper/README describes the metric as DINOv2."
            ),
            "reference_mode": (
                "segmented" if args.segmented_reference_root else "raw_diagnostic"
            ),
            "scores": scores,
        },
    )
    print(f"IP-Sim scoring complete: {len(scores)} images -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
