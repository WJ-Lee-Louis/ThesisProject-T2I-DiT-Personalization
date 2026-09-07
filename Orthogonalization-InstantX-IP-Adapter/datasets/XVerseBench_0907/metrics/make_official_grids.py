"""Create the 2x2 four-seed grids expected by unmodified XVerse scripts."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from PIL import Image

from common import load_benchmark, load_generated_images


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--generated-manifest", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _grid(images: list[Image.Image]) -> Image.Image:
    if len(images) != 4:
        raise ValueError(f"Official-compatible grid requires 4 images, got {len(images)}")
    width, height = images[0].size
    if any(image.size != (width, height) for image in images):
        raise ValueError("All four seed images must have the same resolution")
    canvas = Image.new("RGB", (2 * width, 2 * height), "white")
    for index, image in enumerate(images):
        canvas.paste(image, ((index % 2) * width, (index // 2) * height))
    return canvas


def main() -> int:
    args = build_parser().parse_args()
    manifest, _ = load_benchmark(args.benchmark_root, "subset")
    rows = [
        row
        for row in load_generated_images(args.generated_manifest, manifest)
        if row["condition"] == args.condition
    ]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["record_id"]].append(row)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = {record["record_id"]: record for record in manifest["records"]}
    if set(grouped) != set(records):
        missing = sorted(set(records) - set(grouped))
        raise ValueError(f"Condition does not cover all 29 records; missing={missing}")
    for record_id, record_rows in sorted(grouped.items()):
        record_rows.sort(key=lambda row: row["seed"])
        expected = records[record_id]["seeds"]
        actual = [row["seed"] for row in record_rows]
        if actual != expected:
            raise ValueError(
                f"{record_id} seeds must be {expected}, got {actual}"
            )
        images = []
        for row in record_rows:
            with Image.open(row["image_path"]) as opened:
                images.append(opened.convert("RGB"))
        prompt_slug = "_".join(records[record_id]["prompt"][:40].split())
        filename = f"{records[record_id]['record_index']}_{prompt_slug}.png"
        _grid(images).save(output_dir / filename)
    print(f"Created {len(records)} official-compatible grids -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
