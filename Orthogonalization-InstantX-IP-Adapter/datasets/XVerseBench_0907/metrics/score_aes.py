"""Per-image AES scoring using the model selected by the XVerse evaluator."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
import torch

from common import load_benchmark, load_generated_images, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--benchmark", choices=("subset", "custom"), required=True)
    parser.add_argument("--generated-manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip

    manifest, _ = load_benchmark(args.benchmark_root, args.benchmark)
    rows = load_generated_images(args.generated_manifest, manifest)
    model, preprocessor = convert_v2_5_from_siglip(
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model = model.to(torch.bfloat16).to(args.device).eval()
    scores = []
    with torch.inference_mode():
        for row in rows:
            with Image.open(row["image_path"]) as opened:
                image = opened.convert("RGB")
            pixel_values = preprocessor(images=image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(torch.bfloat16).to(args.device)
            score = float(model(pixel_values).logits.squeeze().float().cpu()) * 10.0
            scores.append(
                {
                    "record_id": row["record_id"],
                    "condition": row["condition"],
                    "seed": row["seed"],
                    "image_path": row["image_path"],
                    "aes": score,
                }
            )
    write_json(
        args.output,
        {
            "schema_version": 1,
            "metric": "AES",
            "benchmark_id": manifest["benchmark_id"],
            "model": "aesthetic-predictor-v2.5 SigLIP",
            "scoring_unit": "individual generated image",
            "scores": scores,
        },
    )
    print(f"AES scoring complete: {len(scores)} images -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
