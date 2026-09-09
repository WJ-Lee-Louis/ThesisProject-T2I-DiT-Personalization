"""Per-image AES scoring using the model selected by the XVerse evaluator."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
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
    parser.add_argument("--predictor-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    score_benchmark(
        benchmark_root=args.benchmark_root,
        benchmark=args.benchmark,
        generated_manifest=args.generated_manifest,
        device=args.device,
        predictor_path=args.predictor_path,
        output=args.output,
    )
    return 0


def load_aes_model(device: str, predictor_path: Path | None):
    from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip

    predictor_path = (
        str(predictor_path.resolve())
        if predictor_path is not None
        else None
    )
    model, preprocessor = convert_v2_5_from_siglip(
        predictor_name_or_path=predictor_path,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    return model.to(torch.bfloat16).to(device).eval(), preprocessor


def score_benchmark(
    *,
    benchmark_root: Path,
    benchmark: str,
    generated_manifest: Path,
    device: str,
    predictor_path: Path | None,
    output: Path,
    model=None,
    preprocessor=None,
) -> None:
    if (model is None) != (preprocessor is None):
        raise ValueError("model and preprocessor must be supplied together")
    if model is None:
        model, preprocessor = load_aes_model(device, predictor_path)
    manifest, _ = load_benchmark(benchmark_root, benchmark)
    rows = load_generated_images(generated_manifest, manifest)
    scores = []
    with torch.inference_mode():
        for row in rows:
            with Image.open(row["image_path"]) as opened:
                image = opened.convert("RGB")
            pixel_values = preprocessor(images=image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(torch.bfloat16).to(device)
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
        output,
        {
            "schema_version": 1,
            "metric": "AES",
            "benchmark_id": manifest["benchmark_id"],
            "model": "aesthetic-predictor-v2.5 SigLIP",
            "package_version": importlib.metadata.version(
                "aesthetic-predictor-v2-5"
            ),
            "predictor_path": str(predictor_path.resolve()) if predictor_path else None,
            "predictor_sha256": (
                _sha256(predictor_path) if predictor_path else None
            ),
            "scoring_unit": "individual generated image",
            "scores": scores,
        },
    )
    print(f"AES scoring complete: {len(scores)} images -> {output}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
