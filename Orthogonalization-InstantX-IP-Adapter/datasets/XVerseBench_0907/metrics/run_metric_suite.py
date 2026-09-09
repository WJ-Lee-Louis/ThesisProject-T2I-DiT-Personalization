"""Run one metric over subset and custom while loading its models only once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable


EXPECTED = {
    "subset": ("xverse_single_animal_29", 696),
    "custom": ("xverse_ref_custom_animal_15x6", 540),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", choices=("dpg", "ip", "aes"), required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--xverse-root", type=Path, required=True)
    parser.add_argument("--subset-manifest", type=Path, required=True)
    parser.add_argument("--custom-manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path)
    parser.add_argument("--aes-predictor", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_suite(args)
    return 0


def run_suite(
    args: argparse.Namespace,
    on_checkpoint: Callable[[], None] | None = None,
) -> None:
    checkpoint = on_checkpoint or (lambda: None)
    manifests = {
        "subset": args.subset_manifest,
        "custom": args.custom_manifest,
    }
    outputs = {
        benchmark: args.run_root / benchmark / f"{args.metric}.json"
        for benchmark in EXPECTED
    }
    pending = [
        benchmark
        for benchmark, output in outputs.items()
        if not valid_score_output(output, benchmark)
    ]
    if not pending:
        print(f"[{args.metric}] both benchmark outputs are complete; reuse")
        return

    if args.metric == "dpg":
        from score_dpg import load_dpg_model, score_benchmark

        model = load_dpg_model(args.xverse_root, "cuda")
        for benchmark in pending:
            score_benchmark(
                benchmark_root=args.benchmark_root,
                benchmark=benchmark,
                generated_manifest=manifests[benchmark],
                xverse_root=args.xverse_root,
                device="cuda",
                output=outputs[benchmark],
                model=model,
            )
            checkpoint()
    elif args.metric == "aes":
        if args.aes_predictor is None or not args.aes_predictor.is_file():
            raise FileNotFoundError("Pinned AES predictor is missing")
        from score_aes import load_aes_model, score_benchmark

        model, preprocessor = load_aes_model("cuda:0", args.aes_predictor)
        for benchmark in pending:
            score_benchmark(
                benchmark_root=args.benchmark_root,
                benchmark=benchmark,
                generated_manifest=manifests[benchmark],
                device="cuda:0",
                predictor_path=args.aes_predictor,
                output=outputs[benchmark],
                model=model,
                preprocessor=preprocessor,
            )
            checkpoint()
    else:
        if args.sam_checkpoint is None or not args.sam_checkpoint.is_file():
            raise FileNotFoundError("SAM2 checkpoint is missing")
        from prepare_segmented_references import prepare_references
        from score_ip_similarity import load_ip_models, score_benchmark

        detector, dino = load_ip_models(args.xverse_root, "cuda:0")
        reference_root = args.run_root / "r"
        if not valid_reference_artifacts(reference_root):
            prepare_references(
                benchmark_root=args.benchmark_root,
                xverse_root=args.xverse_root,
                device="cuda:0",
                output_root=reference_root,
                detector=detector,
            )
            checkpoint()
        for benchmark in pending:
            score_benchmark(
                benchmark_root=args.benchmark_root,
                benchmark=benchmark,
                generated_manifest=manifests[benchmark],
                xverse_root=args.xverse_root,
                device="cuda:0",
                output=outputs[benchmark],
                artifact_root=args.run_root / "a" / benchmark,
                segmented_reference_root=reference_root,
                detector=detector,
                dino=dino,
            )
            checkpoint()

    incomplete = [
        benchmark
        for benchmark, output in outputs.items()
        if not valid_score_output(output, benchmark)
    ]
    if incomplete:
        raise RuntimeError(f"Metric outputs failed validation: {incomplete}")


def valid_score_output(path: Path, benchmark: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected_id, expected_count = EXPECTED[benchmark]
        scores = payload.get("scores", [])
        keys = {
            (str(row["record_id"]), str(row["condition"]), int(row["seed"]))
            for row in scores
        }
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    structurally_valid = (
        payload.get("benchmark_id") == expected_id
        and len(scores) == expected_count
        and len(keys) == expected_count
    )
    if not structurally_valid:
        return False
    if payload.get("metric") == "IP-Sim":
        return any(int(row.get("detected_candidate_count", 0)) > 0 for row in scores)
    return True


def valid_reference_artifacts(root: Path) -> bool:
    manifest = root / "segmentation_manifest.json"
    if not manifest.is_file():
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        records = payload.get("records", [])
        compatibility = payload.get("florence2_compatibility") or {}
        return (
            payload.get("schema_version") == 4
            and payload.get("fallback_method") is None
            and (compatibility.get("after") or {}).get(
                "lm_head_tied_to_shared"
            )
            is True
            and len(records) == 29
            and all(
                row.get("segmentation_method")
                == "florence2_open_vocabulary+sam2"
                and row.get("segmentation_query_attempts")
                and (root / row["foreground_crop"]).is_file()
                and (root / row["binary_mask"]).is_file()
                for row in records
            )
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
