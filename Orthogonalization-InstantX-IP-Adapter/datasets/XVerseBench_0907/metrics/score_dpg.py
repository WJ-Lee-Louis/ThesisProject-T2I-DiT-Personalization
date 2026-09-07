"""Manifest-based DPG scoring adapted from the official XVerse evaluator."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from common import (
    dependency_filtered_scores,
    grouped_question_scores,
    import_official_xverse,
    load_benchmark,
    load_generated_images,
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
    return parser


def main() -> int:
    args = build_parser().parse_args()
    import_official_xverse(args.xverse_root)
    from eval.tools.dpg_score import DPGScore

    manifest, dsg_by_record = load_benchmark(args.benchmark_root, args.benchmark)
    rows = load_generated_images(args.generated_manifest, manifest)
    model = DPGScore(args.device)
    scores = []
    for row in rows:
        dsg = dsg_by_record[row["record_id"]]
        with Image.open(row["image_path"]) as opened:
            image = opened.convert("RGB")
        result = model(image, dsg)
        filtered = dependency_filtered_scores(result)
        factor_scores = grouped_question_scores(filtered, dsg.get("qid2factor"))
        component_scores = grouped_question_scores(
            filtered, dsg.get("qid2component")
        )
        action_background = [
            filtered[qid]
            for qid, factor in dsg.get("qid2factor", {}).items()
            if factor in {"action", "background"}
        ]
        joint_score = None
        if row["benchmark_record"].get("composition") == "O+A+B":
            joint_score = 100.0 * min(action_background)
        scores.append(
            {
                "record_id": row["record_id"],
                "condition": row["condition"],
                "seed": row["seed"],
                "image_path": row["image_path"],
                "dpg_with_dependency": result["average_score_with_dependency"],
                "dpg_without_dependency": result[
                    "average_score_without_dependency"
                ],
                "qid2answer": result["qid2answer"],
                "qid2score_with_dependency": filtered,
                "factor_scores": factor_scores,
                "component_scores": component_scores,
                "joint_action_background_score": joint_score,
            }
        )
    write_json(
        args.output,
        {
            "schema_version": 1,
            "metric": "DPG",
            "benchmark_id": manifest["benchmark_id"],
            "official_model": "mPLUG VQA via official XVerse DPGScore",
            "scores": scores,
        },
    )
    print(f"DPG scoring complete: {len(scores)} images -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
