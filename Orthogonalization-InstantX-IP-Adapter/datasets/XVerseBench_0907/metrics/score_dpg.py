"""Manifest-based DPG scoring adapted from the official XVerse evaluator."""

from __future__ import annotations

import argparse
import os
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
    score_benchmark(
        benchmark_root=args.benchmark_root,
        benchmark=args.benchmark,
        generated_manifest=args.generated_manifest,
        xverse_root=args.xverse_root,
        device=args.device,
        output=args.output,
    )
    return 0


class LeanMPlugVQA:
    """Run the official mPLUG VQA checkpoint without ModelScope's broad pipeline.

    ModelScope's generic multi-modal pipeline imports optional audio/video/OFA
    preprocessors that XVerse DPG never uses.  Keeping the mPLUG checkpoint and
    its exact image/text preprocessing while loading the model directly avoids
    that unrelated dependency chain.
    """

    def __init__(self, checkpoint: str, device: str):
        import torch
        import yaml
        from modelscope.models import Model
        from modelscope.outputs import OutputKeys
        from modelscope.utils.constant import Tasks
        from torchvision import transforms
        from transformers import BertTokenizer

        self.device = torch.device(device)
        self.checkpoint = checkpoint
        self.output_key = OutputKeys.TEXT
        self.model = Model.from_pretrained(
            checkpoint,
            task=Tasks.visual_question_answering,
            device=str(self.device),
        )
        # ModelScope stores ``device`` in the model config but does not move
        # this legacy mPLUG wrapper itself.  Its generic Pipeline normally
        # performs this step, so the lean loader must do it explicitly.
        self.model.to(self.device)
        self.model.eval()
        model_dir = Path(self.model.model_dir)
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        config = yaml.safe_load((model_dir / "config.yaml").read_text(encoding="utf-8"))
        image_resolution = int(config["image_res"])
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_resolution, image_resolution),
                    interpolation=Image.Resampling.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def vqa(self, image: Image.Image, question: str) -> str:
        import torch

        image_tensor = self.image_transform(image.convert("RGB"))
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        question_tokens = self.tokenizer(
            question.lower(),
            padding="max_length",
            truncation=True,
            max_length=25,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            result = self.model({"image": image_tensor, "question": question_tokens})
        return str(result[self.output_key])


def validate_dpg_runtime() -> dict[str, str]:
    """Import the exact lean DPG runtime without downloading the checkpoint."""

    import addict
    import modelscope
    from modelscope.fileio.format import jsonplus  # noqa: F401
    from modelscope.hub.api import HubApi
    from modelscope.models import Model  # noqa: F401
    from modelscope.models.multi_modal.mplug import MPlug  # noqa: F401
    from modelscope.models.multi_modal.mplug_for_all_tasks import (  # noqa: F401
        MPlugForAllTasks,
    )
    from modelscope.outputs import OutputKeys  # noqa: F401
    from modelscope.utils.constant import Tasks  # noqa: F401
    from transformers import BertTokenizer  # noqa: F401

    checkpoint = os.getenv(
        "DPG_VQA_MODEL_PATH",
        "damo/mplug_visual-question-answering_coco_large_en",
    )
    HubApi().get_model(checkpoint)
    return {
        "modelscope": str(modelscope.__version__),
        "addict": str(getattr(addict, "__version__", "installed")),
        "loader": "direct-modelscope-mplug",
        "checkpoint": checkpoint,
    }


def load_dpg_model(xverse_root: Path, device: str):
    import_official_xverse(xverse_root)
    from eval.tools.dpg_score import DPGScore

    checkpoint = os.getenv(
        "DPG_VQA_MODEL_PATH",
        "damo/mplug_visual-question-answering_coco_large_en",
    )
    scorer = DPGScore.__new__(DPGScore)
    scorer.device = device
    scorer.vqa_model = LeanMPlugVQA(checkpoint, device)
    return scorer


def score_benchmark(
    *,
    benchmark_root: Path,
    benchmark: str,
    generated_manifest: Path,
    xverse_root: Path,
    device: str,
    output: Path,
    model=None,
) -> None:
    model = model or load_dpg_model(xverse_root, device)

    manifest, dsg_by_record = load_benchmark(benchmark_root, benchmark)
    rows = load_generated_images(generated_manifest, manifest)
    scores = []
    for row in rows:
        dsg = dsg_by_record[row["record_id"]]
        with Image.open(row["image_path"]) as opened:
            image = opened.convert("RGB")
        result = model(image, dsg)
        filtered = dependency_filtered_scores(result)
        factor_scores = grouped_question_scores(filtered, dsg.get("qid2factor"))
        component_scores = grouped_question_scores(filtered, dsg.get("qid2component"))
        action_background = [
            filtered[qid]
            for qid, factor in dsg.get("qid2factor", {}).items()
            if factor in {"action", "background"}
        ]
        joint_score = None
        if row["benchmark_record"].get("composition") == "O+A+B" and action_background:
            joint_score = 100.0 * min(action_background)
        scores.append(
            {
                "record_id": row["record_id"],
                "condition": row["condition"],
                "seed": row["seed"],
                "image_path": row["image_path"],
                "dpg_with_dependency": result["average_score_with_dependency"],
                "dpg_without_dependency": result["average_score_without_dependency"],
                "qid2answer": result["qid2answer"],
                "qid2score_with_dependency": filtered,
                "factor_scores": factor_scores,
                "component_scores": component_scores,
                "joint_action_background_score": joint_score,
            }
        )
    write_json(
        output,
        {
            "schema_version": 1,
            "metric": "DPG",
            "benchmark_id": manifest["benchmark_id"],
            "official_model": "mPLUG VQA via official XVerse DPGScore",
            "model_checkpoint": getattr(model.vqa_model, "checkpoint", None),
            "scores": scores,
        },
    )
    print(f"DPG scoring complete: {len(scores)} images -> {output}")


if __name__ == "__main__":
    raise SystemExit(main())
