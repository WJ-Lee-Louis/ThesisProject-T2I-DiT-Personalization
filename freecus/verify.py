"""Offline environment checks; no model weights or account connections required."""
import argparse
import importlib.metadata
import json
import platform
from pathlib import Path


def main():
    from .manifest import sha256
    from .runtime import module
    import torch
    import torchvision
    from diffusers import FluxPipeline
    from transformers import Qwen2VLForConditionalGeneration, AutoModelForImageSegmentation
    import qwen_vl_utils
    import timm
    import kornia
    import clip
    from .evaluate import project_evaluator
    evaluator = project_evaluator()
    p = argparse.ArgumentParser()
    p.add_argument("--require-gpu", action="store_true")
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    source = Path(__file__).parent
    provenance = json.loads((source / "UPSTREAM.json").read_text(encoding="utf-8"))
    for name, digest in provenance["files"].items():
        if sha256(source / "upstream" / name) != digest:
            raise ValueError(f"Upstream modified: {name}")
    for name in ["flux_utils", "qwen_for_subject", "seg_birefnet", "sa_handler_for_flux_clean"]:
        module(name)
    gpu = torch.cuda.is_available()
    report = dict(python=platform.python_version(), platform=platform.platform(),
                  versions={name: importlib.metadata.version(name) for name in
                            ["torch", "torchvision", "diffusers", "transformers", "accelerate", "numpy", "Pillow", "timm", "kornia", "clip"]},
                  cuda_available=gpu, cuda_runtime=torch.version.cuda,
                  upstream_commit=provenance["commit"], imports="passed",
                  model_inference="not_run")
    report["metric_protocol"] = evaluator.PROTOCOL["name"]
    if gpu:
        report.update(gpu=torch.cuda.get_device_name(), bfloat16=torch.cuda.is_bf16_supported())
    if a.require_gpu and not (gpu and torch.cuda.is_bf16_supported()):
        raise RuntimeError("CUDA GPU with bfloat16 support is required")
    if a.output:
        from .run import write_json
        write_json(a.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
