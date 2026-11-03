"""1024x1024 single-reference seed-42 generation; outputs compatible with the shared evaluator."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from freecus.manifest import PROJECT, sha256
from freecus.run import artifact_paths, complete, write_json
from .protocol import BASE, selected_rows, validate_config, verify_upstream


def execute(args, event=None):
    cfg = json.loads(args.config.read_text())
    validate_config(cfg)
    verify_upstream()
    rows, manifest = selected_rows(args.benchmark, args.data_root, args.manifest)
    selected = rows
    if args.case_ids is not None:
        wanted = set(args.case_ids)
        selected = [r for r in rows if r["case_id"] in wanted]
        if wanted != {r["case_id"] for r in selected}:
            raise ValueError("Unknown or excluded case IDs")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        selected = selected[:args.limit]
    lock = json.loads((BASE / "models.lock.json").read_text())
    source_files = [*BASE.glob("*.py"), *BASE.glob("*.txt"), *BASE.glob("upstream/**/*.py")]
    shared = [PROJECT / "freecus" / name for name in ["manifest.py", "run.py", "evaluate.py", "project_metrics/benchmark_metrics.py"]]
    code = {str(f.relative_to(PROJECT)): sha256(f) for f in sorted(source_files + shared)}
    cfg["seeds"] = [cfg["seed"]]
    identity = dict(config=cfg, manifest_sha256=sha256(manifest), models=lock, code=code)
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    print(json.dumps(dict(benchmark=args.benchmark, samples=len(rows), selected=len(selected),
                          resolution="1024x1024", fingerprint=fingerprint), indent=2), flush=True)
    if args.dry_run:
        return
    import torch
    from PIL import Image
    from transformers import set_seed
    from .runtime import Engine, task_prompt
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("UNO primary configuration requires a BF16-capable CUDA GPU")
    versions = {n: importlib.metadata.version(n) for n in ["torch", "torchvision", "transformers", "numpy", "Pillow",
                                                         "einops", "safetensors", "tokenizers", "sentencepiece"]}
    marker = args.output / "run.json"
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous["fingerprint"] != fingerprint or previous["versions"] != versions:
            raise ValueError("Output run or installed libraries differ; choose a new output directory")
    else:
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError("Existing output has no run identity")
        write_json(marker, dict(**identity, fingerprint=fingerprint, versions=versions,
                               gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda))
    pending = [r for r in selected if not complete(artifact_paths(args.output, r))]
    if not pending:
        return
    paths = json.loads(args.resolved_models.read_text())
    if set(paths) != set(lock) or any(Path(paths[k]).name != lock[k]["revision"] for k in lock):
        raise ValueError("Resolved model paths must be immutable HF snapshot directories from models.lock.json")
    set_seed(cfg["seed"])
    torch.cuda.reset_peak_memory_stats()
    began = time.perf_counter()
    engine = Engine(cfg, paths, lock, args.output)
    initialization = dict(phase="models_loaded", seconds=time.perf_counter() - began,
                          peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                          peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
    initialization["model_dtypes"] = engine.model_dtypes
    write_json(args.output / "initialization.json", initialization)
    if event:
        event(initialization)
    for row in pending:
        task, prompt = task_prompt(row)
        began = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        condition_path = args.output / "conditioning" / (row["subject"] + ".png")
        if condition_path.exists():
            with Image.open(condition_path) as opened:
                conditioning = opened.convert("RGB")
        else:
            set_seed(row["seed"])
            conditioning = engine.conditioning(args.data_root / row["reference"], task)
            condition_path.parent.mkdir(parents=True, exist_ok=True)
            conditioning.save(condition_path)
        image = engine.generate(row, conditioning)
        torch.cuda.synchronize()
        if image.size != (1024, 1024):
            raise ValueError("Unexpected generated resolution")
        output = artifact_paths(args.output, row)
        for path in output.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output["tgt_image"])
        with Image.open(args.data_root / row["reference"]) as ref:
            ref.convert("RGB").save(output["src_image"])
        output["text"].write_text(row["prompt"], encoding="utf-8")
        timing = dict(seconds=time.perf_counter() - began, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
        write_json(output["metadata"], dict(**row, fingerprint=fingerprint, generation_prompt=prompt,
                   uno_task=task, conditioning_sha256=sha256(condition_path), timing=timing,
                   artifact_sha256={k: sha256(v) for k, v in output.items() if k != "metadata"}))
        if event:
            event(dict(phase="generated", case_id=row["case_id"], **timing))
        print("saved", row["case_id"], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", choices=["dreambench", "dreambench_plus"], required=True)
    p.add_argument("--config", type=Path, default=BASE / "config.json")
    p.add_argument("--manifest", type=Path)
    p.add_argument("--data-root", type=Path, default=PROJECT)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resolved-models", type=Path, default=BASE / "outputs/resolved_models.json")
    p.add_argument("--case-ids", nargs="+")
    p.add_argument("--limit", type=int)
    p.add_argument("--dry-run", action="store_true")
    execute(p.parse_args())


if __name__ == "__main__":
    main()
