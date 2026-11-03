import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

from .manifest import load, sha256


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def validate_config(cfg):
    if (cfg["height"], cfg["width"]) != (1024, 1024):
        raise ValueError("This benchmark integration requires 1024x1024")
    if cfg["implementation"] not in {"paper", "upstream"}:
        raise ValueError("implementation must be paper or upstream")
    if cfg["steps"] != 30 or cfg["guidance"] != 3.5 or cfg["scale"] != 1.1:
        raise ValueError("Baseline requires steps=30, guidance=3.5, scale=1.1")


def artifact_paths(out, row):
    subject, name = row["id"].split("/")
    return {kind: out / kind / subject / f"{name}.{ext}" for kind, ext in
            [("tgt_image", "png"), ("src_image", "png"), ("text", "txt"), ("metadata", "json")]}


def complete(paths):
    if not all(p.is_file() for p in paths.values()):
        return False
    try:
        meta = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        return all(sha256(paths[k]) == v for k, v in meta["artifact_sha256"].items())
    except (ValueError, KeyError):
        return False


def execute(a, on_event=None):
    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    validate_config(cfg)
    rows = load(a.manifest, a.data_root)
    if set(r["seed"] for r in rows) != set(cfg["seeds"]):
        raise ValueError("Manifest seeds differ from config seeds")
    if a.start < 0 or a.start >= len(rows) or (a.limit is not None and a.limit < 1):
        raise ValueError("Invalid start/limit")
    selected = rows[a.start:None if a.limit is None else a.start + a.limit]
    if getattr(a, "case_ids", None) is not None:
        wanted = set(a.case_ids)
        selected = [r for r in selected if r["case_id"] in wanted]
        if {r["case_id"] for r in selected} != wanted:
            raise ValueError("Requested case IDs are missing")
    source = Path(__file__).parent
    files = [*source.glob("*.py"), *source.glob("runtime_src/*.py"), *source.glob("upstream/*.py")]
    code_hashes = {f.relative_to(source).as_posix(): sha256(f) for f in sorted(files)}
    identity = {"config": cfg, "manifest_sha256": sha256(a.manifest), "code": code_hashes}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    print(json.dumps(dict(samples=len(rows), selected=len(selected), resolution="1024x1024",
                          implementation=cfg["implementation"], fingerprint=fingerprint), indent=2))
    if a.dry_run:
        return
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA GPU with bfloat16 support is required")
    marker = a.output / "run.json"
    versions = {name: importlib.metadata.version(name) for name in
                ["torch", "torchvision", "diffusers", "transformers", "accelerate", "numpy", "Pillow"]}
    if marker.exists():
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if previous["fingerprint"] != fingerprint:
            raise ValueError("Output directory belongs to a different run; choose a new directory")
        if previous["versions"] != versions:
            raise ValueError("Installed library versions changed; choose a new output directory")
    else:
        if a.output.exists() and any(a.output.iterdir()):
            raise ValueError("Refusing to mix existing output without run.json")
        write_json(marker, dict(fingerprint=fingerprint, **identity, versions=versions,
                               python=platform.python_version(), gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda))
    # Single writer per output directory. Use distinct output directories for future Modal shards.
    from .runtime import Engine
    engine = Engine(cfg)
    captions = {}
    pending = [r for r in selected if not complete(artifact_paths(a.output, r))]
    for row in pending:
        cache_key = row["subject"] if cfg["implementation"] == "paper" else row["id"].split("/")[0] + "_" + str(row["prompt_index"])
        if cache_key in captions:
            continue
        cache = a.output / "captions" / f"{cache_key}.json"
        if cache.exists():
            captions[cache_key] = json.loads(cache.read_text(encoding="utf-8"))
        else:
            began = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            raw, filtered = engine.describe(row, a.data_root)
            captions[cache_key] = dict(raw=raw, filtered=filtered, reference=row["reference"], seed=cfg["caption_seed"])
            write_json(cache, captions[cache_key])
            if on_event:
                on_event(dict(phase="caption", subject=row["subject"], seconds=time.perf_counter() - began,
                              peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                              peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30))
        print(f"caption ready: {cache_key}", flush=True)
    engine.release_caption_models()
    if on_event:
        on_event(dict(phase="caption_models_released", allocated_gib=torch.cuda.memory_allocated() / 2**30))
    from PIL import Image
    for row in pending:
        key = row["subject"] if cfg["implementation"] == "paper" else row["subject"] + "_" + str(row["prompt_index"])
        began = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        image, augmented_prompt = engine.generate(row, a.data_root, captions[key]["filtered"])
        torch.cuda.synchronize()
        timing = dict(seconds=time.perf_counter() - began,
                      peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
        paths = artifact_paths(a.output, row)
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        if image.size != (1024, 1024):
            raise RuntimeError(f"Unexpected generated resolution: {image.size}")
        image.save(paths["tgt_image"])
        with Image.open(a.data_root / row["reference"]) as ref:
            ref.convert("RGB").save(paths["src_image"])
        paths["text"].write_text(row["prompt"], encoding="utf-8")
        write_json(paths["metadata"], dict(**row, augmented_prompt=augmented_prompt,
                   caption=captions[key], fingerprint=fingerprint, timing=timing,
                   artifact_sha256={k: sha256(v) for k, v in paths.items() if k != "metadata"}))
        print(f"saved: {row['id']}", flush=True)
        if on_event:
            on_event(dict(phase="generated", case_id=row["case_id"], id=row["id"], **timing))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int)
    p.add_argument("--case-ids", nargs="+")
    p.add_argument("--dry-run", action="store_true")
    execute(p.parse_args())


if __name__ == "__main__":
    main()
