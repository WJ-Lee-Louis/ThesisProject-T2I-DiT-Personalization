"""Remote functions, with persistent job identity and bounded generation time."""
import hashlib
import json
from pathlib import Path
import statistics
import time
import traceback

from .config import CONDITIONS, MODELS, SETTINGS, worker_rate
from .io import digest, extract, read, sha, write

WORK = Path("/work")


def stage(bundle_id, runs, cache):
    from huggingface_hub import snapshot_download
    runs.reload()
    cache.reload()
    folder = WORK / "bundles" / bundle_id
    extract(WORK / "uploads" / (bundle_id + ".zip"), folder)
    for name, expected in read(folder / "bundle.json").items():
        if sha(folder / name) != expected:
            raise ValueError("Input hash mismatch: " + name)
    resolved = {}
    patterns = {
        "base": ["model_index.json", "scheduler/*", "tokenizer/*", "tokenizer_2/*",
                 "text_encoder/config.json", "text_encoder/*.safetensors", "text_encoder_2/config.json",
                 "text_encoder_2/*.safetensors", "text_encoder_2/*.index.json", "transformer/config.json",
                 "transformer/*.safetensors", "transformer/*.index.json", "vae/config.json", "vae/*.safetensors"],
        "adapter": ["ip_adapter.safetensors"],
        "clip": ["*.json", "merges.txt", "vocab.json", "model.safetensors"],
    }
    try:
        for name, (repo, revision) in MODELS.items():
            resolved[name] = snapshot_download(repo, revision=revision, allow_patterns=patterns[name],
                                               token=__import__("os").environ.get("HF_TOKEN"))
            cache.commit()
        write(folder / "resolved.json", resolved)
        write(folder / "stage.json", dict(status="ready", models=MODELS))
        runs.commit()
        return dict(status="ready", bundle_id=bundle_id)
    finally:
        cache.commit()


def signature(plan, row, condition):
    return digest(dict(code=plan["code"], models=plan["models"], settings=plan["settings"],
                       row=row, condition=condition))


def result_paths(bundle_id, condition, row):
    stem = WORK / "results" / bundle_id / condition / row["benchmark"] / row["case_id"]
    return stem.with_suffix(".png"), stem.with_suffix(".json")


def complete(plan, row, condition):
    png, sidecar = result_paths(plan["bundle_id"], condition, row)
    if not png.exists() or not sidecar.exists():
        return False
    meta = read(sidecar)
    if meta["signature"] != signature(plan, row, condition) or sha(png) != meta["png_sha256"]:
        raise ValueError("Existing result identity/hash differs")
    return True


def generate(bundle_id, cases, invocation, seconds_cap, gpu, runs, cache, pilot=False, parity=False):
    from importlib.metadata import version
    import torch
    from .inference import Engine
    runs.reload()
    cache.reload()
    folder = WORK / "bundles" / bundle_id
    plan = read(folder / "plan.json")
    gpu_marker = folder / "generation_gpu.json"
    if gpu_marker.exists() and read(gpu_marker)["gpu"] != gpu:
        raise ValueError("GPU changed within this bundle; use a new bundle, never mix pilot hardware")
    write(gpu_marker, dict(gpu=gpu))
    runs.commit()
    selected = {r["case_id"]: r for r in plan["rows"]}
    if len(cases) != len(set(cases)) or any(c not in selected for c in cases):
        raise ValueError("Invalid case selection")
    state_path = WORK / "state" / bundle_id / (invocation + ".json")
    started = time.monotonic()
    result = dict(status="loading", invocation=invocation, gpu=gpu, events=[], seconds_cap=seconds_cap,
                  versions={p: version(p) for p in ("torch", "diffusers", "transformers", "Pillow")})

    def publish():
        result["worker_seconds"] = time.monotonic() - started
        result["estimated_compute_usd"] = (result["worker_seconds"] + 10) * worker_rate(gpu)
        write(state_path, result)
        runs.commit()

    publish()
    try:
        engine = Engine(read(folder / "resolved.json"))
        torch.cuda.synchronize()
        result["initialization_seconds"] = time.monotonic() - started
        result["initialization_peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 1024**3
        result["host_peak_rss_gib"] = __import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss / 1024**2
        result["status"] = "generating"
        publish()
        if parity:
            row = selected[cases[0]]
            payload = read(folder / row["conditioning"])
            image_bytes = (folder / row["reference"]).read_bytes()
            original, _ = engine.generate(row, payload, image_bytes, "baseline")
            zero, _ = engine.generate(row, payload, image_bytes, "source_target", strength=0)
            from PIL import Image, ImageChops
            from io import BytesIO
            a, b = Image.open(BytesIO(original)), Image.open(BytesIO(zero))
            result["parity"] = dict(png_identical=original == zero,
                                     max_pixel_difference=max(x[1] for x in ImageChops.difference(a, b).getextrema()))
            if result["parity"]["max_pixel_difference"] > 1:
                raise ValueError("Strength-zero baseline parity failed")
            publish()
        for case in cases:
            row = selected[case]
            for condition in CONDITIONS:
                if complete(plan, row, condition):
                    continue
                elapsed = time.monotonic() - started
                timings = [e["diagnostics"]["seconds"] for e in result["events"]]
                # Leave room for one complete sample, save/commit and container shutdown.
                allowance = max(timings[-4:] or [75]) * 1.25 + 15
                if elapsed + allowance >= seconds_cap:
                    result["status"] = "budget_checkpoint"
                    publish()
                    return result
                png_bytes, diag = engine.generate(row, read(folder / row["conditioning"]),
                                                  (folder / row["reference"]).read_bytes(), condition)
                png, sidecar = result_paths(bundle_id, condition, row)
                png.parent.mkdir(parents=True, exist_ok=True)
                temp = png.with_suffix(".png.tmp")
                temp.write_bytes(png_bytes)
                temp.replace(png)
                meta = dict(row=row, condition=condition, signature=signature(plan, row, condition),
                            settings=SETTINGS, models=MODELS, code=plan["code"], gpu=gpu,
                            versions=result["versions"], png_sha256=hashlib.sha256(png_bytes).hexdigest(),
                            diagnostics=diag)
                write(sidecar, meta)
                # Full semantic/geometry diagnostics live in the per-image sidecar.
                # Keep the repeatedly committed progress state small and bounded.
                progress_diag = {k: v for k, v in diag.items() if k != "orthogonalization"}
                progress_diag["basis_rank"] = diag["orthogonalization"].get("basis_rank")
                result["events"].append(dict(case_id=case, condition=condition, diagnostics=progress_diag))
                result["host_peak_rss_gib"] = __import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss / 1024**2
                publish()
                print(f"SAVED {case}/{condition} {diag['seconds']:.2f}s", flush=True)
                if diag["peak_reserved_gib"] > diag["gpu_total_gib"] - max(4, .1 * diag["gpu_total_gib"]):
                    result["status"] = "vram_review_required"
                    publish()
                    return result
        result["status"] = "complete"
        result["mean_image_seconds"] = statistics.mean([e["diagnostics"]["seconds"] for e in result["events"]]) if result["events"] else 0
        publish()
        return result
    except Exception:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()[-10000:]
        publish()
        raise
