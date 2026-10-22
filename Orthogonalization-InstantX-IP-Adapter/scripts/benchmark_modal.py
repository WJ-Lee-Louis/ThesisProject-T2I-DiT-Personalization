"""Offline plan/resume/export and opt-in Modal generation from the saved joint caches."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
MANIFEST = ROOT / "outputs/semantic_benchmark_gpt56_sol/generation_manifest.json"
OUTPUT = ROOT / "outputs/benchmark_images_v1"
CONDITIONS = ("baseline", "ssg", "source", "source_target", "source_target_ssg", "target")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def implementation_digest():
    paths = [Path(__file__)] + sorted((ROOT / "orthogonalization_instantx_ip_adapter").glob("*.py"))
    paths += sorted((ROOT.parent / "shared").glob("*.py"))
    return digest([(str(p.relative_to(ROOT.parent)), hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths])


def select_rows(rows, benchmark, shard_index, num_shards):
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("require 0 <= shard-index < num-shards")
    selected = [r for r in rows if benchmark == "all" or r["benchmark"] == benchmark]
    # Partition subjects separately in each benchmark, keeping every subject's prompts together.
    owners = {}
    for name in sorted({r["benchmark"] for r in selected}):
        refs = sorted({r["reference_id"] for r in selected if r["benchmark"] == name})
        owners.update({ref: i % num_shards for i, ref in enumerate(refs)})
    return [r for r in selected if owners[r["reference_id"]] == shard_index]


def paths_for(output, condition, row):
    stem = output / condition / row["benchmark"] / row["case_id"]
    return stem.with_suffix(".png"), stem.with_suffix(".json")


def complete(png, sidecar, signature):
    if not png.exists() or not sidecar.exists():
        return False
    record = read(sidecar)
    if record["job_signature"] != signature:
        raise ValueError(f"Settings/input/code mismatch: {sidecar}. Use a new --output directory.")
    if hashlib.sha256(png.read_bytes()).hexdigest() != record["png_sha256"]:
        raise ValueError(f"Image hash mismatch: {png}")
    from PIL import Image
    with Image.open(png) as image:
        image.verify()
    return True


def prepare(args):
    from orthogonalization_instantx_ip_adapter.semantic_contrasts import ReferenceFingerprint, SemanticContrastConditioning
    rows = read(MANIFEST)
    counts = {b: sum(r["benchmark"] == b for r in rows) for b in ("dreambench", "dreambench_plus")}
    if counts != {"dreambench": 750, "dreambench_plus": 1350} or len({r["case_id"] for r in rows}) != 2100:
        raise ValueError(f"Unexpected manifest: {counts}")
    if any(r["status"] != "READY" for r in rows):
        raise ValueError("Manifest contains non-READY cases")
    selected = select_rows(rows, args.benchmark, args.shard_index, args.num_shards)
    settings = dict(condition=args.condition, strength=args.strength, seed=args.seed,
                    steps=args.steps, width=args.width, height=args.height,
                    ip_adapter_scale=0.7, guidance_scale=3.5, gpu=args.gpu,
                    implementation_sha256=implementation_digest())
    jobs, completed = [], 0
    for row in selected:
        image_bytes = (ROOT / row["image_path"]).read_bytes()
        raw = read(ROOT / row["conditioning_path"])
        # Validate baseline inputs as well. Never instantiate an API provider.
        cached = SemanticContrastConditioning.from_dict(raw)
        cached.validate_reuse(identity_term=row["identity_term"], target_prompt=row["target_prompt"],
                              reference=ReferenceFingerprint.from_image(image_bytes), model="gpt-5.6-sol")
        payload = None
        if args.condition not in ("baseline", "ssg"):
            origin = "source_target" if args.condition.endswith("ssg") else args.condition
            cached = replace(cached, origin=origin)
            payload = cached.to_dict()
        signature = digest(dict(row=row, settings=settings, cache=raw,
                                image_sha256=hashlib.sha256(image_bytes).hexdigest()))
        png, sidecar = paths_for(args.output, args.condition, row)
        if complete(png, sidecar, signature):
            completed += 1
            continue
        jobs.append(dict(row=row, settings=settings, signature=signature,
                         image_bytes=image_bytes, payload=payload))
    print(json.dumps(dict(selected=len(selected), complete=completed, pending=len(jobs),
                          fallbacks=[r["case_id"] for r in selected if r.get("fallback")],
                          shard=args.shard_index, num_shards=args.num_shards, settings=settings), indent=2))
    return jobs[:args.limit] if args.limit else jobs


def export(args):
    from PIL import Image
    rows = select_rows(read(MANIFEST), args.benchmark, 0, 1)
    if args.benchmark == "all":
        raise ValueError("Export one benchmark at a time")
    refs = {r["id"]: r for r in read(ROOT / "datasets/semantic_benchmark/references.json")}
    # Validate the entire result set before creating evaluator inputs.
    signatures = set()
    for row in rows:
        png, sidecar = paths_for(args.output, args.condition, row)
        record = read(sidecar)
        if record["row"] != row or record["settings"]["condition"] != args.condition:
            raise ValueError(f"Provenance mismatch: {sidecar}")
        if not complete(png, sidecar, record["job_signature"]):
            raise ValueError(f"Missing image: {png}")
        signatures.add(digest(record["settings"]))
    if len(signatures) != 1:
        raise ValueError("Mixed generation settings/code revisions")
    destination = args.export_dir.resolve()
    if destination.exists():
        raise ValueError(f"Export destination must be new: {destination}")
    for index, row in enumerate(rows):
        png, _ = paths_for(args.output, args.condition, row)
        if args.benchmark == "dreambench":
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(png, destination / f"{index}_0.png")
        else:
            ref = refs[row["reference_id"]]
            collection = ref["subject_id"].replace("/", "_") + "_" + ref["identity_term"].lower().replace(" ", "_")
            prompt_index = int(row["case_id"].split("-")[-1])
            for folder in ("src_image", "tgt_image", "text"):
                (destination / folder / collection).mkdir(parents=True, exist_ok=True)
            # The upstream GPT scripts require .jpg. Keep original PNG results for audit.
            for source, folder in ((ROOT / row["image_path"], "src_image"), (png, "tgt_image")):
                with Image.open(source) as image:
                    image.convert("RGB").save(destination / folder / collection / f"{prompt_index}_0.jpg", quality=95)
            (destination / "text" / collection / f"{prompt_index}_0.txt").write_text(row["target_prompt"], encoding="utf-8")
    print(f"Exported {len(rows)} cases to {destination}")


def build_generator(args):
    import modal
    from modal_inference import image, hf_cache
    from shared.modal_runtime import HF_SECRET_NAME
    app = modal.App("orthogonalization-dreambench-semantic-v1")

    @app.cls(image=image, gpu=args.gpu, serialized=True, volumes={"/cache": hf_cache},
             secrets=[modal.Secret.from_name(HF_SECRET_NAME)], max_containers=args.workers,
             min_containers=0, scaledown_window=60, timeout=1800)
    class Generator:
        @modal.enter()
        def enter(self):
            self.model = None

        @modal.method()
        def generate(self, job):
            from orthogonalization_instantx_ip_adapter import inference
            from orthogonalization_instantx_ip_adapter.semantic_contrasts import SemanticContrastConditioning, ReferenceFingerprint
            started = time.monotonic()
            row, cfg = job["row"], job["settings"]
            if job["payload"] is not None:
                cached = SemanticContrastConditioning.from_dict(job["payload"])
                cached.validate_reuse(identity_term=row["identity_term"], target_prompt=row["target_prompt"],
                                      reference=ReferenceFingerprint.from_image(job["image_bytes"]), model="gpt-5.6-sol")
            if self.model is None:
                self.model = inference.load_ip_adapter_model()
            loader = inference.load_ip_adapter_model
            try:
                inference.load_ip_adapter_model = lambda: self.model
                png, diagnostics = inference.generate_png_bytes(
                    prompt=row["target_prompt"], reference_image_url="", reference_image_bytes=job["image_bytes"],
                    ip_adapter_scale=cfg["ip_adapter_scale"], guidance_scale=cfg["guidance_scale"],
                    num_inference_steps=cfg["steps"], seed=cfg["seed"], width=cfg["width"], height=cfg["height"],
                    orthogonalization_mode="semantic_local" if job["payload"] is not None else "off",
                    orthogonalization_strength=cfg["strength"], object_prompt="", object_behavior_prompt="",
                    object_background_prompt="", object_behavior_background_prompt="", remove_behavior=True,
                    remove_background=True, include_interaction=False, restore_visual_norm=True,
                    mask_single_text_rows=cfg["condition"] in ("ssg", "source_target_ssg"),
                    contrast_analysis_payload=job["payload"])
            finally:
                inference.load_ip_adapter_model = loader
            return dict(png=png, row=row, settings=cfg, job_signature=job["signature"], diagnostics=diagnostics,
                        elapsed_seconds=time.monotonic() - started, png_sha256=hashlib.sha256(png).hexdigest())

    return app, Generator


def run(args, jobs):
    import modal
    app, Generator = build_generator(args)
    failures = 0
    # One serialized input per container; no concurrent calls into a mutable diffusion pipeline.
    with modal.enable_output(), app.run():
        for result in Generator().generate.map(jobs, order_outputs=False, return_exceptions=True):
            if isinstance(result, BaseException):
                failures += 1
                print(f"FAILED: {result}", file=sys.stderr)
                continue
            png, sidecar = paths_for(args.output, args.condition, result["row"])
            png.parent.mkdir(parents=True, exist_ok=True)
            data = result.pop("png")
            result["modal_profile"] = os.environ["MODAL_PROFILE"]
            temp = png.with_suffix(".png.tmp")
            temp.write_bytes(data)
            temp.replace(png)
            temp = sidecar.with_suffix(".json.tmp")
            temp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            temp.replace(sidecar)
            print(f"SAVED {result['row']['case_id']}: {png}", flush=True)
    if failures:
        raise RuntimeError(f"{failures} jobs failed; rerun the identical command to retry missing outputs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("all", "dreambench", "dreambench_plus"), default="all")
    parser.add_argument("--condition", choices=CONDITIONS, default="source_target")
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gpu", default="A100-80GB")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=0, help="maximum pending jobs to submit; 0 means all")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--export-dir", type=Path)
    parser.add_argument("--run", action="store_true", help="submit paid GPU jobs; otherwise validate offline")
    args = parser.parse_args()
    if args.workers < 1 or args.limit < 0 or args.steps < 1 or min(args.width, args.height) < 16 or not 0 <= args.strength <= 1:
        parser.error("invalid workers/limit/steps/dimensions/strength")
    if args.export_dir:
        export(args)
        return
    jobs = prepare(args)
    if args.run and jobs:
        if not os.environ.get("MODAL_PROFILE"):
            parser.error("set MODAL_PROFILE explicitly for this terminal")
        if os.environ.get("MODAL_TOKEN_ID") or os.environ.get("MODAL_TOKEN_SECRET"):
            parser.error("remove MODAL_TOKEN_ID/SECRET overrides before profile-based execution")
        run(args, jobs)


if __name__ == "__main__":
    main()
