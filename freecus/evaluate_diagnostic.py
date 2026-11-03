"""Alternative diagnostics only; use freecus.evaluate for project comparisons."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from .manifest import load, sha256
from .run import artifact_paths, complete, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--foreground-root", type=Path, help="Prepared SAM views: tgt_image/<id>.png and reference/<dataset relative path>")
    p.add_argument("--foreground-provenance", type=Path, help="JSON describing SAM model/checkpoint, prompts, mask selection and crop/background policy")
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    if bool(a.foreground_root) != bool(a.foreground_provenance):
        p.error("Foreground views require --foreground-provenance and vice versa")
    rows = load(a.manifest, a.data_root)
    run_info = json.loads((a.run / "run.json").read_text(encoding="utf-8"))
    if run_info["manifest_sha256"] != sha256(a.manifest):
        raise ValueError("Run manifest differs from evaluation manifest")
    selected = [r for r in rows if complete(artifact_paths(a.run, r))]
    if not selected or (len(selected) != len(rows) and not a.allow_partial):
        raise ValueError(f"Complete artifacts: {len(selected)}/{len(rows)}; use --allow-partial for diagnostics")
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
    device = a.device
    clip_id = "openai/clip-vit-base-patch32"
    dino_id = "facebook/dino-vits8"
    clip = CLIPModel.from_pretrained(clip_id).eval().to(device)
    processor = CLIPProcessor.from_pretrained(clip_id)
    dino = AutoModel.from_pretrained(dino_id).eval().to(device)
    dino_processor = AutoImageProcessor.from_pretrained(dino_id)
    cache = {}

    def features(path):
        key = str(path.resolve())
        if key not in cache:
            with Image.open(path) as im:
                image = im.convert("RGB")
            c = clip.get_image_features(**processor(images=image, return_tensors="pt").to(device))
            d = dino(**dino_processor(images=image, return_tensors="pt").to(device)).last_hidden_state[:, 0]
            cache[key] = (torch.nn.functional.normalize(c, dim=-1).cpu(), torch.nn.functional.normalize(d, dim=-1).cpu())
        return cache[key]

    scores = []
    with torch.inference_mode():
        for row in selected:
            paths = artifact_paths(a.run, row)
            meta = json.loads(paths["metadata"].read_text(encoding="utf-8"))
            if meta["fingerprint"] != run_info["fingerprint"]:
                raise ValueError("Mixed run fingerprints")
            full_c, _ = features(paths["tgt_image"])
            c, d = features(a.foreground_root / "tgt_image" / f"{row['id']}.png") if a.foreground_root else features(paths["tgt_image"])
            refs = [features((a.foreground_root / "reference" if a.foreground_root else a.data_root) / name)
                    for name in row["evaluation_references"]]
            text_inputs = processor(text=[row["prompt"]], return_tensors="pt", padding=True).to(device)
            if text_inputs.input_ids.shape[-1] > clip.config.text_config.max_position_embeddings:
                raise ValueError(f"CLIP text too long, define a common truncation policy before evaluation: {row['id']}")
            text = torch.nn.functional.normalize(clip.get_text_features(**text_inputs), dim=-1).cpu()
            scores.append(dict(id=row["id"], subject=row["subject"], category=row["category"],
                               clip_t=float((full_c * text).sum()),
                               clip_i=sum(float((c * rc).sum()) for rc, _ in refs) / len(refs),
                               dino=sum(float((d * rd).sum()) for _, rd in refs) / len(refs)))
            # Cache reference features only; generated views need not accumulate.
            for key in list(cache):
                if key == str(paths["tgt_image"].resolve()) or (a.foreground_root and key == str((a.foreground_root / "tgt_image" / f"{row['id']}.png").resolve())):
                    del cache[key]
            print(f"scored: {row['id']}", flush=True)
    grouped = defaultdict(list)
    for score in scores:
        grouped[score["category"]].append(score)
    means = lambda values: {m: sum(s[m] for s in values) / len(values) for m in ["clip_t", "clip_i", "dino"]}
    write_json(a.output, dict(protocol="prepared_SAM_views" if a.foreground_root else "full_image_diagnostic",
               paper_comparable=False, resolution=run_info["config"]["height"], sample_count=len(scores),
               total_expected=len(rows), fingerprint=run_info["fingerprint"],
               model_ids=dict(clip=clip_id, dino=dino_id),
               model_commits=dict(clip=clip.config._commit_hash, dino=dino.config._commit_hash),
               foreground_provenance=json.loads(a.foreground_provenance.read_text(encoding="utf-8")) if a.foreground_provenance else None,
               mean=means(scores), by_category={k: means(v) for k, v in grouped.items()}, samples=scores))


if __name__ == "__main__":
    main()
