"""Lock model revisions and sizes without downloading weights; optional CPU staging."""
import argparse
import json
from pathlib import Path
from freecus.run import write_json

BASE = Path(__file__).parent
FILES = {
    "dreamo": ["dreamo.safetensors", "dreamo_cfg_distill.safetensors", "v1.1/dreamo_sft_lora.safetensors", "v1.1/dreamo_dpo_lora.safetensors"],
    "turbo": ["diffusion_pytorch_model.safetensors"],
    "ben2": ["BEN2_Base.pth"],
}


def stage(lock, output):
    from huggingface_hub import snapshot_download
    paths = {}
    for key, model in lock.items():
        allow = (["model_index.json", "scheduler/*", "tokenizer/*", "tokenizer_2/*", "text_encoder/*",
                  "text_encoder_2/*", "transformer/*", "vae/*"] if key == "flux" else FILES[key])
        paths[key] = snapshot_download(model["repo"], revision=model["revision"], allow_patterns=allow,
                                       ignore_patterns=["*.bin", "*.onnx", "*.h5"] if key == "flux" else None)
    write_json(output, paths)
    return paths


def main():
    from huggingface_hub import HfApi
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--download", action="store_true")
    a = p.parse_args()
    cfg = json.loads((BASE / "config.json").read_text())
    lock_path = BASE / "models.lock.json"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        if {k: v["repo"] for k, v in lock.items()} != cfg["models"]:
            raise ValueError("Model repositories changed")
    else:
        # Reuse the same FLUX revision as the established FreeCus comparison.
        flux_lock = json.loads((BASE.parent / "freecus/models.lock.json").read_text())["flux"]
        api = HfApi()
        lock, sizes = {}, {}
        for key, repo in cfg["models"].items():
            info = api.model_info(repo, revision=flux_lock["revision"] if key == "flux" else None, files_metadata=True)
            lock[key] = dict(repo=repo, revision=info.sha)
            siblings = {s.rfilename: s.size for s in info.siblings}
            selected = (sorted(n for n in siblings if n.endswith(".safetensors") and n.split("/")[0] in
                               {"transformer", "text_encoder", "text_encoder_2", "vae"}) if key == "flux" else FILES[key])
            sizes[key] = dict(repo=repo, revision=info.sha, files={n: siblings[n] for n in selected},
                              total_bytes=sum(siblings[n] for n in selected))
        write_json(lock_path, lock)
        write_json(BASE / "verification/model_sizes.json", sizes)
    if a.download:
        stage(lock, BASE / "outputs/resolved_models.json")
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
