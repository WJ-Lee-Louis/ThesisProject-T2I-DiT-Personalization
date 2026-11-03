"""Stage only the checkpoint and tokenizer files used by official UNO inference."""
import json
from pathlib import Path
from freecus.run import write_json

BASE = Path(__file__).parent
FILES = {
    "flux": ["flux1-dev.safetensors", "ae.safetensors"],
    "t5": ["config.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
           "model.safetensors.index.json", "added_tokens.json", "special_tokens_map.json", "spiece.model", "tokenizer_config.json"],
    "clip": ["config.json", "model.safetensors", "merges.txt", "vocab.json", "tokenizer.json",
             "tokenizer_config.json", "special_tokens_map.json"],
    "lora": ["dit_lora.safetensors"],
}


def stage(lock, output):
    from huggingface_hub import snapshot_download
    if set(lock) != set(FILES):
        raise ValueError("Unexpected model lock")
    paths = {}
    for key, model in lock.items():
        paths[key] = snapshot_download(model["repo"], revision=model["revision"], allow_patterns=FILES[key])
        if any(not (Path(paths[key]) / name).is_file() for name in FILES[key]):
            raise FileNotFoundError("Incomplete model snapshot: " + key)
    write_json(output, paths)
    return paths


def local_download_resolver(lock, paths):
    allowed = {(model["repo"], name): Path(paths[key]) / name
               for key, model in lock.items() for name in FILES[key]}

    def resolve(repo_id, filename, **kwargs):
        path = allowed.get((repo_id, filename))
        if path is None or not path.is_file():
            raise ValueError(f"Unpinned or missing model file: {repo_id}/{filename}")
        return str(path)

    return resolve


if __name__ == "__main__":
    print(json.dumps(json.loads((BASE / "models.lock.json").read_text()), indent=2))
