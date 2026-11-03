"""Resolve immutable HF snapshots now; optionally download for a later GPU run."""
import argparse
import json
from pathlib import Path


def main():
    from huggingface_hub import HfApi, snapshot_download
    from .run import write_json
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    p.add_argument("--lock", type=Path, required=True)
    p.add_argument("--download", action="store_true")
    p.add_argument("--resolved-config", type=Path)
    a = p.parse_args()
    if a.download and a.resolved_config is None:
        p.error("--resolved-config is required with --download")
    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    if a.lock.exists():
        lock = json.loads(a.lock.read_text(encoding="utf-8"))
        if {k: v["repo"] for k, v in lock.items()} != cfg["models"]:
            raise ValueError("Lock repositories differ from config")
    else:
        api = HfApi()
        lock = {k: dict(repo=repo, revision=api.model_info(repo).sha) for k, repo in cfg["models"].items()}
        write_json(a.lock, lock)
    if a.download:
        cfg["models"] = {k: snapshot_download(v["repo"], revision=v["revision"],
                          ignore_patterns=["*.bin", "*.pt", "*.pth", "*.onnx", "*.h5", "*.msgpack"])
                         for k, v in lock.items()}
        write_json(a.resolved_config, cfg)
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
