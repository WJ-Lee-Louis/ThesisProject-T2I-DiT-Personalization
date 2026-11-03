"""Fetch only the official 15 MB DreamBench++ dataset archive."""
import argparse
import zipfile
from pathlib import Path


def main():
    from huggingface_hub import HfApi, hf_hub_download
    from .manifest import sha256
    from .run import write_json
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("freecus/data/dreambench_plus"))
    a = p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        p.error("Output must be empty to avoid mixing dataset versions")
    repo = "yuangpeng/dreambench_plus"
    revision = HfApi().dataset_info(repo).sha
    archive = hf_hub_download(repo, "data.zip", repo_type="dataset", revision=revision)
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            if not (a.output / item.filename).resolve().is_relative_to(a.output.resolve()):
                raise ValueError("Unsafe archive member")
        z.extractall(a.output)
    write_json(a.output / "SOURCE.json", dict(repo=repo, revision=revision, archive_sha256=sha256(archive)))
    print(f"Extracted to {a.output}; use the directory containing images/ and captions/ as --data-root")


if __name__ == "__main__":
    main()
