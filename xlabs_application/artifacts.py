"""Lossless ZIP download with the same 1 GiB bound and hash verification as FreeCus."""
import hashlib
import json
from pathlib import Path
import zipfile

from .config import CONDITIONS
from .io import extract, read, sha, write
from .runtime import complete, result_paths
from .vendor.artifact_bundle import verify

LIMIT = 1024**3


def export(bundle_id, cases, shard, runs):
    runs.reload()
    folder = Path("/work/bundles") / bundle_id
    plan = read(folder / "plan.json")
    selected = [r for r in plan["rows"] if r["case_id"] in cases]
    if len(selected) != len(set(cases)):
        raise ValueError("Invalid shard cases")
    out = Path("/work/exports") / bundle_id / shard
    if (out / "bundle_index.json").exists():
        index = read(out / "bundle_index.json")
        verify(out, index)
        return dict(index=(out / "bundle_index.json").relative_to("/work").as_posix())
    files = {}
    for row in selected:
        for condition in CONDITIONS:
            if not complete(plan, row, condition):
                raise ValueError("Cannot export incomplete case pair: " + row["case_id"])
            png, sidecar = result_paths(bundle_id, condition, row)
            for path in (png, sidecar):
                files["results/" + condition + "/" + row["benchmark"] + "/" + path.name] = path
        files[row["reference"]] = folder / row["reference"]
        files[row["conditioning"]] = folder / row["conditioning"]
    out.mkdir(parents=True, exist_ok=True)
    write(out / "manifest.json", selected)
    write(out / "run.json", {k: plan[k] for k in ("bundle_id", "code", "settings", "models", "source_manifest_sha256")} |
          dict(gpu=read(folder / "generation_gpu.json")["gpu"]))
    files["manifest.json"] = out / "manifest.json"
    files["run.json"] = out / "run.json"
    parts, members, archive = [], {}, None
    try:
        for name, path in sorted(files.items()):
            size = path.stat().st_size
            if size >= LIMIT - 1024**2:
                raise ValueError("Single ZIP member too large")
            if archive is None or archive.fp.tell() + size + 4096 > LIMIT - 1024**2:
                if archive is not None:
                    archive.close()
                part = "images-" + str(len(parts)).zfill(3) + ".zip"
                parts.append(dict(file=part))
                archive = zipfile.ZipFile(out / part, "w", compression=zipfile.ZIP_STORED)
            archive.write(path, name)
            members[name] = dict(size=size, sha256=sha(path))
    finally:
        if archive:
            archive.close()
    for part in parts:
        part.update(size_bytes=(out / part["file"]).stat().st_size, sha256=sha(out / part["file"]))
        if part["size_bytes"] > LIMIT:
            raise ValueError("ZIP exceeds bound")
    index = dict(schema=1, method="xlabs_application", bundle_id=bundle_id, shard=shard,
                 benchmark=selected[0]["benchmark"], generated_images=len(cases) * 2,
                 case_ids=sorted(cases), missing_case_ids=[], members=members, parts=parts)
    verify(out, index)
    write(out / "bundle_index.json", index)
    runs.commit()
    return dict(index=(out / "bundle_index.json").relative_to("/work").as_posix())


def collect(modal, runs, plan, directory, profile):
    state = directory / profile
    receipt = read(state / "export-receipt.json")
    result = modal.FunctionCall.from_id(receipt["call_id"]).get(timeout=0)
    index = json.loads(b"".join(runs.read_file(result["index"])))
    remote = str(Path(result["index"]).parent).replace("\\", "/")
    base = Path(__file__).resolve().parent
    folder = base / "artifacts" / index["benchmark"] / (plan["bundle_id"] + "-" + profile)
    folder.mkdir(parents=True, exist_ok=True)
    for part in index["parts"]:
        path = folder / part["file"]
        if path.exists() and sha(path) == part["sha256"]:
            continue
        temp = path.with_suffix(".zip.tmp")
        with temp.open("wb") as stream:
            for block in runs.read_file(remote + "/" + part["file"]):
                stream.write(block)
        if sha(temp) != part["sha256"]:
            raise ValueError("Downloaded ZIP hash differs")
        temp.replace(path)
    write(folder / "bundle_index.json", index)
    verify(folder, index)
    restored = base / "outputs/restored" / folder.name
    restored.mkdir(parents=True, exist_ok=True)
    # A stopped download/restore must be resumable without deleting good archives.
    needs_restore = any(not (restored / name).is_file() or sha(restored / name) != member["sha256"]
                        for name, member in index["members"].items())
    if needs_restore:
        for part in index["parts"]:
            extract(folder / part["file"], restored)
    for name, member in index["members"].items():
        path = restored / name
        if not path.is_file() or sha(path) != member["sha256"]:
            raise ValueError("Restored member hash differs: " + name)
    results = base / "results" / index["benchmark"] / folder.name
    write(results / "bundle_index.json", index)
    metric_path = state / "evaluate-receipt.json"
    if metric_path.exists():
        metric = modal.FunctionCall.from_id(read(metric_path)["call_id"]).get(timeout=0)
        data = b"".join(runs.read_file(metric["path"]))
        (results / "metrics.json").write_bytes(data)
        import csv
        scores = json.loads(data)["scores"]
        fields = ["condition", "benchmark", "case_id", "reference_id", "category", "target_prompt",
                  "reference_sha256", "generated_sha256", "CLIP-T", "CLIP-I", "DINO-I"]
        with (results / "per_image.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(scores)
    for filename in ("generation_source.zip", "source_hashes.json", "plan.json"):
        __import__("shutil").copyfile(directory / filename, results / filename)
    __import__("shutil").copyfile(base / "manifests" / (index["benchmark"] + "_seed42.jsonl"), results / "manifest.jsonl")
    write(results / "provenance.json", dict(bundle_id=plan["bundle_id"], shard=profile,
          code=plan["code"], models=plan["models"], settings=plan["settings"],
          source_archive_sha256=sha(results / "generation_source.zip"),
          metric_sha256=sha(results / "metrics.json") if (results / "metrics.json").exists() else None,
          manifest_sha256=sha(results / "manifest.jsonl")))
    write(results / "verification.json", dict(archive_hashes_verified=True, members_verified=True,
                                               restored_bytes_verified=True, restored_path=str(restored),
                                               generated_images=index["generated_images"]))
    control = results / "execution_control"
    control.mkdir(exist_ok=True)
    for name in ("supervisor.py", "client.py", "io.py", "artifacts.py", "watch.py", "config.py"):
        __import__("shutil").copyfile(base / name, control / name)
    for name in ("submit-receipt.json", "initial-workflow.json"):
        if (state / name).exists():
            __import__("shutil").copyfile(state / name, control / name)
    write(control / "hashes.json", {p.name: sha(p) for p in control.iterdir() if p.is_file() and p.name != "hashes.json"})
    print(json.dumps(dict(status="verified", archives=str(folder), results=str(results)), indent=2))
