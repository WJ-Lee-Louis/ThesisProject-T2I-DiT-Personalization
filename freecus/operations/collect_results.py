"""Export immutable Modal results on CPU, download one archive and safely extract."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", choices=["dreambench", "dreambench_plus"], required=True)
    a = p.parse_args()
    from freecus.modal_client import connect, BASE, read
    from freecus.modal_remote import RUNS_NAME
    primary = a.benchmark == "dreambench"
    profile = "method-comparison" if primary else "method-comparison-dbpp"
    state = BASE / ("outputs/modal" if primary else f"outputs/modal-{profile}")
    plan = read(state / f"latest-{a.benchmark}.json")
    modal = connect(profile)
    volume = modal.Volume.from_name(RUNS_NAME)
    app = modal.App("freecus-result-export")

    @app.function(name="export", serialized=True, image=modal.Image.debian_slim(python_version="3.11"),
                  volumes={"/work": volume}, cpu=(1, 1), memory=(1024, 1024), timeout=3600,
                  max_containers=1, scaledown_window=2, retries=0)
    def export(bundle_id):
        import hashlib
        import json
        from pathlib import Path
        import zipfile
        volume.reload()
        root = Path("/work")
        result = root / "results" / bundle_id
        archive = root / "exports" / f"{bundle_id}.zip"
        receipt = archive.with_suffix(".json")
        if receipt.exists() and archive.exists():
            return json.loads(receipt.read_text())
        archive.parent.mkdir(parents=True, exist_ok=True)
        files = sorted(f for f in result.rglob("*") if f.is_file())
        files += [f for f in (root / "bundles" / bundle_id).glob("*.json*") if f.is_file()]
        files += sorted((root / "workflows").glob("*.json"))
        with zipfile.ZipFile(archive.with_suffix(".part"), "w", compression=zipfile.ZIP_STORED) as z:
            for f in files:
                z.write(f, f.relative_to(root).as_posix())
        archive.with_suffix(".part").replace(archive)
        digest = hashlib.sha256()
        with archive.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        info = dict(bundle_id=bundle_id, archive=archive.relative_to(root).as_posix(),
                    size_bytes=archive.stat().st_size, sha256=digest.hexdigest(), files=len(files),
                    generated_images=sum("/tgt_image/" in f.as_posix() for f in files))
        receipt.write_text(json.dumps(info, indent=2))
        volume.commit()
        return info

    receipt_path = state / "export.json"
    if receipt_path.exists():
        receipt = read(receipt_path)
        if receipt["bundle_id"] != plan["bundle_id"]:
            raise ValueError("Export receipt belongs to another bundle")
        call = modal.FunctionCall.from_id(receipt["call_id"])
    else:
        with modal.enable_output():
            app.deploy()
        call = export.spawn(plan["bundle_id"])
        receipt_path.write_text(json.dumps(dict(call_id=call.object_id, bundle_id=plan["bundle_id"]), indent=2))
    info = call.get()
    print(json.dumps(info, indent=2), flush=True)
    destination = state / "collected"
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / (plan["bundle_id"] + ".zip")

    def digest_file(path):
        d = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                d.update(chunk)
        return d.hexdigest()

    if not archive.exists() or digest_file(archive) != info["sha256"]:
        part = archive.with_suffix(".part")
        downloaded = 0
        next_report = 256 * 1024 * 1024
        with part.open("wb") as stream:
            for chunk in volume.read_file(info["archive"]):
                stream.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    print(f"Downloaded {downloaded / 2**30:.2f}/{info['size_bytes'] / 2**30:.2f} GiB", flush=True)
                    next_report += 256 * 1024 * 1024
        if part.stat().st_size != info["size_bytes"] or digest_file(part) != info["sha256"]:
            raise ValueError("Archive transfer integrity check failed")
        part.replace(archive)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            if not (destination / member.filename).resolve().is_relative_to(destination.resolve()):
                raise ValueError("Unsafe archive path")
        z.extractall(destination)
    (destination / "export-verified.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("Verified and extracted:", (destination / "results" / plan["bundle_id"]).resolve(), flush=True)


if __name__ == "__main__":
    main()
