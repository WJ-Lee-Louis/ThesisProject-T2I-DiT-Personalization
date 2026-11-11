"""Lossless, bounded ZIP bundles; deduplicated RGB references; portable safe restore."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import zipfile

LIMIT = 1024**3


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def safe_path(root, relative):
    if "\\" in relative or Path(relative).is_absolute():
        raise ValueError("Unsafe path: " + relative)
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("Unsafe path: " + relative)
    return target


def pack(lanes, manifest, input_root, destination, method, benchmark, bundle_id, expected, resolution=(1024, 1024), allow_gpu_variants=False):
    """Archive all completed cases, original references and provenance; never model weights."""
    lanes = [Path(p) for p in lanes]
    manifest, input_root, destination = Path(manifest), Path(input_root), Path(destination)
    receipt = destination / "bundle_index.json"
    if receipt.exists():
        info = json.loads(receipt.read_text(encoding="utf-8"))
        if (info["method"], info["benchmark"], info["bundle_id"], info["generated_images"]) != (method, benchmark, bundle_id, expected):
            raise ValueError("Existing bundle identity differs")
        if info.get('generation_resolution', [1024, 1024]) != list(resolution):
            raise ValueError('Existing bundle resolution differs')
        if all((destination / p["file"]).is_file() and digest(destination / p["file"]) == p["sha256"] for p in info["parts"]):
            return info
        raise ValueError("Existing bundle is incomplete/corrupt; preserve it for investigation")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Unfinished export exists; use a new destination")
    destination.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(s) for s in manifest.read_text(encoding="utf-8").splitlines() if s.strip()]
    canonical = {r["case_id"]: r for r in rows}
    if len(canonical) != (750 if benchmark == "dreambench" else 1350):
        raise ValueError("Incomplete canonical manifest")
    infos = [json.loads((lane / "run.json").read_text(encoding="utf-8")) for lane in lanes]
    for key in ["fingerprint", "config", "manifest_sha256", "code", "versions", "gpu", "cuda"]:
        if key == "gpu" and allow_gpu_variants:
            continue
        if any(i[key] != infos[0][key] for i in infos[1:]):
            raise ValueError("Generation lane identities differ: " + key)
    if digest(manifest) != infos[0]["manifest_sha256"]:
        raise ValueError("Manifest fingerprint differs")
    files = {}
    source_links, seen = {}, set()

    def add(path, relative, expected_hash=None):
        path = Path(path)
        safe_path(Path("/bundle"), relative)
        sha = digest(path)
        if expected_hash is not None and sha != expected_hash:
            raise ValueError("Artifact hash differs: " + str(path))
        if relative in files:
            if files[relative]["sha256"] != sha:
                raise ValueError("Conflicting archive path: " + relative)
            return
        files[relative] = dict(source=path, size=path.stat().st_size, sha256=sha)

    add(lanes[0] / "run.json", "run.json")
    add(manifest, "manifest.jsonl")
    for lane in lanes:
        add(lane / "run.json", "lanes/" + lane.name + "/run.json")
        init = lane / "initialization.json"
        if init.exists():
            add(init, "lanes/" + lane.name + "/initialization.json")
        for path in sorted((lane / "metadata").rglob("*.json")):
            meta = json.loads(path.read_text(encoding="utf-8"))
            case = meta["case_id"]
            if case in seen or case not in canonical:
                raise ValueError("Duplicate/unexpected case: " + case)
            row = canonical[case]
            for field in ["id", "seed", "reference_id", "prompt", "reference_sha256"]:
                if meta[field] != row[field]:
                    raise ValueError("Case input differs: " + case + "/" + field)
            if meta["fingerprint"] != infos[0]["fingerprint"]:
                raise ValueError("Mixed generation fingerprints")
            seen.add(case)
            add(path, "metadata/" + row["id"] + ".json")
            for kind, ext in [("tgt_image", "png"), ("src_image", "png"), ("text", "txt")]:
                original = lane / kind / (row["id"] + "." + ext)
                wanted = meta["artifact_sha256"][kind]
                relative = kind + "/" + row["id"] + "." + ext
                if kind == "src_image":
                    unique = "references/" + wanted + ".png"
                    add(original, unique, wanted)
                    source_links[relative] = unique
                else:
                    add(original, relative, wanted)
                    if kind == "tgt_image":
                        with original.open("rb") as f:
                            header = f.read(24)
                        if header[:8] != b"\x89PNG\r\n\x1a\n" or struct.unpack(">II", header[16:24]) != tuple(resolution):
                            raise ValueError("Unexpected image dimensions")
            if (lane / "text" / (row["id"] + ".txt")).read_text(encoding="utf-8") != row["prompt"]:
                raise ValueError("Target prompt differs")
    if len(seen) != expected:
        raise ValueError(f"Expected {expected} complete cases, found {len(seen)}")
    for row in rows:
        add(input_root / row["reference"], "input/" + row["reference"], row["reference_sha256"])
    write(destination / "source_reference_index.json", source_links)
    add(destination / "source_reference_index.json", "source_reference_index.json")
    parts, archive, count = [], None, 0
    try:
        for relative, item in sorted(files.items()):
            estimate = item["size"] + 2 * len(relative.encode("utf-8")) + 512
            if estimate >= LIMIT:
                raise ValueError("Single member exceeds archive bound")
            if archive is None or archive.fp.tell() + estimate > LIMIT - 1024**2:
                if archive is not None:
                    archive.close()
                name = f"images-{len(parts):03d}.zip"
                parts.append(dict(file=name))
                archive = zipfile.ZipFile(destination / name, "w", compression=zipfile.ZIP_STORED, allowZip64=True)
            archive.write(item["source"], relative)
            count += 1
    finally:
        if archive is not None:
            archive.close()
    for part in parts:
        path = destination / part["file"]
        part.update(size_bytes=path.stat().st_size, sha256=digest(path))
        if part["size_bytes"] > LIMIT:
            raise ValueError("ZIP exceeds 1 GiB")
    info = dict(schema=1, method=method, benchmark=benchmark, bundle_id=bundle_id,
                generation_resolution=list(resolution), generation_devices=sorted({i['gpu'] for i in infos}),
                generated_images=len(seen), manifest_cases=len(rows),
                missing_case_ids=sorted(set(canonical) - seen),
                generation_fingerprint=infos[0]["fingerprint"], manifest_sha256=digest(manifest),
                reference_mode="sha256-deduplicated original src_image PNG; index restores original paths",
                members={name: {k:v for k,v in item.items() if k != "source"} for name,item in files.items()},
                parts=parts)
    write(receipt, info)
    return info


def verify(folder, index):
    """Read all ZIP members without extracting; verify byte hashes, CRC and complete coverage."""
    found = set()
    for part in index["parts"]:
        path = safe_path(folder, part["file"])
        if path.stat().st_size != part["size_bytes"] or digest(path) != part["sha256"]:
            raise ValueError("ZIP hash differs: " + str(path))
        with zipfile.ZipFile(path) as z:
            for member in z.infolist():
                safe_path(folder, member.filename)
                if member.filename in found or member.filename not in index["members"]:
                    raise ValueError("Unexpected or duplicate ZIP member")
                h = hashlib.sha256()
                with z.open(member) as stream:
                    for block in iter(lambda: stream.read(8 * 1024**2), b""):
                        h.update(block)
                expected = index["members"][member.filename]
                if h.hexdigest() != expected["sha256"] or member.file_size != expected["size"]:
                    raise ValueError("ZIP member hash differs: " + member.filename)
                found.add(member.filename)
    if found != set(index["members"]):
        raise ValueError("Missing ZIP members")


def restore(folder, index, destination):
    """Restore into an explicit empty/new method-specific directory; never overwrite different files."""
    folder, destination = Path(folder), Path(destination)
    verify(folder, index)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Restore destination must be empty to protect existing RunPod results")
    destination.mkdir(parents=True, exist_ok=True)
    for part in index["parts"]:
        with zipfile.ZipFile(folder / part["file"]) as z:
            for member in z.infolist():
                target = safe_path(destination, member.filename)
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, 8 * 1024**2)
    links = json.loads((destination / "source_reference_index.json").read_text(encoding="utf-8"))
    for relative, unique in links.items():
        target, source = safe_path(destination, relative), safe_path(destination, unique)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.hardlink_to(source)
        except OSError:
            shutil.copy2(source, target)
    write(destination / "restored_bundle.json", dict(method=index["method"], benchmark=index["benchmark"],
          bundle_id=index["bundle_id"], generated_images=index["generated_images"], verified=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["verify", "restore"])
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--archives", type=Path)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    index = json.loads(args.index.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[2]
    folder = args.archives or root / index["method"] / "artifacts" / index["benchmark"] / index["bundle_id"]
    if args.command == "restore":
        destination = args.destination or root / index["method"] / "outputs/restored" / index["benchmark"] / index["bundle_id"]
        restore(folder, index, destination)
        print("Restored:", destination)
    else:
        verify(folder, index)
    print("Verified", index["method"], index["benchmark"], index["generated_images"], "generated PNGs")


if __name__ == "__main__":
    main()
