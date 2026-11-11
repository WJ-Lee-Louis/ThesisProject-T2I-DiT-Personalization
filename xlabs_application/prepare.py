"""No API/GPU calls: validate all saved caches and prepare portable inputs/source."""
import json
from pathlib import Path
import shutil
import zipfile

from .config import ACCOUNTS, MODELS, SETTINGS
from .io import digest, read, sha, write
from .vendor.semantic_contrasts import ReferenceFingerprint, SemanticContrastConditioning

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
ORIGINAL = ROOT / "Orthogonalization-InstantX-IP-Adapter"


def snapshot():
    files = [p for p in BASE.rglob("*") if p.is_file() and p.suffix in (".py", ".txt", ".json")
             and not set(p.relative_to(BASE).parts) & {"outputs", "results", "artifacts", "manifests", "__pycache__"}
             and not any(part.startswith(".venv") for part in p.parts)]
    identity = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(files)}
    code = digest(identity)
    folder = BASE / "outputs/source" / code
    for p in files:
        target = folder / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    return folder, code, identity


def prepare():
    source = ORIGINAL / "outputs/semantic_benchmark_gpt56_sol/generation_manifest.json"
    original_rows = read(source)
    if (len(original_rows) != 2100 or len({r["case_id"] for r in original_rows}) != 2100
            or sum(r["benchmark"] == "dreambench" for r in original_rows) != 750
            or sum(r["benchmark"] == "dreambench_plus" for r in original_rows) != 1350):
        raise ValueError("Expected exact 750 + 1350 unique cases")
    images, rows, files = {}, [], {}
    for r in original_rows:
        if r["status"] != "READY":
            raise ValueError("Non-ready input")
        image_path = ORIGINAL / r["image_path"]
        if str(image_path) not in images:
            data = image_path.read_bytes()
            images[str(image_path)] = (sha(image_path), ReferenceFingerprint.from_image(data))
            files["references/" + sha(image_path) + ".bin"] = data
        image_hash, reference = images[str(image_path)]
        path = ORIGINAL / r["conditioning_path"]
        cached = SemanticContrastConditioning.from_dict(read(path))
        cached.validate_reuse(identity_term=r["identity_term"], target_prompt=r["target_prompt"],
                              reference=reference, model="gpt-5.6-sol")
        name = "conditioning/" + r["case_id"] + ".json"
        files[name] = path.read_bytes()
        rows.append(dict(case_id=r["case_id"], benchmark=r["benchmark"],
                         reference_id=r["reference_id"], identity_term=r["identity_term"],
                         target_prompt=r["target_prompt"], reference="references/" + image_hash + ".bin",
                         reference_sha256=image_hash, original_reference=r["image_path"],
                         conditioning=name, conditioning_sha256=sha(path),
                         fallback=r.get("fallback"), source_row=r))
    for bench in ("dreambench", "dreambench_plus"):
        path = BASE / "manifests" / (bench + "_seed42.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        content = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows if r["benchmark"] == bench)
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise ValueError("Existing manifest differs")
        path.write_text(content, encoding="utf-8", newline="\n")
    folder, code, identity = snapshot()
    plan = dict(schema=1, code=code, models=MODELS, settings=SETTINGS,
                source_manifest_sha256=sha(source), rows=rows)
    plan["bundle_id"] = digest(plan)[:24]
    state = BASE / "outputs" / plan["bundle_id"]
    state.mkdir(parents=True, exist_ok=True)
    refs = sorted({r["reference_id"] for r in rows if r["benchmark"] == "dreambench"})
    profiles = list(ACCOUNTS)
    owners = {ref: profiles[i % len(profiles)] for i, ref in enumerate(refs)}
    plan["assignments"] = {profile: [r["case_id"] for r in rows if r["benchmark"] == "dreambench"
                                       and owners[r["reference_id"]] == profile] for profile in profiles}
    write(state / "plan.json", plan)
    write(state / "source_hashes.json", identity)
    for name in ("run.json", "prompts.json", "summary.json", "source_fallback_prompt.json"):
        path = source.parent / name
        if path.exists():
            files["provenance/" + name] = path.read_bytes()
    files["plan.json"] = (state / "plan.json").read_bytes()
    files["bundle.json"] = json.dumps({name: __import__("hashlib").sha256(data).hexdigest()
                                         for name, data in files.items()}, sort_keys=True).encode()
    with zipfile.ZipFile(state / "inputs.zip", "w", compression=zipfile.ZIP_STORED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    with zipfile.ZipFile(state / "generation_source.zip", "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in identity:
            z.write(folder / name, name)
    write(BASE / "outputs/current.json", dict(bundle_id=plan["bundle_id"], source=str(folder), code=code))
    print(json.dumps(dict(bundle_id=plan["bundle_id"], validated=len(rows), references=len(images),
                          assignments={k: len(v) * 2 for k, v in plan["assignments"].items()},
                          source=str(folder)), indent=2))
    return plan


if __name__ == "__main__":
    prepare()
