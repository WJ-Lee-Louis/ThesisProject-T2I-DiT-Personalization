"""Build portable manifests without importing torch or copying datasets."""
import argparse
import hashlib
import json
import re
from pathlib import Path

SEEDS = (42,)
PROJECT = Path(__file__).resolve().parents[1]
PROJECT_MANIFEST = PROJECT / "Orthogonalization-InstantX-IP-Adapter/outputs/semantic_benchmark_gpt56_sol/generation_manifest.json"
LIVE = {"cat", "cat2", "dog", "dog2", "dog3", "dog5", "dog6", "dog7", "dog8"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows_for(subject, category, word, reference, references, prompts, root, benchmark, seeds=SEEDS):
    reference_hash = sha256(reference)
    evaluation_hashes = {p.relative_to(root).as_posix(): sha256(p) for p in references}
    for i, prompt in enumerate(prompts):
        for sample, seed in enumerate(seeds):
            yield dict(id=f"{subject}/{i}_{sample}", benchmark=benchmark,
                       subject=subject, category=category, subject_word=word,
                       reference=reference.relative_to(root).as_posix(),
                       reference_sha256=reference_hash,
                       evaluation_reference_sha256=evaluation_hashes,
                       evaluation_references=[p.relative_to(root).as_posix() for p in references],
                       prompt=prompt, prompt_index=i, sample_index=sample, seed=seed)


def dreambench(root, reference_index=0, seeds=SEEDS):
    text = (root / "prompts_and_classes.txt").read_text(encoding="utf-8-sig")
    classes = re.findall(r"^([a-z0-9_]+),([^\r\n]+)$", text, re.M)
    classes = [(s, c) for s, c in classes if s != "subject_name"]
    blocks = text.split("prompt_list = [")[1:]
    if len(classes) != 30 or len(blocks) != 2:
        raise ValueError("Expected official DreamBench 30 classes and 2 prompt lists")
    templates = [re.findall(r"['\"](.*?)['\"]\.format", b.split("]")[0]) for b in blocks]
    if any(len(t) != 25 for t in templates):
        raise ValueError("Expected 25 object and 25 live prompts")
    for subject, word in classes:
        refs = sorted(p for p in (root / subject).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        ref = refs[reference_index]
        prompts = [" ".join(t.format("", word).split()) for t in templates[int(subject in LIVE)]]
        yield from rows_for(subject, "animal" if subject in LIVE else "object", word,
                            ref, refs, prompts, root, "dreambench", seeds)


def dreambench_plus(root, seeds=SEEDS):
    captions = sorted((root / "captions").rglob("*.txt"))
    images = {p.relative_to(root / "images").with_suffix("").as_posix(): p
              for p in (root / "images").rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    if len(captions) != 150 or len(images) != 150:
        raise ValueError("Expected complete official DreamBench++: 150 image/caption pairs")
    for cap in captions:
        relative = cap.relative_to(root / "captions")
        ref = images[relative.with_suffix("").as_posix()]
        lines = cap.read_text(encoding="utf-8-sig").splitlines()
        if len(lines) != 10 or any(not line.strip() for line in lines):
            raise ValueError(f"Expected subject + 9 prompts: {cap}")
        category = cap.parent.name
        prefix = f"live_subject_{category}" if category in {"animal", "human"} else category
        word = lines[0].strip()
        subject = f"{prefix}_{cap.stem}_{re.sub(r'[^a-z0-9_-]+', '_', word.lower()).strip('_')}"
        yield from rows_for(subject, category, word, ref, [ref], lines[1:], root, "dreambench_plus", seeds)


def project_rows(benchmark, source=PROJECT_MANIFEST, root=PROJECT, seeds=SEEDS):
    """Reuse exact active research cases; never reselect references or rewrite targets."""
    source = Path(source)
    original = json.loads(source.read_text(encoding="utf-8"))
    from collections import Counter
    if Counter(r["benchmark"] for r in original) != {"dreambench": 750, "dreambench_plus": 1350}:
        raise ValueError("Expected canonical project manifest with 750 + 1350 cases")
    if len({r["case_id"] for r in original}) != 2100 or any(r["status"] != "READY" for r in original):
        raise ValueError("Canonical project cases must be unique and READY")
    project_dir = source.parents[2]
    source_hash = sha256(source)
    hashes, positions = {}, {}
    for row in original:
        if row["benchmark"] != benchmark:
            continue
        ref = (project_dir / row["image_path"]).resolve()
        relative = ref.relative_to(root.resolve()).as_posix()
        if relative not in hashes:
            hashes[relative] = sha256(ref)
        reference_id = row["reference_id"]
        parts = reference_id.split("/")
        subject = "__".join(parts)
        category = ("animal" if parts[-1] in LIVE else "object") if benchmark == "dreambench" else parts[-2]
        index = positions.get(reference_id, 0)
        positions[reference_id] = index + 1
        for sample, seed in enumerate(seeds):
            yield dict(id=f"{subject}/{index}_{sample}", benchmark=benchmark, subject=subject,
                       category=category, subject_word=row["identity_term"], reference=relative,
                       reference_sha256=hashes[relative], evaluation_references=[relative],
                       evaluation_reference_sha256={relative: hashes[relative]},
                       prompt=row["target_prompt"], prompt_index=index, sample_index=sample, seed=seed,
                       case_id=row["case_id"], reference_id=reference_id,
                       source_manifest_sha256=source_hash, reference_policy="selected_one")


def load(path, root):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = set()
    hashes = {}
    for r in rows:
        if r["id"] in ids or not re.fullmatch(r"[\w-]+/\d+_\d+", r["id"], re.ASCII):
            raise ValueError(f"Duplicate or unsafe id: {r['id']}")
        ids.add(r["id"])
        for name in [r["reference"], *r["evaluation_references"]]:
            p = (root / name).resolve()
            if not p.is_relative_to(root.resolve()) or not p.is_file():
                raise ValueError(f"Missing/unsafe reference: {name}")
            if name not in hashes:
                hashes[name] = sha256(p)
        if hashes[r["reference"]] != r["reference_sha256"]:
            raise ValueError(f"Reference changed: {r['reference']}")
        for name in r["evaluation_references"]:
            if hashes[name] != r["evaluation_reference_sha256"][name]:
                raise ValueError(f"Evaluation reference changed: {name}")
    if not rows:
        raise ValueError("Empty manifest")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("benchmark", choices=["dreambench", "dreambench_plus"])
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference-index", type=int, default=0)
    p.add_argument("--source", choices=["project", "standalone"], default="project")
    p.add_argument("--project-manifest", type=Path, default=PROJECT_MANIFEST)
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    a = p.parse_args()
    if a.reference_index < 0:
        p.error("reference-index must be nonnegative")
    if len(a.seeds) != len(set(a.seeds)):
        p.error("Duplicate seeds")
    if a.source == "project":
        rows = list(project_rows(a.benchmark, a.project_manifest, a.data_root, a.seeds))
    else:
        rows = list(dreambench(a.data_root, a.reference_index, a.seeds) if a.benchmark == "dreambench" else dreambench_plus(a.data_root, a.seeds))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(rows)} samples, {len({r['subject'] for r in rows})} subjects -> {a.output}")


if __name__ == "__main__":
    main()
