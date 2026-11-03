"""Audit UNO selection -> active inputs -> prior inference -> FreeCus reference bytes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .manifest import PROJECT, PROJECT_MANIFEST, load, sha256
from .run import write_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def audit():
    base = PROJECT / "Orthogonalization-InstantX-IP-Adapter"
    uno_dir = base / "datasets/dreambench_uno"
    uno = read(uno_dir / "manifest.json")
    original_file = uno_dir / "provenance/dreambench_singleip.original.json"
    assert sha256(original_file) == uno["selection_sha256"], "UNO selection source changed"
    original = read(original_file)
    selections = {(r["prompt"], r["image_paths"][0].replace("./dreambooth/dataset/", "")) for r in original}
    assert len(selections) == 750 and all(len(r["image_paths"]) == 1 for r in original)
    uno_refs = {r["id"]: r for r in uno["references"]}
    references = {r["id"]: r for r in read(base / "datasets/semantic_benchmark/references.json")}
    active = read(PROJECT_MANIFEST)
    by_case = {r["case_id"]: r for r in active}
    freecus = [r for name in ["dreambench", "dreambench_plus"]
               for r in load(PROJECT / f"freecus/manifests/{name}_project_seed42.jsonl", PROJECT)]
    assert len(freecus) == len({r["case_id"] for r in freecus}) == 2100
    per_reference = {}
    for row in freecus:
        old = by_case[row["case_id"]]
        ref = references[row["reference_id"]]
        assert row["reference"] == "Orthogonalization-InstantX-IP-Adapter/" + old["image_path"]
        assert old["image_path"] == ref["image_path"]
        assert row["reference_sha256"] == ref["image_sha256"]
        assert row["evaluation_references"] == [row["reference"]]
        assert row["seed"] == 42 and row["sample_index"] == 0
        if row["benchmark"] == "dreambench":
            selected = uno_refs[row["reference_id"]]
            assert selected["image_sha256"] == row["reference_sha256"]
            relative = selected["uno_image_path"].replace("./dreambooth/dataset/", "")
            assert (row["prompt"], relative) in selections
        per_reference[row["reference_id"]] = dict(path=row["reference"], sha256=row["reference_sha256"])

    # Verify saved inference sidecars, including the signature binding the original input hash.
    sidecars = Counter()
    conditioning = {}
    hashes = {key: value["sha256"] for key, value in per_reference.items()}
    for condition in ["baseline", "source_target_ssg"]:
        for old in active:
            path = base / "outputs/benchmark_images_v1" / condition / old["benchmark"] / f"{old['case_id']}.json"
            meta = read(path)
            assert meta["row"] == old
            assert meta["settings"]["seed"] == 42
            assert (meta["settings"]["width"], meta["settings"]["height"]) == (1024, 1024)
            cache_path = old["conditioning_path"]
            if cache_path not in conditioning:
                conditioning[cache_path] = read(base / cache_path)
            value = dict(row=old, settings=meta["settings"], cache=conditioning[cache_path],
                         image_sha256=hashes[old["reference_id"]])
            signature = hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            assert signature == meta["job_signature"], f"Prior input signature mismatch: {path}"
            sidecars[condition + "/" + old["benchmark"]] += 1

    pairs = read(base / "outputs/benchmark_metrics_v1/pairs.json")
    for record in pairs["records"]:
        assert record["reference_sha256"] == hashes[record["row"]["reference_id"]]
        assert Path(record["reference"]).resolve() == (base / record["row"]["image_path"]).resolve()
    assert len(pairs["records"]) == 4200

    return dict(status="passed", uno_commit=uno["uno_commit"],
                uno_selection_sha256=uno["selection_sha256"], uno_reference_count=len(uno_refs),
                freecus_cases=2100, unique_references=len(per_reference),
                reference_mismatches=0, previous_inference_sidecars=dict(sidecars),
                previous_metric_pairs=4200, best_of_n_selection=False,
                audit_scope="reference selection/bytes and metadata signatures; generated PNG contents were not rehashed",
                per_reference=per_reference)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=PROJECT / "freecus/verification/reference-selection-audit.json")
    a = p.parse_args()
    result = audit()
    write_json(a.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "per_reference"}, indent=2))


if __name__ == "__main__":
    main()
