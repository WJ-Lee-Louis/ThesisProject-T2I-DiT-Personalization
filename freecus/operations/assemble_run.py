"""Verify GPU lane artifacts and assemble exactly the requested evaluation subset."""
import argparse
import json
from pathlib import Path
import shutil

from freecus.evaluate import prepare, project_evaluator, validate
from freecus.manifest import PROJECT, sha256
from freecus.run import artifact_paths, complete, write_json


def assemble_lanes(source, manifest, root, destination, selection):
    lanes = sorted(source.glob("lane-*"))
    if len(lanes) != 2:
        raise ValueError("Expected both downloaded GPU lanes")
    run_bytes = [(lane / "run.json").read_bytes() for lane in lanes]
    infos = [json.loads(data) for data in run_bytes]
    keys = ["fingerprint", "config", "manifest_sha256", "code", "versions", "gpu", "cuda"]
    if any({k: x[k] for k in keys} != {k: infos[0][k] for k in keys} for x in infos[1:]):
        raise ValueError("Lane identities/environments differ")
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / "run.json"
    if marker.exists() and marker.read_bytes() != run_bytes[0]:
        raise ValueError("Destination belongs to another run")
    marker.write_bytes(run_bytes[0])
    evaluation = prepare(manifest, root, destination, project_evaluator(), selection)
    for record in evaluation["records"]:
        row = dict(id=record["freecus_id"])
        found = [(lane, artifact_paths(lane, row)) for lane in lanes]
        found = [(lane, paths) for lane, paths in found if paths["metadata"].exists()]
        if len(found) != 1:
            raise ValueError(f"Missing or duplicate case: {record['row']['case_id']}")
        lane, paths = found[0]
        if not complete(paths):
            raise ValueError(f"Missing or corrupt artifacts: {row['id']}")
        targets = artifact_paths(destination, row)
        for kind, src in paths.items():
            dst = targets[kind]
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                if sha256(dst) != sha256(src):
                    raise ValueError(f"Existing assembled artifact differs: {dst}")
            else:
                shutil.copy2(src, dst)
    validate(evaluation)
    write_json(destination / "assembly.json", dict(status="validated",
               selection=evaluation["selection"], source=str(source.resolve()),
               manifest_sha256=sha256(manifest), fingerprint=infos[0]["fingerprint"],
               evaluation_input=str(destination.resolve())))
    print(json.dumps(dict(status="validated", cases=len(evaluation["records"]),
                          selection=selection, run=str(destination.resolve())), indent=2))
    return evaluation


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", choices=["dreambench", "dreambench_plus"], required=True)
    a = p.parse_args()
    base = Path(__file__).resolve().parents[1]
    primary = a.benchmark == "dreambench"
    state = base / ("outputs/modal" if primary else "outputs/modal-method-comparison-dbpp")
    plan = json.loads((state / f"latest-{a.benchmark}.json").read_text())
    assemble_lanes(state / "collected/results" / plan["bundle_id"],
                   base / f"manifests/{a.benchmark}_project_seed42.jsonl", PROJECT,
                   base / "outputs/evaluation_inputs" / ("dreambench" if primary else "dreambench_plus_no_style"),
                   "all" if primary else "dbpp-no-style")


if __name__ == "__main__":
    main()
