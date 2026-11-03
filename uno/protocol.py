"""Freeze the exact FreeCus comparison cohort; no model imports."""
import argparse
import json
from pathlib import Path
from freecus.manifest import PROJECT, load, sha256
from freecus.run import write_json

BASE = Path(__file__).parent


def selected_rows(benchmark, root=PROJECT, manifest=None):
    path = Path(manifest) if manifest else PROJECT / f"freecus/manifests/{benchmark}_project_seed42.jsonl"
    rows = load(path, root)
    expected = 750 if benchmark == "dreambench" else 1350
    if len(rows) != expected or {r["seed"] for r in rows} != {42}:
        raise ValueError("Canonical input differs")
    return rows, path


def validate_config(config):
    expected = dict(model_type="flux-dev", width=1024, height=1024, seed=42, steps=25, guidance=4.0,
                    reference_resolution=512, reference_task="single_image_all_categories", only_lora=True,
                    lora_rank=512, pe="d", quantization="none", cpu_offload=False)
    if config != expected:
        raise ValueError("Primary UNO single-image BF16 configuration changed; define a separate experiment")



def verify_upstream():
    info = json.loads((BASE / "UPSTREAM.json").read_text(encoding="utf-8"))
    for name, digest in info["files"].items():
        if sha256(BASE / "upstream" / name) != digest:
            raise ValueError(f"Upstream source changed: {name}")
    return info


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    verify_upstream()
    validate_config(json.loads((BASE / "config.json").read_text()))
    report = {}
    for benchmark in ["dreambench", "dreambench_plus"]:
        rows, source = selected_rows(benchmark)
        path = BASE / f"manifests/{benchmark}_seed42.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        report[benchmark] = dict(cases=len(rows), references=len({r["reference_id"] for r in rows}),
                                 source_manifest_sha256=sha256(source), manifest_sha256=sha256(path),
                                 excluded_category=None)
    write_json(BASE / "verification/protocol.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
