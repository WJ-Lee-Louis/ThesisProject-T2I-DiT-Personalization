"""CPU audit of persisted images after a billing interruption."""
from datetime import datetime, timezone
from pathlib import Path

from .io import read, write


def audit(bundle_id, runs):
    from .evaluation import make_plan, validate_pairs
    from .runtime import signature
    runs.reload()
    root = Path("/work/bundles") / bundle_id
    plan = read(root / "plan.json")
    cases = plan["assignments"]["wj-lee-louis"] + [r["case_id"] for r in plan["rows"] if r["benchmark"] == "dreambench_plus"]
    pairs = make_plan(plan, root, cases)
    present, missing, counts, errors = [], [], {}, []
    for record in pairs["records"]:
        key = record["row"]["benchmark"] + "/" + record["condition"]
        counts.setdefault(key, dict(expected=0, present=0))
        counts[key]["expected"] += 1
        png = Path(record["generated"])
        meta = png.with_suffix(".json")
        identity = dict(benchmark=record["row"]["benchmark"], condition=record["condition"], case_id=record["row"]["case_id"])
        if not png.is_file() or not meta.is_file():
            missing.append(dict(**identity, png_exists=png.is_file(), metadata_exists=meta.is_file()))
            continue
        counts[key]["present"] += 1
        if read(meta)["signature"] != signature(plan, record["row"], record["condition"]):
            errors.append(dict(**identity, error="Generation signature mismatch"))
        present.append(record)
    checked = dict(pairs, records=present)
    try:
        validate_pairs(checked)
    except Exception:
        for record in present:
            try:
                validate_pairs(dict(pairs, records=[record]))
            except Exception as error:
                errors.append(dict(case_id=record["row"]["case_id"], condition=record["condition"], error=str(error)))
    result = dict(observed_at=datetime.now(timezone.utc).isoformat(), bundle_id=bundle_id,
                  workspace="wj-lee-louis", snapshot_during_recovery=True,
                  checks="PNG SHA256/CRC/size; metadata, prompt/reference, models/settings/code/GPU/signature; projection and no SSG",
                  counts=counts, existing_images=len(present), missing_images=len(missing), errors=errors, missing=missing)
    out = Path("/work/recovery") / bundle_id / "generation_audit.json"
    write(out, result)
    runs.commit()
    return dict(path=out.relative_to("/work").as_posix(), **{k: v for k, v in result.items() if k != "missing"})


def main():
    from .client import connect, current
    from .config import RUNS
    modal = connect("wj-lee-louis")
    info, directory, plan = current()
    runs = modal.Volume.from_name(RUNS)
    image = (modal.Image.debian_slim(python_version="3.11").pip_install("Pillow==11.3.0")
             .add_local_dir(info["source"], "/root", copy=True)
             .add_local_file(Path(__file__), "/root/xlabs_application/generation_audit.py", copy=True))
    app = modal.App("xlabs-application-generation-audit-v1")

    @app.function(name="audit", serialized=True, image=image, volumes={"/work": runs},
                  cpu=(.5, .5), memory=(1024, 1024), timeout=3600, retries=0, max_containers=1, scaledown_window=2)
    def execute(bundle_id):
        from xlabs_application.generation_audit import audit
        return audit(bundle_id, runs)

    with modal.enable_output():
        app.deploy()
    call = execute.spawn(plan["bundle_id"])
    write(directory / "recovery/audit-receipt.json", dict(call_id=call.object_id))
    print(call.object_id, flush=True)


if __name__ == "__main__":
    main()
