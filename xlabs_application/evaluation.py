"""The unchanged selected-reference evaluator behind XLabs-specific validation."""
from pathlib import Path

from .config import CONDITIONS, MODELS, SETTINGS
from .io import read, sha, write
from .runtime import complete, result_paths
from .vendor import benchmark_metrics as metrics


def make_plan(plan, folder, cases):
    records = []
    for row in plan["rows"]:
        if row["case_id"] not in cases:
            continue
        for condition in CONDITIONS:
            png, _ = result_paths(plan["bundle_id"], condition, row)
            records.append(dict(condition=condition, category=metrics.category(row), row=row,
                                reference=str(folder / row["reference"]), reference_sha256=row["reference_sha256"],
                                generated=str(png)))
    if len(records) != len(cases) * 2:
        raise ValueError("Missing or duplicate evaluation cases")
    return dict(schema_version=1, protocol=metrics.PROTOCOL, records=records,
                generation_gpu=read(folder / "generation_gpu.json")["gpu"], generation_code=plan["code"])


def validate_pairs(pair_plan, account=None):
    from PIL import Image
    checked = set()
    for record in pair_plan["records"]:
        row = record["row"]
        ref = Path(record["reference"])
        png = Path(record["generated"])
        if ref not in checked:
            if sha(ref) != record["reference_sha256"]:
                raise ValueError("Evaluation reference differs")
            checked.add(ref)
        meta = read(png.with_suffix(".json"))
        if (meta["row"] != row or meta["condition"] != record["condition"]
                or meta["settings"] != SETTINGS or meta["models"] != MODELS
                or meta["gpu"] != pair_plan["generation_gpu"] or meta["code"] != pair_plan["generation_code"]
                or sha(png) != meta["png_sha256"]):
            raise ValueError("Evaluation input/provenance mismatch")
        diag = meta["diagnostics"]
        if diag["mask_single_text_rows"] or diag["orthogonalization"]["enabled"] != (record["condition"] == "source_target"):
            raise ValueError("Projection/SSG condition mismatch")
        with Image.open(png) as image:
            if image.size != (1024, 1024) or image.format != "PNG":
                raise ValueError("Expected original 1024 PNG")
            image.verify()


def evaluate(bundle_id, cases, runs, cache):
    runs.reload()
    cache.reload()
    folder = Path("/work/bundles") / bundle_id
    plan = read(folder / "plan.json")
    pairs = make_plan(plan, folder, cases)
    identity = __import__("hashlib").sha256("\n".join(sorted(cases)).encode()).hexdigest()[:16]
    out = Path("/work/metrics") / bundle_id / identity
    if (out / "metrics.json").exists():
        result = read(out / "metrics.json")
        validate_pairs(pairs)
    else:
        result = metrics.evaluate(pairs, out / "metrics.json", "cuda", 32,
                                  checkpoint=out / "checkpoint.json", validator=validate_pairs,
                                  commit=lambda _: runs.commit())
    if len(result["scores"]) != len(cases) * 2:
        raise ValueError("Incomplete metric part")
    result["generation_provenance"] = dict(code=plan["code"], gpu=pairs["generation_gpu"], settings=SETTINGS, models=MODELS)
    write(out / "metrics.json", result)
    write(out / "provenance.json", dict(bundle_id=bundle_id, code=plan["code"], cases=cases,
                                         settings=SETTINGS, models=MODELS))
    runs.commit()
    cache.commit()
    return dict(path=(out / "metrics.json").relative_to("/work").as_posix(),
                cases=len(result["scores"]), summary=result["summary"])


def merge(parts, plan, benchmark, output):
    """Combine XLabs account shards only; never alter other methods' tables."""
    expected = {(c, r["case_id"]): r for r in plan["rows"] if r["benchmark"] == benchmark for c in CONDITIONS}
    scores, seen, environment, generation = [], set(), None, None
    for path in parts:
        data = read(path)
        env = {k: data[k] for k in ("device", "device_name", "versions")}
        if data["protocol"] != metrics.PROTOCOL or environment not in (None, env):
            raise ValueError("Metric environment/protocol differs")
        environment = env
        if generation not in (None, data["generation_provenance"]):
            raise ValueError("Generation GPU/code/settings differ between metric shards")
        generation = data["generation_provenance"]
        for score in data["scores"]:
            key = score["condition"], score["case_id"]
            if key not in expected or key in seen:
                raise ValueError("Unexpected or duplicate score")
            row = expected[key]
            if any(score[k] != row[k] for k in ("reference_id", "target_prompt", "reference_sha256")):
                raise ValueError("Score reference/prompt binding differs")
            if not all(__import__("math").isfinite(score[k]) for k in metrics.METRICS):
                raise ValueError("Invalid metric value")
            seen.add(key)
            scores.append(score)
    if seen != set(expected):
        raise ValueError(f"Missing {len(set(expected) - seen)} scores; final summary withheld")
    scores.sort(key=lambda r: (r["case_id"], r["condition"]))
    result = dict(protocol=metrics.PROTOCOL, generation_provenance=generation, **environment,
                  summary=metrics.summarize(scores), scores=scores)
    write(output, result)
    return result
