"""Resumable GPT-5.6 Sol A/B benchmark preparation; never generates images."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from orthogonalization_instantx_ip_adapter import contrast_prompt as prompts
from orthogonalization_instantx_ip_adapter.llm_provider import ResponsesJSONProvider, LLMTransportError
from orthogonalization_instantx_ip_adapter.semantic_contrasts import (
    CaptionRecord, ReferenceFingerprint, SemanticLLM,
    SemanticContrastConditioning, prepare_semantic_conditioning, identity_occurrences,
    StageTrace, prepare_source_only_fallback,
)
from orthogonalization_instantx_ip_adapter.contrast_directions import contrast_plan

MODEL = "gpt-5.6-sol"
LEDGER = ROOT / "datasets/semantic_benchmark/references.json"
OUT = ROOT / "outputs/semantic_benchmark_gpt56_sol"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Windows readers/antivirus can briefly hold the destination open. Retry
    # the atomic rename rather than aborting a completed, paid batch.
    for attempt in range(20):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def validate_references(refs):
    if len(refs) != 180 or sum(len(r["target_prompts"]) for r in refs) != 2100:
        raise ValueError("Expected 180 references and 2100 targets")
    if len({r["id"] for r in refs}) != len(refs):
        raise ValueError("Duplicate reference IDs")
    manifest = {r["subject_id"]: r for r in read(ROOT / "datasets/dreambench_uno/manifest.json")["references"]}
    for r in refs:
        data = (ROOT / r["image_path"]).read_bytes()
        if sha(data) != r["image_sha256"]:
            raise ValueError(f"Reference changed: {r['id']}")
        if r["benchmark"] == "dreambench_plus":
            raw = (ROOT / r["label_source"]).read_bytes()
            lines = raw.decode("utf-8").splitlines()
            if sha(raw) != r["label_source_sha256"] or lines[0].strip() != r["identity_term"]:
                raise ValueError("Benchmark label mismatch")
            targets = [line.strip() for line in lines[1:] if line.strip()]
        else:
            source = manifest[r["subject_id"]]
            if source["identity_term"] != r["identity_term"] or source["image_sha256"] != r["image_sha256"]:
                raise ValueError("DreamBench reference binding mismatch")
            targets = source["target_prompts"]
        if targets != r["target_prompts"]:
            raise ValueError("Benchmark targets changed")
        # Identity spelling is checked by the runtime per pair. Keep original
        # benchmark text here; input errors are recorded, never silently rewritten.


def new_llm():
    provider = ResponsesJSONProvider.from_env(MODEL)
    provider.timeout = 240
    return SemanticLLM(provider)


def request(operation):
    # Transport retries are distinct from SemanticLLM's two output-validation retries.
    # Authentication and capability failures are never retried.
    for attempt in range(3):
        try:
            return operation()
        except LLMTransportError as error:
            transient = any(s in str(error) for s in ("HTTP 429", "HTTP 500", "HTTP 502", "HTTP 503", "HTTP 504", "network request failed"))
            if not transient or attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))


def failure(error):
    result = {"status": "ERROR", "error": f"{type(error).__name__}: {error}"}
    if getattr(error, "trace", None):
        result["trace"] = error.trace.to_dict()
    return result


def retain_previous_failure(path):
    """Retain paid failed response traces when explicitly retrying a case."""
    if path.exists():
        history = OUT / "recovered_failures" / path.name
        previous = read(history) if history.exists() else []
        if not isinstance(previous, list):
            previous = [previous]
        save(history, previous + [read(path)])
        path.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Authorize paid API requests")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--source-only-fallback", nargs="*", default=[], metavar="CASE_ID",
                        help="Explicit target-conflict fallback for selected cases only")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("workers must be 1..16")
    refs = read(LEDGER)
    validate_references(refs)
    valid_ids = {f"{i:03d}-{j:02d}" for i, r in enumerate(refs) for j in range(len(r["target_prompts"]))}
    if not set(args.source_only_fallback) <= valid_ids:
        parser.error("Unknown fallback case ID")
    if args.source_only_fallback:
        save(OUT / "source_fallback_prompt.json", dict(version=prompts.SOURCE_FALLBACK_PROMPT_VERSION,
             sha256=prompts.SOURCE_FALLBACK_PROMPT_SHA256, prompt=prompts.SOURCE_FALLBACK_SYSTEM_PROMPT))
    marker = dict(model=MODEL, references_sha256=sha(LEDGER.read_bytes()),
                  caption_prompt_version=prompts.CAPTION_PROMPT_VERSION,
                  contrast_prompt_version=prompts.CONTRAST_PROMPT_VERSION,
                  caption_prompt_sha256=prompts.CAPTION_PROMPT_SHA256,
                  contrast_prompt_sha256=prompts.CONTRAST_PROMPT_SHA256,
                  caption_policy=prompts.CAPTION_POLICY_VERSION,
                  contrast_policy=prompts.CONTRAST_POLICY_VERSION,
                  references=180, pairs=2100)
    if (OUT / "run.json").exists() and read(OUT / "run.json") != marker:
        raise ValueError("Run inputs/model/prompts changed; do not mix experiments")
    save(OUT / "run.json", marker)
    save(OUT / "prompts.json", dict(caption=prompts.CAPTION_SYSTEM_PROMPT, contrast=prompts.CONTRAST_SYSTEM_PROMPT))
    print("Preflight PASS: 180 references / 2100 original targets", flush=True)
    if not args.run:
        return
    new_llm()  # Check credentials before scheduling work.

    def caption(i):
        r = refs[i]
        path = OUT / "captions" / f"{i:03d}.json"
        error_path = OUT / "errors" / f"A-{i:03d}.json"
        data = (ROOT / r["image_path"]).read_bytes()
        if path.exists():
            record = CaptionRecord.from_dict(read(path))
            record.validate_reuse(identity_term=r["identity_term"], reference=ReferenceFingerprint.from_image(data), model=MODEL)
            return record
        if error_path.exists() and not args.retry_errors:
            return None
        retain_previous_failure(error_path)
        try:
            record = request(lambda: new_llm().caption(data, r["identity_term"]))
            save(path, record.to_dict())
            if error_path.exists():
                error_path.unlink()
            return record
        except Exception as error:
            save(error_path, failure(error))
            return None

    records = {}
    # Verify vision and structured output before the full fanout.
    records[0] = caption(0)
    if records[0] is None:
        raise RuntimeError("First caption failed; inspect errors/A-000.json before fanout")
    def progress(stage, done, total, errors):
        value = dict(stage=stage, done=done, total=total, errors=errors,
                     updated_at=datetime.now(timezone.utc).isoformat())
        save(OUT / "progress.json", value)
        print(json.dumps(value), flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(caption, i): i for i in range(1, len(refs))}
        for future in as_completed(futures):
            records[futures[future]] = future.result()
            progress("A", len(records), len(refs), sum(r is None for r in records.values()))

    def contrast(i, j):
        r = refs[i]
        case_id = f"{i:03d}-{j:02d}"
        path = OUT / "conditioning" / f"{case_id}.json"
        error_path = OUT / "errors" / f"B-{case_id}.json"
        row = dict(case_id=case_id, reference_id=r["id"], benchmark=r["benchmark"],
                   image_path=r["image_path"], identity_term=r["identity_term"],
                   target_prompt=r["target_prompts"][j])
        if records[i] is None:
            return dict(row, status="BLOCKED", error="Caption failed")
        data = (ROOT / r["image_path"]).read_bytes()
        if path.exists():
            result = SemanticContrastConditioning.from_dict(read(path))
        else:
            fallback = False
            if error_path.exists():
                prior = read(error_path)
                fallback = (case_id in args.source_only_fallback and prior["error"].startswith("IdentityConflictError:"))
                # A validated semantic conflict is terminal, not a malformed
                # response to retry until a preferred answer appears.
                if not fallback and (not args.retry_errors or prior["error"].startswith("IdentityConflictError:")):
                    return dict(row, **prior)
            if not fallback:
                retain_previous_failure(error_path)
            try:
                if fallback:
                    result = request(lambda: prepare_source_only_fallback(
                        llm=new_llm(), caption_record=records[i], target_prompt=row["target_prompt"],
                        reference_image_bytes=data, failed_trace=StageTrace.from_dict(prior["trace"])))
                else:
                    result = request(lambda: prepare_semantic_conditioning(
                        llm=new_llm(), identity_term=r["identity_term"], target_prompt=row["target_prompt"],
                        reference_image_bytes=data, caption_record=records[i]))
                save(path, result.to_dict())
                if error_path.exists() and not fallback:
                    error_path.unlink()
            except Exception as error:
                value = failure(error)
                save(OUT / "errors" / f"fallback-{case_id}.json" if fallback else error_path, value)
                return dict(row, **value)
        result.validate_reuse(identity_term=r["identity_term"], target_prompt=row["target_prompt"],
                              reference=ReferenceFingerprint.from_image(data), source_caption=records[i].text,
                              preserve=records[i].preserve, model=MODEL)
        plan = contrast_plan(result)
        if any(p.positive != (records[i].text if p.origin == "source" else row["target_prompt"]) for p in plan):
            raise ValueError("Local positive changed")
        return dict(row, status="READY", conditioning_path=path.relative_to(ROOT).as_posix(),
                    source_directions=len(result.analysis.source), target_directions=len(result.analysis.target),
                    contrast_attempts=len(result.analysis.trace.attempts),
                    fallback="target_conflict_source_only_v1" if result.fallback_trace else None)

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(contrast, i, j) for i, r in enumerate(refs) for j in range(len(r["target_prompts"]))]
        for future in as_completed(futures):
            results.append(future.result())
            progress("B", len(results), 2100, sum(r["status"] != "READY" for r in results))
    results.sort(key=lambda r: r["case_id"])
    save(OUT / "generation_manifest.json", results)
    summary = dict(model=MODEL, captions_ready=sum(r is not None for r in records.values()),
                   pairs_ready=sum(r["status"] == "READY" for r in results), pairs_total=2100,
                   fallback_cases=[r["case_id"] for r in results if r.get("fallback")],
                   errors=[{"case_id": r["case_id"], "status": r["status"], "error": r["error"]}
                           for r in results if r["status"] != "READY"],
                   caption_validation_attempts=sum(len(r.trace.attempts) for r in records.values() if r),
                   contrast_validation_attempts=sum(r.get("contrast_attempts", 0) for r in results),
                   source_directions=sum(r.get("source_directions", 0) for r in results),
                   target_directions=sum(r.get("target_directions", 0) for r in results),
                   image_generation=False, real_embedding_evaluation=False, visual_caption_review=False)
    save(OUT / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
