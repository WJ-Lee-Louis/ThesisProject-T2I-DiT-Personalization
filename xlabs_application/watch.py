"""Local collection monitor; cloud generation/evaluation runs independently."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

from .config import ACCOUNTS
from .io import read, write

BASE = Path(__file__).resolve().parent


def cli(*args):
    result = subprocess.run([sys.executable, "-X", "utf8", "-m", "xlabs_application.client", *args],
                            cwd=BASE.parent, capture_output=True, encoding="utf-8", timeout=3600)
    print(result.stdout, flush=True)
    if result.returncode:
        raise RuntimeError(result.stderr[-6000:])


def finalize(bundle_id):
    from .evaluation import merge
    directory = BASE / "outputs" / bundle_id
    plan = read(directory / "plan.json")
    parts = [BASE / "results/dreambench" / (bundle_id + "-" + p) / "metrics.json" for p in ACCOUNTS]
    report = BASE / "results/reports" / bundle_id
    result = merge(parts, plan, "dreambench", report / "dreambench_scores.json")
    fields = ["condition", "benchmark", "case_id", "reference_id", "category", "target_prompt",
              "reference_sha256", "generated_sha256", "CLIP-T", "CLIP-I", "DINO-I"]
    with (report / "dreambench_per_image.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["scores"])
    write(report / "summary.json", dict(generation_provenance=result["generation_provenance"],
                                        protocol=result["protocol"], summary=result["summary"],
                                        scored_images=len(result["scores"])))
    write(directory / "collection-status.json", dict(status="complete", scored_images=len(result["scores"]),
                                                     reports=str(report)))
    print(json.dumps(dict(status="complete", reports=str(report)), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-id", required=True)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    if args.finalize_only:
        finalize(args.bundle_id)
        return
    directory = BASE / "outputs" / args.bundle_id
    done, failed, transient = set(), {}, 0
    deadline = time.monotonic() + 12 * 3600
    while len(done) + len(failed) < len(ACCOUNTS):
        if read(BASE / "outputs/current.json")["bundle_id"] != args.bundle_id:
            raise ValueError("Current bundle changed; do not collect into another run")
        if time.monotonic() > deadline:
            raise TimeoutError("Collection monitor timed out; cloud checkpoints remain")
        for profile in ACCOUNTS:
            if profile in done or profile in failed:
                continue
            try:
                cli("status", "--profile", profile)
                status = read(directory / profile / "submit-receipt-status.json")
                if status["status"] == "complete":
                    cli("collect", "--profile", profile)
                    done.add(profile)
                elif status["status"] in ("failed", "remote_error", "budget_checkpoint", "vram_review_required"):
                    failed[profile] = status
                transient = 0
            except Exception as error:
                transient += 1
                print("Collection retry:", str(error), flush=True)
                if transient >= 5:
                    write(directory / "collection-status.json", dict(status="collection_error", error=str(error)))
                    raise
        write(directory / "collection-status.json", dict(status="waiting", collected=sorted(done), failed=failed))
        if len(done) + len(failed) < len(ACCOUNTS):
            time.sleep(60)
    if failed:
        write(directory / "collection-status.json", dict(status="incomplete", collected=sorted(done), failed=failed))
        raise RuntimeError("Some cloud shards are incomplete; no final metric summary was published")
    finalize(args.bundle_id)


if __name__ == "__main__":
    main()
