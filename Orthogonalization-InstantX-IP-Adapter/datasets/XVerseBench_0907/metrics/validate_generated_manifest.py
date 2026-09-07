"""Reject incomplete or seed-mismatched generated-image manifests."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from common import load_benchmark, load_generated_images


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--benchmark", choices=("subset", "custom"), required=True)
    parser.add_argument("--generated-manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest, _ = load_benchmark(args.benchmark_root, args.benchmark)
    rows = load_generated_images(args.generated_manifest, manifest)
    actual: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for row in rows:
        actual[row["condition"]].add((row["record_id"], row["seed"]))

    expected = set()
    for record in manifest["records"]:
        seeds = record.get("seeds", [record.get("seed")])
        for seed in seeds:
            if seed is None:
                raise ValueError(f"Record has no seed policy: {record['record_id']}")
            expected.add((record["record_id"], int(seed)))

    errors = []
    for condition, keys in sorted(actual.items()):
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        if missing or extra:
            errors.append(
                f"{condition}: missing={len(missing)}, extra={len(extra)}"
            )
        else:
            print(f"  {condition}: {len(keys)} images, complete")
    if errors:
        raise ValueError("Generated manifest is incomplete: " + "; ".join(errors))
    print(
        f"Generated manifest validation: OK ({len(actual)} conditions, "
        f"{len(rows)} images)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
