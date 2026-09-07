"""Build and validate the frozen XVerseBench 0907 evaluation metadata."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


DATASET_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = DATASET_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from orthogonalization_instantx_ip_adapter.xversebench_0907 import (  # noqa: E402
    build_evaluation_assets,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate references and regenerate benchmark/DSG manifests."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
    )
    parser.add_argument(
        "--official-dataset-root",
        type=Path,
        default=DATASET_ROOT.parent / "XVerseBench",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate without rewriting generated JSON artifacts.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = build_evaluation_assets(
        args.dataset_root,
        args.official_dataset_root,
        write=not args.check_only,
    )
    subset = payload["subset_manifest"]
    custom = payload["custom_manifest"]
    integrity = payload["integrity_manifest"]
    print("XVerseBench_0907 validation: OK")
    print(
        "  subset_bench: "
        f"{subset['reference_count']} references, {subset['record_count']} records"
    )
    print(
        "  custom_bench: "
        f"{custom['reference_count']} references, {custom['record_count']} records"
    )
    print(f"  verified copies: {len(integrity['records'])}")
    print(f"  official commit: {subset['official_xverse_commit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
