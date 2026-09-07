"""Validate the 29 detailed captions later used by source-aware projection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("caption_file", type=Path)
    parser.add_argument(
        "--template",
        type=Path,
        default=ROOT / "source_captions.template.json",
    )
    parser.add_argument(
        "--runtime-output",
        type=Path,
        help=(
            "Optionally write a projection-ready manifest containing O, O+A, "
            "O+B, and O+A+B source prompts."
        ),
    )
    args = parser.parse_args()
    expected = json.loads(args.template.read_text(encoding="utf-8"))["records"]
    actual = json.loads(args.caption_file.read_text(encoding="utf-8"))["records"]
    expected_keys = {
        (
            int(row["official_index"]),
            row["reference_filename"],
            row["content_sha256"],
        )
        for row in expected
    }
    actual_keys = {
        (
            int(row["official_index"]),
            row["reference_filename"],
            row["content_sha256"],
        )
        for row in actual
    }
    if len(actual) != len(actual_keys):
        raise ValueError("Caption file contains duplicate index/filename records")
    if actual_keys != expected_keys:
        raise ValueError(
            "Caption index/filename/hash set differs from the frozen template"
        )
    missing = []
    for row in actual:
        for field in ("behavior", "background", "full_caption"):
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                missing.append(f"{row['official_index']}:{field}")
    if missing:
        raise ValueError("Incomplete source captions: " + ", ".join(missing))
    print(f"Source caption validation: OK ({len(actual)} references)")
    if args.runtime_output is not None:
        records = []
        for row in sorted(actual, key=lambda item: int(item["official_index"])):
            subject = " ".join(row["subject_caption"].split())
            behavior = " ".join(row["behavior"].split())
            background = " ".join(row["background"].split())
            full_caption = " ".join(row["full_caption"].split())
            expected_full = f"{subject} {behavior} {background}"
            if full_caption.casefold() != expected_full.casefold():
                raise ValueError(
                    "full_caption must equal subject + behavior + background for "
                    f"{row['reference_filename']}: {full_caption!r} != "
                    f"{expected_full!r}"
                )
            records.append(
                {
                    "official_index": int(row["official_index"]),
                    "reference_filename": row["reference_filename"],
                    "reference_stem": Path(row["reference_filename"]).stem,
                    "content_sha256": row["content_sha256"],
                    "subject_caption": subject,
                    "source_behavior": behavior,
                    "source_background": background,
                    "source_prompts": {
                        "object_only": subject,
                        "object_behavior": f"{subject} {behavior}",
                        "object_background": f"{subject} {background}",
                        "object_behavior_background": expected_full,
                    },
                    "notes": row.get("notes"),
                }
            )
        canonical = json.dumps(
            records,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        payload = {
            "schema_version": 1,
            "benchmark_id": "xverse_single_animal_29",
            "semantic_digest": hashlib.sha256(canonical).hexdigest(),
            "reference_count": len(records),
            "records": records,
        }
        args.runtime_output.parent.mkdir(parents=True, exist_ok=True)
        args.runtime_output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Projection runtime manifest: {args.runtime_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
