"""Merge and validate downloaded UNO GPU lanes with the established artifact checks."""
import argparse
from pathlib import Path
from freecus.manifest import PROJECT
from freecus.operations.assemble_run import assemble_lanes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", choices=["dreambench", "dreambench_plus"], required=True)
    p.add_argument("--source", type=Path, required=True, help="Folder containing lane-0/ and lane-1/")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    assemble_lanes(a.source, PROJECT / f"freecus/manifests/{a.benchmark}_project_seed42.jsonl",
                   PROJECT, a.output, "all")


if __name__ == "__main__":
    main()
