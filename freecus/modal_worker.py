"""GPU child process. Parent supervises time and commits the mounted Volume."""
import argparse
import json
import resource
import time
from pathlib import Path
from types import SimpleNamespace

from .run import execute, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("job", type=Path)
    a = p.parse_args()
    job = json.loads(a.job.read_text(encoding="utf-8"))
    events = a.job.with_name("events.jsonl")
    began = time.monotonic()

    def event(value):
        value["worker_elapsed_seconds"] = time.monotonic() - began
        value["host_peak_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
        with events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value) + "\n")
            stream.flush()

    args = SimpleNamespace(config=Path(job["config"]), manifest=Path(job["manifest"]),
                           data_root=Path(job["data_root"]), output=Path(job["output"]),
                           start=0, limit=None, dry_run=False, case_ids=job["case_ids"])
    try:
        execute(args, on_event=event)
    except BaseException as exc:
        write_json(a.job.with_name("child_result.json"), dict(status="failed", error=repr(exc)))
        raise
    write_json(a.job.with_name("child_result.json"), dict(status="complete", seconds=time.monotonic() - began))


if __name__ == "__main__":
    main()
