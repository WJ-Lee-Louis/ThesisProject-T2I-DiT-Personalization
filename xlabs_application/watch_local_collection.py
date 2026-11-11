"""Finish local publication after authenticated download clients verify all lanes."""
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time

from .collect import BASE, SHARDS
from .io import read, write


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-id", required=True)
    args = parser.parse_args()
    state = BASE / "outputs/local-collection-status.json"
    deadline = time.monotonic() + 4 * 3600
    while time.monotonic() < deadline:
        done = []
        for benchmark, shards in SHARDS.items():
            for shard in shards:
                path = BASE / "results" / benchmark / args.bundle_id / "lanes" / shard / "local_verification.json"
                if path.is_file() and read(path).get("restored_bytes_verified"):
                    done.append(benchmark + "/" + shard)
        write(state, dict(status="downloading_and_verifying", bundle_id=args.bundle_id,
                          verified_lanes=done, total_lanes=4))
        if len(done) == 4:
            break
        time.sleep(20)
    else:
        raise TimeoutError("Local transfers are incomplete; do not publish a complete bundle")
    subprocess.run([sys.executable, "-X", "utf8", "-m", "xlabs_application.collect", "assemble",
                    "--benchmark", "all", "--bundle-id", args.bundle_id], cwd=BASE.parent, check=True)
    parts = [read(BASE / "verification" / (benchmark + ".json")) for benchmark in SHARDS]
    result = dict(status="complete", bundle_id=args.bundle_id,
                  completed_at=datetime.now(timezone.utc).isoformat(),
                  generated_images=sum(p["generated_images"] for p in parts),
                  scored_images=sum(p["scored_images"] for p in parts),
                  archive_parts=sum(p["archive_parts"] for p in parts),
                  archive_bytes=sum(p["archive_bytes"] for p in parts),
                  original_pngs_restored_and_verified=True, git_lfs_archives=True,
                  no_commit_or_push_performed=True, benchmarks=parts)
    if result["generated_images"] != 4200 or result["scored_images"] != 4200:
        raise ValueError("Expected exactly 4,200 images and scores")
    write(BASE / "verification/collection.json", result)
    write(state, result)
    print("Local collection complete: 4,200 images and scores verified.", flush=True)


if __name__ == "__main__":
    main()
