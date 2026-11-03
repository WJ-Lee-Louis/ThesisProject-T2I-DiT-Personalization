"""Prime local LFS cache with verified immutable ZIP hardlinks, avoiding a second disk copy."""
import argparse
from pathlib import Path
import subprocess
from .artifact_bundle import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()).resolve()
    git_dir = Path(subprocess.check_output(["git", "rev-parse", "--absolute-git-dir"], text=True).strip())
    for path in args.paths:
        path = path.resolve()
        relative = path.relative_to(root)
        if relative.parts[0] not in {"freecus", "dreamo", "uno"} or relative.parts[1] != "artifacts" or path.suffix != ".zip":
            raise ValueError("Only immutable comparison image ZIPs are eligible")
        sha = digest(path)
        cached = git_dir / "lfs/objects" / sha[:2] / sha[2:4] / sha
        cached.parent.mkdir(parents=True, exist_ok=True)
        if cached.exists():
            if digest(cached) != sha:
                raise ValueError("Corrupt pre-existing LFS object")
        else:
            cached.hardlink_to(path)
        print(relative.as_posix(), sha, "local LFS object ready")


if __name__ == "__main__":
    main()
