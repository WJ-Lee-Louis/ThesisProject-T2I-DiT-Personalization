"""Collect immutable model-comparison ZIP shards, with explicit Modal profiles and no Git network writes."""
import argparse
import json
from pathlib import Path
import shutil
import zipfile
from .artifact_bundle import pack, digest, verify, write

ROOT = Path(__file__).resolve().parents[2]
PROFILES = {("dreamo", "dreambench"): "method-comparison", ("dreamo", "dreambench_plus"): "method-comparison-dbpp",
            ("uno", "dreambench"): "wj-lee-louis", ("uno", "dreambench_plus"): "woojin716"}


def preserve_provenance(method, benchmark, plan, folder):
    metrics = ROOT / method / "outputs/metrics" / (("dreambench_plus_no_style" if method == "freecus" and benchmark == "dreambench_plus" else benchmark) + "_scores.json")
    shutil.copyfile(metrics, folder / "metrics.json")
    manifest = ROOT / "freecus/manifests" / (benchmark + "_project_seed42.jsonl")
    shutil.copyfile(manifest, folder / "manifest.jsonl")
    source = Path(plan["source"])
    if not source.is_dir():
        raise FileNotFoundError("Original generation source snapshot missing: " + str(source))
    target = folder / "generation_source.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(source.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                z.write(path, path.relative_to(source).as_posix())
    write(folder / "provenance.json", dict(method=method, benchmark=benchmark, bundle_id=plan["bundle_id"],
          source_archive_sha256=digest(target), metrics_sha256=digest(folder / "metrics.json"),
          manifest_sha256=digest(folder / "manifest.jsonl"),
          metric_selection="dbpp-no-style" if method == "freecus" and benchmark == "dreambench_plus" else "all",
          metric_cases=len(json.loads(metrics.read_text(encoding="utf-8"))["scores"]),
          layout="ZIP shards in method/artifacts/benchmark/bundle_id; original bytes verified; restore script rebuilds src_image"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["freecus", "dreamo", "uno"], required=True)
    parser.add_argument("--benchmark", choices=["dreambench", "dreambench_plus"], required=True)
    args = parser.parse_args()
    method, benchmark = args.method, args.benchmark
    base = ROOT / method
    if method == "freecus":
        state = base / ("outputs/modal" if benchmark == "dreambench" else "outputs/modal-method-comparison-dbpp")
        plan = json.loads((state / ("latest-" + benchmark + ".json")).read_text(encoding="utf-8"))
        bundle_id = plan["bundle_id"]
        archives = base / "artifacts" / benchmark / bundle_id
        lanes = sorted((state / "collected/results" / bundle_id).glob("lane-*"))
        if len(lanes) != 2:
            raise ValueError("Expected both already downloaded FreeCus lanes")
        count = 750 if benchmark == "dreambench" else 1188
        info = pack(lanes, ROOT / "freecus/manifests" / (benchmark + "_project_seed42.jsonl"), ROOT,
                    archives, method, benchmark, bundle_id, count)
    else:
        profile = PROFILES[method, benchmark]
        state = base / "outputs/modal" / profile
        plan = json.loads((state / (benchmark + ".json")).read_text(encoding="utf-8"))
        status = json.loads((state / (benchmark + "-workflow-status.json")).read_text(encoding="utf-8"))
        if status["status"] != "complete":
            raise ValueError("Generation is not complete")
        bundle_id = plan["bundle_id"]
        archives = base / "artifacts" / benchmark / bundle_id
        count = 750 if benchmark == "dreambench" else 1350
        from freecus.modal_client import connect
        modal = connect(profile)
        volume = modal.Volume.from_name(method + "-runs")
        receipt_path = state / (benchmark + "-artifact-export.json")
        if receipt_path.exists():
            call = modal.FunctionCall.from_id(json.loads(receipt_path.read_text())["call_id"])
        else:
            image = modal.Image.debian_slim(python_version="3.11").add_local_file(
                ROOT / "freecus/operations/artifact_bundle.py", "/root/freecus/operations/artifact_bundle.py")
            app = modal.App(method + "-portable-artifact-export")

            @app.function(name="export", serialized=True, image=image, volumes={"/work": volume},
                          cpu=(2, 2), memory=(2048, 2048), timeout=3600, retries=0,
                          max_containers=1, scaledown_window=2)
            def export(method, benchmark, bundle_id, gpu, expected):
                from pathlib import Path
                from freecus.operations.artifact_bundle import pack
                volume.reload()
                root = Path("/work")
                folder = root / "portable_exports" / bundle_id
                bundle = root / "bundles" / bundle_id
                info = pack(sorted((root / "results" / bundle_id / gpu).glob("lane-*")),
                            bundle / "manifest.jsonl", bundle / "input", folder,
                            method, benchmark, bundle_id, expected)
                volume.commit()
                return info

            with modal.enable_output():
                app.deploy()
            call = export.spawn(method, benchmark, bundle_id, status["selected_gpu"], count)
            write(receipt_path, dict(call_id=call.object_id, bundle_id=bundle_id, method=method, benchmark=benchmark))
        info = call.get()
        archives.mkdir(parents=True, exist_ok=True)
        for part in info["parts"]:
            target = archives / part["file"]
            if target.exists() and digest(target) == part["sha256"]:
                continue
            required = part["size_bytes"] + 1024**3
            if shutil.disk_usage(archives).free < required:
                raise RuntimeError("Insufficient disk space; remote export preserved")
            temp = target.with_suffix(".download")
            progress = [0, 128 * 1024**2]

            def report_progress(advance=0, **kwargs):
                progress[0] += advance
                if progress[0] >= progress[1]:
                    print(method, benchmark, part["file"], round(progress[0] / 2**30, 2), "GiB transferred", flush=True)
                    progress[1] += 128 * 1024**2

            with temp.open("wb") as f:
                volume.read_file_into_fileobj("portable_exports/" + bundle_id + "/" + part["file"],
                                              f, progress_cb=report_progress)
            if temp.stat().st_size != part["size_bytes"] or digest(temp) != part["sha256"]:
                raise ValueError("Downloaded ZIP hash mismatch")
            temp.replace(target)
            print(method, benchmark, "downloaded", target.name, round(target.stat().st_size / 2**30, 3), "GiB", flush=True)
    results = base / "results" / benchmark / bundle_id
    results.mkdir(parents=True, exist_ok=True)
    write(results / "bundle_index.json", info)
    preserve_provenance(method, benchmark, plan, results)
    verify(archives, info)
    write(results / "verification.json", dict(status="verified", generated_images=info["generated_images"],
          zip_files=len(info["parts"]), total_archive_bytes=sum(p["size_bytes"] for p in info["parts"]),
          all_member_sha256_verified=True, source_reference_paths_recoverable=True))
    print(json.dumps(dict(status="verified", method=method, benchmark=benchmark,
          generated_images=info["generated_images"], parts=len(info["parts"]),
          gib=sum(p["size_bytes"] for p in info["parts"])/2**30)), flush=True)


if __name__ == "__main__":
    main()
