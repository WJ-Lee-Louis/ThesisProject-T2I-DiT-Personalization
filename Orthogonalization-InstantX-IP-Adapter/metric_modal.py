"""Modal GPU workers for the XVerse subset/custom metric suite."""

# ruff: noqa: E402

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace
from urllib.request import urlopen
from uuid import uuid4

import modal


EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parent
for source_root in (EXPERIMENT_ROOT, PROJECT_ROOT):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from orthogonalization_instantx_ip_adapter.config import (
    AES_PREDICTOR_URL,
    EVAL_BATCH_RESULTS_VOLUME_NAME,
    METRIC_APP_NAME,
    METRIC_GPU,
    METRIC_MAX_CONTAINERS,
    METRIC_RESULTS_VOLUME_NAME,
    SAM2_CHECKPOINT_URL,
    XVERSE_REPOSITORY,
    XVERSE_REVISION,
)
from orthogonalization_instantx_ip_adapter.evaluation_manifest import (
    EVALUATION_CONDITIONS,
)
from orthogonalization_instantx_ip_adapter.implementation import (
    compute_metric_implementation_digest,
)
from shared.modal_runtime import HF_SECRET_NAME, build_hf_cache_volume


BENCHMARK_ROOT = Path("/root/datasets/XVerseBench_0907")
METRIC_SCRIPT_ROOT = BENCHMARK_ROOT / "metrics"
XVERSE_ROOT = Path("/opt/xverse")
GENERATION_RESULTS_ROOT = Path("/generation_results")
METRIC_RESULTS_ROOT = Path("/metric_results")
CACHE_ROOT = Path("/cache/xverse_metrics")
SAM_CHECKPOINT = CACHE_ROOT / "sam2.1_hiera_large.pt"
AES_PREDICTOR = CACHE_ROOT / "aesthetic_predictor_v2_5.pth"
METRICS = ("dpg", "ip", "aes")
EXPECTED_IMAGES_PER_CONDITION = {"subset": 116, "custom": 90}
IMPLEMENTATION_DIGEST_ENV = "XVERSE_METRIC_IMPLEMENTATION_DIGEST"
IMPLEMENTATION_DIGEST = os.environ.get(
    IMPLEMENTATION_DIGEST_ENV
) or compute_metric_implementation_digest(EXPERIMENT_ROOT)


metric_packages = (
    "addict==2.4.0",
    "aesthetic-predictor-v2-5==2024.12.18.1",
    "accelerate==1.2.1",
    "einops==0.8.1",
    "huggingface_hub==0.28.1",
    "hydra-core==1.3.2",
    "iopath==0.1.10",
    "modelscope==1.23.2",
    "numpy==1.26.4",
    "omegaconf==2.3.0",
    "opencv-python-headless==4.11.0.86",
    "pandas==1.5.3",
    "pillow==11.2.1",
    "protobuf==5.29.3",
    "pycocotools==2.0.8",
    "pyyaml==6.0.2",
    "requests==2.32.4",
    "safetensors==0.5.3",
    "sentencepiece==0.2.0",
    "simplejson==3.20.1",
    "sortedcontainers==2.4.0",
    "supervision==0.25.1",
    "timm==1.0.15",
    "tqdm==4.67.1",
    "transformers==4.49.0",
    "webdataset==0.2.111",
)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .run_commands(
        "pip install torch==2.6.0 torchvision==0.21.0 "
        "--index-url https://download.pytorch.org/whl/cu124"
    )
    .pip_install(*metric_packages)
    .run_commands(
        f"git clone {XVERSE_REPOSITORY} /opt/xverse",
        f"git -C /opt/xverse checkout --detach {XVERSE_REVISION}",
    )
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "HUGGINGFACE_HUB_CACHE": "/cache/huggingface/hub",
            "MODELSCOPE_CACHE": "/cache/modelscope",
            "TORCH_HOME": "/cache/torch",
            "SAM2_MODEL_PATH": "/cache/xverse_metrics/sam2.1_hiera_large.pt",
            "DINO_MODEL_PATH": "facebook/dino-vits16",
            "FLORENCE2_MODEL_PATH": "microsoft/Florence-2-large",
            "DPG_VQA_MODEL_PATH": (
                "damo/mplug_visual-question-answering_coco_large_en"
            ),
            IMPLEMENTATION_DIGEST_ENV: IMPLEMENTATION_DIGEST,
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_python_source("shared")
    .add_local_dir(
        EXPERIMENT_ROOT / "orthogonalization_instantx_ip_adapter",
        remote_path="/root/orthogonalization_instantx_ip_adapter",
    )
    .add_local_dir(
        EXPERIMENT_ROOT / "datasets" / "XVerseBench_0907" / "metrics",
        remote_path="/root/datasets/XVerseBench_0907/metrics",
    )
    .add_local_dir(
        EXPERIMENT_ROOT / "datasets" / "XVerseBench_0907" / "subset_bench",
        remote_path="/root/datasets/XVerseBench_0907/subset_bench",
    )
    .add_local_dir(
        EXPERIMENT_ROOT / "datasets" / "XVerseBench_0907" / "custom_bench",
        remote_path="/root/datasets/XVerseBench_0907/custom_bench",
    )
)

app = modal.App(METRIC_APP_NAME)
hf_cache = build_hf_cache_volume()
generation_results = modal.Volume.from_name(EVAL_BATCH_RESULTS_VOLUME_NAME)
metric_results = modal.Volume.from_name(
    METRIC_RESULTS_VOLUME_NAME, create_if_missing=True
)


@app.function(
    image=image,
    gpu=METRIC_GPU,
    timeout=30 * 60,
    volumes={
        "/cache": hf_cache,
        "/generation_results": generation_results.with_mount_options(read_only=True),
    },
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
)
def diagnose_florence2(
    batch_id: str = "xverse-m2-0908-subset-baseline",
    record_id: str = "official_animal_000",
    seed: int = 42,
) -> dict:
    """Compare XVerse Florence-2 before/after the missing weight tie repair."""

    from PIL import Image

    if str(METRIC_SCRIPT_ROOT) not in sys.path:
        sys.path.insert(0, str(METRIC_SCRIPT_ROOT))
    from common import import_official_xverse
    from florence2_compat import (
        florence2_weight_report,
        pin_florence2_checkpoint,
        repair_xverse_florence2_weight_tying,
        run_florence2_diagnostic,
    )

    import_official_xverse(XVERSE_ROOT)
    from eval.tools.florence_sam import ObjectDetector

    _ensure_download(SAM2_CHECKPOINT_URL, SAM_CHECKPOINT)
    hf_cache.commit()
    generation_results.reload()
    batch_root = _safe_child(GENERATION_RESULTS_ROOT, batch_id)
    generated_manifest = _read_json(
        batch_root / "generated_images_manifest.json"
    )
    generated_row = next(
        row
        for row in generated_manifest["images"]
        if str(row["record_id"]) == record_id and int(row["seed"]) == int(seed)
    )
    benchmark_manifest = _read_json(
        BENCHMARK_ROOT / "subset_bench" / "benchmark_manifest.json"
    )
    record = next(
        row for row in benchmark_manifest["records"]
        if str(row["record_id"]) == record_id
    )
    reference_path = BENCHMARK_ROOT / record["reference_path"]
    generated_path = _safe_child(batch_root, generated_row["image_path"])
    checkpoint = pin_florence2_checkpoint()
    detector = ObjectDetector("cuda:0")
    detector._florence2_checkpoint = checkpoint
    before_weights = florence2_weight_report(detector)

    with Image.open(reference_path) as opened:
        reference = opened.convert("RGB")
    with Image.open(generated_path) as opened:
        generated = opened.convert("RGB")

    before_reference = run_florence2_diagnostic(
        detector, reference, record["subject_label"]
    )
    before_generated = run_florence2_diagnostic(
        detector, generated, record["subject_label"]
    )
    repair_report = repair_xverse_florence2_weight_tying(detector)
    after_reference = run_florence2_diagnostic(
        detector, reference, record["subject_label"]
    )
    after_generated = run_florence2_diagnostic(
        detector, generated, record["subject_label"]
    )
    reference_instances = detector.get_multiple_instances(
        reference, record["subject_label"], min_size=max(1, reference.width // 20)
    )
    generated_instances = detector.get_multiple_instances(
        generated, record["subject_label"], min_size=max(1, generated.width // 20)
    )
    return {
        "batch_id": batch_id,
        "record_id": record_id,
        "seed": int(seed),
        "subject_label": record["subject_label"],
        "reference_path": str(reference_path),
        "generated_path": str(generated_path),
        "before_weights": before_weights,
        "repair_report": repair_report,
        "before": {
            "reference": before_reference,
            "generated": before_generated,
        },
        "after": {
            "reference": after_reference,
            "generated": after_generated,
            "reference_instance_count": len(
                reference_instances.get("instance_images", [])
            ),
            "generated_instance_count": len(
                generated_instances.get("instance_images", [])
            ),
            "reference_mask_scores": [
                float(value)
                for value in reference_instances.get("instance_mask_scores", [])
            ],
            "generated_mask_scores": [
                float(value)
                for value in generated_instances.get("instance_mask_scores", [])
            ],
        },
    }


@app.function(
    image=image,
    gpu=METRIC_GPU,
    timeout=60 * 60,
    volumes={
        "/cache": hf_cache,
        "/generation_results": generation_results.with_mount_options(read_only=True),
    },
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
)
def validate_florence2_gate(
    batch_prefix: str = "xverse-m2-0908-subset",
    generated_record_id: str = "official_animal_000",
    generated_seed: int = 42,
) -> dict:
    """Require all references and one generated sample/condition to reach SAM2."""

    from PIL import Image

    if str(METRIC_SCRIPT_ROOT) not in sys.path:
        sys.path.insert(0, str(METRIC_SCRIPT_ROOT))
    from common import import_official_xverse
    from florence2_compat import (
        pin_florence2_checkpoint,
        repair_xverse_florence2_weight_tying,
    )
    from prepare_segmented_references import detect_reference_subject

    import_official_xverse(XVERSE_ROOT)
    from eval.tools.florence_sam import ObjectDetector

    _ensure_download(SAM2_CHECKPOINT_URL, SAM_CHECKPOINT)
    hf_cache.commit()
    generation_results.reload()
    checkpoint = pin_florence2_checkpoint()
    detector = ObjectDetector("cuda:0")
    detector._florence2_checkpoint = checkpoint
    repair_report = repair_xverse_florence2_weight_tying(detector)
    benchmark_manifest = _read_json(
        BENCHMARK_ROOT / "subset_bench" / "benchmark_manifest.json"
    )

    reference_rows = []
    for record in benchmark_manifest["records"]:
        reference_path = BENCHMARK_ROOT / record["reference_path"]
        with Image.open(reference_path) as opened:
            image = opened.convert("RGB")
        result, query, attempts = detect_reference_subject(
            detector,
            image,
            record["subject_label"],
            min_size=max(1, image.width // 20),
        )
        instance_count = len(result.get("instance_images", [])) if result else 0
        reference_rows.append(
            {
                "record_id": record["record_id"],
                "subject_label": record["subject_label"],
                "query": query,
                "query_attempts": attempts,
                "instance_count": instance_count,
                "mask_scores": (
                    [float(value) for value in result["instance_mask_scores"]]
                    if result
                    else []
                ),
            }
        )

    generated_record = next(
        row
        for row in benchmark_manifest["records"]
        if str(row["record_id"]) == generated_record_id
    )
    generated_rows = []
    for condition in EVALUATION_CONDITIONS:
        batch_id = f"{batch_prefix}-{condition}"
        batch_root = _safe_child(GENERATION_RESULTS_ROOT, batch_id)
        manifest = _read_json(batch_root / "generated_images_manifest.json")
        row = next(
            item
            for item in manifest["images"]
            if str(item["record_id"]) == generated_record_id
            and int(item["seed"]) == int(generated_seed)
        )
        image_path = _safe_child(batch_root, row["image_path"])
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        result = detector.get_multiple_instances(
            image,
            generated_record["subject_label"],
            min_size=max(1, image.width // 20),
        )
        generated_rows.append(
            {
                "condition": condition,
                "batch_id": batch_id,
                "instance_count": len(result.get("instance_images", [])),
                "mask_scores": [
                    float(value) for value in result.get("instance_mask_scores", [])
                ],
            }
        )

    reference_failures = [
        row for row in reference_rows if row["instance_count"] == 0
    ]
    generated_failures = [
        row for row in generated_rows if row["instance_count"] == 0
    ]
    return {
        "valid": not reference_failures and not generated_failures,
        "repair_report": repair_report,
        "reference_count": len(reference_rows),
        "reference_detected_count": len(reference_rows) - len(reference_failures),
        "reference_failures": reference_failures,
        "generated_sample_count": len(generated_rows),
        "generated_detected_count": len(generated_rows) - len(generated_failures),
        "generated_failures": generated_failures,
        "references": reference_rows,
        "generated_samples": generated_rows,
    }


@app.function(
    image=image,
    timeout=20 * 60,
    volumes={"/cache": hf_cache},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
)
def check_access() -> dict:
    """Verify that public/HF model resources needed by metrics are reachable."""
    from huggingface_hub import HfApi

    if str(METRIC_SCRIPT_ROOT) not in sys.path:
        sys.path.insert(0, str(METRIC_SCRIPT_ROOT))
    from score_dpg import validate_dpg_runtime

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    repositories = (
        "microsoft/Florence-2-large",
        "facebook/dino-vits16",
        "google/siglip-so400m-patch14-384",
    )
    checked = []
    for repo_id in repositories:
        info = api.model_info(repo_id=repo_id)
        checked.append({"repo_id": repo_id, "sha": info.sha})
    for url in (SAM2_CHECKPOINT_URL, AES_PREDICTOR_URL):
        with urlopen(url, timeout=60) as response:
            response.read(1)
        checked.append({"url": url, "reachable": True})
    return {
        "access": "OK",
        "gpu_metrics": list(METRICS),
        "resources": checked,
        "implementation_digest": IMPLEMENTATION_DIGEST,
        "dpg_runtime": validate_dpg_runtime(),
    }


@app.cls(
    image=image,
    gpu=METRIC_GPU,
    max_containers=METRIC_MAX_CONTAINERS,
    retries=modal.Retries(
        max_retries=1,
        initial_delay=10.0,
        backoff_coefficient=2.0,
    ),
    scaledown_window=300,
    startup_timeout=30 * 60,
    timeout=24 * 60 * 60,
    volumes={
        "/cache": hf_cache,
        "/generation_results": generation_results.with_mount_options(read_only=True),
        "/metric_results": metric_results,
    },
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
)
class XVerseMetricEvaluator:
    """Evaluate both benchmarks with one loaded model stack per metric."""

    @modal.method()
    def evaluate(
        self,
        metric: str,
        run_id: str,
        generation_batches: dict[str, dict[str, str]],
        implementation_digest: str,
    ) -> dict:
        metric = _validate_metric(metric)
        run_id = _validate_component(run_id, "run_id")
        deployed_digest = os.environ.get(IMPLEMENTATION_DIGEST_ENV)
        if implementation_digest != deployed_digest:
            raise RuntimeError(
                "Local metric code does not match the deployed worker. "
                "Redeploy metric_modal.py before submitting."
            )
        generation_results.reload()
        metric_results.reload()
        run_root = _safe_child(METRIC_RESULTS_ROOT, run_id)
        metadata_root = run_root / "_metadata"
        status_path = metadata_root / f"status_{metric}.json"
        config_path = metadata_root / f"config_{metric}.json"
        metadata_root.mkdir(parents=True, exist_ok=True)
        config = {
            "schema_version": 1,
            "run_id": run_id,
            "metric": metric,
            "generation_batches": generation_batches,
            "implementation_digest": implementation_digest,
            "xverse_revision": XVERSE_REVISION,
            "requested_gpu": METRIC_GPU,
        }
        _validate_existing_config(config_path, status_path, config)
        _atomic_write_json(config_path, config)
        status = {
            "schema_version": 1,
            "run_id": run_id,
            "metric": metric,
            "state": "running",
            "current_benchmark": None,
            "completed_benchmarks": [],
            "started_at": _utc_now(),
            "updated_at": _utc_now(),
            "completed_at": None,
            "error": None,
        }
        _atomic_write_json(status_path, status)
        metric_results.commit()

        try:
            manifests = {
                benchmark: _assemble_remote_manifest(
                    run_root, benchmark, generation_batches[benchmark]
                )
                for benchmark in ("subset", "custom")
            }
            if metric == "ip":
                _ensure_download(SAM2_CHECKPOINT_URL, SAM_CHECKPOINT)
                hf_cache.commit()
            elif metric == "aes":
                _ensure_download(AES_PREDICTOR_URL, AES_PREDICTOR)
                hf_cache.commit()

            status["current_benchmark"] = "subset+custom"
            status["updated_at"] = _utc_now()
            _atomic_write_json(status_path, status)
            metric_results.commit()
            _run_metric_suite(metric, manifests, run_root)
            for benchmark in ("subset", "custom"):
                output = run_root / benchmark / f"{metric}.json"
                expected = EXPECTED_IMAGES_PER_CONDITION[benchmark] * len(
                    EVALUATION_CONDITIONS
                )
                if not _valid_score_output(output, benchmark, expected):
                    raise RuntimeError(
                        f"Invalid {metric} output after scoring: {benchmark}"
                    )
            status["completed_benchmarks"] = ["subset", "custom"]

            status["state"] = "completed"
            status["current_benchmark"] = None
            status["completed_at"] = _utc_now()
            status["updated_at"] = status["completed_at"]
            _atomic_write_json(status_path, status)
            metric_results.commit()
            return status
        except Exception as error:
            status["state"] = "failed"
            status["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
            status["updated_at"] = _utc_now()
            _atomic_write_json(status_path, status)
            metric_results.commit()
            raise


def _assemble_remote_manifest(
    run_root: Path,
    benchmark: str,
    batches: dict[str, str],
) -> Path:
    if set(batches) != set(EVALUATION_CONDITIONS):
        raise ValueError(
            f"{benchmark} batch conditions differ from the six-condition protocol"
        )
    rows = []
    benchmark_id = None
    expected_per_condition = EXPECTED_IMAGES_PER_CONDITION[benchmark]
    expected_keys = _expected_benchmark_keys(benchmark)
    for condition in EVALUATION_CONDITIONS:
        batch_id = _validate_component(batches[condition], "batch_id")
        batch_root = _safe_child(GENERATION_RESULTS_ROOT, batch_id)
        status = _read_json(batch_root / "_metadata" / "status.json")
        config = _read_json(batch_root / "_metadata" / "config.json")
        generated = _read_json(batch_root / "generated_images_manifest.json")
        if status.get("state") != "completed":
            raise RuntimeError(f"Generation batch is not complete: {batch_id}")
        if int(status.get("completed_count", -1)) != expected_per_condition:
            raise RuntimeError(f"Generation count mismatch in {batch_id}")
        if (
            config.get("condition") != condition
            or generated.get("condition") != condition
        ):
            raise RuntimeError(f"Generation condition mismatch in {batch_id}")
        current_id = generated.get("benchmark_id")
        benchmark_id = current_id if benchmark_id is None else benchmark_id
        if current_id != benchmark_id:
            raise RuntimeError(f"Benchmark ID mismatch in {batch_id}")
        batch_rows = generated.get("images", [])
        if len(batch_rows) != expected_per_condition:
            raise RuntimeError(f"Generated manifest count mismatch in {batch_id}")
        seen = set()
        for raw in batch_rows:
            key = (str(raw["record_id"]), int(raw["seed"]))
            if key in seen:
                raise RuntimeError(f"Duplicate generated row in {batch_id}: {key}")
            seen.add(key)
            image_path = _safe_child(batch_root, str(raw["image_path"]))
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing generated image: {image_path}")
            row = dict(raw)
            row["condition"] = condition
            row["image_path"] = str(image_path)
            rows.append(row)
        if seen != expected_keys:
            raise RuntimeError(
                f"Generated key set mismatch in {batch_id}: "
                f"missing={len(expected_keys - seen)}, extra={len(seen - expected_keys)}"
            )
    output = run_root / "_metadata" / f"generated_{benchmark}.json"
    _atomic_write_json(
        output,
        {
            "schema_version": 1,
            "benchmark_id": benchmark_id,
            "conditions": list(EVALUATION_CONDITIONS),
            "images": rows,
        },
    )
    return output


def _expected_benchmark_keys(benchmark: str) -> set[tuple[str, int]]:
    payload = _read_json(
        BENCHMARK_ROOT / f"{benchmark}_bench" / "benchmark_manifest.json"
    )
    keys = set()
    for record in payload["records"]:
        seeds = record.get("seeds", [record.get("seed")])
        keys.update((str(record["record_id"]), int(seed)) for seed in seeds)
    return keys


def _run_metric_suite(
    metric: str,
    manifests: dict[str, Path],
    run_root: Path,
) -> None:
    if str(METRIC_SCRIPT_ROOT) not in sys.path:
        sys.path.insert(0, str(METRIC_SCRIPT_ROOT))
    previous_cwd = Path.cwd()
    try:
        # The released SAM2 builder derives its package/config path from cwd.
        os.chdir(XVERSE_ROOT)
        from run_metric_suite import run_suite

        args = SimpleNamespace(
            metric=metric,
            benchmark_root=BENCHMARK_ROOT,
            xverse_root=XVERSE_ROOT,
            subset_manifest=manifests["subset"],
            custom_manifest=manifests["custom"],
            run_root=run_root,
            sam_checkpoint=SAM_CHECKPOINT if metric == "ip" else None,
            aes_predictor=AES_PREDICTOR if metric == "aes" else None,
        )
        run_suite(args, on_checkpoint=metric_results.commit)
    finally:
        os.chdir(previous_cwd)


def _valid_score_output(path: Path, benchmark: str, expected: int) -> bool:
    if not path.is_file():
        return False
    try:
        payload = _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    expected_id = {
        "subset": "xverse_single_animal_29",
        "custom": "xverse_ref_custom_animal_15x6",
    }[benchmark]
    scores = payload.get("scores", [])
    keys = {
        (str(row["record_id"]), str(row["condition"]), int(row["seed"]))
        for row in scores
    }
    structurally_valid = (
        payload.get("benchmark_id") == expected_id
        and len(scores) == expected
        and len(keys) == expected
    )
    if not structurally_valid:
        return False
    if payload.get("metric") == "IP-Sim":
        return any(int(row.get("detected_candidate_count", 0)) > 0 for row in scores)
    return True


def _ensure_download(url: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size > 1024:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".d-{uuid4().hex[:12]}.tmp")
    try:
        with urlopen(url, timeout=120) as response, temporary.open("wb") as stream:
            while chunk := response.read(8 * 1024 * 1024):
                stream.write(chunk)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_existing_config(path: Path, status_path: Path, expected: dict) -> None:
    if not path.is_file():
        return
    actual = _read_json(path)
    if actual == expected:
        return
    actual_semantics = dict(actual)
    expected_semantics = dict(expected)
    actual_semantics.pop("implementation_digest", None)
    expected_semantics.pop("implementation_digest", None)
    previous_status = _read_json(status_path) if status_path.is_file() else {}
    if (
        previous_status.get("state") == "failed"
        and actual_semantics == expected_semantics
    ):
        return
    raise RuntimeError(
        "This metric run_id already exists with different settings, or its "
        "previous job is not failed; choose a new run_id."
    )


def _validate_metric(value: str) -> str:
    if value not in METRICS:
        raise ValueError(f"Unsupported metric: {value!r}")
    return value


def _validate_component(value: str, label: str) -> str:
    if (
        not value
        or value in {".", ".."}
        or any(
            char
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            for char in value
        )
    ):
        raise ValueError(f"Unsafe {label}: {value!r}")
    return value


def _safe_child(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    resolved_root = root.resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ValueError(f"Path escapes root: {relative!r}")
    return candidate


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".w-{uuid4().hex[:12]}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "EXPECTED_IMAGES_PER_CONDITION",
    "IMPLEMENTATION_DIGEST",
    "METRICS",
    "XVerseMetricEvaluator",
]
