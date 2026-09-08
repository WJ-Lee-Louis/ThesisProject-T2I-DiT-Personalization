"""Validated XVerse subset/custom manifests for Modal image generation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Mapping


EvaluationBenchmark = Literal["subset", "custom"]

EVALUATION_CONDITIONS = (
    "baseline",
    "ssg",
    "target_projection_ssg",
    "source_projection_ssg",
    "st_matched_projection_ssg",
    "st_all_projection_ssg",
)

DEFAULT_EVALUATION_DATASET_ROOT = (
    Path(__file__).resolve().parents[1] / "datasets" / "XVerseBench_0907"
)
DEFAULT_EVALUATION_SOURCE_CAPTIONS = (
    DEFAULT_EVALUATION_DATASET_ROOT / "source_captions.runtime.json"
)

_BENCHMARK_MANIFESTS = {
    "subset": "subset_bench/benchmark_manifest.json",
    "custom": "custom_bench/benchmark_manifest.json",
}


@dataclass(frozen=True, slots=True)
class EvaluationReference:
    filename: str
    stem: str
    path: str
    content_sha256: str
    subject_caption: str
    subject_label: str

    def upload_dict(self) -> dict[str, str]:
        return {
            "filename": self.filename,
            "stem": self.stem,
            "content_sha256": self.content_sha256,
            "subject_caption": self.subject_caption,
            "subject_label": self.subject_label,
        }


@dataclass(frozen=True, slots=True)
class EvaluationJob:
    record_id: str
    reference_filename: str
    reference_stem: str
    reference_path: str
    subject_caption: str
    subject_label: str
    target_object_prompt: str
    prompt_id: str
    prompt: str
    composition: str
    action: str | None
    background: str | None
    attribute: str | None
    seed: int
    output_relative_path: str

    @property
    def target_factors(self) -> tuple[str, ...]:
        values = []
        if self.action is not None:
            values.append("behavior")
        if self.background is not None:
            values.append("background")
        if self.attribute is not None:
            values.append("attribute")
        return tuple(values)

    def upload_dict(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "reference_filename": self.reference_filename,
            "reference_stem": self.reference_stem,
            "subject_caption": self.subject_caption,
            "subject_label": self.subject_label,
            "target_object_prompt": self.target_object_prompt,
            "prompt_id": self.prompt_id,
            "prompt": self.prompt,
            "composition": self.composition,
            "action": self.action,
            "background": self.background,
            "attribute": self.attribute,
            "seed": self.seed,
            "output_relative_path": self.output_relative_path,
        }


@dataclass(frozen=True, slots=True)
class EvaluationManifest:
    benchmark: EvaluationBenchmark
    benchmark_id: str
    references: tuple[EvaluationReference, ...]
    jobs: tuple[EvaluationJob, ...]

    def _canonical_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "benchmark": self.benchmark,
            "benchmark_id": self.benchmark_id,
            "references": [item.upload_dict() for item in self.references],
            "jobs": [item.upload_dict() for item in self.jobs],
        }

    @property
    def semantic_digest(self) -> str:
        encoded = json.dumps(
            self._canonical_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_upload_dict(self) -> dict[str, object]:
        payload = self._canonical_dict()
        payload["semantic_digest"] = self.semantic_digest
        return payload


@dataclass(frozen=True, slots=True)
class EvaluationSourceCaption:
    reference_filename: str
    reference_stem: str
    content_sha256: str
    object_only: str
    source_behavior: str
    source_background: str

    @property
    def object_behavior(self) -> str:
        return f"{self.object_only} {self.source_behavior}"

    @property
    def object_background(self) -> str:
        return f"{self.object_only} {self.source_background}"

    @property
    def object_behavior_background(self) -> str:
        return (
            f"{self.object_only} {self.source_behavior} "
            f"{self.source_background}"
        )

    def upload_dict(self) -> dict[str, str]:
        return {
            "reference_filename": self.reference_filename,
            "reference_stem": self.reference_stem,
            "content_sha256": self.content_sha256,
            "object_only": self.object_only,
            "source_behavior": self.source_behavior,
            "source_background": self.source_background,
        }


@dataclass(frozen=True, slots=True)
class EvaluationSourceCaptions:
    records: tuple[EvaluationSourceCaption, ...]

    @property
    def semantic_digest(self) -> str:
        canonical = {
            "schema_version": 1,
            "records": [record.upload_dict() for record in self.records],
        }
        encoded = json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def for_reference(self, stem: str) -> EvaluationSourceCaption:
        matches = [record for record in self.records if record.reference_stem == stem]
        if len(matches) != 1:
            raise KeyError(f"No unique source caption for {stem!r}")
        return matches[0]

    def to_upload_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "semantic_digest": self.semantic_digest,
            "records": [record.upload_dict() for record in self.records],
        }


def load_evaluation_manifest(
    dataset_root: str | Path,
    benchmark: EvaluationBenchmark,
) -> EvaluationManifest:
    root = Path(dataset_root).expanduser().resolve()
    try:
        manifest_path = root / _BENCHMARK_MANIFESTS[benchmark]
    except KeyError as error:
        raise ValueError(f"Unknown evaluation benchmark: {benchmark!r}") from error
    payload = _load_json(manifest_path)
    annotations = _load_target_annotations(root, benchmark, payload)

    references_by_filename: dict[str, EvaluationReference] = {}
    jobs: list[EvaluationJob] = []
    seen_jobs: set[tuple[str, int]] = set()
    for raw in payload["records"]:
        filename = _required_text(raw, "reference_filename")
        relative_reference = Path(_required_text(raw, "reference_path"))
        reference_path = (root / relative_reference).resolve()
        _ensure_within(reference_path, root)
        if not reference_path.is_file():
            raise FileNotFoundError(f"Missing evaluation reference: {reference_path}")
        expected_sha = _required_text(raw, "content_sha256")
        actual_sha = _sha256_file(reference_path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"Reference hash mismatch for {filename}: "
                f"expected {expected_sha}, got {actual_sha}"
            )
        reference = EvaluationReference(
            filename=filename,
            stem=Path(filename).stem,
            path=str(reference_path),
            content_sha256=expected_sha,
            subject_caption=_required_text(raw, "subject_caption"),
            subject_label=_required_text(raw, "subject_label"),
        )
        previous = references_by_filename.setdefault(filename, reference)
        if previous != reference:
            raise ValueError(f"Inconsistent repeated reference record: {filename}")

        record_id = _required_text(raw, "record_id")
        factor = annotations[record_id]
        _validate_target_factorization(raw, factor)
        seeds_value = raw.get("seeds", [raw.get("seed")])
        if not isinstance(seeds_value, list) or not seeds_value:
            raise ValueError(f"Invalid seed policy for {record_id}")
        for seed_value in seeds_value:
            if isinstance(seed_value, bool) or not isinstance(seed_value, int):
                raise TypeError(f"Seed for {record_id} must be an integer")
            key = (record_id, seed_value)
            if key in seen_jobs:
                raise ValueError(f"Duplicate evaluation job: {key}")
            seen_jobs.add(key)
            jobs.append(
                EvaluationJob(
                    record_id=record_id,
                    reference_filename=filename,
                    reference_stem=reference.stem,
                    reference_path=str(reference_path),
                    subject_caption=reference.subject_caption,
                    subject_label=reference.subject_label,
                    target_object_prompt=factor["target_object_prompt"],
                    prompt_id=_required_text(raw, "prompt_id"),
                    prompt=_required_text(raw, "prompt", strip=False),
                    composition=_required_text(raw, "composition"),
                    action=factor["action"],
                    background=factor["background"],
                    attribute=factor["attribute"],
                    seed=seed_value,
                    output_relative_path=(
                        f"images/{record_id}/seed_{seed_value:04d}.png"
                    ),
                )
            )
    return EvaluationManifest(
        benchmark=benchmark,
        benchmark_id=_required_text(payload, "benchmark_id"),
        references=tuple(references_by_filename.values()),
        jobs=tuple(jobs),
    )


def build_evaluation_manifest_from_upload(
    payload: Mapping[str, Any],
    reference_root: str | Path,
) -> EvaluationManifest:
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError("Evaluation upload manifest schema_version must be 1")
    benchmark = payload.get("benchmark")
    if benchmark not in _BENCHMARK_MANIFESTS:
        raise ValueError(f"Unknown uploaded benchmark: {benchmark!r}")
    root = Path(reference_root).resolve()
    references = []
    reference_by_filename = {}
    for raw in payload.get("references", []):
        filename = _required_text(raw, "filename")
        path = (root / filename).resolve()
        _ensure_within(path, root)
        expected_sha = _required_text(raw, "content_sha256")
        if not path.is_file() or _sha256_file(path) != expected_sha:
            raise ValueError(f"Uploaded reference validation failed: {filename}")
        reference = EvaluationReference(
            filename=filename,
            stem=_required_text(raw, "stem"),
            path=str(path),
            content_sha256=expected_sha,
            subject_caption=_required_text(raw, "subject_caption"),
            subject_label=_required_text(raw, "subject_label"),
        )
        references.append(reference)
        reference_by_filename[filename] = reference

    jobs = []
    for raw in payload.get("jobs", []):
        filename = _required_text(raw, "reference_filename")
        if filename not in reference_by_filename:
            raise ValueError(f"Job references unknown file: {filename}")
        jobs.append(
            EvaluationJob(
                record_id=_required_text(raw, "record_id"),
                reference_filename=filename,
                reference_stem=_required_text(raw, "reference_stem"),
                reference_path=reference_by_filename[filename].path,
                subject_caption=_required_text(raw, "subject_caption"),
                subject_label=_required_text(raw, "subject_label"),
                target_object_prompt=_required_text(raw, "target_object_prompt"),
                prompt_id=_required_text(raw, "prompt_id"),
                prompt=_required_text(raw, "prompt", strip=False),
                composition=_required_text(raw, "composition"),
                action=_optional_text(raw.get("action")),
                background=_optional_text(raw.get("background")),
                attribute=_optional_text(raw.get("attribute")),
                seed=_required_integer(raw, "seed"),
                output_relative_path=_required_text(raw, "output_relative_path"),
            )
        )
    manifest = EvaluationManifest(
        benchmark=benchmark,
        benchmark_id=_required_text(payload, "benchmark_id"),
        references=tuple(references),
        jobs=tuple(jobs),
    )
    if manifest.semantic_digest != payload.get("semantic_digest"):
        raise ValueError("Uploaded evaluation manifest digest mismatch")
    return manifest


def load_evaluation_source_captions(
    path: str | Path,
    manifest: EvaluationManifest,
) -> EvaluationSourceCaptions:
    payload = _load_json(Path(path).expanduser().resolve())
    if payload.get("schema_version") != 1:
        raise ValueError("Evaluation source captions schema_version must be 1")
    raw_by_filename = {
        _required_text(raw, "reference_filename"): raw
        for raw in payload.get("records", [])
    }
    records = []
    for reference in manifest.references:
        raw = raw_by_filename.get(reference.filename)
        if raw is None:
            raise ValueError(f"Missing source caption for {reference.filename}")
        if _required_text(raw, "content_sha256") != reference.content_sha256:
            raise ValueError(f"Source caption hash mismatch: {reference.filename}")
        prompts = raw.get("source_prompts")
        if not isinstance(prompts, Mapping):
            raise TypeError(f"Missing source_prompts for {reference.filename}")
        records.append(
            EvaluationSourceCaption(
                reference_filename=reference.filename,
                reference_stem=reference.stem,
                content_sha256=reference.content_sha256,
                object_only=_required_text(prompts, "object_only"),
                source_behavior=_required_text(raw, "source_behavior"),
                source_background=_required_text(raw, "source_background"),
            )
        )
    return EvaluationSourceCaptions(tuple(records))


def build_evaluation_source_captions_from_upload(
    payload: Mapping[str, Any],
    manifest: EvaluationManifest,
) -> EvaluationSourceCaptions:
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError("Uploaded source captions schema_version must be 1")
    records = tuple(
        EvaluationSourceCaption(
            reference_filename=_required_text(raw, "reference_filename"),
            reference_stem=_required_text(raw, "reference_stem"),
            content_sha256=_required_text(raw, "content_sha256"),
            object_only=_required_text(raw, "object_only"),
            source_behavior=_required_text(raw, "source_behavior"),
            source_background=_required_text(raw, "source_background"),
        )
        for raw in payload.get("records", [])
    )
    captions = EvaluationSourceCaptions(records)
    if captions.semantic_digest != payload.get("semantic_digest"):
        raise ValueError("Uploaded source-caption digest mismatch")
    expected = {reference.stem for reference in manifest.references}
    actual = {record.reference_stem for record in captions.records}
    if expected != actual:
        raise ValueError("Uploaded source captions do not match references exactly")
    for reference in manifest.references:
        record = captions.for_reference(reference.stem)
        if record.content_sha256 != reference.content_sha256:
            raise ValueError(f"Source-caption image mismatch: {reference.filename}")
    return captions


def _load_target_annotations(
    root: Path,
    benchmark: EvaluationBenchmark,
    manifest_payload: Mapping[str, Any],
) -> dict[str, dict[str, str | None]]:
    if benchmark == "subset":
        payload = _load_json(root / "subset_bench" / "target_factor_annotations.json")
        raw_records = payload.get("records", [])
    else:
        raw_records = [
            {
                "record_id": raw["record_id"],
                "target_object_prompt": raw["subject_caption"],
                "action": raw.get("action"),
                "background": raw.get("background"),
                "attribute": raw.get("attribute"),
            }
            for raw in manifest_payload["records"]
        ]
    annotations = {}
    for raw in raw_records:
        record_id = _required_text(raw, "record_id")
        if record_id in annotations:
            raise ValueError(f"Duplicate target factor annotation: {record_id}")
        annotations[record_id] = {
            "target_object_prompt": _required_text(raw, "target_object_prompt"),
            "action": _optional_text(raw.get("action")),
            "background": _optional_text(raw.get("background")),
            "attribute": _optional_text(raw.get("attribute")),
        }
    expected = {str(raw["record_id"]) for raw in manifest_payload["records"]}
    if set(annotations) != expected:
        raise ValueError("Target factor annotations do not match benchmark records")
    return annotations


def _validate_target_factorization(
    manifest_record: Mapping[str, Any],
    factor: Mapping[str, str | None],
) -> None:
    record_id = _required_text(manifest_record, "record_id")
    prompt = _required_text(manifest_record, "prompt", strip=False)
    attribute = factor["attribute"]
    if attribute is not None:
        if factor["action"] is not None or factor["background"] is not None:
            raise ValueError(
                f"Attribute-only record {record_id} must not also define "
                "action/background"
            )
        if _required_text(manifest_record, "composition") != "O+attribute":
            raise ValueError(
                f"Attribute factor is only supported for O+attribute: {record_id}"
            )
        object_prompt = factor["target_object_prompt"]
        if _normalize_prompt(object_prompt) != _normalize_prompt(
            _required_text(manifest_record, "subject_caption")
        ):
            raise ValueError(
                f"Attribute contrast baseline must equal subject_caption: {record_id}"
            )
        if _normalize_prompt(prompt) == _normalize_prompt(object_prompt):
            raise ValueError(
                f"Attribute prompt must differ from its object baseline: {record_id}"
            )
        return

    reconstructed_prompt = " ".join(
        value
        for value in (
            factor["target_object_prompt"],
            factor["action"],
            factor["background"],
        )
        if value is not None
    )
    if _normalize_prompt(reconstructed_prompt) != _normalize_prompt(prompt):
        raise ValueError(
            f"Target factorization does not reconstruct {record_id}: "
            f"{reconstructed_prompt!r}"
        )


def _required_text(
    value: Mapping[str, Any], field: str, *, strip: bool = True
) -> str:
    raw = value.get(field)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return raw.strip() if strip else raw


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Optional prompt factor must be null or non-empty text")
    return " ".join(value.split())


def _required_integer(value: Mapping[str, Any], field: str) -> int:
    raw = value.get(field)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise TypeError(f"{field} must be an integer")
    return raw


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_within(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Path escapes evaluation dataset root: {path}") from error


def _normalize_prompt(value: str) -> str:
    return " ".join(value.split()).rstrip(".").casefold()


__all__ = [
    "DEFAULT_EVALUATION_DATASET_ROOT",
    "DEFAULT_EVALUATION_SOURCE_CAPTIONS",
    "EVALUATION_CONDITIONS",
    "EvaluationJob",
    "EvaluationManifest",
    "EvaluationSourceCaption",
    "EvaluationSourceCaptions",
    "build_evaluation_manifest_from_upload",
    "build_evaluation_source_captions_from_upload",
    "load_evaluation_manifest",
    "load_evaluation_source_captions",
]
