"""Build and validate the 2026-09-07 XVerseBench evaluation subsets.

This module deliberately has no GPU or model dependencies.  It freezes the
reference/prompt/question manifests before generation and leaves the expensive
metric runners to ``datasets/XVerseBench_0907/metrics``.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


OFFICIAL_XVERSE_COMMIT = "65e2581aac0faffe4607881a10cc486186f6499d"
OFFICIAL_SEEDS = (42, 43, 44, 45)
CUSTOM_BASE_SEED = 42
EXPECTED_SUBSET_REFERENCES = 29
EXPECTED_CUSTOM_REFERENCES = 15


@dataclass(frozen=True, slots=True)
class CustomPromptSpec:
    prompt_id: str
    suffix: str
    composition: str
    action: str | None = None
    background: str | None = None
    background_entity: str | None = None
    background_relation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "suffix": self.suffix,
            "composition": self.composition,
            "action": self.action,
            "background": self.background,
            "background_entity": self.background_entity,
            "background_relation": self.background_relation,
        }


CUSTOM_PROMPTS = (
    CustomPromptSpec(
        prompt_id="oa_sleeping",
        suffix="sleeping",
        composition="O+A",
        action="sleeping",
    ),
    CustomPromptSpec(
        prompt_id="oa_playing_guitar",
        suffix="playing the guitar",
        composition="O+A",
        action="playing the guitar",
    ),
    CustomPromptSpec(
        prompt_id="ob_snow",
        suffix="in the snow",
        composition="O+B",
        background="in the snow",
        background_entity="snow",
        background_relation="in the snow",
    ),
    CustomPromptSpec(
        prompt_id="ob_sofa",
        suffix="on a sofa",
        composition="O+B",
        background="on a sofa",
        background_entity="sofa",
        background_relation="on the sofa",
    ),
    CustomPromptSpec(
        prompt_id="oab_sleeping_snow",
        suffix="sleeping in the snow",
        composition="O+A+B",
        action="sleeping",
        background="in the snow",
        background_entity="snow",
        background_relation="in the snow",
    ),
    CustomPromptSpec(
        prompt_id="oab_playing_guitar_sofa",
        suffix="playing the guitar on a sofa",
        composition="O+A+B",
        action="playing the guitar",
        background="on a sofa",
        background_entity="sofa",
        background_relation="on the sofa",
    ),
)


# This is an analysis label, not an official XVerseBench annotation.  Attribute
# modifiers such as a collar or hat are retained in the prompt but are not
# treated as A or B in this project's action/background grouping.
OFFICIAL_COMPOSITION_BY_INDEX = {
    0: "O+A+B",
    1: "O+A+B",
    2: "O+A+B",
    3: "O+A",
    4: "O+A+B",
    5: "O+A+B",
    6: "O+A+B",
    7: "O+A+B",
    8: "O+A+B",
    9: "O+attribute",
    10: "O+A+B",
    11: "O+A+B",
    12: "O+A+B",
    13: "O+A+B",
    14: "O+A+B",
    15: "O+A+B",
    16: "O+B",
    17: "O+A+B",
    18: "O+A+B",
    19: "O+attribute",
    20: "O+attribute",
    21: "O+A+B",
    22: "O+A+B",
    23: "O+A+B",
    24: "O+B",
    25: "O+attribute",
    26: "O+A+B",
    27: "O+attribute",
    28: "O+A+B",
}


def build_evaluation_assets(
    dataset_root: str | Path,
    official_dataset_root: str | Path,
    *,
    write: bool = True,
) -> dict[str, Any]:
    """Validate references and build all CPU-side benchmark metadata."""

    dataset_root = Path(dataset_root).resolve()
    official_dataset_root = Path(official_dataset_root).resolve()
    official_metadata = official_dataset_root / "official_metadata"
    official_reference_root = (
        official_dataset_root
        / "prepared"
        / "assets"
        / "XVerseBench"
        / "animal"
    )

    official_records = _load_json(
        official_metadata / "XVerseBench_single.json"
    )
    official_dsg = _load_json(
        official_metadata / "XVerseBench_single_DSG.json"
    )
    if len(official_records) != len(official_dsg):
        raise ValueError("Official single metadata and DSG lengths differ")

    animal_pairs = []
    for record, dsg in zip(official_records, official_dsg):
        src = record["modulation"][0]["src_inputs"][0]
        if "/animal/" not in src["image_path"].replace("\\", "/"):
            continue
        index = int(record["index"])
        if dsg["prompt"] != record["prompt"]:
            raise ValueError(f"Official prompt/DSG mismatch at index {index}")
        animal_pairs.append((record, dsg))

    if len(animal_pairs) != EXPECTED_SUBSET_REFERENCES:
        raise ValueError(
            "Expected exactly 29 official single-animal records, got "
            f"{len(animal_pairs)}"
        )
    indices = [int(record["index"]) for record, _ in animal_pairs]
    if indices != list(range(EXPECTED_SUBSET_REFERENCES)):
        raise ValueError(f"Unexpected official animal indices: {indices}")

    official_by_filename = {
        Path(record["modulation"][0]["src_inputs"][0]["image_path"]).name:
        (record, dsg)
        for record, dsg in animal_pairs
    }
    if len(official_by_filename) != EXPECTED_SUBSET_REFERENCES:
        raise ValueError("Official animal reference filenames are not unique")

    subset_files = _discover_images(dataset_root / "subset_bench")
    custom_files = _discover_images(dataset_root / "custom_bench")
    _validate_reference_selection(
        "subset_bench",
        subset_files,
        official_by_filename,
        EXPECTED_SUBSET_REFERENCES,
    )
    _validate_reference_selection(
        "custom_bench",
        custom_files,
        official_by_filename,
        EXPECTED_CUSTOM_REFERENCES,
    )

    integrity_records = []
    for bench_name, files in (
        ("subset_bench", subset_files),
        ("custom_bench", custom_files),
    ):
        for path in files:
            official_path = official_reference_root / path.name
            if not official_path.is_file():
                raise FileNotFoundError(
                    f"Official prepared reference is missing: {official_path}"
                )
            copied_hash = _sha256_file(path)
            official_hash = _sha256_file(official_path)
            if copied_hash != official_hash:
                raise ValueError(
                    "Copied reference differs from the reconstructed prepared "
                    f"source: {path}"
                )
            integrity_records.append(
                {
                    "benchmark": bench_name,
                    "reference_filename": path.name,
                    "sha256": copied_hash,
                    "bytes": path.stat().st_size,
                    "matches_reconstructed_prepared_source": True,
                }
            )

    subset_records = []
    subset_dsg = []
    subset_official_json = []
    for record, dsg in animal_pairs:
        index = int(record["index"])
        src = record["modulation"][0]["src_inputs"][0]
        filename = Path(src["image_path"]).name
        subject_caption = src["caption"].strip()
        subset_records.append(
            {
                "record_id": f"official_animal_{index:03d}",
                "prompt_id": f"official_animal_{index:03d}",
                "record_index": index,
                "official_index": index,
                "reference_filename": filename,
                "reference_path": f"subset_bench/{filename}",
                "content_sha256": _sha256_file(
                    dataset_root / "subset_bench" / filename
                ),
                "subject_caption": subject_caption,
                "subject_label": _subject_label(subject_caption),
                "prompt": record["prompt"],
                "composition": OFFICIAL_COMPOSITION_BY_INDEX[index],
                "seeds": list(OFFICIAL_SEEDS),
                "dpg_record_index": index,
            }
        )
        subset_official_json.append(record)
        subset_dsg.append(dsg)

    custom_records = []
    custom_dsg = []
    custom_official_indices = []
    for reference_order, path in enumerate(custom_files):
        official_record, _ = official_by_filename[path.name]
        official_index = int(official_record["index"])
        src = official_record["modulation"][0]["src_inputs"][0]
        subject_caption = src["caption"].strip()
        subject_label = _subject_label(subject_caption)
        custom_official_indices.append(official_index)
        for prompt_order, spec in enumerate(CUSTOM_PROMPTS):
            record_index = reference_order * len(CUSTOM_PROMPTS) + prompt_order
            prompt = f"{subject_caption} {spec.suffix}"
            record_id = f"custom_{record_index:03d}"
            seed = CUSTOM_BASE_SEED + record_index
            custom_records.append(
                {
                    "record_id": record_id,
                    "record_index": record_index,
                    "official_source_index": official_index,
                    "reference_order": reference_order,
                    "prompt_order": prompt_order,
                    "reference_filename": path.name,
                    "reference_path": f"custom_bench/{path.name}",
                    "content_sha256": _sha256_file(path),
                    "subject_caption": subject_caption,
                    "subject_label": subject_label,
                    "prompt_id": spec.prompt_id,
                    "prompt_suffix": spec.suffix,
                    "prompt": prompt,
                    "composition": spec.composition,
                    "action": spec.action,
                    "background": spec.background,
                    "seed": seed,
                    "dpg_record_index": record_index,
                }
            )
            custom_dsg.append(
                build_custom_dpg_record(
                    record_id=record_id,
                    prompt=prompt,
                    subject_label=subject_label,
                    spec=spec,
                )
            )

    expected_custom_records = EXPECTED_CUSTOM_REFERENCES * len(CUSTOM_PROMPTS)
    if len(custom_records) != expected_custom_records:
        raise AssertionError("Custom record construction produced wrong count")

    subset_manifest = {
        "schema_version": 1,
        "benchmark_name": "XVerseBench single-animal official subset",
        "benchmark_id": "xverse_single_animal_29",
        "official_xverse_commit": OFFICIAL_XVERSE_COMMIT,
        "official_target_prompts": True,
        "official_dpg_questions": True,
        "reference_count": len(subset_records),
        "record_count": len(subset_records),
        "seed_policy": {
            "kind": "four fixed seeds per record",
            "seeds": list(OFFICIAL_SEEDS),
        },
        "records": subset_records,
    }
    custom_manifest = {
        "schema_version": 1,
        "benchmark_name": "XVerse-reference controlled animal benchmark",
        "benchmark_id": "xverse_ref_custom_animal_15x6",
        "official_xverse_commit": OFFICIAL_XVERSE_COMMIT,
        "official_target_prompts": False,
        "official_dpg_questions": False,
        "reference_count": len(custom_files),
        "prompt_count": len(CUSTOM_PROMPTS),
        "record_count": len(custom_records),
        "selected_official_indices": custom_official_indices,
        "seed_policy": {
            "kind": "one deterministic seed per reference-prompt pair",
            "base_seed": CUSTOM_BASE_SEED,
            "formula": "base_seed + record_index",
            "same_pair_seed_must_be_reused_across_conditions": True,
        },
        "prompt_specs": [spec.to_dict() for spec in CUSTOM_PROMPTS],
        "records": custom_records,
    }
    integrity_manifest = {
        "schema_version": 1,
        "official_xverse_commit": OFFICIAL_XVERSE_COMMIT,
        "reconstructed_prepared_reference_root": str(official_reference_root),
        "records": integrity_records,
    }
    source_caption_template = {
        "schema_version": 1,
        "description": (
            "Fill behavior/background/full_caption after visual inspection. "
            "This file is for source-aware projection only, not DPG scoring."
        ),
        "records": [
            {
                "official_index": record["official_index"],
                "reference_filename": record["reference_filename"],
                "content_sha256": record["content_sha256"],
                "subject_caption": record["subject_caption"],
                "behavior": None,
                "background": None,
                "full_caption": None,
                "notes": None,
            }
            for record in subset_records
        ],
    }
    generation_schema = _generation_results_schema()

    payload = {
        "subset_manifest": subset_manifest,
        "subset_official_json": subset_official_json,
        "subset_dsg": subset_dsg,
        "custom_manifest": custom_manifest,
        "custom_dsg": custom_dsg,
        "integrity_manifest": integrity_manifest,
        "source_caption_template": source_caption_template,
        "generation_results_schema": generation_schema,
    }
    if write:
        _write_assets(dataset_root, payload)
    return payload


def build_custom_dpg_record(
    *,
    record_id: str,
    prompt: str,
    subject_label: str,
    spec: CustomPromptSpec,
) -> dict[str, Any]:
    """Construct an official-compatible DSG record plus analysis labels."""

    qid2tuple: dict[str, str] = {}
    qid2dependency: dict[str, list[int]] = {}
    qid2question: dict[str, str] = {}
    qid2component: dict[str, str] = {}
    qid2factor: dict[str, str] = {}

    def add(
        tuple_name: str,
        dependencies: Iterable[int],
        question: str,
        component: str,
        factor: str,
    ) -> int:
        qid = str(len(qid2question) + 1)
        qid2tuple[qid] = tuple_name
        qid2dependency[qid] = list(dependencies)
        qid2question[qid] = question
        qid2component[qid] = component
        qid2factor[qid] = factor
        return int(qid)

    subject_qid = add(
        "entity - whole",
        [0],
        f"Is there { _indefinite_phrase(subject_label) }?",
        "subject_presence",
        "object",
    )

    guitar_qid = None
    if spec.action == "playing the guitar":
        guitar_qid = add(
            "entity - whole",
            [0],
            "Is there a guitar?",
            "action_object_presence",
            "action",
        )

    background_qid = None
    if spec.background_entity is not None:
        article = "a " if spec.background_entity == "sofa" else ""
        background_qid = add(
            "entity - whole",
            [0],
            f"Is there {article}{spec.background_entity}?",
            "background_presence",
            "background",
        )

    if spec.action == "sleeping":
        add(
            "attribute - state",
            [subject_qid],
            f"Is the {subject_label} sleeping?",
            "action_state",
            "action",
        )
    elif spec.action == "playing the guitar":
        if guitar_qid is None:
            raise AssertionError("Guitar action requires a guitar entity")
        add(
            "action -",
            [subject_qid, guitar_qid],
            f"Is the {subject_label} playing the guitar?",
            "action_relation",
            "action",
        )

    if spec.background_relation is not None:
        if background_qid is None:
            raise AssertionError("Background relation requires an entity")
        add(
            "relation - spatial",
            [subject_qid, background_qid],
            f"Is the {subject_label} {spec.background_relation}?",
            "background_relation",
            "background",
        )

    return {
        "record_id": record_id,
        "prompt": prompt,
        "composition": spec.composition,
        "prompt_id": spec.prompt_id,
        "qid2tuple": qid2tuple,
        "qid2dependency": qid2dependency,
        "qid2question": qid2question,
        "qid2component": qid2component,
        "qid2factor": qid2factor,
    }


def _write_assets(dataset_root: Path, payload: dict[str, Any]) -> None:
    mappings = (
        (
            dataset_root / "subset_bench" / "benchmark_manifest.json",
            payload["subset_manifest"],
        ),
        (
            dataset_root / "subset_bench" / "XVerseBench_single_animal_29.json",
            payload["subset_official_json"],
        ),
        (
            dataset_root
            / "subset_bench"
            / "XVerseBench_single_animal_29_DSG.json",
            payload["subset_dsg"],
        ),
        (
            dataset_root / "custom_bench" / "benchmark_manifest.json",
            payload["custom_manifest"],
        ),
        (
            dataset_root / "custom_bench" / "XVerseBench_custom_animal_15x6_DSG.json",
            payload["custom_dsg"],
        ),
        (
            dataset_root / "custom_bench" / "custom_prompt_templates.json",
            [spec.to_dict() for spec in CUSTOM_PROMPTS],
        ),
        (dataset_root / "dataset_integrity.json", payload["integrity_manifest"]),
        (
            dataset_root / "generated_images_manifest.schema.json",
            payload["generation_results_schema"],
        ),
    )
    for path, value in mappings:
        _write_json(path, value)

    _write_reference_mapping_csv(
        dataset_root / "reference_index_mapping.csv",
        payload["subset_manifest"]["records"],
        set(payload["custom_manifest"]["selected_official_indices"]),
    )

    source_path = dataset_root / "source_captions.template.json"
    _write_json(source_path, payload["source_caption_template"])


def _generation_results_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "XVerseBench 0907 generated-image manifest",
        "type": "object",
        "required": ["schema_version", "benchmark_id", "images"],
        "properties": {
            "schema_version": {"const": 1},
            "benchmark_id": {
                "enum": [
                    "xverse_single_animal_29",
                    "xverse_ref_custom_animal_15x6",
                ]
            },
            "images": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": [
                        "record_id",
                        "condition",
                        "seed",
                        "image_path",
                    ],
                    "properties": {
                        "record_id": {"type": "string"},
                        "condition": {"type": "string"},
                        "seed": {"type": "integer"},
                        "image_path": {"type": "string"},
                    },
                    "additionalProperties": True,
                },
            },
        },
        "additionalProperties": True,
    }


def _write_reference_mapping_csv(
    path: Path,
    subset_records: list[dict[str, Any]],
    custom_indices: set[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "official_index",
                "reference_filename",
                "subject_caption",
                "official_target_prompt",
                "composition",
                "included_in_custom_bench",
            ),
        )
        writer.writeheader()
        for record in subset_records:
            writer.writerow(
                {
                    "official_index": record["official_index"],
                    "reference_filename": record["reference_filename"],
                    "subject_caption": record["subject_caption"],
                    "official_target_prompt": record["prompt"].strip(),
                    "composition": record["composition"],
                    "included_in_custom_bench": (
                        record["official_index"] in custom_indices
                    ),
                }
            )


def _validate_reference_selection(
    name: str,
    files: tuple[Path, ...],
    official_by_filename: dict[str, Any],
    expected_count: int,
) -> None:
    if len(files) != expected_count:
        raise ValueError(
            f"{name} must contain exactly {expected_count} images; got {len(files)}"
        )
    unknown = sorted(path.name for path in files if path.name not in official_by_filename)
    if unknown:
        raise ValueError(f"{name} contains non-official animal files: {unknown}")


def _discover_images(root: Path) -> tuple[Path, ...]:
    if not root.is_dir():
        raise FileNotFoundError(f"Reference directory does not exist: {root}")
    supported = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return tuple(
        sorted(
            (path for path in root.iterdir() if path.suffix.casefold() in supported),
            key=lambda path: (path.name.casefold(), path.name),
        )
    )


def _subject_label(caption: str) -> str:
    lowered = caption.casefold()
    for prefix in ("an ", "a "):
        if lowered.startswith(prefix):
            return caption[len(prefix):].strip()
    return caption.strip()


def _indefinite_phrase(label: str) -> str:
    article = "an" if label[:1].casefold() in {"a", "e", "i", "o", "u"} else "a"
    return f"{article} {label}"


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
