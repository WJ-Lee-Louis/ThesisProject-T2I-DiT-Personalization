"""Validated per-reference captions for source-aware orthogonalization."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .batch_manifest import BatchManifest, Species


SOURCE_CAPTION_SCHEMA_VERSION = 1
DEFAULT_SOURCE_CAPTIONS_PATH = (
    Path(__file__).resolve().parents[1] / "source_captions.json"
)


@dataclass(frozen=True, slots=True)
class SourceCaptionRecord:
    """Factorized nuisance caption bound to one exact reference image."""

    reference_stem: str
    species: Species
    content_sha256: str
    source_behavior: str
    source_background: str
    notes: str = ""

    @property
    def object_only(self) -> str:
        return f"a {self.species}"

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

    def prompt_dict(self) -> dict[str, str]:
        return {
            "object_only": self.object_only,
            "object_behavior": self.object_behavior,
            "object_background": self.object_background,
            "object_behavior_background": (
                self.object_behavior_background
            ),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "reference_stem": self.reference_stem,
            "species": self.species,
            "content_sha256": self.content_sha256,
            "source_behavior": self.source_behavior,
            "source_background": self.source_background,
            "source_prompts": self.prompt_dict(),
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class SourceCaptionManifest:
    """Complete caption set for every reference in one image manifest."""

    records: tuple[SourceCaptionRecord, ...]

    @property
    def semantic_digest(self) -> str:
        canonical = {
            "schema_version": SOURCE_CAPTION_SCHEMA_VERSION,
            "references": [
                {
                    "reference_stem": record.reference_stem,
                    "species": record.species,
                    "content_sha256": record.content_sha256,
                    "source_behavior": record.source_behavior,
                    "source_background": record.source_background,
                }
                for record in self.records
            ],
        }
        encoded = json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def for_reference(self, reference_stem: str) -> SourceCaptionRecord:
        matches = [
            record
            for record in self.records
            if record.reference_stem == reference_stem
        ]
        if len(matches) != 1:
            raise KeyError(
                f"No unique source caption for reference {reference_stem!r}"
            )
        return matches[0]

    def to_upload_dict(self) -> dict[str, object]:
        """Return the minimal user-editable representation used remotely."""

        return {
            "schema_version": SOURCE_CAPTION_SCHEMA_VERSION,
            "references": {
                record.reference_stem: {
                    "source_behavior": record.source_behavior,
                    "source_background": record.source_background,
                    **({"notes": record.notes} if record.notes else {}),
                }
                for record in self.records
            },
        }

    def to_upload_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.to_upload_dict(),
            ensure_ascii=False,
            indent=indent,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SOURCE_CAPTION_SCHEMA_VERSION,
            "semantic_digest": self.semantic_digest,
            "reference_count": len(self.records),
            "records": [record.to_dict() for record in self.records],
        }


def load_source_caption_manifest(
    path: str | Path,
    image_manifest: BatchManifest,
) -> SourceCaptionManifest:
    source_path = Path(path).expanduser()
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Source caption file does not exist: {source_path}"
        )
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Source caption file is not valid JSON: {source_path}"
        ) from error
    return build_source_caption_manifest(payload, image_manifest)


def build_source_caption_manifest(
    payload: Mapping[str, Any],
    image_manifest: BatchManifest,
) -> SourceCaptionManifest:
    """Validate JSON data against the exact ten reference image records."""

    if not isinstance(payload, Mapping):
        raise TypeError("Source caption payload must be a mapping")
    unknown_top = set(payload) - {"schema_version", "references"}
    if unknown_top:
        raise ValueError(
            "Unknown source caption top-level field(s): "
            + ", ".join(sorted(str(value) for value in unknown_top))
        )
    if payload.get("schema_version") != SOURCE_CAPTION_SCHEMA_VERSION:
        raise ValueError(
            "source caption schema_version must be "
            f"{SOURCE_CAPTION_SCHEMA_VERSION}"
        )

    references_payload = payload.get("references")
    if not isinstance(references_payload, Mapping):
        raise TypeError("source caption references must be a mapping")

    expected_stems = {
        record.reference_stem for record in image_manifest.references
    }
    actual_stems = set(references_payload)
    missing = sorted(expected_stems - actual_stems)
    unknown = sorted(actual_stems - expected_stems)
    if missing or unknown:
        raise ValueError(
            "Source captions must match the image manifest exactly: "
            f"missing={missing}, unknown={unknown}"
        )

    records: list[SourceCaptionRecord] = []
    for reference in image_manifest.references:
        raw = references_payload[reference.reference_stem]
        if not isinstance(raw, Mapping):
            raise TypeError(
                "Source caption entry must be a mapping: "
                f"{reference.reference_stem}"
            )
        unknown_fields = set(raw) - {
            "source_behavior",
            "source_background",
            "notes",
        }
        if unknown_fields:
            raise ValueError(
                f"Unknown source caption field(s) for "
                f"{reference.reference_stem}: "
                + ", ".join(sorted(str(value) for value in unknown_fields))
            )

        behavior = _clean_required_phrase(
            raw.get("source_behavior"),
            field_name="source_behavior",
            reference_stem=reference.reference_stem,
        )
        background = _clean_required_phrase(
            raw.get("source_background"),
            field_name="source_background",
            reference_stem=reference.reference_stem,
        )
        notes_value = raw.get("notes", "")
        if not isinstance(notes_value, str):
            raise TypeError(
                f"notes for {reference.reference_stem} must be a string"
            )

        records.append(
            SourceCaptionRecord(
                reference_stem=reference.reference_stem,
                species=reference.species,
                content_sha256=reference.content_sha256,
                source_behavior=behavior,
                source_background=background,
                notes=notes_value.strip(),
            )
        )

    return SourceCaptionManifest(records=tuple(records))


def _clean_required_phrase(
    value: object,
    *,
    field_name: str,
    reference_stem: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{field_name} for {reference_stem} must be a non-empty string"
        )
    return " ".join(value.split())


__all__ = [
    "DEFAULT_SOURCE_CAPTIONS_PATH",
    "SOURCE_CAPTION_SCHEMA_VERSION",
    "SourceCaptionManifest",
    "SourceCaptionRecord",
    "build_source_caption_manifest",
    "load_source_caption_manifest",
]
