from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from orthogonalization_instantx_ip_adapter.batch_manifest import (
    BatchManifest,
    ReferenceRecord,
)
from orthogonalization_instantx_ip_adapter.source_captions import (
    SOURCE_CAPTION_SCHEMA_VERSION,
    build_source_caption_manifest,
    load_source_caption_manifest,
)


REFERENCE_SPECIES = (
    ("cat2_00", "cat"),
    ("cat_03", "cat"),
    ("dog2_04", "dog"),
    ("dog3_04", "dog"),
    ("dog5_02", "dog"),
    ("dog6_02", "dog"),
    ("dog7_03", "dog"),
    ("dog8_04", "dog"),
    ("dog_02", "dog"),
    ("PA_white_cat_background", "cat"),
)


def _image_manifest() -> BatchManifest:
    references = tuple(
        ReferenceRecord(
            reference_path=f"C:/references/{stem}.jpg",
            reference_stem=stem,
            species=species,
            content_sha256=f"{index:064x}",
        )
        for index, (stem, species) in enumerate(
            REFERENCE_SPECIES,
            start=1,
        )
    )
    return BatchManifest(
        reference_root="C:/references",
        references=references,
        jobs=(),
    )


def _caption_payload(*, reverse_order: bool = False) -> dict[str, object]:
    items = list(enumerate(REFERENCE_SPECIES, start=1))
    if reverse_order:
        items.reverse()
    references = {
        stem: {
            "source_behavior": f"source behavior {index}",
            "source_background": f"in source background {index}",
            "notes": f"annotation {index}",
        }
        for index, (stem, _species) in items
    }
    return {
        "schema_version": SOURCE_CAPTION_SCHEMA_VERSION,
        "references": references,
    }


def test_manifest_requires_and_binds_exactly_all_ten_references() -> None:
    image_manifest = _image_manifest()

    captions = build_source_caption_manifest(
        _caption_payload(reverse_order=True),
        image_manifest,
    )

    assert len(captions.records) == 10
    assert tuple(record.reference_stem for record in captions.records) == tuple(
        record.reference_stem for record in image_manifest.references
    )
    assert tuple(record.species for record in captions.records) == tuple(
        record.species for record in image_manifest.references
    )
    assert tuple(record.content_sha256 for record in captions.records) == tuple(
        record.content_sha256 for record in image_manifest.references
    )
    assert len(captions.semantic_digest) == 64
    assert set(captions.to_upload_dict()["references"]) == {
        stem for stem, _species in REFERENCE_SPECIES
    }


def test_record_builds_the_four_factorial_prompts_from_reference_caption() -> None:
    captions = build_source_caption_manifest(
        _caption_payload(),
        _image_manifest(),
    )

    record = captions.for_reference("dog_02")

    assert record.object_only == "a dog"
    assert record.object_behavior == "a dog source behavior 9"
    assert record.object_background == "a dog in source background 9"
    assert (
        record.object_behavior_background
        == "a dog source behavior 9 in source background 9"
    )
    assert record.prompt_dict() == {
        "object_only": record.object_only,
        "object_behavior": record.object_behavior,
        "object_background": record.object_background,
        "object_behavior_background": record.object_behavior_background,
    }


def test_caption_json_key_order_does_not_change_record_order_or_digest() -> None:
    image_manifest = _image_manifest()

    forward = build_source_caption_manifest(
        _caption_payload(),
        image_manifest,
    )
    reversed_json_order = build_source_caption_manifest(
        _caption_payload(reverse_order=True),
        image_manifest,
    )

    assert reversed_json_order.records == forward.records
    assert reversed_json_order.semantic_digest == forward.semantic_digest


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("source_behavior", "a materially different behavior"),
        ("source_background", "in a materially different background"),
    ),
)
def test_semantic_caption_text_change_changes_digest(
    field: str,
    replacement: str,
) -> None:
    image_manifest = _image_manifest()
    original_payload = _caption_payload()
    changed_payload = deepcopy(original_payload)
    changed_payload["references"]["dog_02"][field] = replacement

    original = build_source_caption_manifest(
        original_payload,
        image_manifest,
    )
    changed = build_source_caption_manifest(
        changed_payload,
        image_manifest,
    )

    assert changed.semantic_digest != original.semantic_digest


def test_whitespace_is_normalized_and_notes_do_not_change_semantic_digest() -> None:
    image_manifest = _image_manifest()
    original_payload = _caption_payload()
    presentation_only_change = deepcopy(original_payload)
    presentation_only_change["references"]["dog_02"][
        "source_behavior"
    ] = "  source   behavior  9 "
    presentation_only_change["references"]["dog_02"][
        "source_background"
    ] = "\tin source\nbackground 9  "
    presentation_only_change["references"]["dog_02"][
        "notes"
    ] = "a different non-semantic note"

    original = build_source_caption_manifest(
        original_payload,
        image_manifest,
    )
    changed = build_source_caption_manifest(
        presentation_only_change,
        image_manifest,
    )

    assert changed.semantic_digest == original.semantic_digest
    assert changed.for_reference("dog_02").source_behavior == "source behavior 9"
    assert (
        changed.for_reference("dog_02").source_background
        == "in source background 9"
    )


def test_missing_or_unknown_reference_breaks_exact_coverage() -> None:
    image_manifest = _image_manifest()
    missing = _caption_payload()
    del missing["references"]["dog_02"]

    with pytest.raises(ValueError, match=r"missing=.*dog_02"):
        build_source_caption_manifest(missing, image_manifest)

    unknown = _caption_payload()
    unknown["references"]["unknown_dog"] = unknown["references"].pop("dog_02")

    with pytest.raises(ValueError, match=r"unknown=.*unknown_dog"):
        build_source_caption_manifest(unknown, image_manifest)


@pytest.mark.parametrize(
    ("mutate", "error_type", "message"),
    (
        (
            lambda payload: payload.update({"schema_version": 999}),
            ValueError,
            "schema_version",
        ),
        (
            lambda payload: payload.update({"unexpected": True}),
            ValueError,
            "top-level",
        ),
        (
            lambda payload: payload.update({"references": []}),
            TypeError,
            "references must be a mapping",
        ),
        (
            lambda payload: payload["references"].update({"dog_02": "caption"}),
            TypeError,
            "entry must be a mapping",
        ),
        (
            lambda payload: payload["references"]["dog_02"].pop(
                "source_behavior"
            ),
            ValueError,
            "source_behavior",
        ),
        (
            lambda payload: payload["references"]["dog_02"].update(
                {"source_behavior": " "}
            ),
            ValueError,
            "source_behavior",
        ),
        (
            lambda payload: payload["references"]["dog_02"].update(
                {"source_background": 17}
            ),
            ValueError,
            "source_background",
        ),
        (
            lambda payload: payload["references"]["dog_02"].update(
                {"extra_caption": "not allowed"}
            ),
            ValueError,
            "Unknown source caption field",
        ),
        (
            lambda payload: payload["references"]["dog_02"].update(
                {"notes": ["not", "text"]}
            ),
            TypeError,
            "notes",
        ),
    ),
)
def test_invalid_caption_fields_are_rejected(
    mutate,
    error_type: type[Exception],
    message: str,
) -> None:
    payload = _caption_payload()
    mutate(payload)

    with pytest.raises(error_type, match=message):
        build_source_caption_manifest(payload, _image_manifest())


def test_non_mapping_payload_is_rejected() -> None:
    with pytest.raises(TypeError, match="payload must be a mapping"):
        build_source_caption_manifest([], _image_manifest())


def test_load_rejects_missing_and_invalid_json_files(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_source_caption_manifest(missing_path, _image_manifest())

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_source_caption_manifest(invalid_path, _image_manifest())


def test_upload_json_round_trips_to_the_same_semantics() -> None:
    image_manifest = _image_manifest()
    captions = build_source_caption_manifest(
        _caption_payload(),
        image_manifest,
    )

    reloaded = build_source_caption_manifest(
        json.loads(captions.to_upload_json()),
        image_manifest,
    )

    assert reloaded.records == captions.records
    assert reloaded.semantic_digest == captions.semantic_digest
