from pathlib import PurePosixPath
from types import SimpleNamespace

import pytest

import batch_client
from orthogonalization_instantx_ip_adapter.batch_config import (
    BatchGenerationConfig,
)
from orthogonalization_instantx_ip_adapter.source_captions import (
    DEFAULT_SOURCE_CAPTIONS_PATH,
)


def test_default_batch_id_is_a_valid_safe_component():
    value = batch_client.default_batch_id("projection")

    BatchGenerationConfig(batch_id=value, condition="projection")
    assert value.startswith("dreambench-projection-")


@pytest.mark.parametrize(
    ("remote_path", "expected"),
    [
        (
            "run-01/cat_03/cat_03_running.png",
            PurePosixPath("cat_03/cat_03_running.png"),
        ),
        (
            "/run-01/_metadata/status.json",
            PurePosixPath("_metadata/status.json"),
        ),
    ],
)
def test_relative_remote_path_accepts_only_the_selected_batch(
    remote_path,
    expected,
):
    assert (
        batch_client._relative_remote_path(remote_path, "run-01")
        == expected
    )


def test_relative_remote_path_rejects_another_batch():
    with pytest.raises(ValueError, match="outside batch"):
        batch_client._relative_remote_path(
            "run-02/cat_03/cat_03_running.png",
            "run-01",
        )


def test_safe_local_child_rejects_parent_traversal(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        batch_client._safe_local_child(tmp_path, "../outside.png")


def test_submit_requires_an_explicit_condition():
    parser = batch_client.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["submit"])


@pytest.mark.parametrize(
    "condition",
    [
        "baseline",
        "projection",
        "gate",
        "combined",
        "sa_projection",
        "sa_combined",
    ],
)
def test_submit_parser_exposes_all_six_non_conflicting_conditions(
    condition: str,
) -> None:
    args = batch_client.build_parser().parse_args(
        ["submit", "--condition", condition]
    )

    assert args.condition == condition
    assert args.source_captions == DEFAULT_SOURCE_CAPTIONS_PATH


@pytest.mark.parametrize("condition", ["sa_projection", "sa_combined"])
def test_default_source_aware_batch_id_is_safe(
    condition: str,
) -> None:
    value = batch_client.default_batch_id(condition)

    BatchGenerationConfig(
        batch_id=value,
        condition=condition,
        source_caption_digest="ab" * 32,
    )
    assert value.startswith(f"dreambench-{condition}-")


@pytest.fixture
def submit_harness(monkeypatch, tmp_path):
    reference_path = tmp_path / "dog_01.jpg"
    reference_path.write_bytes(b"reference")
    manifest = SimpleNamespace(
        semantic_digest="11" * 32,
        references=(
            SimpleNamespace(
                reference_path=str(reference_path),
                reference_stem="dog_01",
                species="dog",
            ),
        ),
        to_json=lambda: '{"test_manifest": true}',
    )
    captions = SimpleNamespace(
        semantic_digest="33" * 32,
        records=(),
        to_upload_dict=lambda: {
            "schema_version": 1,
            "references": {
                "dog_01": {
                    "source_behavior": "lying down",
                    "source_background": "on a beach",
                }
            },
        },
    )
    captured = {
        "caption_loads": [],
        "uploads": [],
        "spawn_calls": [],
        "job_records": [],
        "remote_lookup": [],
    }

    monkeypatch.setattr(
        batch_client,
        "build_batch_manifest",
        lambda _root: manifest,
    )
    monkeypatch.setattr(
        batch_client,
        "load_source_caption_manifest",
        lambda path, image_manifest: (
            captured["caption_loads"].append((path, image_manifest))
            or captions
        ),
    )
    monkeypatch.setattr(
        batch_client,
        "compute_batch_implementation_digest",
        lambda _root: "22" * 32,
    )
    monkeypatch.setattr(
        batch_client,
        "print_manifest_summary",
        lambda _manifest: None,
    )
    monkeypatch.setattr(
        batch_client,
        "print_source_caption_summary",
        lambda _captions, *, source_path: None,
    )
    monkeypatch.setattr(
        batch_client,
        "_write_job_record",
        lambda record: captured["job_records"].append(record),
    )

    class FakeUpload:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def put_file(self, source, remote_path):
            captured["uploads"].append((source, remote_path))

    class FakeVolume:
        def batch_upload(self, *, force):
            assert force is True
            return FakeUpload()

    class FakeCall:
        object_id = "fc-test-source-aware"

        @staticmethod
        def get_dashboard_url():
            return "https://modal.test/call"

    class FakeGenerateBatch:
        @staticmethod
        def spawn(*args):
            captured["spawn_calls"].append(args)
            return FakeCall()

    class FakeRemoteObject:
        generate_batch = FakeGenerateBatch()

    class FakeRemoteClass:
        def __call__(self):
            return FakeRemoteObject()

    fake_modal = SimpleNamespace(
        Volume=SimpleNamespace(
            from_name=lambda *args, **kwargs: FakeVolume(),
        ),
        Cls=SimpleNamespace(
            from_name=lambda app_name, class_name: (
                captured["remote_lookup"].append((app_name, class_name))
                or FakeRemoteClass()
            ),
        ),
    )
    monkeypatch.setattr(batch_client, "modal", fake_modal)
    return captured, manifest, captions


@pytest.mark.parametrize(
    "condition",
    ["baseline", "projection", "gate", "combined"],
)
def test_original_submit_conditions_do_not_load_or_upload_source_captions(
    condition: str,
    submit_harness,
) -> None:
    captured, _manifest, _captions = submit_harness
    args = batch_client.build_parser().parse_args(
        [
            "submit",
            "--condition",
            condition,
            "--batch-id",
            f"regression-{condition}",
            "--yes",
        ]
    )

    assert batch_client.command_submit(args) == 0

    assert captured["caption_loads"] == []
    config_payload, caption_payload = captured["spawn_calls"][0]
    assert config_payload["condition"] == condition
    assert config_payload["source_caption_digest"] is None
    assert caption_payload is None


@pytest.mark.parametrize("condition", ["sa_projection", "sa_combined"])
def test_source_aware_submit_validates_and_sends_caption_payload(
    condition: str,
    submit_harness,
) -> None:
    captured, manifest, captions = submit_harness
    caption_path = "captions-for-test.json"
    args = batch_client.build_parser().parse_args(
        [
            "submit",
            "--condition",
            condition,
            "--batch-id",
            f"source-{condition}",
            "--source-captions",
            caption_path,
            "--yes",
        ]
    )

    assert batch_client.command_submit(args) == 0

    assert captured["caption_loads"] == [
        (args.source_captions, manifest),
    ]
    config_payload, caption_payload = captured["spawn_calls"][0]
    assert config_payload["condition"] == condition
    assert (
        config_payload["source_caption_digest"]
        == captions.semantic_digest
    )
    assert caption_payload == captions.to_upload_dict()
    assert (
        captured["job_records"][0]["source_caption_digest"]
        == captions.semantic_digest
    )
