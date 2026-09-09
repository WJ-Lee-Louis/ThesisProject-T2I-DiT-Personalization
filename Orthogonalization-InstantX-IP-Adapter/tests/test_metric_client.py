from pathlib import Path

import metric_client


def test_metric_download_temporary_name_stays_short() -> None:
    destination = Path("C:/") / ("x" * 180) / ("y" * 100 + ".json")

    temporary = metric_client._download_temporary_path(destination)

    assert temporary.parent == destination.parent
    assert temporary.name.startswith(".d-")
    assert temporary.name.endswith(".tmp")
    assert len(temporary.name) == 19


def test_generation_batch_contract_has_six_conditions_per_benchmark() -> None:
    batches = metric_client._generation_batches("animal-6cond-v1")

    assert set(batches) == {"subset", "custom"}
    assert all(len(items) == 6 for items in batches.values())
    assert (
        batches["custom"]["st_all_projection_ssg"]
        == "xverse-animal-6cond-v1-custom-st_all_projection_ssg"
    )


def test_submit_parser_can_decouple_metric_and_generation_run_ids() -> None:
    args = metric_client.build_parser().parse_args(
        [
            "submit",
            "--run-id",
            "animal-ip-fix-v1",
            "--generation-run-id",
            "m2-0908",
            "--metric",
            "ip",
        ]
    )

    assert args.run_id == "animal-ip-fix-v1"
    assert args.generation_run_id == "m2-0908"


def test_metric_job_record_path_is_not_profile_partitioned() -> None:
    path = metric_client._record_path("modal-research-2", "animal-ip-fix-v1", "ip")

    assert path == metric_client.JOB_RECORD_ROOT / "animal-ip-fix-v1" / "ip.json"
    assert "modal-research-2" not in path.parts
