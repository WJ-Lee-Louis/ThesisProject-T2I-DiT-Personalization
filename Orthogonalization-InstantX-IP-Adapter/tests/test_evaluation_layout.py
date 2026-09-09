from pathlib import Path, PurePosixPath

from orthogonalization_instantx_ip_adapter.evaluation_layout import EvaluationLayout


def test_layout_is_research_oriented_and_profile_independent() -> None:
    layout = EvaluationLayout(Path("C:/bench"), "animal_eval")

    assert layout.generated_manifest("subset") == (
        Path("C:/bench").resolve()
        / "experiments"
        / "animal_eval"
        / "generation"
        / "subset"
        / "generated_images_manifest.json"
    )
    assert "modal" not in str(layout.root).casefold()


def test_metric_remote_paths_map_to_canonical_sections() -> None:
    layout = EvaluationLayout(Path("C:/bench"), "animal_eval")

    assert layout.metric_download_path(PurePosixPath("subset/dpg.json")) == (
        layout.evaluation_root / "scores" / "subset" / "dpg.json"
    )
    assert layout.metric_download_path(PurePosixPath("r/masks/cat.png")) == (
        layout.evaluation_root / "ip" / "references" / "masks" / "cat.png"
    )
    assert layout.metric_download_path(
        PurePosixPath(
            "a/subset/generated/st_matched_projection_ssg/"
            "official_animal_028/seed_45/candidate_00_mask.png"
        )
    ) == (
        layout.evaluation_root
        / "ip"
        / "gen"
        / "subset"
        / "st_matched_projection_ssg"
        / "official_animal_028"
        / "seed_45"
        / "candidate_00_mask.png"
    )
