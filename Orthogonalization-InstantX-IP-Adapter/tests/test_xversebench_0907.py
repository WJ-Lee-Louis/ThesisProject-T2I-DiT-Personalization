from collections import Counter

from orthogonalization_instantx_ip_adapter.xversebench_0907 import (
    CUSTOM_PROMPTS,
    OFFICIAL_COMPOSITION_BY_INDEX,
    build_custom_dpg_record,
)


def test_official_animal_composition_counts_are_frozen() -> None:
    assert Counter(OFFICIAL_COMPOSITION_BY_INDEX.values()) == {
        "O+A+B": 21,
        "O+A": 1,
        "O+B": 2,
        "O+attribute": 5,
    }


def test_custom_prompt_design_is_balanced_by_composition() -> None:
    assert len(CUSTOM_PROMPTS) == 6
    assert Counter(spec.composition for spec in CUSTOM_PROMPTS) == {
        "O+A": 2,
        "O+B": 2,
        "O+A+B": 2,
    }
    assert len({spec.prompt_id for spec in CUSTOM_PROMPTS}) == 6


def test_custom_dpg_question_counts_and_dependencies() -> None:
    expected_counts = {
        "oa_sleeping": 2,
        "oa_playing_guitar": 3,
        "ob_snow": 3,
        "ob_sofa": 3,
        "oab_sleeping_snow": 4,
        "oab_playing_guitar_sofa": 5,
    }
    for spec in CUSTOM_PROMPTS:
        record = build_custom_dpg_record(
            record_id="custom_000",
            prompt=f"a polar bear {spec.suffix}",
            subject_label="polar bear",
            spec=spec,
        )
        assert len(record["qid2question"]) == expected_counts[spec.prompt_id]
        assert set(record["qid2question"]) == set(record["qid2tuple"])
        assert set(record["qid2question"]) == set(record["qid2dependency"])
        assert set(record["qid2question"]) == set(record["qid2component"])
        assert set(record["qid2question"]) == set(record["qid2factor"])
        for qid, dependencies in record["qid2dependency"].items():
            assert all(parent == 0 or parent < int(qid) for parent in dependencies)


def test_oab_guitar_questions_separate_action_and_background() -> None:
    spec = next(
        item for item in CUSTOM_PROMPTS
        if item.prompt_id == "oab_playing_guitar_sofa"
    )
    record = build_custom_dpg_record(
        record_id="custom_005",
        prompt="a dog playing the guitar on a sofa",
        subject_label="dog",
        spec=spec,
    )

    assert record["qid2question"] == {
        "1": "Is there a dog?",
        "2": "Is there a guitar?",
        "3": "Is there a sofa?",
        "4": "Is the dog playing the guitar?",
        "5": "Is the dog on the sofa?",
    }
    assert record["qid2dependency"]["4"] == [1, 2]
    assert record["qid2dependency"]["5"] == [1, 3]
    assert record["qid2factor"] == {
        "1": "object",
        "2": "action",
        "3": "background",
        "4": "action",
        "5": "background",
    }
