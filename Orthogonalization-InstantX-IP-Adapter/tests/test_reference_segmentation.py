"""Tests for Florence2 reference-segmentation queries."""

# ruff: noqa: E402

from pathlib import Path
import sys


METRIC_ROOT = (
    Path(__file__).resolve().parents[1] / "datasets" / "XVerseBench_0907" / "metrics"
)
if str(METRIC_ROOT) not in sys.path:
    sys.path.insert(0, str(METRIC_ROOT))

from prepare_segmented_references import (
    detect_reference_subject,
    segmentation_queries,
)


def test_segmentation_queries_use_exact_then_head_noun_only() -> None:
    assert segmentation_queries("polar bear") == (
        "polar bear",
        "polar bear.",
        "bear",
        "bear.",
    )
    assert segmentation_queries("dog") == ("dog", "dog.")
    assert "animal" not in segmentation_queries("French bulldog")


def test_reference_detector_records_successful_fallback_query() -> None:
    class FakeDetector:
        def __init__(self):
            self.queries = []

        def get_multiple_instances(self, image, query, min_size):
            self.queries.append((query, min_size))
            boxes = [[1, 2, 3, 4]] if query == "bear" else []
            return {"instance_bboxes": boxes, "instance_images": [object()]}

    detector = FakeDetector()
    result, query, attempts = detect_reference_subject(
        detector, object(), "polar bear", 20
    )

    assert result is not None
    assert query == "bear"
    assert attempts == ["polar bear", "polar bear.", "bear"]
    assert detector.queries == [
        ("polar bear", 20),
        ("polar bear.", 20),
        ("bear", 20),
    ]
