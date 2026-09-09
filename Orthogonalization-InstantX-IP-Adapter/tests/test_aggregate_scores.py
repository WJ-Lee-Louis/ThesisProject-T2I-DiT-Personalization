# ruff: noqa: E402

import json
from pathlib import Path
import sys


METRIC_ROOT = (
    Path(__file__).resolve().parents[1] / "datasets" / "XVerseBench_0907" / "metrics"
)
if str(METRIC_ROOT) not in sys.path:
    sys.path.insert(0, str(METRIC_ROOT))

from aggregate_scores import _load_metric


def test_aggregate_excludes_ip_when_all_detections_failed(tmp_path: Path) -> None:
    path = tmp_path / "ip.json"
    path.write_text(
        json.dumps(
            {
                "metric": "IP-Sim",
                "benchmark_id": "benchmark",
                "scores": [
                    {
                        "record_id": "r",
                        "condition": "baseline",
                        "seed": 42,
                        "detected_candidate_count": 0,
                        "ip_similarity": 0.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert _load_metric(path, "benchmark") == []
