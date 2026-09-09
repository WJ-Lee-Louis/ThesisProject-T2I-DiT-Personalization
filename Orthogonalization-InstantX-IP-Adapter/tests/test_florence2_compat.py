"""Tests for the pinned XVerse Florence-2 compatibility repair."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import sys


METRIC_ROOT = (
    Path(__file__).resolve().parents[1] / "datasets" / "XVerseBench_0907" / "metrics"
)
if str(METRIC_ROOT) not in sys.path:
    sys.path.insert(0, str(METRIC_ROOT))

from florence2_compat import (  # noqa: E402
    florence2_weight_report,
    repair_xverse_florence2_weight_tying,
)


class FakeWeight:
    def __init__(self, pointer: int):
        self.pointer = pointer
        self.shape = (8, 4)

    def data_ptr(self) -> int:
        return self.pointer


def _fake_detector():
    shared = SimpleNamespace(weight=FakeWeight(1))
    language_model = SimpleNamespace(
        config=SimpleNamespace(tie_word_embeddings=True),
        model=SimpleNamespace(
            shared=shared,
            encoder=SimpleNamespace(
                embed_tokens=SimpleNamespace(weight=FakeWeight(2))
            ),
            decoder=SimpleNamespace(
                embed_tokens=SimpleNamespace(weight=FakeWeight(3))
            ),
        ),
        lm_head=SimpleNamespace(weight=FakeWeight(4)),
    )
    model = SimpleNamespace(language_model=language_model)
    return SimpleNamespace(
        detector=SimpleNamespace(florence2_model=model)
    )


def test_repair_ties_lm_head_and_token_embeddings() -> None:
    detector = _fake_detector()

    before = florence2_weight_report(detector)
    report = repair_xverse_florence2_weight_tying(detector)

    assert not before["lm_head_tied_to_shared"]
    assert report["after"]["encoder_tied_to_shared"]
    assert report["after"]["decoder_tied_to_shared"]
    assert report["after"]["lm_head_tied_to_shared"]
    assert detector._florence2_compat_report == report
