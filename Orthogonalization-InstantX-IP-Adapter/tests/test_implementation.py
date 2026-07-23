from pathlib import Path

from orthogonalization_instantx_ip_adapter.implementation import (
    compute_batch_implementation_digest,
)


def test_implementation_digest_is_stable_sha256():
    experiment_root = Path(__file__).resolve().parents[1]

    first = compute_batch_implementation_digest(experiment_root)
    second = compute_batch_implementation_digest(experiment_root)

    assert first == second
    assert len(first) == 64
    assert set(first) <= set("0123456789abcdef")
