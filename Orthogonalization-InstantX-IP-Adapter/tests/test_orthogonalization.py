import pytest
import torch

from orthogonalization_instantx_ip_adapter.orthogonalization import (
    PromptSet,
    build_factorial_directions,
    build_local_directions,
    estimate_orthonormal_basis,
    normalize_fp32,
    orthogonalize_pooled_embedding,
    project_pooled_embedding,
)


def test_prompt_set_accepts_partial_p2_and_rejects_blank_prompts() -> None:
    prompts = PromptSet(
        object_only="a dog",
        object_behavior="a dog running",
    )
    assert prompts.object_background is None

    with pytest.raises(ValueError, match="object_background"):
        PromptSet(object_only="a dog", object_background="  ")


def test_normalize_fp32_uses_fp32_and_preserves_zero_vectors() -> None:
    values = torch.tensor([[3.0, 4.0], [0.0, 0.0]], dtype=torch.bfloat16)

    normalized = normalize_fp32(values)

    assert normalized.dtype == torch.float32
    torch.testing.assert_close(normalized[0], torch.tensor([0.6, 0.8]))
    torch.testing.assert_close(normalized[1], torch.zeros(2))


def test_local_directions_hold_the_other_factor_fixed() -> None:
    e00 = torch.tensor([1.0, 0.0, 0.0])
    e10 = torch.tensor([1.0, 1.0, 0.0])
    e01 = torch.tensor([1.0, 0.0, 1.0])
    e11 = torch.tensor([1.0, 2.0, 3.0])

    result = build_local_directions(
        e00,
        object_behavior=e10,
        object_background=e01,
        object_behavior_background=e11,
        normalize_directions=False,
    )

    normalized = [normalize_fp32(value) for value in (e00, e10, e01, e11)]
    expected = torch.stack(
        [
            normalized[3] - normalized[2],
            normalized[3] - normalized[1],
        ]
    )
    assert result.names == ("behavior_local", "background_local")
    torch.testing.assert_close(result.values, expected)


def test_local_direction_falls_back_to_object_only_contrast() -> None:
    e00 = torch.tensor([1.0, 0.0])
    e10 = torch.tensor([1.0, 1.0])

    result = build_local_directions(
        e00,
        object_behavior=e10,
        include_background=False,
    )

    expected = normalize_fp32(normalize_fp32(e10) - normalize_fp32(e00))
    torch.testing.assert_close(result.values[0], expected)


def test_factorial_main_effects_and_interaction_follow_two_by_two_contrasts() -> None:
    e00 = torch.tensor([1.0, 0.0, 0.0])
    e10 = torch.tensor([1.0, 1.0, 0.0])
    e01 = torch.tensor([1.0, 0.0, 1.0])
    e11 = torch.tensor([1.0, 2.0, 3.0])

    result = build_factorial_directions(
        e00,
        e10,
        e01,
        e11,
        include_interaction=True,
        normalize_directions=False,
    )

    n00, n10, n01, n11 = [normalize_fp32(value) for value in (e00, e10, e01, e11)]
    expected = torch.stack(
        [
            0.5 * ((n10 - n00) + (n11 - n01)),
            0.5 * ((n01 - n00) + (n11 - n10)),
            n11 - n10 - n01 + n00,
        ]
    )
    assert result.names == (
        "behavior_main_effect",
        "background_main_effect",
        "behavior_background_interaction",
    )
    torch.testing.assert_close(result.values, expected)


def test_svd_basis_drops_collinear_and_zero_directions() -> None:
    directions = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    result = estimate_orthonormal_basis(directions, relative_tolerance=1e-3)

    assert result.basis.dtype == torch.float32
    assert result.basis.shape == (3, 1)
    assert result.diagnostics.retained_rank == 1
    torch.testing.assert_close(
        result.basis.transpose(0, 1) @ result.basis,
        torch.eye(1),
    )


def test_full_projection_is_orthogonal_and_restores_input_norm() -> None:
    embedding = torch.tensor([3.0, 4.0, 12.0])
    basis = torch.tensor([[1.0], [0.0], [0.0]])

    result = project_pooled_embedding(embedding, basis, strength=1.0)

    assert torch.dot(normalize_fp32(result.embedding), basis[:, 0]).abs() < 1e-6
    torch.testing.assert_close(
        torch.linalg.vector_norm(result.embedding),
        torch.linalg.vector_norm(embedding),
    )
    assert result.diagnostics.max_abs_basis_dot < 1e-6
    assert not result.diagnostics.degenerate


def test_partial_projection_attenuates_but_is_not_fully_orthogonal() -> None:
    embedding = torch.tensor([1.0, 1.0])
    basis = torch.tensor([[1.0], [0.0]])

    result = project_pooled_embedding(embedding, basis, strength=0.5)

    output_unit = normalize_fp32(result.embedding)
    assert 0.0 < output_unit[0] < normalize_fp32(embedding)[0]
    torch.testing.assert_close(
        torch.linalg.vector_norm(result.embedding),
        torch.linalg.vector_norm(embedding),
    )
    assert result.diagnostics.removed_component_norm > 0


def test_zero_rank_basis_is_a_no_op_and_preserves_bfloat16() -> None:
    embedding = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    basis = torch.empty((2, 0))

    result = project_pooled_embedding(embedding, basis, strength=1.0)

    assert result.embedding.dtype == torch.bfloat16
    torch.testing.assert_close(result.embedding, embedding)
    torch.testing.assert_close(
        result.diagnostics.max_abs_basis_dot,
        torch.zeros(1),
    )


def test_convenience_pipeline_reports_rank_and_rejects_bad_strength() -> None:
    embedding = torch.tensor([1.0, 2.0, 3.0])
    directions = torch.tensor([[1.0, 0.0, 0.0]])

    result = orthogonalize_pooled_embedding(
        embedding,
        directions,
        strength=1.0,
    )

    assert result.basis_diagnostics.retained_rank == 1
    assert result.projection_diagnostics.max_abs_basis_dot < 1e-6

    with pytest.raises(ValueError, match="strength"):
        project_pooled_embedding(embedding, result.basis, strength=1.01)
