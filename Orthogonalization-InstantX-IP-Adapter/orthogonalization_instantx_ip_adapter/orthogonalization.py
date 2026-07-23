"""Pure PyTorch utilities for pooled-space SigLIP orthogonalization.

The functions in this module operate on already-computed pooled embeddings.
They intentionally do not import Transformers or any InstantX implementation,
which keeps the geometry independently testable.

Tensor conventions:

* pooled embeddings have shape ``[..., embedding_dim]``;
* contrast directions are stacked by row as ``[direction_count, embedding_dim]``;
* an orthonormal basis is stored by column as ``[embedding_dim, rank]``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor


@dataclass(frozen=True)
class PromptSet:
    """Four prompts used by local and factorial contrast construction.

    ``object_only`` is always required. The other prompts may be omitted when
    an experiment only needs a single-factor P2 contrast.
    """

    object_only: str
    object_behavior: Optional[str] = None
    object_background: Optional[str] = None
    object_behavior_background: Optional[str] = None

    def __post_init__(self) -> None:
        for field_name in (
            "object_only",
            "object_behavior",
            "object_background",
            "object_behavior_background",
        ):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{field_name} must be a non-empty string when provided")


@dataclass(frozen=True)
class ContrastDirections:
    """Named, row-stacked contrast directions in FP32."""

    values: Tensor
    names: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("values must have shape [direction_count, embedding_dim]")
        if self.values.shape[0] != len(self.names):
            raise ValueError("the number of names must match the number of directions")


@dataclass(frozen=True)
class BasisDiagnostics:
    """Numerical diagnostics produced while estimating a semantic basis."""

    direction_count: int
    embedding_dim: int
    singular_values: Tensor
    relative_singular_values: Tensor
    threshold: float
    retained_rank: int
    dropped_rank: int


@dataclass(frozen=True)
class BasisResult:
    """An orthonormal column basis and its SVD diagnostics."""

    basis: Tensor
    diagnostics: BasisDiagnostics


@dataclass(frozen=True)
class ProjectionDiagnostics:
    """Per-example measurements for a pooled-space projection."""

    strength: float
    original_norm: Tensor
    projection_component_norm: Tensor
    removed_component_norm: Tensor
    adjusted_unit_norm: Tensor
    output_norm: Tensor
    edit_norm: Tensor
    cosine_to_input: Tensor
    max_abs_basis_dot: Tensor
    degenerate: Tensor


@dataclass(frozen=True)
class ProjectionResult:
    """Projected embedding and measurements of the applied edit."""

    embedding: Tensor
    diagnostics: ProjectionDiagnostics


@dataclass(frozen=True)
class OrthogonalizationResult:
    """Combined result from basis estimation and embedding projection."""

    embedding: Tensor
    basis: Tensor
    basis_diagnostics: BasisDiagnostics
    projection_diagnostics: ProjectionDiagnostics


def normalize_fp32(value: Tensor, *, dim: int = -1, eps: float = 1e-12) -> Tensor:
    """L2-normalize ``value`` in FP32 while preserving zero vectors as zero."""

    _validate_float_tensor("value", value)
    _validate_eps(eps)
    value_fp32 = value.float()
    norm = torch.linalg.vector_norm(value_fp32, dim=dim, keepdim=True)
    return torch.where(norm > eps, value_fp32 / norm.clamp_min(eps), torch.zeros_like(value_fp32))


def build_local_directions(
    object_only: Tensor,
    *,
    object_behavior: Optional[Tensor] = None,
    object_background: Optional[Tensor] = None,
    object_behavior_background: Optional[Tensor] = None,
    include_behavior: bool = True,
    include_background: bool = True,
    include_joint: bool = False,
    normalize_directions: bool = True,
    eps: float = 1e-12,
) -> ContrastDirections:
    """Build target-local P2 directions from normalized pooled embeddings.

    When the joint prompt is available, the behavior contrast holds background
    fixed and the background contrast holds behavior fixed:

    * behavior: ``e(object+behavior+background) - e(object+background)``;
    * background: ``e(object+behavior+background) - e(object+behavior)``.

    Without the joint prompt, each available factor is contrasted against
    ``e(object)``. The optional joint direction is always
    ``e(object+behavior+background) - e(object)``.
    """

    embeddings = _prepare_four_embeddings(
        object_only=object_only,
        object_behavior=object_behavior,
        object_background=object_background,
        object_behavior_background=object_behavior_background,
        eps=eps,
    )
    e00, e10, e01, e11 = embeddings

    directions = []
    names = []

    if include_behavior:
        if e11 is not None and e01 is not None:
            directions.append(e11 - e01)
        elif e10 is not None:
            directions.append(e10 - e00)
        else:
            raise ValueError(
                "behavior direction requires object_behavior, or both "
                "object_background and object_behavior_background"
            )
        names.append("behavior_local")

    if include_background:
        if e11 is not None and e10 is not None:
            directions.append(e11 - e10)
        elif e01 is not None:
            directions.append(e01 - e00)
        else:
            raise ValueError(
                "background direction requires object_background, or both "
                "object_behavior and object_behavior_background"
            )
        names.append("background_local")

    if include_joint:
        if e11 is None:
            raise ValueError("joint direction requires object_behavior_background")
        directions.append(e11 - e00)
        names.append("behavior_background_joint")

    return _finalize_directions(
        directions,
        names,
        embedding_dim=e00.shape[-1],
        device=e00.device,
        normalize_directions=normalize_directions,
        eps=eps,
    )


def build_factorial_directions(
    object_only: Tensor,
    object_behavior: Tensor,
    object_background: Tensor,
    object_behavior_background: Tensor,
    *,
    include_interaction: bool = False,
    normalize_directions: bool = True,
    eps: float = 1e-12,
) -> ContrastDirections:
    """Build P3 2-by-2 factorial main-effect directions.

    The behavior and background vectors average their counterfactual
    differences across the other factor's two levels. ``include_interaction``
    appends the non-additive residual as a diagnostic/ablation direction.
    """

    e00, e10, e01, e11 = _prepare_four_embeddings(
        object_only=object_only,
        object_behavior=object_behavior,
        object_background=object_background,
        object_behavior_background=object_behavior_background,
        eps=eps,
        require_all=True,
    )
    assert e10 is not None and e01 is not None and e11 is not None

    behavior = 0.5 * ((e10 - e00) + (e11 - e01))
    background = 0.5 * ((e01 - e00) + (e11 - e10))

    directions = [behavior, background]
    names = ["behavior_main_effect", "background_main_effect"]
    if include_interaction:
        directions.append(e11 - e10 - e01 + e00)
        names.append("behavior_background_interaction")

    return _finalize_directions(
        directions,
        names,
        embedding_dim=e00.shape[-1],
        device=e00.device,
        normalize_directions=normalize_directions,
        eps=eps,
    )


def estimate_orthonormal_basis(
    directions: Tensor,
    *,
    relative_tolerance: float = 1e-3,
    absolute_tolerance: float = 0.0,
    max_rank: Optional[int] = None,
) -> BasisResult:
    """Estimate a rank-filtered orthonormal basis with an FP32 SVD.

    ``directions`` must be row-stacked. Singular values are retained when they
    are strictly greater than
    ``max(absolute_tolerance, relative_tolerance * largest_singular_value)``.
    """

    _validate_float_tensor("directions", directions)
    if directions.ndim != 2:
        raise ValueError("directions must have shape [direction_count, embedding_dim]")
    if not torch.isfinite(directions).all():
        raise ValueError("directions must contain only finite values")
    if not 0.0 <= relative_tolerance < 1.0:
        raise ValueError("relative_tolerance must be in [0, 1)")
    if absolute_tolerance < 0.0 or not math.isfinite(absolute_tolerance):
        raise ValueError("absolute_tolerance must be finite and non-negative")
    if max_rank is not None and max_rank <= 0:
        raise ValueError("max_rank must be positive when provided")

    direction_count, embedding_dim = directions.shape
    matrix = directions.float().transpose(0, 1)

    if direction_count == 0 or embedding_dim == 0:
        singular_values = torch.empty(0, dtype=torch.float32, device=directions.device)
        relative = singular_values.clone()
        basis = torch.empty(
            (embedding_dim, 0), dtype=torch.float32, device=directions.device
        )
        diagnostics = BasisDiagnostics(
            direction_count=direction_count,
            embedding_dim=embedding_dim,
            singular_values=singular_values,
            relative_singular_values=relative,
            threshold=float(absolute_tolerance),
            retained_rank=0,
            dropped_rank=0,
        )
        return BasisResult(basis=basis, diagnostics=diagnostics)

    left, singular_values, _ = torch.linalg.svd(matrix, full_matrices=False)
    largest = singular_values[0]
    threshold_tensor = torch.maximum(
        largest * relative_tolerance,
        singular_values.new_tensor(absolute_tolerance),
    )
    retained = singular_values > threshold_tensor
    retained_rank = int(retained.sum().item())
    if max_rank is not None:
        retained_rank = min(retained_rank, max_rank)

    basis = left[:, :retained_rank].contiguous()
    relative = torch.where(
        largest > 0,
        singular_values / largest,
        torch.zeros_like(singular_values),
    )
    diagnostics = BasisDiagnostics(
        direction_count=direction_count,
        embedding_dim=embedding_dim,
        singular_values=singular_values,
        relative_singular_values=relative,
        threshold=float(threshold_tensor.item()),
        retained_rank=retained_rank,
        dropped_rank=int(singular_values.numel()) - retained_rank,
    )
    return BasisResult(basis=basis, diagnostics=diagnostics)


def project_pooled_embedding(
    embedding: Tensor,
    basis: Tensor,
    *,
    strength: float = 1.0,
    restore_norm: bool = True,
    preserve_dtype: bool = True,
    eps: float = 1e-12,
) -> ProjectionResult:
    """Remove some or all of an orthonormal basis component.

    A strength of 1 is a full orthogonal projection. Values strictly between
    0 and 1 are partial attenuation and should not be described as producing a
    vector fully orthogonal to the basis.
    """

    _validate_float_tensor("embedding", embedding)
    _validate_float_tensor("basis", basis)
    _validate_eps(eps)
    if embedding.ndim < 1:
        raise ValueError("embedding must have at least one dimension")
    if basis.ndim != 2:
        raise ValueError("basis must have shape [embedding_dim, rank]")
    if basis.shape[0] != embedding.shape[-1]:
        raise ValueError("basis and embedding dimensions do not match")
    if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be finite and in [0, 1]")
    if not torch.isfinite(embedding).all() or not torch.isfinite(basis).all():
        raise ValueError("embedding and basis must contain only finite values")

    embedding_fp32 = embedding.float()
    basis_fp32 = basis.to(device=embedding.device, dtype=torch.float32)
    _validate_orthonormal_columns(basis_fp32)

    original_norm_keep = torch.linalg.vector_norm(
        embedding_fp32, dim=-1, keepdim=True
    )
    unit = torch.where(
        original_norm_keep > eps,
        embedding_fp32 / original_norm_keep.clamp_min(eps),
        torch.zeros_like(embedding_fp32),
    )

    if basis_fp32.shape[1] == 0:
        component = torch.zeros_like(unit)
    else:
        component = torch.matmul(torch.matmul(unit, basis_fp32), basis_fp32.transpose(0, 1))

    removed = strength * component
    adjusted = unit - removed
    adjusted_norm_keep = torch.linalg.vector_norm(adjusted, dim=-1, keepdim=True)
    degenerate_keep = adjusted_norm_keep <= eps

    if restore_norm:
        output_fp32 = torch.where(
            ~degenerate_keep,
            adjusted / adjusted_norm_keep.clamp_min(eps) * original_norm_keep,
            torch.zeros_like(adjusted),
        )
    else:
        output_fp32 = adjusted

    output_unit = normalize_fp32(output_fp32, eps=eps)
    if basis_fp32.shape[1] == 0:
        max_abs_basis_dot = torch.zeros(
            embedding.shape[:-1], dtype=torch.float32, device=embedding.device
        )
    else:
        basis_dot = torch.matmul(output_unit, basis_fp32)
        max_abs_basis_dot = basis_dot.abs().amax(dim=-1)

    input_unit = normalize_fp32(embedding_fp32, eps=eps)
    cosine = (input_unit * output_unit).sum(dim=-1)
    original_norm = original_norm_keep.squeeze(-1)
    adjusted_norm = adjusted_norm_keep.squeeze(-1)
    output_norm = torch.linalg.vector_norm(output_fp32, dim=-1)
    component_norm = torch.linalg.vector_norm(component, dim=-1)
    removed_norm = torch.linalg.vector_norm(removed, dim=-1)
    edit_norm = torch.linalg.vector_norm(output_fp32 - embedding_fp32, dim=-1)

    output = output_fp32.to(dtype=embedding.dtype) if preserve_dtype else output_fp32
    diagnostics = ProjectionDiagnostics(
        strength=float(strength),
        original_norm=original_norm,
        projection_component_norm=component_norm,
        removed_component_norm=removed_norm,
        adjusted_unit_norm=adjusted_norm,
        output_norm=output_norm,
        edit_norm=edit_norm,
        cosine_to_input=cosine,
        max_abs_basis_dot=max_abs_basis_dot,
        degenerate=degenerate_keep.squeeze(-1),
    )
    return ProjectionResult(embedding=output, diagnostics=diagnostics)


def orthogonalize_pooled_embedding(
    embedding: Tensor,
    directions: Tensor,
    *,
    strength: float = 1.0,
    relative_tolerance: float = 1e-3,
    absolute_tolerance: float = 0.0,
    max_rank: Optional[int] = None,
    restore_norm: bool = True,
    preserve_dtype: bool = True,
    eps: float = 1e-12,
) -> OrthogonalizationResult:
    """Estimate a basis and project an embedding in one convenience call."""

    basis_result = estimate_orthonormal_basis(
        directions,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        max_rank=max_rank,
    )
    projection_result = project_pooled_embedding(
        embedding,
        basis_result.basis,
        strength=strength,
        restore_norm=restore_norm,
        preserve_dtype=preserve_dtype,
        eps=eps,
    )
    return OrthogonalizationResult(
        embedding=projection_result.embedding,
        basis=basis_result.basis,
        basis_diagnostics=basis_result.diagnostics,
        projection_diagnostics=projection_result.diagnostics,
    )


def _prepare_four_embeddings(
    *,
    object_only: Tensor,
    object_behavior: Optional[Tensor],
    object_background: Optional[Tensor],
    object_behavior_background: Optional[Tensor],
    eps: float,
    require_all: bool = False,
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    values = (
        object_only,
        object_behavior,
        object_background,
        object_behavior_background,
    )
    names = (
        "object_only",
        "object_behavior",
        "object_background",
        "object_behavior_background",
    )
    if require_all and any(value is None for value in values):
        raise ValueError("factorial directions require all four pooled embeddings")

    reference_shape = object_only.shape
    if object_only.ndim != 1:
        raise ValueError("pooled prompt embeddings must be one-dimensional")

    normalized = []
    for name, value in zip(names, values):
        if value is None:
            normalized.append(None)
            continue
        _validate_float_tensor(name, value)
        if value.shape != reference_shape:
            raise ValueError("all pooled prompt embeddings must have the same shape")
        if value.device != object_only.device:
            raise ValueError("all pooled prompt embeddings must be on the same device")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values")
        normalized.append(normalize_fp32(value, eps=eps))

    e00, e10, e01, e11 = normalized
    assert e00 is not None
    return e00, e10, e01, e11


def _finalize_directions(
    directions: Sequence[Tensor],
    names: Sequence[str],
    *,
    embedding_dim: int,
    device: torch.device,
    normalize_directions: bool,
    eps: float,
) -> ContrastDirections:
    if directions:
        values = torch.stack(tuple(directions), dim=0).float()
        if normalize_directions:
            values = normalize_fp32(values, eps=eps)
    else:
        values = torch.empty((0, embedding_dim), dtype=torch.float32, device=device)
    return ContrastDirections(values=values, names=tuple(names))


def _validate_float_tensor(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")


def _validate_eps(eps: float) -> None:
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive")


def _validate_orthonormal_columns(basis: Tensor) -> None:
    rank = basis.shape[1]
    if rank == 0:
        return
    gram = basis.transpose(0, 1) @ basis
    identity = torch.eye(rank, dtype=torch.float32, device=basis.device)
    if not torch.allclose(gram, identity, atol=1e-4, rtol=1e-4):
        raise ValueError("basis columns must be orthonormal")
