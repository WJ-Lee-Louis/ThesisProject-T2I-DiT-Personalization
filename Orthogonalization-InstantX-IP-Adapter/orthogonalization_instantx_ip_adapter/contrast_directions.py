"""Local positive-minus-negative directions from the joint B response."""
from dataclasses import asdict, dataclass

import torch

from .orthogonalization import ContrastDirections, normalize_fp32
from .semantic_contrasts import SemanticContrastConditioning


@dataclass(frozen=True)
class LocalContrast:
    name: str
    origin: str
    unit: str
    positive: str
    negative: str


def contrast_plan(conditioning):
    if not isinstance(conditioning, SemanticContrastConditioning):
        raise ValueError("local contrasts require SemanticContrastConditioning")
    analysis = conditioning.analysis
    return tuple(LocalContrast(f"{side}:removal_{i}", side, row.unit,
                               analysis.source_caption if side == "source" else analysis.target_prompt,
                               row.negative)
                 for side, rows in conditioning.selected for i, row in enumerate(rows))


def contrast_diagnostics(conditioning):
    plan = contrast_plan(conditioning)
    return dict(direction_strategy="semantic_local", origin=conditioning.origin,
                contrasts=[asdict(row) for row in plan],
                source_preserved=list(conditioning.analysis.source_preserved),
                target_projection_skipped=conditioning.fallback_trace is not None,
                fallback_policy="target_conflict_source_only_v1" if conditioning.fallback_trace else None,
                selected_direction_count=len(plan),
                unique_text_context_count=len({s for row in plan for s in (row.positive, row.negative)}),
                contrast_pair_count=len(plan),
                empty_shape_probe=not plan,
                semantic_conditioning=conditioning.to_dict())


def build_contrast_directions(conditioning, encode):
    plan = contrast_plan(conditioning)
    prompts = list(dict.fromkeys(s for row in plan for s in (row.positive, row.negative)))
    # The encode-only interface uses a shape probe, never a fabricated removal.
    if not prompts:
        prompts = [conditioning.analysis.identity_term]
    pooled = encode(prompts)
    if (not isinstance(pooled, torch.Tensor) or pooled.ndim != 2 or pooled.shape[0] != len(prompts)
            or pooled.shape[1] == 0 or not pooled.is_floating_point() or not torch.isfinite(pooled).all()):
        raise ValueError("encoder must return a finite floating [prompt_count, embedding_dim] tensor")
    lookup = dict(zip(prompts, normalize_fp32(pooled)))
    raw = (torch.stack([lookup[row.positive] - lookup[row.negative] for row in plan]) if plan
           else pooled.new_empty((0, pooled.shape[1]), dtype=torch.float32))
    return ContrastDirections(normalize_fp32(raw), tuple(row.name for row in plan),
                              torch.linalg.vector_norm(raw, dim=-1))
