"""Generalized versions of the existing local and factorial text contrasts."""
from itertools import combinations

import torch

from .orthogonalization import ContrastDirections, normalize_fp32


def component_contrast_plan(conditioning):
    """Return (name, positive/negative context pairs) for each selected component."""
    rows = []
    for label, analysis in conditioning.selected:
        count = len(analysis.components)
        full = set(range(count))
        for index in range(count):
            others = sorted(full - {index})
            if conditioning.strategy == "local":
                contexts = [others]
            else:
                contexts = [subset for size in range(len(others) + 1)
                            for subset in combinations(others, size)]
            pairs = tuple((analysis.prompt_for_subset((*context, index)), analysis.prompt_for_subset(context))
                          for context in contexts)
            rows.append((f"{label}:component_{index}", pairs))
    return tuple(rows)


def build_component_directions(conditioning, encode):
    plan = component_contrast_plan(conditioning)
    prompts = list(dict.fromkeys(prompt for _, pairs in plan for pair in pairs for prompt in pair))
    # Obtain embedding dimension/device even for a valid zero-component identity-only input.
    if not prompts:
        prompts = [conditioning.target.identity_term]
    pooled = encode(prompts)
    normalized = normalize_fp32(pooled)
    lookup = dict(zip(prompts, normalized))
    rows = []
    for _, pairs in plan:
        # Match existing factorial: normalize pooled prompts, average raw differences,
        # then normalize each resulting direction. Do not normalize individual pairs.
        rows.append(torch.stack([lookup[pos] - lookup[neg] for pos, neg in pairs]).mean(dim=0))
    raw = torch.stack(rows) if rows else pooled.new_empty((0, pooled.shape[-1]), dtype=torch.float32)
    return ContrastDirections(normalize_fp32(raw), tuple(name for name, _ in plan),
                              torch.linalg.vector_norm(raw, dim=-1))
