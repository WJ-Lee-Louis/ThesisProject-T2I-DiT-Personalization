"""Structured prompt catalog for the animal personalization experiment.

The catalog keeps the human-facing evaluation category separate from the
semantic factors used to construct counterfactual text directions.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping


PromptFactor = Literal["behavior", "background"]
PromptCategory = Literal[
    "single_behavior",
    "object_related",
    "background_related",
    "expression_related",
    "complex",
    "background_only",
]
PROMPT_CATEGORIES = frozenset(
    {
        "single_behavior",
        "object_related",
        "background_related",
        "expression_related",
        "complex",
        "background_only",
    }
)


@dataclass(frozen=True, slots=True)
class PromptSpec:
    """One target prompt and its evaluation/conditioning decomposition."""

    suffix: str
    category: PromptCategory
    behavior: str | None = None
    background: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.suffix, str) or not self.suffix.strip():
            raise ValueError("PromptSpec suffix must be a non-empty string")
        if self.suffix != self.suffix.strip():
            raise ValueError("PromptSpec suffix must not have surrounding whitespace")
        if self.category not in PROMPT_CATEGORIES:
            raise ValueError(f"Unknown prompt category: {self.category!r}")
        if self.behavior is None and self.background is None:
            raise ValueError("PromptSpec must contain behavior and/or background")
        for name, value in (
            ("behavior", self.behavior),
            ("background", self.background),
        ):
            if value is not None and (
                not isinstance(value, str)
                or not value.strip()
                or value != value.strip()
            ):
                raise ValueError(
                    f"PromptSpec {name} must be None or a trimmed non-empty string"
                )

        reconstructed = " ".join(
            value for value in (self.behavior, self.background) if value is not None
        )
        if reconstructed != self.suffix:
            raise ValueError(
                "PromptSpec suffix must equal its behavior/background composition: "
                f"expected {reconstructed!r}, got {self.suffix!r}"
            )

    @property
    def factors(self) -> tuple[PromptFactor, ...]:
        """Semantic factors present in deterministic behavior/background order."""

        factors: list[PromptFactor] = []
        if self.behavior is not None:
            factors.append("behavior")
        if self.background is not None:
            factors.append("background")
        return tuple(factors)

    @property
    def is_factorial(self) -> bool:
        """Whether both factors require a four-cell counterfactual design."""

        return self.behavior is not None and self.background is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "suffix": self.suffix,
            "category": self.category,
            "behavior": self.behavior,
            "background": self.background,
            "factors": list(self.factors),
            "is_factorial": self.is_factorial,
        }


PROMPT_CATALOG: tuple[PromptSpec, ...] = (
    PromptSpec(
        suffix="flying",
        category="single_behavior",
        behavior="flying",
    ),
    PromptSpec(
        suffix="sleeping",
        category="single_behavior",
        behavior="sleeping",
    ),
    PromptSpec(
        suffix="playing the piano",
        category="object_related",
        behavior="playing the piano",
    ),
    PromptSpec(
        suffix="holding an umbrella",
        category="object_related",
        behavior="holding an umbrella",
    ),
    PromptSpec(
        suffix="eating watermelon",
        category="object_related",
        behavior="eating watermelon",
    ),
    PromptSpec(
        suffix="riding a skateboard",
        category="object_related",
        behavior="riding a skateboard",
    ),
    PromptSpec(
        suffix="flying in the sky",
        category="background_related",
        behavior="flying",
        background="in the sky",
    ),
    PromptSpec(
        suffix="sleeping in the snow",
        category="background_related",
        behavior="sleeping",
        background="in the snow",
    ),
    PromptSpec(
        suffix="smiling",
        category="expression_related",
        behavior="smiling",
    ),
    PromptSpec(
        suffix="crying",
        category="expression_related",
        behavior="crying",
    ),
    PromptSpec(
        suffix="playing the guitar on a stage",
        category="complex",
        behavior="playing the guitar",
        background="on a stage",
    ),
    PromptSpec(
        suffix="swimming in the pool",
        category="complex",
        behavior="swimming",
        background="in the pool",
    ),
    PromptSpec(
        suffix="jumping over a wooden fence",
        category="complex",
        behavior="jumping",
        background="over a wooden fence",
    ),
    PromptSpec(
        suffix="reading a book on a sofa",
        category="complex",
        behavior="reading a book",
        background="on a sofa",
    ),
    PromptSpec(
        suffix="on a stage",
        category="background_only",
        background="on a stage",
    ),
    PromptSpec(
        suffix="in the snow",
        category="background_only",
        background="in the snow",
    ),
)

PROMPT_SUFFIXES: tuple[str, ...] = tuple(spec.suffix for spec in PROMPT_CATALOG)

_PROMPT_SPECS_BY_SUFFIX = {spec.suffix: spec for spec in PROMPT_CATALOG}
if len(_PROMPT_SPECS_BY_SUFFIX) != len(PROMPT_CATALOG):
    raise RuntimeError("Prompt catalog suffixes must be unique")

PROMPT_SPECS_BY_SUFFIX: Mapping[str, PromptSpec] = MappingProxyType(
    _PROMPT_SPECS_BY_SUFFIX
)


def get_prompt_spec(prompt_suffix: str) -> PromptSpec:
    """Return the immutable specification for a known prompt suffix."""

    try:
        return PROMPT_SPECS_BY_SUFFIX[prompt_suffix]
    except KeyError as exc:
        raise ValueError(f"Unknown prompt suffix: {prompt_suffix!r}") from exc


__all__ = [
    "PROMPT_CATALOG",
    "PROMPT_CATEGORIES",
    "PROMPT_SPECS_BY_SUFFIX",
    "PROMPT_SUFFIXES",
    "PromptCategory",
    "PromptFactor",
    "PromptSpec",
    "get_prompt_spec",
]
