"""Frozen 16-prompt catalog for the FreeGraftor baseline."""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True, slots=True)
class PromptSpec:
    suffix: str
    slug: str
    category: str
    composition: str

    def full_prompt(self, species: str) -> str:
        if species not in {"dog", "cat"}:
            raise ValueError(f"Unsupported species: {species!r}")
        return f"a {species} {self.suffix}"


PROMPT_CATALOG = (
    PromptSpec("flying", "flying", "single_behavior", "O+A"),
    PromptSpec("sleeping", "sleeping", "single_behavior", "O+A"),
    PromptSpec("in the snow", "in_the_snow", "single_background", "O+B"),
    PromptSpec("on a stage", "on_a_stage", "single_background", "O+B"),
    PromptSpec(
        "playing the piano",
        "playing_the_piano",
        "object_related",
        "O+A",
    ),
    PromptSpec(
        "holding an umbrella",
        "holding_an_umbrella",
        "object_related",
        "O+A",
    ),
    PromptSpec(
        "eating watermelon",
        "eating_watermelon",
        "object_related",
        "O+A",
    ),
    PromptSpec(
        "riding a skateboard",
        "riding_a_skateboard",
        "object_related",
        "O+A",
    ),
    PromptSpec(
        "flying in the sky",
        "flying_in_the_sky",
        "background_related",
        "O+A+B",
    ),
    PromptSpec(
        "sleeping in the snow",
        "sleeping_in_the_snow",
        "background_related",
        "O+A+B",
    ),
    PromptSpec("smiling", "smiling", "expression", "O+A"),
    PromptSpec("crying", "crying", "expression", "O+A"),
    PromptSpec(
        "playing the guitar on a stage",
        "playing_the_guitar_on_a_stage",
        "complex",
        "O+A+B",
    ),
    PromptSpec(
        "swimming in the pool",
        "swimming_in_the_pool",
        "complex",
        "O+A+B",
    ),
    PromptSpec(
        "jumping over a wooden fence",
        "jumping_over_a_wooden_fence",
        "complex",
        "O+A+B",
    ),
    PromptSpec(
        "reading a book on a sofa",
        "reading_a_book_on_a_sofa",
        "complex",
        "O+A+B",
    ),
)

PROMPT_SUFFIXES = tuple(item.suffix for item in PROMPT_CATALOG)


def _validate_catalog() -> None:
    if len(PROMPT_CATALOG) != 16:
        raise RuntimeError("The baseline catalog must contain exactly 16 prompts")
    suffixes = {item.suffix for item in PROMPT_CATALOG}
    slugs = {item.slug for item in PROMPT_CATALOG}
    if len(suffixes) != 16 or len(slugs) != 16:
        raise RuntimeError("Prompt suffixes and slugs must be unique")
    for item in PROMPT_CATALOG:
        if not re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", item.slug):
            raise RuntimeError(f"Unsafe prompt slug: {item.slug!r}")


_validate_catalog()
