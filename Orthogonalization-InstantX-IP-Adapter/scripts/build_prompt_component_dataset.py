"""Build minimal prompt-component datasets for three personalization benchmarks.

Each generated record stores the input metadata and only the ordered component strings
needed as projection directions. The API-facing contract is intentionally smaller still:
``{"components": ["..."]}``.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "datasets" / "prompt_component_analysis_v1"
DBPLUS_CAPTIONS = (
    REPO_ROOT
    / "datasets"
    / "XVerseBench"
    / "dreambench_plus_raw"
    / "extracted"
    / "data"
    / "captions"
)
XVERSE_SINGLE = (
    REPO_ROOT
    / "datasets"
    / "XVerseBench"
    / "official_metadata"
    / "XVerseBench_single.json"
)

SCHEMA_VERSION = "1.0.0"

GENERIC_WRAPPER_PATTERNS = (
    r"\ba photo of\b",
    r"\ba photograph of\b",
    r"\ba detailed photo of\b",
    r"\ba detailed photograph of\b",
    r"\ba high-resolution photo of\b",
    r"\ba photorealistic image of\b",
    r"\ba picture of\b",
    r"\ba depiction of\b",
    r"\ba photo capturing\b",
)

STYLE_MARKERS: tuple[tuple[str, str], ...] = (
    ("watercolor", "watercolor style"),
    ("anime-style", "anime style"),
    ("anime style", "anime style"),
    ("retro-style", "retro style"),
    ("retro style", "retro style"),
    ("retro comic", "retro comic style"),
    ("vintage-style", "vintage style"),
    ("cartoon-style", "cartoon style"),
    ("art deco", "art deco style"),
    ("pop art", "pop art style"),
    ("surrealistic", "surrealistic style"),
    ("surreal", "surreal style"),
    ("abstract", "abstract style"),
    ("cubist", "cubist style"),
    ("minimalist", "minimalist style"),
    ("charcoal drawing", "charcoal drawing style"),
    ("ink sketch", "ink sketch style"),
    ("line drawing", "line drawing style"),
    ("digital painting", "digital painting style"),
    ("digital art", "digital art style"),
    ("digital illustration", "digital illustration style"),
    ("oil painting", "oil painting style"),
    ("fantasy scene", "fantasy style"),
)

MEDIA_NOUNS = (
    "photo",
    "photograph",
    "image",
    "picture",
    "depiction",
    "illustration",
    "drawing",
    "painting",
    "sketch",
    "artwork",
    "digital art",
    "watercolor",
    "art piece",
    "poster",
    "graphic",
    "rendering",
    "cartoon",
    "comic strip",
    "scene",
)

ACTION_WORDS = {
    "balancing", "basking", "battling", "blending", "blowing", "broadcasting",
    "attached", "broken", "bringing", "carrying", "catching", "changing", "chasing", "chatting", "clinging", "curled",
    "chugging", "clearing", "climbing", "conducting", "crossing", "dancing",
    "displaying", "diving", "drifting", "driving", "eating", "emitting",
    "exploring", "falling", "feeding", "floating", "flying", "focusing", "galloping",
    "gardening", "gazing", "gliding", "grazing", "grooming", "guided", "hanging",
    "helping", "hiking", "holding", "hovering", "howling", "interacting",
    "jumping", "landing", "leading", "leaning", "leaping", "looking",
    "lounging", "meditating", "melting", "napping", "navigating", "nestled", "nibbling",
    "draped", "guided", "illuminated", "observed", "operating", "orbiting", "overlooking", "parked", "peeking", "perched", "performing", "piloting",
    "planting", "playing", "practicing", "prowling", "racing", "reading",
    "releasing", "resting", "riding", "roaring", "rowing", "running", "sailing",
    "radiating", "serving", "shooting", "sipping", "sitting", "sleeping", "soaring",
    "solving", "spinning", "splashing", "sprouting", "standing", "strolling",
    "swimming", "swinging", "taking", "teaching", "trading", "transforming",
    "surrounded", "transformed", "tucked", "traveling", "tumbling", "turning", "using", "waiting", "walking", "watching",
    "wearing", "weaving", "zooming",
}

PROTECTED_RELATION_STARTS = (
    "adorned with", "attached to", "covered in", "covered with", "crafted from", "decorated with",
    "draped over", "dressed in", "filled with", "guided by", "illuminated by", "made from", "made of",
    "morphing into", "observed through", "surrounded by", "transforming into", "turning into",
)

LOCATIVE_MARKERS = (
    "in front of", "on top of", "out of", "next to", "surrounded by",
    "above", "against", "along", "amid", "among", "around", "atop", "before", "behind", "below",
    "beneath", "beside", "by", "down", "inside", "near", "on", "outside", "over",
    "through", "under", "within", "during", "in", "at",
)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def trim_phrase(text: str) -> str:
    return normalize_space(text).strip(" ,.;:")


def is_action_word(word: str) -> bool:
    word = word.lower().strip(" ,.;:")
    non_verbal_ing = {
        "painting", "drawing", "setting", "morning", "lighting", "ring", "spring",
        "icing", "ceiling", "shimmering", "sparkling", "glowing", "bustling", "building", "living", "recording", "steaming", "trading", "blossoming",
    }
    return word in ACTION_WORDS or (word.endswith("ing") and word not in non_verbal_ing)


def articleless(text: str) -> str:
    return re.sub(r"^(?:a|an|the)\s+", "", trim_phrase(text), flags=re.I)


def make_component(text: str) -> str:
    """Normalize one final direction string without attaching semantic labels."""

    return trim_phrase(text)


def record(
    *,
    record_id: str,
    benchmark: str,
    category: str,
    subject_id: str,
    identity_hint: str,
    prompt_index: int,
    prompt: str,
    components: list[str],
    annotation_method: str,
    source_path: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "record_id": record_id,
        "benchmark": benchmark,
        "category": category,
        "subject_id": subject_id,
        "identity_hint": identity_hint,
        "prompt_index": prompt_index,
        "prompt": normalize_space(prompt),
        "components": components,
        "annotation_method": annotation_method,
        "source_path": source_path,
    }
    if extra:
        item.update(extra)
    return item


DREAMBENCH_SUBJECTS: tuple[tuple[str, str], ...] = (
    ("backpack", "backpack"),
    ("backpack_dog", "backpack"),
    ("bear_plushie", "stuffed animal"),
    ("berry_bowl", "bowl"),
    ("can", "can"),
    ("candle", "candle"),
    ("cat", "cat"),
    ("cat2", "cat"),
    ("clock", "clock"),
    ("colorful_sneaker", "sneaker"),
    ("dog", "dog"),
    ("dog2", "dog"),
    ("dog3", "dog"),
    ("dog5", "dog"),
    ("dog6", "dog"),
    ("dog7", "dog"),
    ("dog8", "dog"),
    ("duck_toy", "toy"),
    ("fancy_boot", "boot"),
    ("grey_sloth_plushie", "stuffed animal"),
    ("monster_toy", "toy"),
    ("pink_sunglasses", "glasses"),
    ("poop_emoji", "toy"),
    ("rc_car", "toy"),
    ("red_cartoon", "cartoon"),
    ("robot_toy", "toy"),
    ("shiny_sneaker", "sneaker"),
    ("teapot", "teapot"),
    ("vase", "vase"),
    ("wolf_plushie", "stuffed animal"),
)

LIVE_SUBJECT_IDS = {"cat", "cat2", "dog", "dog2", "dog3", "dog5", "dog6", "dog7", "dog8"}

# Each entry is (rendered suffix/prefix template, [(component text, type), ...]).
DREAMBENCH_OBJECT_TEMPLATES: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    ("{identity} in the jungle", (("in the jungle", "background_or_location"),)),
    ("{identity} in the snow", (("in the snow", "background_or_location"),)),
    ("{identity} on the beach", (("on the beach", "background_or_location"),)),
    ("{identity} on a cobblestone street", (("on a cobblestone street", "background_or_location"),)),
    ("{identity} on top of pink fabric", (("on top of pink fabric", "background_or_location"),)),
    ("{identity} on top of a wooden floor", (("on top of a wooden floor", "background_or_location"),)),
    ("{identity} with a city in the background", (("with a city in the background", "background_or_location"),)),
    ("{identity} with a mountain in the background", (("with a mountain in the background", "background_or_location"),)),
    ("{identity} with a blue house in the background", (("with a blue house in the background", "background_or_location"),)),
    ("{identity} on top of a purple rug in a forest", (("on top of a purple rug", "background_or_location"), ("in a forest", "background_or_location"))),
    ("{identity} with a wheat field in the background", (("with a wheat field in the background", "background_or_location"),)),
    ("{identity} with a tree and autumn leaves in the background", (("with a tree and autumn leaves in the background", "background_or_location"),)),
    ("{identity} with the Eiffel Tower in the background", (("with the Eiffel Tower in the background", "background_or_location"),)),
    ("{identity} floating on top of water", (("floating", "action"), ("on top of water", "background_or_location"))),
    ("{identity} floating in an ocean of milk", (("floating", "action"), ("in an ocean of milk", "background_or_location"))),
    ("{identity} on top of green grass with sunflowers around it", (("on top of green grass", "background_or_location"), ("with sunflowers around it", "relation"))),
    ("{identity} on top of a mirror", (("on top of a mirror", "background_or_location"),)),
    ("{identity} on top of the sidewalk in a crowded street", (("on top of the sidewalk", "background_or_location"), ("in a crowded street", "background_or_location"))),
    ("{identity} on top of a dirt road", (("on top of a dirt road", "background_or_location"),)),
    ("{identity} on top of a white rug", (("on top of a white rug", "background_or_location"),)),
    ("a red {identity_np}", (("red", "attribute"),)),
    ("a purple {identity_np}", (("purple", "attribute"),)),
    ("a shiny {identity_np}", (("shiny", "attribute"),)),
    ("a wet {identity_np}", (("wet", "attribute"),)),
    ("a cube shaped {identity_np}", (("cube shaped", "attribute"),)),
)

DREAMBENCH_LIVE_TEMPLATES = DREAMBENCH_OBJECT_TEMPLATES[:10] + (
    ("{identity} wearing a red hat", (("wearing a red hat", "attire_or_accessory"),)),
    ("{identity} wearing a santa hat", (("wearing a santa hat", "attire_or_accessory"),)),
    ("{identity} wearing a rainbow scarf", (("wearing a rainbow scarf", "attire_or_accessory"),)),
    ("{identity} wearing a black top hat and a monocle", (("wearing a black top hat and a monocle", "attire_or_accessory"),)),
    ("{identity} in a chef outfit", (("in a chef outfit", "attire_or_accessory"),)),
    ("{identity} in a firefighter outfit", (("in a firefighter outfit", "attire_or_accessory"),)),
    ("{identity} in a police outfit", (("in a police outfit", "attire_or_accessory"),)),
    ("{identity} wearing pink glasses", (("wearing pink glasses", "attire_or_accessory"),)),
    ("{identity} wearing a yellow shirt", (("wearing a yellow shirt", "attire_or_accessory"),)),
    ("{identity} in a purple wizard outfit", (("in a purple wizard outfit", "attire_or_accessory"),)),
) + DREAMBENCH_OBJECT_TEMPLATES[20:]


def build_dreambench() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    source_url = "https://github.com/google/dreambooth/blob/main/dataset/prompts_and_classes.txt"
    for subject_id, class_name in DREAMBENCH_SUBJECTS:
        category = "live_subject" if subject_id in LIVE_SUBJECT_IDS else "object"
        templates = DREAMBENCH_LIVE_TEMPLATES if category == "live_subject" else DREAMBENCH_OBJECT_TEMPLATES
        unique_token = f"<{subject_id}>"
        identity_hint = f"{unique_token} {class_name}"
        identity = f"a {identity_hint}"
        for prompt_index, (template, component_specs) in enumerate(templates):
            prompt = template.format(identity=identity, identity_np=identity_hint)
            components = [make_component(text) for text, _kind in component_specs]
            records.append(
                record(
                    record_id=f"dreambench:{subject_id}:{prompt_index:02d}",
                    benchmark="dreambench",
                    category=category,
                    subject_id=subject_id,
                    identity_hint=identity_hint,
                    prompt_index=prompt_index,
                    prompt=prompt,
                    components=components,
                    annotation_method="verbatim_surface_span_decomposition_v2",
                    source_path=source_url,
                    extra={
                        "class_name": class_name,
                        "unique_token_policy": "deterministic stand-in for the benchmark's learned unique token",
                    },
                )
            )
    return records


def split_on_conjunctions(text: str) -> list[str]:
    text = trim_phrase(text)
    if not text:
        return []
    parts = re.split(r"\s*(?:;|\bwhile\b|\bwhereas\b)\s*", text, flags=re.I)
    output: list[str] = []
    for part in parts:
        part = trim_phrase(part)
        if not part:
            continue
        # Split 'and' only when it clearly starts a second predicate or prepositional unit.
        chunks = re.split(
            r"\s+and\s+(?=(?:" + "|".join(sorted(ACTION_WORDS)) + r")\b|(?:in|on|at|under|over|beside|near|with|without|by)\b)",
            part,
            flags=re.I,
        )
        output.extend(trim_phrase(chunk) for chunk in chunks if trim_phrase(chunk))
    return output


def split_action_and_context(text: str) -> list[str]:
    text = trim_phrase(text)
    if not text:
        return []
    lower = text.lower()
    if lower.startswith(PROTECTED_RELATION_STARTS):
        return [text]
    if lower.startswith("with ") or lower.startswith("without "):
        pose_match = re.search(r"\s+(in\s+(?:a|an|the)\s+[^,]+\bpose)$", text, flags=re.I)
        if pose_match:
            return [trim_phrase(text[: pose_match.start()]), trim_phrase(pose_match.group(1))]
        return [text]

    # These verbs select the following preposition as an argument rather than an
    # independently editable background. Keeping the complement avoids fragments
    # such as ``gazing`` + ``at a sunset``.
    if re.match(
        r"^(?:gazing\s+at|looking\s+at|focusing\s+on|interacting\s+with|"
        r"reflecting\s+on|serving\s+as|transforming\s+into|morphing\s+into)\b",
        lower,
    ):
        return [text]

    tokens = text.split()
    if not tokens:
        return []

    first_action = next((index for index, token in enumerate(tokens) if is_action_word(token)), None)
    if first_action is not None and first_action > 0:
        prefix_words = tokens[:first_action]
        if all(word.lower().strip(" ,.;:").endswith("ly") for word in prefix_words):
            lower_text = text.lower()
            best: tuple[int, str] | None = None
            for marker in LOCATIVE_MARKERS:
                match = re.search(rf"\s\b{re.escape(marker)}\b\s", lower_text)
                if match and match.start() > 0 and (best is None or match.start() < best[0]):
                    best = (match.start(), marker)
            if best:
                pos = best[0] + 1
                return [trim_phrase(text[:pos]), *split_context_sequence(trim_phrase(text[pos:]))]
            return [text]
        if first_action <= 3:
            return [trim_phrase(" ".join(prefix_words)), *split_action_and_context(" ".join(tokens[first_action:]))]

    # A leading action/pose is independent from a later locative/background phrase.
    if is_action_word(tokens[0]):
        lower_text = text.lower()
        best: tuple[int, str] | None = None
        for marker in LOCATIVE_MARKERS:
            match = re.search(rf"\s\b{re.escape(marker)}\b\s", lower_text)
            if match and match.start() > 0:
                if best is None or match.start() < best[0]:
                    best = (match.start(), marker)
        if best:
            pos = best[0] + 1
            action = trim_phrase(text[:pos])
            context = trim_phrase(text[pos:])
            # Keep tightly bound verb complements together for known relations.
            if action.lower() not in {"looking", "focusing", "interacting"}:
                return [action, *split_context_sequence(context)]

    # A leading prepositional phrase followed by another predicate contains two units.
    if re.match(r"^(?:in|on|at|under|over|beside|near|with|without|by|amid|among)\b", lower):
        words = text.split()
        for index in range(2, len(words)):
            if is_action_word(words[index]):
                return [trim_phrase(" ".join(words[:index])), trim_phrase(" ".join(words[index:]))]
    if re.match(r"^(?:in|on|at|under|over|beside|near|with|without|by|amid|among|down)\b", lower):
        return split_context_sequence(text)
    return [text]


def split_context_sequence(text: str) -> list[str]:
    """Split sequential PP-like contexts while keeping each PP internally intact."""

    text = trim_phrase(text)
    lower = text.lower()
    if lower.startswith("with ") or lower.startswith("without "):
        return [text]
    # A trailing with-clause generally introduces one added entity/relation. Keep
    # nested prepositions inside it (for example, ``with Earth in the background``).
    with_match = re.search(r"\s+(with|without)\s+", lower)
    if with_match and not re.search(r"\b(?:filled|covered|decorated|adorned)\s*$", lower[: with_match.start()]):
        left = trim_phrase(text[: with_match.start()])
        right = trim_phrase(text[with_match.start() + 1 :])
        return [item for item in (left, right) if item]
    candidates: list[tuple[int, int, str]] = []
    for marker in sorted((*LOCATIVE_MARKERS, "with", "without"), key=len, reverse=True):
        for match in re.finditer(rf"(?<![A-Za-z]){re.escape(marker)}(?![A-Za-z])", lower):
            if marker in {"with", "without"} and re.search(
                r"\b(?:filled|covered|decorated|adorned)\s*$", lower[: match.start()]
            ):
                continue
            candidates.append((match.start(), match.end(), marker))
    candidates.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    starts: list[int] = []
    last_end = -1
    for start, end, _ in candidates:
        if start < last_end:
            continue
        if not starts or start - starts[-1] >= 4:
            starts.append(start)
            last_end = end
    if len(starts) <= 1 or starts[0] != 0:
        return [text]
    output = [trim_phrase(text[start:end]) for start, end in zip(starts, starts[1:] + [len(text)])]
    refined: list[str] = []
    for item in output:
        words = item.split()
        split_at = next((index for index in range(2, len(words)) if is_action_word(words[index])), None)
        if split_at is None:
            refined.append(item)
        else:
            refined.extend((trim_phrase(" ".join(words[:split_at])), trim_phrase(" ".join(words[split_at:]))))
    return [item for item in refined if item]


def split_suffix(text: str) -> list[str]:
    text = trim_phrase(text)
    if not text:
        return []
    action_pattern = "|".join(sorted(ACTION_WORDS | {"guided", "illuminated", "transformed"}))
    raw_clauses = re.split(
        rf"\s*,\s*(?=(?:{action_pattern})\b|(?:with|without|its|their|his|her|the|a|an|this|that)\b)|"
        rf"\s*;\s*|\s+—\s+|\s+-\s+",
        text,
        flags=re.I,
    )
    phrases: list[str] = []
    for clause in raw_clauses:
        for conjunct in split_on_conjunctions(clause):
            wearing_match = re.match(r"^wearing\s+(.+?)\s+and\s+((?:a|an|the)?\s*[^,]+)$", conjunct, flags=re.I)
            if wearing_match and len(wearing_match.group(2).split()) <= 4:
                second_item = trim_phrase(wearing_match.group(2))
                if not re.match(r"^(?:a|an|the)\b", second_item, flags=re.I) and len(second_item.split()) == 1 and not second_item.lower().endswith("s"):
                    second_item = f"a {second_item}"
                phrases.extend(
                    [
                        f"wearing {trim_phrase(wearing_match.group(1))}",
                        f"wearing {second_item}",
                    ]
                )
                continue
            added_entity = re.match(
                r"^(?:and\s+)?((?:a|an|the)\s+.+?)\s+(?=(?:" + action_pattern + r")\b)",
                conjunct,
                flags=re.I,
            )
            if added_entity:
                phrases.append(trim_phrase(added_entity.group(1)))
                conjunct = trim_phrase(conjunct[added_entity.end():])
            phrases.extend(split_action_and_context(conjunct))
    return [phrase for phrase in (trim_phrase(item) for item in phrases) if phrase]


def analyze_dbplus_prefix(prefix: str) -> list[str]:
    original = normalize_space(prefix)
    lower = original.lower()
    components: list[str] = []
    if not original:
        return components

    # Capture explicit non-photographic styles before discarding carrier wording.
    seen_style: set[str] = set()
    captured_style_markers: set[str] = set()
    occupied_style_spans: list[tuple[int, int]] = []
    intrinsic_style_markers = {
        "watercolor", "anime-style", "anime style", "retro-style", "retro style", "retro comic",
        "vintage-style", "cartoon-style", "art deco", "pop art", "charcoal drawing",
        "ink sketch", "line drawing", "digital painting", "digital art",
        "digital illustration", "oil painting", "fantasy scene",
    }
    for marker, canonical in sorted(STYLE_MARKERS, key=lambda item: len(item[0]), reverse=True):
        marker_match = re.search(rf"(?<![A-Za-z]){re.escape(marker)}(?![A-Za-z])", lower)
        following = lower[marker_match.end() :] if marker_match else ""
        marker_is_style = bool(
            marker_match
            and (
                marker in intrinsic_style_markers
                or re.match(r"(?:\s+|-)(?:style|photo|image|depiction|illustration|drawing|painting|sketch|artwork|scene)\b", following)
            )
        )
        if marker_is_style and marker_match and not any(
            marker_match.start() >= start and marker_match.end() <= end
            for start, end in occupied_style_spans
        ) and canonical not in seen_style:
            components.append(canonical)
            seen_style.add(canonical)
            captured_style_markers.add(marker)
            occupied_style_spans.append(marker_match.span())

    if "close-up" in lower:
        components.append("close-up composition")

    # Identify and remove a generic carrier that fills the entire pre-identity
    # prefix. Explicit styles were captured above and are retained.
    media_pattern = "|".join(re.escape(noun) for noun in sorted(MEDIA_NOUNS, key=len, reverse=True))
    carrier_match = re.match(
        r"^(?:an|a)\s+.*?\b(?:" + media_pattern + r")\b.*?\b(?:of|featuring|capturing)\s*(?:an|a|the)?\s*$",
        original,
        flags=re.I,
    )
    if not carrier_match:
        carrier_match = re.match(
            r"^(?:an|a)\s+(?:scene|image|depiction)\s+depicting\s*(?:an|a|the)?\s*$",
            original,
            flags=re.I,
        )
    remainder = original
    connector_match = re.search(r"\bon\s+(?:a|an|the)\s*$", remainder, flags=re.I)
    if connector_match:
        remainder = trim_phrase(remainder[: connector_match.start()])
        original = remainder
    if carrier_match:
        carrier = carrier_match.group(0)
        remainder = original[carrier_match.end():]
        carrier_lower = carrier.lower()
        if not seen_style:
            for medium in ("illustration", "drawing", "painting", "sketch", "artwork", "poster", "graphic", "rendering", "cartoon"):
                if medium in carrier_lower:
                    components.append(f"{medium} medium")
                    break
    else:
        partial_carrier = re.match(
            r"^(?:an|a)\s+.*?\b(?:" + media_pattern + r")\b.*?\b(?:of|featuring|capturing)\s*(?:an|a|the)?\s+",
            remainder,
            flags=re.I,
        )
        if not partial_carrier:
            partial_carrier = re.match(
                r"^(?:an|a)\s+(?:scene|image|depiction)\s+(?:depicting|featuring|capturing)\s+",
                remainder,
                flags=re.I,
            )
        if partial_carrier:
            remainder = remainder[partial_carrier.end():]
        # Plain leading article is grammatical scaffolding, not a projection direction.
        article_match = re.match(r"^(?:a|an|the)(?:\s+|$)", remainder, flags=re.I)
        if article_match:
            remainder = remainder[article_match.end():]

    # Remove style words already represented as style components from the leftover.
    for marker in captured_style_markers:
        remainder = re.sub(rf"\b{re.escape(marker)}\b", " ", remainder, flags=re.I)
    remainder = re.sub(
        r"\b(?:photo|photograph|image|picture|depiction|illustration|drawing|painting|sketch|artwork|poster|graphic|rendering|cartoon|scene)\b",
        " ",
        remainder,
        flags=re.I,
    )
    remainder = re.sub(r"\b(?:detailed|high-resolution|photorealistic)\b", " ", remainder, flags=re.I)
    remainder = trim_phrase(remainder)

    if remainder:
        # Preserve a newly introduced entity separately when the identity is an object.
        entity_match = re.match(r"^(a|an|the)\s+(character|person|man|woman|boy|girl|rider|guy)\b", remainder, flags=re.I)
        if entity_match:
            components.append(entity_match.group(0))
            remainder = trim_phrase(remainder[entity_match.end():])
        if remainder:
            components.extend(split_suffix(remainder))

    return components


def dedupe_phrases(phrases: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        phrase = trim_phrase(phrase)
        key = phrase.lower()
        if phrase and key not in seen:
            output.append(phrase)
            seen.add(key)
    return output


def repair_phrases(phrases: Iterable[str]) -> list[str]:
    """Remove parser scaffolding and merge deterministically broken fragments."""

    cleaned: list[str] = []
    generic_fragments = {
        "in", "in a style", "in the style", "featuring", "featuring a", "featuring an",
        "depicting", "depicting a", "depicting an", "capturing", "capturing a", "capturing an",
    }
    for raw in phrases:
        phrase = trim_phrase(raw)
        if not phrase or phrase.lower() in generic_fragments:
            continue
        if cleaned and cleaned[-1].lower() in {
            "illuminated", "guided", "observed", "reflecting", "tumbling", "spinning"
        } and re.match(r"^(?:by|through|on|around)\b", phrase, flags=re.I):
            cleaned[-1] = trim_phrase(f"{cleaned[-1]} {phrase}")
            continue
        if cleaned and re.search(r"\b(?:a|an|the)$", cleaned[-1], flags=re.I):
            cleaned[-1] = trim_phrase(f"{cleaned[-1]} {phrase}")
            continue
        if phrase.lower() in {"above", "below", "inside", "outside", "around", "nearby"} and cleaned:
            cleaned[-1] = trim_phrase(f"{cleaned[-1]} {phrase}")
            continue
        cleaned.append(phrase)

    # Prefer the more specific marker when style names overlap.
    result: list[str] = []
    for phrase in cleaned:
        if phrase.lower() == "surreal style" and any(
            item.lower() == "surrealistic style" for item in cleaned
        ):
            continue
        result.append(phrase)
    return dedupe_phrases(result)


DBPLUS_CARRIER_CONNECTOR = re.compile(
    r"^(?P<carrier>.+?\b(?:of|featuring|depicting|capturing))\s+(?P<remainder>.*)$",
    flags=re.I,
)

BOUND_PREPOSITION_PAIRS = {
    ("adorned", "with"),
    ("attached", "to"),
    ("captured", "during"),
    ("captured", "in"),
    ("captured", "with"),
    ("caught", "in"),
    ("covered", "in"),
    ("covered", "with"),
    ("depicted", "in"),
    ("depicted", "with"),
    ("dotted", "with"),
    ("draped", "over"),
    ("dressed", "in"),
    ("engaged", "in"),
    ("entwined", "with"),
    ("filled", "with"),
    ("focusing", "on"),
    ("gazing", "at"),
    ("guided", "by"),
    ("held", "by"),
    ("held", "in"),
    ("looking", "at"),
    ("made", "from"),
    ("made", "of"),
    ("morphing", "into"),
    ("outlined", "against"),
    ("pouncing", "on"),
    ("set", "against"),
    ("tethered", "by"),
    ("surrounded", "by"),
    ("transforming", "into"),
    ("turning", "into"),
    ("wrapped", "in"),
}


def _verbatim_piece(text: str) -> str:
    """Remove separators without changing any character inside a component."""

    return text.strip().strip(" ,;").strip()


def split_dbplus_prefix_verbatim(prefix: str) -> list[str]:
    """Split pre-identity content while retaining exact prompt wording.

    The carrier is deliberately kept as one surface span.  In particular, style
    wording is never canonicalized to labels such as ``watercolor style``.
    """

    prefix = prefix.strip()
    if not prefix:
        return []

    carrier_match = DBPLUS_CARRIER_CONNECTOR.match(prefix)
    if carrier_match:
        carrier = _verbatim_piece(carrier_match.group("carrier"))
        remainder = carrier_match.group("remainder").strip()
        remainder = re.sub(r"^(?:a|an|the)(?:\s+|$)", "", remainder, count=1, flags=re.I)
        # These words introduce the grammatical number of the identity; they are
        # not an independently removable visual direction.
        if re.fullmatch(r"(?:group|pair|cluster|collection)\s+of", remainder, flags=re.I):
            remainder = ""
        return [piece for piece in (carrier, _verbatim_piece(remainder)) if piece]

    # In "A tiny kitten", the article stays with the excluded noun phrase while
    # the exact attribute span remains removable.
    without_article = re.sub(r"^(?:a|an|the)(?:\s+|$)", "", prefix, count=1, flags=re.I)
    piece = _verbatim_piece(without_article)
    return [piece] if piece else []


def _split_comma_clauses_verbatim(text: str) -> list[str]:
    action_pattern = "|".join(sorted(ACTION_WORDS | {
        "guided", "illuminated", "observed", "surrounded", "transformed",
    }))
    boundary = re.compile(
        rf"[,;]\s+(?=(?:{action_pattern}|with|without|its|their|his|her)\b)",
        flags=re.I,
    )
    pieces: list[str] = []
    start = 0
    for match in boundary.finditer(text):
        piece = _verbatim_piece(text[start:match.start()])
        if piece:
            pieces.append(piece)
        start = match.end()
    final = _verbatim_piece(text[start:])
    if final:
        pieces.append(final)
    return pieces


def _is_bound_preposition(text: str, marker_start: int, marker: str) -> bool:
    before = text[:marker_start].rstrip()
    previous_match = re.search(r"([A-Za-z-]+)\s*$", before)
    if not previous_match:
        return False
    return (previous_match.group(1).lower(), marker.lower()) in BOUND_PREPOSITION_PAIRS


def _split_contexts_verbatim(text: str) -> list[str]:
    """Split independent settings/relations without inventing bridge words."""

    candidates: list[tuple[int, int]] = []
    for marker in sorted((*LOCATIVE_MARKERS, "with", "without"), key=len, reverse=True):
        for match in re.finditer(rf"(?<![A-Za-z]){re.escape(marker)}(?![A-Za-z])", text, flags=re.I):
            if match.start() == 0 or _is_bound_preposition(text, match.start(), marker):
                continue
            candidates.append(match.span())
    candidates.sort(key=lambda span: (span[0], -(span[1] - span[0])))

    starts: list[int] = []
    occupied_end = -1
    for start, end in candidates:
        if start < occupied_end:
            continue
        preceding = _verbatim_piece(text[(starts[-1] if starts else 0):start])
        if not preceding or re.search(r"\b(?:a|an|the|of)$", preceding, flags=re.I):
            continue
        starts.append(start)
        occupied_end = end

    if not starts:
        piece = _verbatim_piece(text)
        return [piece] if piece else []

    boundaries = [0, *starts, len(text)]
    return [
        piece
        for piece in (
            _verbatim_piece(text[start:end])
            for start, end in zip(boundaries, boundaries[1:])
        )
        if piece
    ]


def split_dbplus_suffix_verbatim(suffix: str) -> list[str]:
    suffix = suffix.strip()
    if not suffix:
        return []
    output: list[str] = []
    for clause in _split_comma_clauses_verbatim(suffix):
        output.extend(_split_contexts_verbatim(clause))
    return output


def _identity_match(prompt: str, label: str) -> re.Match[str]:
    candidates = [label]
    without_article = re.sub(r"^(?:a|an|the)\s+", "", label, count=1, flags=re.I)
    if without_article != label:
        candidates.append(without_article)
    for candidate in candidates:
        match = re.search(
            rf"(?<![A-Za-z]){re.escape(candidate)}(?![A-Za-z])", prompt, flags=re.I
        )
        if match:
            return match
    raise ValueError(f"Identity label {label!r} is absent from prompt {prompt!r}")


def validate_verbatim_components(prompt: str, label: str, components: list[str]) -> None:
    """Enforce exact, ordered, non-overlapping surface spans and no identity leak."""

    identity = _identity_match(prompt, label).span()
    cursor = 0
    spans: list[tuple[int, int]] = []
    for component in components:
        if component != component.strip() or not component:
            raise AssertionError(f"Invalid component whitespace in {prompt!r}: {component!r}")
        start = prompt.find(component, cursor)
        if start < 0:
            raise AssertionError(f"Component is not an ordered verbatim span: {component!r} in {prompt!r}")
        end = start + len(component)
        if start < identity[1] and end > identity[0]:
            raise AssertionError(f"Identity leaked into component: {component!r} in {prompt!r}")
        spans.append((start, end))
        cursor = end
    if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
        raise AssertionError(f"Overlapping components in {prompt!r}")

    # The complement of the marked component and identity spans is the excluded
    # material (determiners, separators, and any identity-side wording).  Slice
    # every boundary and verify that the partition reconstructs the source byte
    # for byte; no normalization is allowed during this check.
    boundaries = sorted({0, len(prompt), *identity, *(point for span in spans for point in span)})
    reconstructed = "".join(
        prompt[start:end] for start, end in zip(boundaries, boundaries[1:])
    )
    if reconstructed != prompt:
        raise AssertionError(f"Lossless prompt reconstruction failed: {prompt!r}")


def analyze_dreambench_plus_prompt(prompt: str, label: str) -> list[str]:
    match = _identity_match(prompt, label)
    suffix = prompt[match.end():]
    # These immediately attached nouns complete the depicted identity rather
    # than introducing an independently removable direction.
    identity_extensions = {
        "avocado": "character",
        "pineapple": "character",
        "vintage television": "set",
    }
    extension = identity_extensions.get(label.casefold())
    if extension:
        extension_match = re.match(rf"\s+{re.escape(extension)}\b", suffix, flags=re.I)
        if extension_match:
            suffix = suffix[extension_match.end():]
    components = [
        *split_dbplus_prefix_verbatim(prompt[:match.start()]),
        *split_dbplus_suffix_verbatim(suffix),
    ]
    validate_verbatim_components(prompt, label, components)
    return components


def build_dreambench_plus() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(DBPLUS_CAPTIONS.rglob("*.txt")):
        relative = path.relative_to(DBPLUS_CAPTIONS)
        if "style" in relative.parts:
            continue
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(lines) != 10:
            raise ValueError(f"Expected label plus 9 prompts in {path}, got {len(lines)} lines")
        label = lines[0]
        category = "animal" if "animal" in relative.parts else "human" if "human" in relative.parts else "object"
        subject_id = "_".join((*relative.parent.parts, path.stem))
        for prompt_index, prompt in enumerate(lines[1:]):
            components = analyze_dreambench_plus_prompt(prompt, label)
            records.append(
                record(
                    record_id=f"dreambench_plus:{subject_id}:{prompt_index}",
                    benchmark="dreambench_plus",
                    category=category,
                    subject_id=subject_id,
                    identity_hint=label,
                    prompt_index=prompt_index,
                    prompt=prompt,
                    components=components,
                    annotation_method="verbatim_surface_span_decomposition_v2",
                    source_path=str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
                    extra={
                        "difficulty_group": "photorealistic" if prompt_index < 4 else "stylized" if prompt_index < 7 else "imaginative",
                    },
                )
            )
    return records


XVERSE_SELECTED_OBJECTS = {50, 52, 61, 64, 70, 73, 78, 83, 84, 86, 89}

# Explicit phrase-level decompositions for the requested 60 XVerseBench records.
XVERSE_COMPONENTS: dict[int, tuple[tuple[str, str], ...]] = {
    0: (("standing", "pose_or_state"), ("on iceberg", "background_or_location")),
    1: (("swimming", "action"), ("in the sea", "background_or_location")),
    2: (("standing", "pose_or_state"), ("in a forest", "background_or_location")),
    3: (("wearing a red collar", "attire_or_accessory"), ("running", "action")),
    4: (("sitting", "pose_or_state"), ("on a sofa", "background_or_location")),
    5: (("colorful", "attribute"), ("flying", "action"), ("in the garden", "background_or_location")),
    6: (("jumping", "action"), ("on the grass", "background_or_location")),
    7: (("standing", "pose_or_state"), ("in a forest", "background_or_location")),
    8: (("jumping", "action"), ("out of the water", "background_or_location")),
    9: (("cute", "attribute"), ("wearing a bowtie", "attire_or_accessory")),
    10: (("standing", "pose_or_state"), ("in the savannah", "background_or_location")),
    11: (("swimming", "action"), ("in the blue sea", "background_or_location")),
    12: (("flying", "action"), ("in the blue sky", "background_or_location")),
    13: (("sitting", "pose_or_state"), ("on a bamboo mat", "background_or_location")),
    14: (("standing", "pose_or_state"), ("on a wooden fence", "background_or_location")),
    15: (("cute", "attribute"), ("sitting", "pose_or_state"), ("on a mat", "background_or_location")),
    16: (("on a rocky background", "background_or_location"),),
    17: (("standing", "pose_or_state"), ("on an iceberg", "background_or_location")),
    18: (("standing", "pose_or_state"), ("in the grass", "background_or_location")),
    19: (("in a cozy sweater", "attire_or_accessory"),),
    20: (("cute", "attribute"), ("wearing a tiny hat", "attire_or_accessory")),
    21: (("walking", "action"), ("on the sandy beach", "background_or_location")),
    22: (("standing", "pose_or_state"), ("on a grassy field", "background_or_location")),
    23: (("howling", "action"), ("in the forest", "background_or_location")),
    24: (("sly", "attribute"), ("in a forest setting", "background_or_location")),
    25: (("cute", "attribute"), ("in a red bow", "attire_or_accessory")),
    26: (("cute", "attribute"), ("sitting", "pose_or_state"), ("on a mat", "background_or_location")),
    27: (("cute", "attribute"), ("in a red collar", "attire_or_accessory")),
    28: (("standing", "pose_or_state"), ("by a pond", "background_or_location")),
    29: (("standing", "pose_or_state"), ("in a city street", "background_or_location")),
    30: (("smiling", "pose_or_state"), ("in a flower-filled garden", "background_or_location")),
    31: (("cute", "attribute"), ("smiling", "pose_or_state"), ("in a crib", "background_or_location")),
    32: (("smiling", "pose_or_state"), ("in a sunny park", "background_or_location")),
    33: (("standing", "pose_or_state"), ("in a city street", "background_or_location")),
    34: (("smiling", "pose_or_state"), ("in a flower-filled garden", "background_or_location")),
    35: (("in a red dress", "attire_or_accessory"), ("smiling", "pose_or_state")),
    36: (("standing", "pose_or_state"), ("in a city street", "background_or_location")),
    37: (("standing", "pose_or_state"), ("in a park", "background_or_location")),
    38: (("wearing a hat", "attire_or_accessory"), ("standing", "pose_or_state"), ("in the forest", "background_or_location")),
    39: (("standing", "pose_or_state"), ("in a city street", "background_or_location")),
    40: (("standing", "pose_or_state"), ("in a garden", "background_or_location")),
    41: (("standing", "pose_or_state"), ("in a flower-filled garden", "background_or_location")),
    42: (("standing", "pose_or_state"), ("in a city street", "background_or_location")),
    43: (("standing", "pose_or_state"), ("in a city street", "background_or_location")),
    44: (("standing", "pose_or_state"), ("in a city street", "background_or_location")),
    45: (("standing", "pose_or_state"), ("in a city street", "background_or_location")),
    46: (("standing", "pose_or_state"), ("in a city street", "background_or_location")),
    47: (("sitting", "pose_or_state"), ("on a bench", "background_or_location")),
    48: (("smiling", "pose_or_state"), ("in a garden", "background_or_location")),
    50: (("in a city background", "background_or_location"),),
    52: (("in a cute pose", "pose_or_state"),),
    61: (("in a cute pose", "pose_or_state"),),
    64: (("sitting", "pose_or_state"), ("on a cozy sofa", "background_or_location")),
    70: (("in a cool pose", "pose_or_state"),),
    73: (("standing", "pose_or_state"), ("in a city street", "background_or_location")),
    78: (("chugging", "action"), ("on tracks", "background_or_location")),
    83: (("sitting", "pose_or_state"), ("on a sofa", "background_or_location")),
    84: (("in a traditional pose", "pose_or_state"),),
    86: (("on a pedestal", "background_or_location"),),
    89: (("standing", "pose_or_state"), ("in a pixel world", "background_or_location")),
}


def build_xverse() -> list[dict[str, Any]]:
    data = json.loads(XVERSE_SINGLE.read_text(encoding="utf-8"))
    selected = set(range(49)) | XVERSE_SELECTED_OBJECTS
    records: list[dict[str, Any]] = []
    for item in data:
        index = int(item["index"])
        if index not in selected:
            continue
        modulation = item["modulation"][0]
        src = modulation["src_inputs"][0]
        identity_hint = src["caption"]
        image_path = src["image_path"]
        category = image_path.split("/")[-2]
        specs = XVERSE_COMPONENTS[index]
        components = [make_component(text) for text, _kind in specs]
        records.append(
            record(
                record_id=f"xversebench:single:{index:03d}",
                benchmark="xversebench",
                category=category,
                subject_id=f"single_{index:03d}",
                identity_hint=identity_hint,
                prompt_index=0,
                prompt=item["prompt"],
                components=components,
                annotation_method="verbatim_surface_span_decomposition_v2",
                source_path=str(XVERSE_SINGLE.relative_to(REPO_ROOT)).replace("\\", "/"),
                extra={"official_record_index": index, "reference_image": image_path},
            )
        )
    return records


def markdown_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=False) + "\n")


def write_markdown(path: Path, title: str, records: list[dict[str, Any]], note: str) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in records:
        grouped.setdefault((item["category"], item["subject_id"]), []).append(item)

    lines = [
        f"# {title}",
        "",
        note,
        "",
        f"총 레코드: **{len(records)}개**",
        "",
    ]
    for (category, subject_id), items in grouped.items():
        identity = items[0]["identity_hint"]
        lines.extend(
            [
                f"## {category} / {subject_id} — `{markdown_escape(identity)}`",
                "",
                "| # | Prompt | Components |",
                "|---:|---|---|",
            ]
        )
        for item in items:
            components = "; ".join(f"`{markdown_escape(x)}`" for x in item["components"]) or "[]"
            lines.append(
                "| {idx} | {prompt} | {components} |".format(
                    idx=item["prompt_index"],
                    prompt=markdown_escape(item["prompt"]),
                    components=components,
                )
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def distribution(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(item[key]) for item in records).items()))


def validate(records_by_benchmark: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    expected = {"dreambench": 750, "dreambench_plus": 1170, "xversebench": 60}
    all_ids: list[str] = []
    summary: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "benchmarks": {}}
    for name, records in records_by_benchmark.items():
        if len(records) != expected[name]:
            raise AssertionError(f"{name}: expected {expected[name]} records, got {len(records)}")
        ids = [item["record_id"] for item in records]
        if len(ids) != len(set(ids)):
            raise AssertionError(f"{name}: duplicate record_id")
        all_ids.extend(ids)
        for item in records:
            if not item["prompt"] or not item["identity_hint"]:
                raise AssertionError(f"Empty prompt or identity: {item['record_id']}")
            if not isinstance(item["components"], list) or not all(
                isinstance(component, str) and component for component in item["components"]
            ):
                raise AssertionError(f"Invalid components: {item['record_id']}")
        summary["benchmarks"][name] = {
            "record_count": len(records),
            "category_counts": distribution(records, "category"),
            "components_per_prompt_distribution": dict(
                sorted(Counter(len(item["components"]) for item in records).items())
            ),
            "annotation_methods": distribution(records, "annotation_method"),
        }
        for item in records:
            validate_verbatim_components(
                item["prompt"], item["identity_hint"], item["components"]
            )
        summary["benchmarks"][name]["verbatim_span_validation"] = {
            "checked_records": len(records),
            "all_components_are_ordered_non_overlapping_prompt_substrings": True,
            "identity_overlap_count": 0,
            "lossless_original_offset_reconstruction": True,
        }
    if len(all_ids) != len(set(all_ids)):
        raise AssertionError("Duplicate record IDs across benchmarks")
    summary["total_record_count"] = len(all_ids)
    if summary["total_record_count"] != 1980:
        raise AssertionError(f"Expected 1980 total records, got {summary['total_record_count']}")
    return summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records_by_benchmark = {
        "dreambench": build_dreambench(),
        "dreambench_plus": build_dreambench_plus(),
        "xversebench": build_xverse(),
    }
    summary = validate(records_by_benchmark)

    filenames = {
        "dreambench": ("dreambench_components.jsonl", "dreambench_components.md"),
        "dreambench_plus": ("dreambench_plus_non_style_components.jsonl", "dreambench_plus_non_style_components.md"),
        "xversebench": ("xversebench_selected_components.jsonl", "xversebench_selected_components.md"),
    }
    notes = {
        "dreambench": "공식 30개 identity × 25개 prompt를 원문 verbatim surface span으로 분해했습니다. `<subject_id>`는 원 논문의 learned unique token을 검토 가능하게 표현한 stand-in입니다.",
        "dreambench_plus": "Style-reference category를 제외한 130개 identity × 9개 prompt입니다. component는 원문에 실제로 존재하는 대소문자·하이픈·문장부호를 보존한 비중첩 surface span입니다.",
        "xversebench": "Single-subject의 animal 29개, human 20개와 비개인화 인물 상호작용이 없는 object 11개를 원문 verbatim surface span으로 분해했습니다.",
    }
    titles = {
        "dreambench": "DreamBench prompt-component pairs",
        "dreambench_plus": "DreamBench++ non-style prompt-component pairs",
        "xversebench": "XVerseBench selected single-subject prompt-component pairs",
    }
    for name, records in records_by_benchmark.items():
        jsonl_name, md_name = filenames[name]
        write_jsonl(OUTPUT_DIR / jsonl_name, records)
        write_markdown(OUTPUT_DIR / md_name, titles[name], records, notes[name])

    (OUTPUT_DIR / "validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
