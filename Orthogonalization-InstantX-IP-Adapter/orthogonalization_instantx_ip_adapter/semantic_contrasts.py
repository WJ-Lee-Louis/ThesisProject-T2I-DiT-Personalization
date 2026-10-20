"""Joint local contrasts, independent of span partitions, encoders, and generation.

Validation establishes the wire contract, not grammatical or semantic correctness.
All stored positives come from the caller, never from a model-generated summary.
"""
from __future__ import annotations

from dataclasses import dataclass, replace, field
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import re
from typing import Protocol

from PIL import Image

from .contrast_prompt import (
    CAPTION_POLICY_VERSION, CAPTION_PROMPT_VERSION, CAPTION_PROMPT_SHA256,
    CAPTION_SCHEMA, CAPTION_SYSTEM_PROMPT,
    CONTRAST_POLICY_VERSION, CONTRAST_PROMPT_VERSION, CONTRAST_PROMPT_SHA256,
    CONTRAST_SCHEMA, CONTRAST_SYSTEM_PROMPT,
    SOURCE_FALLBACK_PROMPT_VERSION, SOURCE_FALLBACK_PROMPT_SHA256, SOURCE_FALLBACK_SYSTEM_PROMPT,
)

SCHEMA_VERSION = 4
CAPTION_SCHEMA_VERSION = 2
MAX_REMOVALS = 64
ORIGINS = ("source", "target", "source_target")


class StructuredLLMProvider(Protocol):
    def complete(self, *, system_prompt: str, payload: dict, schema: dict,
                 image_bytes: bytes | None = None, feedback: str | None = None) -> str: ...


def _text(value, name):
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty string without outer whitespace")
    return value


def _keys(value, names, name):
    if not isinstance(value, dict) or set(value) != set(names.split()):
        raise ValueError(f"{name} requires exactly: {names}")


def _array(value, name):
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    return value


def _hash(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a SHA-256 hex digest")


def _version(value, expected, name):
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"unsupported {name}: expected {expected!r}")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _input_hash(value: dict) -> str:
    return _sha(json.dumps(value, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"), allow_nan=False).encode("utf-8"))


def identity_occurrences(text: str, identity_term: str, *, allow_initial_capital: bool = False) -> tuple[tuple[int, int], ...]:
    """Case-sensitive literal phrase with the legacy lexical boundary rule."""
    _text(text, "text")
    _text(identity_term, "identity_term")
    pattern = re.escape(identity_term)
    if identity_term[0].isalnum() or identity_term[0] == "_":
        pattern = r"(?<!\w)" + pattern
    if identity_term[-1].isalnum() or identity_term[-1] == "_":
        pattern += r"(?!\w)"
    matches = tuple(m.span() for m in re.finditer(pattern, text))
    if allow_initial_capital and identity_term[0].islower():
        initial = identity_term[0].upper() + identity_term[1:]
        boundary = r"(?!\w)" if identity_term[-1].isalnum() or identity_term[-1] == "_" else ""
        match = re.match(re.escape(initial) + boundary, text)
        if match and match.span() not in matches:
            matches = tuple(sorted((match.span(), *matches)))
    if not matches:
        raise ValueError(f"Required phrase {identity_term!r} must occur verbatim at lexical boundaries in {text!r}")
    return matches


def preservation_phrases(value, source_caption, identity_term):
    """Validate evidence only; semantic protection and grammar remain LLM decisions."""
    if not isinstance(value, (list, tuple)) or len(value) > 32:
        raise ValueError("preserve must be an explicit list of at most 32 source phrases")
    for phrase in value:
        identity_occurrences(source_caption, phrase)
        try:
            identity_occurrences(phrase, identity_term)
        except ValueError:
            pass
        else:
            raise ValueError("preserve must not repeat identity_term; supply additional features only")
    if len(set(value)) != len(value):
        raise ValueError("duplicate preserve phrase")
    return tuple(value)


def parse_json_object(response: str) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"non-JSON constant: {value}")

    if not isinstance(response, str):
        raise ValueError("provider response must be a JSON string")
    result = json.loads(response, object_pairs_hook=unique, parse_constant=invalid_constant)
    if not isinstance(result, dict):
        raise ValueError("response must be a JSON object")
    return result


@dataclass(frozen=True)
class ReferenceFingerprint:
    width: int
    height: int
    rgb_sha256: str
    bytes_sha256: str

    def __post_init__(self):
        if any(type(n) is not int or n < 1 for n in (self.width, self.height)):
            raise ValueError("reference dimensions must be positive integers")
        _hash(self.rgb_sha256, "rgb_sha256")
        _hash(self.bytes_sha256, "bytes_sha256")

    @classmethod
    def from_image(cls, image_bytes: bytes):
        if not isinstance(image_bytes, bytes) or not image_bytes:
            raise ValueError("nonempty reference image bytes are required")
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                rgb = image.convert("RGB")  # Same conversion as the provider, no resize/EXIF edit.
                width, height = rgb.size
                pixels = rgb.tobytes()
        except (OSError, ValueError) as error:
            raise ValueError("invalid reference image") from error
        header = f"sha256_rgb_v1:{width}:{height}:".encode("ascii")
        return cls(width, height, _sha(header + pixels), _sha(image_bytes))

    @property
    def binding(self):
        return {"algorithm": "sha256_rgb_v1", "width": self.width,
                "height": self.height, "rgb_sha256": self.rgb_sha256}

    def to_dict(self):
        return {**self.binding, "bytes_sha256": self.bytes_sha256}

    @classmethod
    def from_dict(cls, value):
        _keys(value, "algorithm width height rgb_sha256 bytes_sha256", "reference")
        _version(value["algorithm"], "sha256_rgb_v1", "reference algorithm")
        return cls(value["width"], value["height"], value["rgb_sha256"], value["bytes_sha256"])


@dataclass(frozen=True)
class AttemptRecord:
    raw_response: str | None
    validation_error: str | None

    def __post_init__(self):
        if self.raw_response is not None and not isinstance(self.raw_response, str):
            raise ValueError("raw_response must be a string or null")
        if self.validation_error is not None:
            _text(self.validation_error, "validation_error")
        elif self.raw_response is None:
            raise ValueError("successful attempt requires raw_response")

    def to_dict(self):
        return {"raw_response": self.raw_response, "validation_error": self.validation_error}

    @classmethod
    def from_dict(cls, value):
        _keys(value, "raw_response validation_error", "attempt")
        return cls(**value)


@dataclass(frozen=True)
class StageTrace:
    stage: str
    model: str
    input_sha256: str
    prompt_version: str
    prompt_sha256: str
    attempts: tuple[AttemptRecord, ...]

    def __post_init__(self):
        if self.stage not in ("caption", "contrast", "contrast_source_fallback"):
            raise ValueError("invalid trace stage")
        _text(self.model, "model")
        _hash(self.input_sha256, "input_sha256")
        version, digest = ((CAPTION_PROMPT_VERSION, CAPTION_PROMPT_SHA256) if self.stage == "caption"
                           else (CONTRAST_PROMPT_VERSION, CONTRAST_PROMPT_SHA256))
        if self.stage == "contrast_source_fallback":
            version, digest = SOURCE_FALLBACK_PROMPT_VERSION, SOURCE_FALLBACK_PROMPT_SHA256
        _version(self.prompt_version, version, "prompt version")
        _version(self.prompt_sha256, digest, "prompt hash")
        if (not isinstance(self.attempts, tuple) or not 1 <= len(self.attempts) <= 6
                or any(not isinstance(a, AttemptRecord) for a in self.attempts)):
            raise ValueError("trace requires 1..6 attempts")
        if any(a.validation_error is None for a in self.attempts[:-1]):
            raise ValueError("cannot retry a successful attempt")

    def validate_success(self, stage, input_sha256, response):
        if self.stage != stage or self.input_sha256 != input_sha256:
            raise ValueError("trace input binding mismatch")
        last = self.attempts[-1]
        if last.validation_error is not None or parse_json_object(last.raw_response) != response:
            raise ValueError("trace does not match successful response")

    def to_dict(self):
        return {"stage": self.stage, "model": self.model, "input_sha256": self.input_sha256,
                "prompt_version": self.prompt_version, "prompt_sha256": self.prompt_sha256,
                "attempts": [a.to_dict() for a in self.attempts]}

    @classmethod
    def from_dict(cls, value):
        _keys(value, "stage model input_sha256 prompt_version prompt_sha256 attempts", "trace")
        return cls(value["stage"], value["model"], value["input_sha256"], value["prompt_version"],
                   value["prompt_sha256"], tuple(AttemptRecord.from_dict(a)
                                               for a in _array(value["attempts"], "attempts")))


@dataclass(frozen=True)
class LegacyCaptionProvenance:
    path: str
    source_sha256: str
    model: str | None
    prompt_sha256: str | None

    def __post_init__(self):
        _text(self.path, "legacy path")
        _hash(self.source_sha256, "legacy source hash")
        if self.model is not None:
            _text(self.model, "legacy model")
        if self.prompt_sha256 is not None:
            _hash(self.prompt_sha256, "legacy prompt hash")

    def to_dict(self):
        return dict(path=self.path, source_sha256=self.source_sha256,
                    model=self.model, prompt_sha256=self.prompt_sha256)

    @classmethod
    def from_dict(cls, value):
        _keys(value, "path source_sha256 model prompt_sha256", "legacy provenance")
        return cls(**value)


@dataclass(frozen=True)
class CaptionRecord:
    text: str
    identity_term: str
    reference: ReferenceFingerprint | None
    origin: str
    trace: StageTrace | None = None
    imported_from: LegacyCaptionProvenance | None = None
    preserve: tuple[str, ...] = field(kw_only=True)

    def __post_init__(self):
        identity_occurrences(self.text, self.identity_term)
        if not isinstance(self.preserve, tuple):
            raise ValueError("caption preserve must be a tuple")
        preservation_phrases(self.preserve, self.text, self.identity_term)
        if self.reference is not None and not isinstance(self.reference, ReferenceFingerprint):
            raise ValueError("invalid caption reference")
        if self.origin == "generated_v2":
            if self.reference is None or not isinstance(self.trace, StageTrace) or self.imported_from is not None:
                raise ValueError("generated caption requires reference and trace, without import metadata")
            self.trace.validate_success("caption", self.input_sha256, {"source_caption": self.text, "preserve": list(self.preserve)})
        elif self.origin in ("manual", "imported_legacy"):
            if self.trace is not None:
                raise ValueError("supplied captions cannot claim a new A trace")
            if self.origin == "imported_legacy":
                if not isinstance(self.imported_from, LegacyCaptionProvenance):
                    raise ValueError("legacy caption requires explicit provenance")
            elif self.imported_from is not None:
                raise ValueError("manual caption cannot claim legacy import metadata")
        else:
            raise ValueError("unsupported caption origin")

    @property
    def input_sha256(self):
        return _input_hash({"identity_term": self.identity_term,
                            "reference": self.reference.binding if self.reference else None})

    @classmethod
    def manual(cls, text, identity_term, reference=None, *, preserve=None):
        return cls(text, identity_term, reference, "manual", preserve=preservation_phrases(preserve, text, identity_term))

    @classmethod
    def import_legacy(cls, text, identity_term, *, provenance, reference=None, status, preserve=None):
        if status != "PASS":
            raise ValueError("cannot import a failed legacy caption")
        return cls(text, identity_term, reference, "imported_legacy", imported_from=provenance, preserve=preservation_phrases(preserve, text, identity_term))

    def validate_reuse(self, *, identity_term, reference, model=None):
        if self.identity_term != identity_term:
            raise ValueError("caption identity mismatch")
        _same_reference(self.reference, reference)
        if self.trace is not None and model is not None and self.trace.model != model:
            raise ValueError("caption model mismatch")

    def to_dict(self):
        return {"caption_schema_version": CAPTION_SCHEMA_VERSION,
                "caption_policy_version": CAPTION_POLICY_VERSION if self.origin == "generated_v2" else None,
                "text": self.text, "preserve": list(self.preserve), "identity_term": self.identity_term, "origin": self.origin,
                "reference": self.reference.to_dict() if self.reference else None,
                "input_sha256": self.input_sha256,
                "trace": self.trace.to_dict() if self.trace else None,
                "imported_from": self.imported_from.to_dict() if self.imported_from else None}

    @classmethod
    def from_dict(cls, value):
        _keys(value, "caption_schema_version caption_policy_version text preserve identity_term origin reference "
              "input_sha256 trace imported_from", "caption record")
        _version(value["caption_schema_version"], CAPTION_SCHEMA_VERSION, "caption schema")
        expected = CAPTION_POLICY_VERSION if value["origin"] == "generated_v2" else None
        _version(value["caption_policy_version"], expected, "caption policy")
        result = cls(value["text"], value["identity_term"],
                     ReferenceFingerprint.from_dict(value["reference"]) if value["reference"] is not None else None,
                     value["origin"], StageTrace.from_dict(value["trace"]) if value["trace"] is not None else None,
                     LegacyCaptionProvenance.from_dict(value["imported_from"])
                     if value["imported_from"] is not None else None,
                     preserve=preservation_phrases(_array(value["preserve"], "preserve"), value["text"], value["identity_term"]))
        if value["input_sha256"] != result.input_sha256:
            raise ValueError("caption input hash mismatch")
        return result


@dataclass(frozen=True)
class Removal:
    unit: str
    negative: str

    def __post_init__(self):
        _text(self.unit, "unit")
        _text(self.negative, "negative")

    def to_dict(self):
        return {"unit": self.unit, "negative": self.negative}

    @classmethod
    def from_dict(cls, value):
        _keys(value, "unit negative", "removal")
        return cls(**value)


@dataclass(frozen=True)
class AnalysisIssue:
    kind: str
    side: str
    evidence: str
    reason: str

    def __post_init__(self):
        if self.kind != "identity_conflict":
            raise ValueError("unsupported issue kind")
        if self.side not in ("source", "target"):
            raise ValueError("unsupported issue side")
        if self.kind == "identity_conflict" and self.side != "target":
            raise ValueError("identity conflict requires target evidence")
        _text(self.evidence, "evidence")
        _text(self.reason, "reason")

    def to_dict(self):
        return dict(kind=self.kind, side=self.side, evidence=self.evidence, reason=self.reason)

    @classmethod
    def from_dict(cls, value):
        _keys(value, "kind side evidence reason", "issue")
        return cls(**value)


class ContrastAnalysisError(RuntimeError):
    """A valid model-reported issue. Not an output-validation retry."""
    def __init__(self, issues, *, trace=None):
        self.issues, self.trace = tuple(issues), trace
        super().__init__("; ".join(f"{i.kind} ({i.side}): {i.reason}" for i in self.issues))


class IdentityConflictError(ContrastAnalysisError):
    pass


class InvalidLLMOutputError(ValueError):
    def __init__(self, trace):
        self.trace = trace
        super().__init__(f"LLM output invalid after {len(trace.attempts)} attempts: "
                         f"{trace.attempts[-1].validation_error}")


@dataclass(frozen=True)
class JointContrastAnalysis:
    source_caption: str
    target_prompt: str
    identity_term: str
    source_preserved: tuple[str, ...]
    source: tuple[Removal, ...]
    target: tuple[Removal, ...]
    trace: StageTrace | None = None
    preserve: tuple[str, ...] = field(kw_only=True)

    def __post_init__(self):
        if not isinstance(self.preserve, tuple):
            raise ValueError("analysis preserve must be a tuple")
        preservation_phrases(self.preserve, self.source_caption, self.identity_term)
        identity_occurrences(self.source_caption, self.identity_term)
        identity_occurrences(self.target_prompt, self.identity_term, allow_initial_capital=True)
        if not isinstance(self.source_preserved, tuple) or self.identity_term not in self.source_preserved:
            raise ValueError("source_preserved must include exact identity_term")
        for phrase in self.source_preserved:
            identity_occurrences(self.source_caption, phrase)
        if len(set(self.source_preserved)) != len(self.source_preserved):
            raise ValueError("duplicate preserved phrase")
        for side, original, target_side in ((self.source, self.source_caption, False), (self.target, self.target_prompt, True)):
            if (not isinstance(side, tuple) or len(side) > MAX_REMOVALS
                    or any(not isinstance(row, Removal) for row in side)):
                raise ValueError(f"removals must be a tuple of at most {MAX_REMOVALS} Removal objects")
            if len({r.unit for r in side}) != len(side) or len({r.negative for r in side}) != len(side):
                raise ValueError("duplicate removal unit or negative within side")
            count = len(identity_occurrences(original, self.identity_term, allow_initial_capital=target_side))
            for row in side:
                if row.negative == original:
                    raise ValueError("negative must differ from original")
                # Antecedent expansion may repeat identity; original occurrences cannot be lost.
                if len(identity_occurrences(row.negative, self.identity_term, allow_initial_capital=target_side)) < count:
                    raise ValueError("negative must retain all identity occurrences")
        # Preserved meaning may survive grammatical glue edits; do not enforce verbatim negatives.
        if self.trace is not None:
            if not isinstance(self.trace, StageTrace):
                raise ValueError("invalid contrast trace")
            if self.trace.stage not in ("contrast", "contrast_source_fallback"):
                raise ValueError("analysis requires a contrast trace")
            if self.trace.stage == "contrast_source_fallback" and self.target:
                raise ValueError("source fallback cannot contain target removals")
            self.trace.validate_success(self.trace.stage, _input_hash(self.inputs), self.response_dict())

    @property
    def inputs(self):
        return dict(source_caption=self.source_caption, target_prompt=self.target_prompt,
                    identity_term=self.identity_term, preserve=list(self.preserve))

    def response_dict(self):
        return dict(identity_term=self.identity_term, source_preserved=list(self.source_preserved),
                    source=[r.to_dict() for r in self.source], target=[r.to_dict() for r in self.target], issues=[])

    @classmethod
    def from_response(cls, response, *, source_caption, target_prompt, identity_term, preserve=None):
        preserve = preservation_phrases(preserve, source_caption, identity_term)
        identity_occurrences(source_caption, identity_term)
        identity_occurrences(target_prompt, identity_term, allow_initial_capital=True)
        _keys(response, "identity_term source_preserved source target issues", "joint response")
        if response["identity_term"] != identity_term:
            raise ValueError("response identity mismatch")
        for key in ("source_preserved", "source", "target", "issues"):
            _array(response[key], key)
        issues = tuple(AnalysisIssue.from_dict(i) for i in response["issues"])
        if issues:
            if response["source_preserved"] or response["source"] or response["target"]:
                raise ValueError("issue response requires empty preservation and removal arrays")
            for issue in issues:
                original = source_caption if issue.side == "source" else target_prompt
                if issue.evidence not in original:
                    raise ValueError("issue evidence is absent from its original text")
            raise IdentityConflictError(issues)
        return cls(source_caption, target_prompt, identity_term, tuple(response["source_preserved"]),
                   tuple(Removal.from_dict(r) for r in response["source"]),
                   tuple(Removal.from_dict(r) for r in response["target"]), preserve=preserve)

    def to_dict(self):
        return dict(schema_version=SCHEMA_VERSION, kind="joint_contrast_analysis",
                    policy_version=CONTRAST_POLICY_VERSION, inputs=self.inputs,
                    input_sha256=_input_hash(self.inputs), response=self.response_dict(),
                    trace=self.trace.to_dict() if self.trace else None)

    @classmethod
    def from_dict(cls, value):
        _keys(value, "schema_version kind policy_version inputs input_sha256 response trace", "analysis record")
        _version(value["schema_version"], SCHEMA_VERSION, "schema version")
        _version(value["kind"], "joint_contrast_analysis", "analysis kind")
        _version(value["policy_version"], CONTRAST_POLICY_VERSION, "policy version")
        _keys(value["inputs"], "source_caption target_prompt identity_term preserve", "analysis inputs")
        if value["input_sha256"] != _input_hash(value["inputs"]):
            raise ValueError("analysis input hash mismatch")
        result = cls.from_response(value["response"], **value["inputs"])
        return replace(result, trace=StageTrace.from_dict(value["trace"])) if value["trace"] is not None else result


def _same_reference(left, right):
    if left is None or right is None:
        raise ValueError("reference binding is required for cache reuse")
    if not isinstance(right, ReferenceFingerprint) or left.binding != right.binding:
        raise ValueError("reference fingerprint mismatch")


@dataclass(frozen=True)
class SemanticContrastConditioning:
    analysis: JointContrastAnalysis
    caption: CaptionRecord
    origin: str = "source_target"
    strategy: str = "local"
    created_at: str = ""
    fallback_trace: StageTrace | None = None

    def __post_init__(self):
        _selection(self.origin, self.strategy)
        if not isinstance(self.analysis, JointContrastAnalysis) or not isinstance(self.caption, CaptionRecord):
            raise ValueError("new joint analysis and caption records are required")
        if self.analysis.trace is None:
            raise ValueError("conditioning requires a validated B trace")
        if (self.caption.text != self.analysis.source_caption
                or self.caption.identity_term != self.analysis.identity_term
                or self.caption.preserve != self.analysis.preserve):
            raise ValueError("caption/analysis binding mismatch")
        if self.analysis.trace.stage == "contrast_source_fallback":
            if self.fallback_trace is None or self.origin == "target":
                raise ValueError("source fallback requires conflict provenance and source projection")
            validate_target_conflict(self.fallback_trace, self.analysis.inputs, self.analysis.trace.model)
        elif self.fallback_trace is not None:
            raise ValueError("normal conditioning cannot carry fallback provenance")
        if self.created_at == "":
            object.__setattr__(self, "created_at", datetime.now(timezone.utc).isoformat())
        _text(self.created_at, "created_at")
        if datetime.fromisoformat(self.created_at).utcoffset() is None:
            raise ValueError("created_at requires timezone")

    @property
    def selected(self):
        return tuple((side, getattr(self.analysis, side)) for side in ("source", "target")
                     if self.origin in (side, "source_target"))

    @property
    def direction_count(self):
        return sum(len(rows) for _, rows in self.selected)

    def validate_reuse(self, *, identity_term, target_prompt, reference, source_caption=None, model=None, preserve=None):
        if self.analysis.identity_term != identity_term or self.analysis.target_prompt != target_prompt:
            raise ValueError("cached identity or target mismatch")
        if source_caption is not None and self.analysis.source_caption != source_caption:
            raise ValueError("cached source caption mismatch")
        if preserve is not None and preservation_phrases(preserve, self.analysis.source_caption, identity_term) != self.caption.preserve:
            raise ValueError("cached preserve mismatch")
        self.caption.validate_reuse(identity_term=identity_term, reference=reference, model=model)
        if model is not None and self.analysis.trace.model != model:
            raise ValueError("contrast model mismatch")

    def to_dict(self):
        value = dict(kind="semantic_local_contrast", schema_version=SCHEMA_VERSION,
                    policy_version=CONTRAST_POLICY_VERSION, inputs=self.analysis.inputs,
                    input_sha256=_input_hash(self.analysis.inputs),
                    reference=self.caption.reference.to_dict() if self.caption.reference else None,
                    caption=self.caption.to_dict(), analysis=self.analysis.response_dict(),
                    llm={"caption": self.caption.trace.to_dict() if self.caption.trace else None,
                         "contrast": self.analysis.trace.to_dict()},
                    selection={"origin": self.origin, "strategy": self.strategy},
                    limits={"max_removals_per_side": MAX_REMOVALS}, created_at=self.created_at)
        if self.fallback_trace is not None:
            value["kind"] = "semantic_source_fallback"
            value["fallback"] = {"policy_version": "target_conflict_source_only_v1",
                                 "original_contrast_trace": self.fallback_trace.to_dict()}
        return value

    @classmethod
    def from_dict(cls, value):
        fallback = isinstance(value, dict) and value.get("kind") == "semantic_source_fallback"
        _keys(value, "kind schema_version policy_version inputs input_sha256 reference caption analysis "
              "llm selection limits created_at" + (" fallback" if fallback else ""), "conditioning record")
        _version(value["kind"], "semantic_source_fallback" if fallback else "semantic_local_contrast", "conditioning kind")
        original_trace = None
        if fallback:
            _keys(value["fallback"], "policy_version original_contrast_trace", "fallback")
            _version(value["fallback"]["policy_version"], "target_conflict_source_only_v1", "fallback policy")
            original_trace = StageTrace.from_dict(value["fallback"]["original_contrast_trace"])
        _version(value["schema_version"], SCHEMA_VERSION, "schema version")
        _version(value["policy_version"], CONTRAST_POLICY_VERSION, "policy version")
        _keys(value["inputs"], "source_caption target_prompt identity_term preserve", "inputs")
        _keys(value["llm"], "caption contrast", "llm")
        _keys(value["selection"], "origin strategy", "selection")
        _keys(value["limits"], "max_removals_per_side", "limits")
        _version(value["limits"]["max_removals_per_side"], MAX_REMOVALS, "removal limit")
        _text(value["created_at"], "created_at")
        if value["input_sha256"] != _input_hash(value["inputs"]):
            raise ValueError("conditioning input hash mismatch")
        caption = CaptionRecord.from_dict(value["caption"])
        if (value["reference"] != (caption.reference.to_dict() if caption.reference else None)
                or value["llm"]["caption"] != (caption.trace.to_dict() if caption.trace else None)):
            raise ValueError("caption metadata mismatch")
        analysis = JointContrastAnalysis.from_response(value["analysis"], **value["inputs"])
        analysis = replace(analysis, trace=StageTrace.from_dict(value["llm"]["contrast"]))
        return cls(analysis, caption, **value["selection"], created_at=value["created_at"], fallback_trace=original_trace)


def _selection(origin, strategy):
    if origin not in ORIGINS:
        raise ValueError("origin must be source, target, or source_target")
    if strategy != "local":
        raise ValueError("new semantic contrasts support local only; factorial is a legacy research option")


class SemanticLLM:
    """New A and joint B only. No analyze_source/analyze_target fallback."""
    def __init__(self, provider: StructuredLLMProvider, *, max_retries: int = 2, model: str | None = None):
        if type(max_retries) is not int or not 0 <= max_retries <= 5:
            raise ValueError("max_retries must be an integer in [0, 5]")
        self.provider = provider
        self.max_retries = max_retries
        provider_model = getattr(provider, "model", None)
        if model is not None and provider_model is not None and model != provider_model:
            raise ValueError("explicit model must match provider model")
        self.model = _text(model if model is not None else provider_model or "unspecified", "model")

    def _run(self, *, stage, payload, validate, image_bytes=None, binding=None):
        prompt, schema, version, digest = (
            (CAPTION_SYSTEM_PROMPT, CAPTION_SCHEMA, CAPTION_PROMPT_VERSION, CAPTION_PROMPT_SHA256)
            if stage == "caption" else
            (CONTRAST_SYSTEM_PROMPT, CONTRAST_SCHEMA, CONTRAST_PROMPT_VERSION, CONTRAST_PROMPT_SHA256))
        if stage == "contrast_source_fallback":
            prompt, version, digest = SOURCE_FALLBACK_SYSTEM_PROMPT, SOURCE_FALLBACK_PROMPT_VERSION, SOURCE_FALLBACK_PROMPT_SHA256
        attempts = []
        feedback = None

        def trace():
            return StageTrace(stage, self.model, _input_hash(binding if binding is not None else payload),
                              version, digest, tuple(attempts))

        for attempt in range(self.max_retries + 1):
            raw = None
            try:
                raw = self.provider.complete(system_prompt=prompt, payload=dict(payload), schema=schema,
                                             image_bytes=image_bytes, feedback=feedback)
                result = validate(parse_json_object(raw))
            except ContrastAnalysisError as error:
                attempts.append(AttemptRecord(raw, None))
                error.trace = trace()
                raise
            except ValueError as error:
                feedback = str(error).strip() or type(error).__name__
                attempts.append(AttemptRecord(raw if isinstance(raw, str) else None, feedback))
                if attempt == self.max_retries:
                    raise InvalidLLMOutputError(trace()) from error
            else:
                attempts.append(AttemptRecord(raw, None))
                return result, trace()

    def caption(self, image_bytes: bytes, identity_term: str) -> CaptionRecord:
        identity_occurrences(identity_term, identity_term)
        reference = ReferenceFingerprint.from_image(image_bytes)

        def validate(value):
            _keys(value, "source_caption preserve", "caption response")
            identity_occurrences(value["source_caption"], identity_term)
            preservation_phrases(_array(value["preserve"], "preserve"), value["source_caption"], identity_term)
            return value

        caption, trace = self._run(stage="caption", payload={"identity_term": identity_term},
                                   validate=validate, image_bytes=image_bytes,
                                   binding={"identity_term": identity_term, "reference": reference.binding})
        return CaptionRecord(caption["source_caption"], identity_term, reference, "generated_v2", trace,
                             preserve=tuple(caption["preserve"]))

    def contrast(self, source_caption: str, target_prompt: str, identity_term: str, preserve=None) -> JointContrastAnalysis:
        identity_occurrences(source_caption, identity_term)
        identity_occurrences(target_prompt, identity_term, allow_initial_capital=True)
        payload = dict(source_caption=source_caption, target_prompt=target_prompt, identity_term=identity_term,
                       preserve=list(preservation_phrases(preserve, source_caption, identity_term)))
        analysis, trace = self._run(stage="contrast", payload=payload,
                                    validate=lambda value: JointContrastAnalysis.from_response(value, **payload))
        return replace(analysis, trace=trace)


def prepare_semantic_conditioning(*, llm: SemanticLLM, identity_term: str, target_prompt: str,
                                  reference_image_bytes: bytes | None = None, source_caption: str | None = None,
                                  caption_record: CaptionRecord | None = None, preserve=None,
                                  origin: str = "source_target", strategy: str = "local") -> SemanticContrastConditioning:
    """Prepare only: no encoding/generation. Every origin performs the same joint B.

    Manual text may be analyzed without an image; such unbound records cannot be
    reused as reference-bound caches. Generation must provide the actual image.
    """
    _selection(origin, strategy)
    identity_occurrences(target_prompt, identity_term, allow_initial_capital=True)
    if source_caption is not None and caption_record is not None:
        raise ValueError("source_caption and caption_record are mutually exclusive")
    if source_caption is not None:
        identity_occurrences(source_caption, identity_term)
        preservation_phrases(preserve, source_caption, identity_term)
    elif preserve is not None:
        raise ValueError("preserve is only supplied with a manual source_caption; caption records carry their own")
    reference = (ReferenceFingerprint.from_image(reference_image_bytes)
                 if reference_image_bytes is not None else None)
    if caption_record is not None:
        if not isinstance(caption_record, CaptionRecord):
            raise ValueError("caption_record must be a new CaptionRecord")
        caption_record.validate_reuse(identity_term=identity_term, reference=reference, model=llm.model)
    elif source_caption is not None:
        caption_record = CaptionRecord.manual(source_caption, identity_term, reference, preserve=preserve)
    else:
        if reference is None:
            raise ValueError("source caption or reference image is required for every origin, including target")
        caption_record = llm.caption(reference_image_bytes, identity_term)
    analysis = llm.contrast(caption_record.text, target_prompt, identity_term, preserve=caption_record.preserve)
    return SemanticContrastConditioning(analysis, caption_record, origin, strategy)


def validate_target_conflict(trace, inputs, model):
    """Validate prior failure against these exact inputs before any paid fallback."""
    if not isinstance(trace, StageTrace) or trace.stage != "contrast" or trace.model != model:
        raise ValueError("fallback requires an original joint contrast trace with the same model")
    response = parse_json_object(trace.attempts[-1].raw_response)
    trace.validate_success("contrast", _input_hash(inputs), response)
    try:
        JointContrastAnalysis.from_response(response, **inputs)
    except IdentityConflictError as error:
        if any(issue.side != "target" for issue in error.issues):
            raise ValueError("fallback supports target-side identity conflicts only") from error
    else:
        raise ValueError("fallback requires a validated target identity conflict")


def prepare_source_only_fallback(*, llm, caption_record, target_prompt,
                                 reference_image_bytes, failed_trace):
    """Explicit opt-in: one additional source analysis, never hidden normal B retries."""
    caption_record.validate_reuse(identity_term=caption_record.identity_term,
                                 reference=ReferenceFingerprint.from_image(reference_image_bytes), model=llm.model)
    inputs = dict(source_caption=caption_record.text, target_prompt=target_prompt,
                  identity_term=caption_record.identity_term, preserve=list(caption_record.preserve))
    validate_target_conflict(failed_trace, inputs, llm.model)

    def validate(response):
        if response.get("target") != [] or response.get("issues") != []:
            raise ValueError("source fallback requires empty target and issues arrays")
        return JointContrastAnalysis.from_response(response, **inputs)

    analysis, trace = llm._run(stage="contrast_source_fallback", payload=inputs, validate=validate)
    return SemanticContrastConditioning(replace(analysis, trace=trace), caption_record,
                                       origin="source", fallback_trace=failed_trace)
