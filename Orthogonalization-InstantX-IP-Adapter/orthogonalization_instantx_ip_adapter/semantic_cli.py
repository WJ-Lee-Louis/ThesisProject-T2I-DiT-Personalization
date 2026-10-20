"""CPU-side semantic CLI preparation, shared by Modal and offline tests."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import warnings

from .llm_provider import ResponsesJSONProvider
from .semantic_contrasts import (
    CaptionRecord, LegacyCaptionProvenance, ReferenceFingerprint, SemanticContrastConditioning,
    SemanticLLM, identity_occurrences, preservation_phrases, parse_json_object, prepare_semantic_conditioning,
)


def resolve_contrast_options(*, contrast_origin="", component_origin="", contrast_analysis_path="",
                             component_analysis_path="", component_strategy="", component_max_contexts=None):
    if contrast_origin and component_origin or contrast_analysis_path and component_analysis_path:
        raise ValueError("cannot specify both new and deprecated option aliases")
    if component_max_contexts is not None:
        raise ValueError("component_max_contexts is not used by local contrasts; remove it")
    if component_strategy not in ("", "local"):
        raise ValueError("semantic contrasts support local only; factorial is a separate legacy research path")
    if component_origin or component_analysis_path or component_strategy:
        warnings.warn("component-* options are deprecated; use contrast-* options and schema 4", FutureWarning)
    origin = contrast_origin or component_origin or None
    if origin is not None and origin not in ("source", "target", "source_target"):
        raise ValueError("invalid contrast origin")
    return origin, contrast_analysis_path or component_analysis_path


def validate_caption_options(*, identity_term, source_caption="", caption_cache_path="",
                             import_legacy_caption="", analysis_path="", preserve_json=""):
    if preserve_json:
        try:
            phrases = json.loads(preserve_json)
        except (ValueError, TypeError) as error:
            raise ValueError("preserve_json must be a JSON array") from error
        if not isinstance(phrases, list):
            raise ValueError("preserve_json must be a JSON array")
        if not (source_caption or import_legacy_caption):
            raise ValueError("preserve_json requires a manual or imported caption")
        if source_caption:
            preservation_phrases(phrases, source_caption, identity_term)
    elif source_caption or import_legacy_caption:
        raise ValueError("manual/imported captions require preserve_json (use [] for identity only)")
    count = sum(bool(v) for v in (source_caption, caption_cache_path, import_legacy_caption))
    if count > 1 or (count and analysis_path):
        raise ValueError("caption options are mutually exclusive and cannot mix with full analysis cache")
    if count and not identity_term:
        raise ValueError("caption options require identity_term")


def _read(path):
    return parse_json_object(Path(path).read_text(encoding="utf-8"))


def prepare_cli_conditioning(*, image_bytes, target_prompt, identity_term="", source_caption="",
                             caption_cache_path="", import_legacy_caption="", analysis_path="",
                             origin=None, model=None, provider_factory=None, preserve_json=""):
    validate_caption_options(identity_term=identity_term, source_caption=source_caption,
                             caption_cache_path=caption_cache_path, import_legacy_caption=import_legacy_caption,
                             analysis_path=analysis_path, preserve_json=preserve_json)
    if origin is not None and origin not in ("source", "target", "source_target"):
        raise ValueError("invalid contrast origin")
    reference = ReferenceFingerprint.from_image(image_bytes)
    if analysis_path:
        cached = SemanticContrastConditioning.from_dict(_read(analysis_path))
        identity = identity_term or cached.analysis.identity_term
        cached.validate_reuse(identity_term=identity, target_prompt=target_prompt, reference=reference, model=model)
        return replace(cached, origin=origin) if origin is not None else cached
    identity_occurrences(target_prompt, identity_term, allow_initial_capital=True)
    record = None
    if source_caption:
        identity_occurrences(source_caption, identity_term)
    elif caption_cache_path:
        record = CaptionRecord.from_dict(_read(caption_cache_path))
        record.validate_reuse(identity_term=identity_term, reference=reference, model=model)
    elif import_legacy_caption:
        # One explicitly selected historical record, not an aggregate's arbitrary first row.
        raw = Path(import_legacy_caption).read_bytes()
        old = parse_json_object(raw.decode("utf-8"))
        if old.get("identity_term") != identity_term:
            raise ValueError("legacy caption identity mismatch")
        if old.get("image_sha256") != reference.bytes_sha256:
            raise ValueError("legacy caption original image hash mismatch")
        provenance = LegacyCaptionProvenance(str(Path(import_legacy_caption).resolve()),
                                             hashlib.sha256(raw).hexdigest(), old.get("requested_model"),
                                             old.get("prompt_sha256"))
        record = CaptionRecord.import_legacy(old.get("source_caption"), identity_term, reference=reference,
                                             provenance=provenance, status=old.get("status"), preserve=json.loads(preserve_json))
    llm = SemanticLLM((provider_factory or ResponsesJSONProvider.from_env)(model))
    return prepare_semantic_conditioning(llm=llm, identity_term=identity_term, target_prompt=target_prompt,
                                         reference_image_bytes=image_bytes, source_caption=source_caption or None,
                                         caption_record=record, origin=origin or "source_target",
                                         preserve=json.loads(preserve_json) if source_caption else None)
