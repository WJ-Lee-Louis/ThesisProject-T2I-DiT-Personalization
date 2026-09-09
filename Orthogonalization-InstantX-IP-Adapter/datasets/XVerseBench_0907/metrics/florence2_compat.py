"""Compatibility repair and diagnostics for XVerse's vendored Florence-2.

The pinned XVerse revision vendors an older Florence-2 model implementation
that declares tied language weights but does not implement the `_tie_weights`
hook present in Microsoft's corrected implementation.  The public checkpoint
does not store a separate LM-head tensor, so inference must share the loaded
token embedding with the output head.
"""

from __future__ import annotations

import os
from typing import Any


FLORENCE2_MODEL_ID = "microsoft/Florence-2-large"
FLORENCE2_REVISION = "21a599d414c4d928c9032694c424fb94458e3594"
FLORENCE2_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def pin_florence2_checkpoint() -> dict[str, str]:
    """Resolve the exact Florence-2 snapshot used by the validated evaluator."""

    from huggingface_hub import snapshot_download

    snapshot_path = snapshot_download(
        repo_id=FLORENCE2_MODEL_ID,
        revision=FLORENCE2_REVISION,
        allow_patterns=list(FLORENCE2_FILES),
    )
    os.environ["FLORENCE2_MODEL_PATH"] = snapshot_path
    return {
        "repo_id": FLORENCE2_MODEL_ID,
        "revision": FLORENCE2_REVISION,
        "snapshot_path": snapshot_path,
    }


def _language_components(object_detector):
    try:
        model = object_detector.detector.florence2_model
        language_model = model.language_model
        shared = language_model.model.shared
        encoder = language_model.model.encoder.embed_tokens
        decoder = language_model.model.decoder.embed_tokens
        lm_head = language_model.lm_head
    except AttributeError as error:
        raise RuntimeError(
            "Unexpected XVerse Florence-2 model structure; refusing to apply "
            "the weight-tying compatibility repair"
        ) from error
    return model, language_model, shared, encoder, decoder, lm_head


def _same_storage(left, right) -> bool:
    return left.data_ptr() == right.data_ptr()


def florence2_weight_report(object_detector) -> dict[str, Any]:
    """Describe whether the three language projections share loaded storage."""

    model, language_model, shared, encoder, decoder, lm_head = _language_components(
        object_detector
    )
    return {
        "model_class": type(model).__name__,
        "language_model_class": type(language_model).__name__,
        "tie_word_embeddings": bool(
            getattr(language_model.config, "tie_word_embeddings", False)
        ),
        "shared_shape": list(shared.weight.shape),
        "encoder_tied_to_shared": _same_storage(
            encoder.weight, shared.weight
        ),
        "decoder_tied_to_shared": _same_storage(
            decoder.weight, shared.weight
        ),
        "lm_head_tied_to_shared": _same_storage(
            lm_head.weight, shared.weight
        ),
    }


def repair_xverse_florence2_weight_tying(object_detector) -> dict[str, Any]:
    """Apply Microsoft's missing Florence-2 weight-sharing hook to XVerse."""

    _, language_model, shared, encoder, decoder, lm_head = _language_components(
        object_detector
    )
    before = florence2_weight_report(object_detector)

    # Equivalent to the `_tie_weights` hook in Microsoft's corrected
    # Florence2LanguageForConditionalGeneration implementation.
    encoder.weight = shared.weight
    decoder.weight = shared.weight
    lm_head.weight = shared.weight

    after = florence2_weight_report(object_detector)
    if not all(
        after[key]
        for key in (
            "encoder_tied_to_shared",
            "decoder_tied_to_shared",
            "lm_head_tied_to_shared",
        )
    ):
        raise RuntimeError("Florence-2 weight-tying repair did not take effect")
    report = {
        "repair": "xverse_vendored_florence2_missing_tie_weights_hook",
        "checkpoint": getattr(object_detector, "_florence2_checkpoint", None),
        "before": before,
        "after": after,
    }
    object_detector._florence2_compat_report = report
    return report


def run_florence2_diagnostic(
    object_detector,
    image,
    text_input: str,
    task_prompt: str = "<OPEN_VOCABULARY_DETECTION>",
) -> dict[str, Any]:
    """Run XVerse's exact generation settings while retaining raw output."""

    import torch

    florence = object_detector.detector
    model = florence.florence2_model
    processor = florence.florence2_processor
    prompt = task_prompt if text_input is None else task_prompt + text_input
    inputs = processor(
        text=prompt,
        images=image,
        max_length=1024,
        truncation=True,
        return_tensors="pt",
    ).to(florence.device, florence.torch_dtype)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float32):
        generated_ids = model.generate(
            input_ids=inputs["input_ids"].to(florence.device),
            pixel_values=inputs["pixel_values"].to(florence.device),
            max_new_tokens=768,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=False
    )[0]
    parsed = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height),
    )
    task_result = parsed.get(task_prompt, {})
    return {
        "task_prompt": task_prompt,
        "text_input": text_input,
        "generated_text": generated_text,
        "parsed": parsed,
        "bbox_count": len(task_result.get("bboxes", [])),
        "labels": list(task_result.get("bboxes_labels", [])),
    }


__all__ = [
    "FLORENCE2_MODEL_ID",
    "FLORENCE2_REVISION",
    "florence2_weight_report",
    "pin_florence2_checkpoint",
    "repair_xverse_florence2_weight_tying",
    "run_florence2_diagnostic",
]
