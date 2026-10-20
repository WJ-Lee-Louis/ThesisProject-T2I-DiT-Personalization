"""Offline full-run binding/serialization/encoder-wiring verification, not an LLM evaluation."""
from dataclasses import replace
import hashlib
import torch

from run_full_semantic_benchmark import (
    ROOT, OUT, LEDGER, MODEL, read, save, validate_references,
    CaptionRecord, SemanticContrastConditioning, ReferenceFingerprint, contrast_plan,
)
from orthogonalization_instantx_ip_adapter.contrast_directions import build_contrast_directions


def vector(text):
    return torch.tensor(list(hashlib.sha256(text.encode("utf-8")).digest()), dtype=torch.float32) - 127.5


def main():
    refs = read(LEDGER)
    validate_references(refs)
    checked = plans = directions = 0
    blocked = []
    caption_count = 0
    for i, ref in enumerate(refs):
        fingerprint = ReferenceFingerprint.from_image((ROOT / ref["image_path"]).read_bytes())
        caption_path = OUT / "captions" / f"{i:03d}.json"
        if not caption_path.exists():
            blocked.append(f"A-{i:03d}")
            continue
        caption = CaptionRecord.from_dict(read(caption_path))
        caption.validate_reuse(identity_term=ref["identity_term"], reference=fingerprint, model=MODEL)
        caption_count += 1
        for j, target in enumerate(ref["target_prompts"]):
            path = OUT / "conditioning" / f"{i:03d}-{j:02d}.json"
            if not path.exists():
                blocked.append(f"B-{i:03d}-{j:02d}")
                continue
            result = SemanticContrastConditioning.from_dict(read(path))
            result.validate_reuse(identity_term=ref["identity_term"], reference=fingerprint,
                                  target_prompt=target, source_caption=caption.text,
                                  preserve=caption.preserve, model=MODEL)
            assert SemanticContrastConditioning.from_dict(result.to_dict()) == result
            if result.fallback_trace is not None:
                assert not result.analysis.target
                try:
                    replace(result, origin="target")
                except ValueError:
                    pass
                else:
                    raise AssertionError("fallback must retain source projection")
            for origin in (("source", "source_target") if result.fallback_trace else ("source", "target", "source_target")):
                selected = replace(result, origin=origin)
                plan = contrast_plan(selected)
                expected_texts = list(dict.fromkeys(t for row in plan for t in (row.positive, row.negative)))
                calls = []

                def encode(texts):
                    calls.append(list(texts))
                    return torch.stack([vector(t) for t in texts])

                actual = build_contrast_directions(selected, encode)
                assert calls == [expected_texts or [ref["identity_term"]]]
                assert actual.values.shape[0] == len(plan) == selected.direction_count
                for k, row in enumerate(plan):
                    assert row.positive == (caption.text if row.origin == "source" else target)
                    positive = torch.nn.functional.normalize(vector(row.positive), dim=0)
                    negative = torch.nn.functional.normalize(vector(row.negative), dim=0)
                    expected = torch.nn.functional.normalize(positive - negative, dim=0)
                    torch.testing.assert_close(actual.values[k], expected)
                plans += 1
            directions += result.direction_count
            checked += 1
    report = dict(captions_verified=caption_count, pairs_verified=checked,
                  origin_plans_verified=plans, generated_direction_rows=directions,
                  missing=blocked, encoder="deterministic SHA256 mock; not SigLIP",
                  semantic_accuracy_evaluated=False, image_quality_evaluated=False)
    save(OUT / "verification.json", report)
    print(report)


if __name__ == "__main__":
    main()
