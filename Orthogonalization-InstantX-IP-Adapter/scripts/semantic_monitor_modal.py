"""Isolated GPU runner using the existing inference function and pinned models."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

import modal
from modal_inference import image as base_image, hf_cache
from shared.modal_runtime import HF_SECRET_NAME, DEFAULT_GPU

app = modal.App('orthogonalization-semantic-monitor')
image = base_image.add_local_file(ROOT / 'modal_inference.py', '/root/modal_inference.py')


@app.function(image=image, gpu=DEFAULT_GPU, volumes={'/cache': hf_cache},
              secrets=[modal.Secret.from_name(HF_SECRET_NAME)], timeout=7200)
def run_cases(requests):
    import time
    from orthogonalization_instantx_ip_adapter import inference
    from orthogonalization_instantx_ip_adapter.semantic_contrasts import SemanticContrastConditioning, ReferenceFingerprint
    # Validate every payload/reference before loading a model or requesting model artifacts.
    checked = []
    for request in requests:
        conditioning = SemanticContrastConditioning.from_dict(request['contrast_analysis_payload'])
        conditioning.validate_reuse(identity_term=conditioning.analysis.identity_term,
            target_prompt=conditioning.analysis.target_prompt,
            reference=ReferenceFingerprint.from_image(request['image_bytes']))
        checked.append((request, conditioning))
    model = None
    for request, conditioning in checked:
        started = time.monotonic()
        event = dict(case_id=request['case_id'], strategy='local')
        try:
            if model is None:
                model = inference.load_ip_adapter_model()
            # Reuse one model without permanently replacing the module loader.
            original_loader = inference.load_ip_adapter_model
            try:
                inference.load_ip_adapter_model = lambda: model
                png, diagnostics = inference.generate_png_bytes(
                    prompt=conditioning.analysis.target_prompt, reference_image_url='',
                    reference_image_bytes=request['image_bytes'], ip_adapter_scale=0.7, guidance_scale=3.5,
                    num_inference_steps=24, seed=42, width=1024, height=1024,
                    orthogonalization_mode='semantic_local', orthogonalization_strength=1.0,
                    object_prompt='', object_behavior_prompt='', object_background_prompt='',
                    object_behavior_background_prompt='', remove_behavior=True, remove_background=True,
                    include_interaction=False, restore_visual_norm=True, mask_single_text_rows=True,
                    contrast_analysis_payload=conditioning.to_dict())
            finally:
                inference.load_ip_adapter_model = original_loader
            event.update(status='generated', png_bytes=png, diagnostics=diagnostics)
        except Exception as error:
            event.update(status='generation_failed', error=f'{type(error).__name__}: {error}')
        event['elapsed_seconds'] = time.monotonic() - started
        yield event
