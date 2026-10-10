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
    import torch
    from huggingface_hub import hf_hub_download
    from orthogonalization_instantx_ip_adapter import inference
    from orthogonalization_instantx_ip_adapter.config import BASE_MODEL_ID, BASE_MODEL_REVISION
    from orthogonalization_instantx_ip_adapter.model_loader import get_hf_token, load_ip_adapter_model
    from orthogonalization_instantx_ip_adapter.semantic_components import SemanticConditioning
    from orthogonalization_instantx_ip_adapter.component_directions import build_component_directions
    from orthogonalization_instantx_ip_adapter.orthogonalization import estimate_orthonormal_basis, normalize_fp32

    try:
        print('Checking pinned FLUX access before model allocation', flush=True)
        hf_hub_download(BASE_MODEL_ID, 'model_index.json', revision=BASE_MODEL_REVISION, token=get_hf_token())
        print('Loading existing InstantX/FLUX model once for all cases', flush=True)
        model = load_ip_adapter_model()
        # Reuse this same loaded model through the existing generation entry point.
        inference.load_ip_adapter_model = lambda: model
    except Exception as error:
        yield dict(status='gpu_initialization_failed', error=f'{type(error).__name__}: {error}')
        return

    for request in requests:
        for strategy in ('local', 'factorial'):
            started = time.monotonic()
            event = dict(case_id=request['case_id'], strategy=strategy)
            try:
                print('Generating', request['case_id'], strategy, flush=True)
                conditioning = SemanticConditioning.from_dict(request[strategy])
                with torch.inference_mode():
                    directions = build_component_directions(conditioning, model._encode_component_text_pool)
                    basis = estimate_orthonormal_basis(directions.values).basis
                    fixed_prompts = [a.prompt_for_subset([]) for _, a in conditioning.selected]
                    fixed = normalize_fp32(model._encode_component_text_pool(fixed_prompts))
                event['text_geometry'] = dict(names=directions.names,
                    raw_difference_norms=directions.raw_norms.cpu().tolist(),
                    direction_cosines=(directions.values @ directions.values.T).cpu().tolist(),
                    fixed_context_prompts=fixed_prompts,
                    fixed_context_projection_fraction=(fixed @ basis).norm(dim=-1).cpu().tolist())
                png, diagnostics = inference.generate_png_bytes(
                    prompt=conditioning.target.text, reference_image_url='', reference_image_bytes=request['image_bytes'],
                    ip_adapter_scale=0.7, guidance_scale=3.5, num_inference_steps=24, seed=42,
                    width=1024, height=1024, orthogonalization_mode='components', orthogonalization_strength=1.0,
                    object_prompt='', object_behavior_prompt='', object_background_prompt='',
                    object_behavior_background_prompt='', remove_behavior=True, remove_background=True,
                    include_interaction=False, restore_visual_norm=True, mask_single_text_rows=True,
                    component_analysis_payload=conditioning.to_dict())
                event.update(status='generated', png_bytes=png, diagnostics=diagnostics)
            except Exception as error:
                event.update(status='generation_failed', error=f'{type(error).__name__}: {error}')
            event['elapsed_seconds'] = time.monotonic() - started
            yield event
