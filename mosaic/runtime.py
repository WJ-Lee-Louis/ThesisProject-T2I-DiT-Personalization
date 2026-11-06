"""Adapter for the unmodified official MOSAIC single-reference inference path."""
import importlib.util
import sys
from pathlib import Path


class Mosaic:
    def __init__(self, paths, config):
        import torch
        from diffusers import FluxPipeline
        root = Path(__file__).resolve().parent / 'upstream'
        # The official script imports src.flux_omini and utils from its own root.
        sys.path.insert(0, str(root))
        from src.flux_omini import Condition, generate
        spec = importlib.util.spec_from_file_location('mosaic_official_utils', root/'utils.py')
        utils = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(utils)
        self.condition_type = Condition
        self.generate_official = generate
        self.process_image = utils.process_image
        self.config = config
        self.pipe = FluxPipeline.from_pretrained(paths['flux'], torch_dtype=torch.bfloat16).to('cuda')
        self.pipe.load_lora_weights(paths['mosaic'], weight_name='subject_512.safetensors', adapter_name='subject')
        self.pipe.set_adapters(['subject'], [config['adapter_weight']])
        self.pipe.set_progress_bar_config(disable=True)

    def generate(self, row, reference):
        import torch
        cfg = self.config
        appearance = self.process_image(str(reference), target_size=cfg['reference_size'],
            pad_color=tuple(cfg['pad_color']), scale=cfg['reference_scale'])
        condition = self.condition_type(appearance, cfg['adapter_name'], cfg['position_delta'])
        result = self.generate_official(self.pipe, prompt=row['prompt'], conditions=[condition],
            num_inference_steps=cfg['steps'], num_images_per_prompt=1,
            height=cfg['height'], width=cfg['width'], guidance_scale=cfg['guidance'],
            generator=torch.Generator('cuda').manual_seed(row['seed']),
            image_guidance_scale=cfg['image_guidance_scale'], kv_cache=cfg['kv_cache'])
        if len(result.images) != 1:
            raise ValueError('Official MOSAIC inference returned an unexpected image count')
        return result.images[0], row['prompt']
