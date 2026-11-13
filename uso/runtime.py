"""Official USO content and style routes, with immutable checkpoint paths."""
import os
from pathlib import Path


class Engine:
    def __init__(self, config, paths):
        import torch
        from transformers import SiglipVisionModel, SiglipImageProcessor
        os.environ.update(FLUX_DEV=str(Path(paths['flux']) / 'flux1-dev.safetensors'),
                          AE=str(Path(paths['flux']) / 'ae.safetensors'),
                          T5=paths['t5'], CLIP=paths['clip'],
                          LORA=str(Path(paths['lora']) / 'uso_flux_v1.0/dit_lora.safetensors'),
                          PROJECTION_MODEL=str(Path(paths['lora']) / 'uso_flux_v1.0/projector.safetensors'),
                          HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        from uso.flux.pipeline import USOPipeline
        from uso.flux import util
        util.configs['flux-dev'].ckpt_path = os.environ['FLUX_DEV']
        util.configs['flux-dev'].ae_path = os.environ['AE']
        self.pipe = USOPipeline('flux-dev', torch.device('cuda'), offload=False,
                                only_lora=True, lora_rank=128, hf_download=False)
        self.config = config
        self.processor = SiglipImageProcessor.from_pretrained(paths['siglip'], local_files_only=True)
        self.vision = SiglipVisionModel.from_pretrained(paths['siglip'], local_files_only=True).eval().to('cuda')
        self.pipe.model.vision_encoder = self.vision
        for module in [self.pipe.model, self.pipe.t5, self.pipe.clip, self.pipe.ae]:
            if any(p.is_meta for p in module.parameters()):
                raise ValueError('USO has uninitialized meta parameters')
            module.eval().requires_grad_(False)

    def generate(self, row, reference):
        import torch
        from transformers import set_seed
        from uso.flux.pipeline import preprocess_ref
        set_seed(row['seed'])
        style = row['category'] == 'style'
        refs = [] if style else [preprocess_ref(reference, self.config['reference_resolution'])]
        with torch.inference_mode():
            styles = [self.processor(reference, return_tensors='pt').to(self.pipe.device)] if style else None
            # The official decoder is offloaded after every inference, even without cpu_offload.
            self.pipe.ae.decoder.to(self.pipe.device)
            return self.pipe(prompt=row['prompt'], width=1024, height=1024, guidance=4.0,
                             num_steps=25, seed=row['seed'], ref_imgs=refs, pe='d', siglip_inputs=styles)
