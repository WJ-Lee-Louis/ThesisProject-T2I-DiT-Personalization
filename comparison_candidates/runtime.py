"""Thin adapters around pinned official inference implementations."""
import ast
import os
import re
import sys
from pathlib import Path


def caption(row):
    match = re.search(re.escape(row['subject_word']), row['prompt'], re.IGNORECASE)
    if not match:
        raise ValueError('Subject phrase missing from original target prompt')
    return match.group()


class Diptych:
    def __init__(self, paths, config):
        import torch
        from transformers import AutoProcessor, AutoModelForMaskGeneration, pipeline
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'diptych/upstream'))
        import diptych_prompting_inference as official
        self.official = official
        self.config = config
        control = official.FluxControlNetModel.from_pretrained(paths['controlnet'], torch_dtype=torch.bfloat16)
        self.pipe = official.FluxControlNetInpaintingPipeline.from_pretrained(
            paths['flux'], controlnet=control, torch_dtype=torch.bfloat16).to('cuda')
        self.segmentator = AutoModelForMaskGeneration.from_pretrained(paths['sam']).cuda()
        self.processor = AutoProcessor.from_pretrained(paths['sam'])
        self.detector = pipeline(model=paths['detector'], task='zero-shot-object-detection', device=0)
        self.cache = {}
        self.pipe.set_progress_bar_config(disable=True)

    def generate(self, row, reference):
        import numpy as np
        import torch
        from PIL import Image
        subject = row['subject_word']
        # Official 8-pixel border, cropped from the generated right panel.
        offset = self.config['pixel_offset']
        width = self.config['width'] + offset*2
        height = self.config['height'] + offset*2
        if row['reference_id'] not in self.cache:
            image = Image.open(reference).convert('RGB').resize((width, height))
            if row['category'] != 'style':
                arr, detections = self.official.grounded_segmentation(
                    self.detector, self.segmentator, self.processor, image=image,
                    labels=[subject], threshold=0.3, polygon_refinement=True)
                if not detections:
                    raise ValueError(f'Official detector found no subject: {subject}')
                alpha = np.expand_dims(detections[0].mask / 255, axis=-1)
                image = Image.fromarray((arr * alpha + np.ones_like(arr) * (1-alpha)*255).astype(np.uint8))
            self.cache[row['reference_id']] = image
        image = self.cache[row['reference_id']]
        control = Image.fromarray(np.concatenate([np.array(image), np.zeros_like(np.array(image))], axis=1))
        mask = Image.fromarray(np.concatenate([np.zeros((height,width,3)), np.ones((height,width,3))*255], axis=1).astype(np.uint8))
        if row['category']=='style':
            # Paper appendix G: retain full style reference and disable enhancement.
            prompt = (f'A diptych with two side-by-side images of same style. On the left, '
                      f'{subject}. On the right, replicate this style exactly but as {row["prompt"]}')
        else:
            prompt = (f'A diptych with two side-by-side images of same {subject}. On the left, '
                      f'a photo of {subject}. On the right, replicate this {subject} exactly but as {row["prompt"]}')
        processors = {name:self.official.CustomFluxAttnProcessor2_0(
            height=height//16, width=width//16*2,
            attn_enforce=1.0 if row['category']=='style' else self.config['attn_enforce'])
            for name in self.pipe.transformer.attn_processors}
        self.pipe.transformer.set_attn_processor(processors)
        output = self.pipe(prompt=prompt,height=height,width=width*2,control_image=control,control_mask=mask,
            num_inference_steps=30,generator=torch.Generator('cuda').manual_seed(row['seed']),
            controlnet_conditioning_scale=0.95,guidance_scale=3.5,negative_prompt='',true_guidance_scale=3.5).images[0]
        return output.crop((width,0,width*2,height)).crop((offset,offset,width-offset,height-offset)), prompt


class XVerse:
    def __init__(self, paths, config):
        import torch
        self.config = config
        root = Path('/root/xverse/upstream')
        sys.path.insert(0, str(root))
        os.environ['FLUX_MODEL_PATH'] = paths['flux']
        os.environ['CLIP_MODEL_PATH'] = paths['clip']
        from src.flux.pipeline_tools import CustomFluxPipeline, load_modulation_adapter, load_dit_lora
        from src.utils.data_utils import get_train_config, image_grid
        from src.flux.generate import generate_from_test_sample
        cfg = get_train_config(str(root/'train/config/XVerse_config_demo.yaml'))
        # Official unquantized branch; do not introduce int8 as a speed shortcut.
        cfg['model']['dit_quant'] = 'None'
        cfg['model']['use_dit_lora'] = False
        self.model = CustomFluxPipeline(cfg, torch.device('cuda'), torch_dtype=torch.bfloat16)
        self.model.pipe.unload_lora_weights()
        # Match official CLI: instantiate without LoRA, then restore demo controls.
        cfg = get_train_config(str(root/'train/config/XVerse_config_demo.yaml'))
        self.model.config = cfg
        folder = Path(paths['xverse'])/'modulation_adapter'
        if not (folder/'diffusion_pytorch_model.safetensors').is_file():
            raise FileNotFoundError('Missing pretrained modulation adapter')
        adapter = load_modulation_adapter(self.model,cfg,torch.bfloat16,torch.device('cuda'),str(folder),is_training=False)
        self.model.add_modulation_adapter(adapter.to('cuda'))
        load_dit_lora(self.model,self.model.pipe,cfg,torch.bfloat16,torch.device('cuda'),
                      paths['xverse'],is_training=False)
        self.model.pipe.set_progress_bar_config(disable=True)
        # Load the exact official single-sample function without optional demo UI,
        # face-evaluation and detector imports (none are used by this function).
        source = root/'inference_single_sample.py'
        tree = ast.parse(source.read_text())
        node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='generate_image')
        import math
        namespace = dict(torch=torch,generate_from_test_sample=generate_from_test_sample,
                         image_grid=image_grid,math=math,store_attn_map=False)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
        self.generate_image = namespace['generate_image']

    def generate(self, row, reference):
        import torch
        # Exact substring preserves token matching; original target stays unchanged.
        subject = caption(row)
        image = self.generate_image(self.model,row['prompt'],256,self.config['height'],self.config['width'],row['seed'],
            '0-0.05:1,0.8-1:1','0-1:1/3/5',0.85,'0-1:0.85','0-1:1.3',
            [0],1,torch.device('cuda'),None,28,str(reference),subject,row['category']=='human')
        return image,row['prompt']
