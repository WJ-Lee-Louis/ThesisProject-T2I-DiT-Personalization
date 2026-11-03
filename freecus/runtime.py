"""Load audited upstream derivatives in an isolated module namespace."""
import importlib.util
from pathlib import Path

VITAL_LAYERS = [0, 1, 2, 17, 18, 25, 28, 53, 54, 56]


def module(name):
    path = Path(__file__).parent / "runtime_src" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"freecus.runtime_src.{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Engine:
    """One engine per process: upstream attention globals are not thread safe."""
    def __init__(self, config):
        import torch
        self.torch = torch
        self.config = config
        self.qwen = None
        self.segmenter = None
        self.pipe = None
        self.attention = module("sa_handler_for_flux_clean")
        self.flux = module("flux_utils")

    def caption(self, path, word, seed):
        from transformers import set_seed
        set_seed(seed)
        if self.qwen is None:
            models = self.config["models"]
            self.qwen = module("qwen_for_subject").Qwen2VL(models["qwen_vl"], models["qwen_llm"])
        if self.config["implementation"] == "upstream":
            raise ValueError("upstream captions require the target prompt; use describe()")
        raw = self.qwen.get_description(str(path.resolve()), word)[0]
        return raw, self.qwen.get_filtered_description(raw, word)

    def describe(self, row, root):
        if self.config["implementation"] == "paper":
            return self.caption(root / row["reference"], row["subject_word"], self.config["caption_seed"])
        if self.qwen is None:
            models = self.config["models"]
            self.qwen = module("qwen_for_subject").Qwen2VL(models["qwen_vl"], models["qwen_llm"])
        from transformers import set_seed
        set_seed(self.config["caption_seed"])
        return row["prompt"], self.qwen.get_filtered_description(row["prompt"], row["subject_word"])

    def release_caption_models(self):
        import gc
        self.qwen = None
        gc.collect()
        self.torch.cuda.empty_cache()

    def generate(self, row, root, description):
        from PIL import Image
        import numpy as np
        from diffusers import FluxPipeline
        from transformers import set_seed
        torch = self.torch
        cfg = self.config
        height, width = cfg["height"], cfg["width"]
        set_seed(row["seed"])
        if self.segmenter is None:
            self.segmenter = module("seg_birefnet").BiRefNet(cfg["models"]["birefnet"], height, width)
        if self.pipe is None:
            self.pipe = FluxPipeline.from_pretrained(cfg["models"]["flux"], torch_dtype=torch.bfloat16).to("cuda")
        path = root / row["reference"]
        with Image.open(path) as original:
            rgb = original.convert("RGB")
            image = (self.segmenter.center_crop(rgb) if cfg["implementation"] == "paper" else rgb).resize((width, height))
        _, mask = self.segmenter.extract(str(path), pil_image=image if cfg["implementation"] == "paper" else None)
        mask = torch.from_numpy(np.array(mask).copy()).to("cuda", dtype=torch.bfloat16)[None, None]
        mask = torch.nn.functional.interpolate(mask, (height // 16, width // 16), mode="bilinear")
        indices = (mask.reshape(-1) > 0).nonzero(as_tuple=True)[0]
        if not len(indices):
            raise ValueError(f"Empty foreground mask: {path}")
        self.flux.PAPER_SEQUENCE_LENGTH = cfg["implementation"] == "paper"
        prompt = row["prompt"] + ". " + description
        handler = self.attention.Handler()
        originals = dict(self.pipe.transformer.attn_processors)
        handler.clear()
        try:
            with torch.inference_mode():
                set_seed(row["seed"])
                _, trajectory, _ = self.flux.pil_to_latents_flux(image, self.pipe, cfg["steps"], height, width, row["seed"], "negative")
                def callback(pipeline, i, t, kwargs):
                    return {"latents": trajectory[min(i + 1, cfg["steps"] - 1)].to(kwargs["latents"])}
                common = dict(prompt=prompt, height=height, width=width, guidance_scale=cfg["guidance"],
                              num_inference_steps=cfg["steps"], max_sequence_length=512)
                handler.register(self.pipe, "save", cfg["steps"], cfg["scale"], indices, VITAL_LAYERS)
                self.pipe(**common, latents=trajectory[0], callback_on_step_end=callback,
                          generator=torch.Generator("cpu").manual_seed(row["seed"]), output_type="latent")
                handler.register(self.pipe, "use", cfg["steps"], cfg["scale"], indices, VITAL_LAYERS)
                return self.pipe(**common, generator=torch.Generator("cpu").manual_seed(row["seed"])).images[0], prompt
        finally:
            handler.clear()
            self.pipe.transformer.set_attn_processor(originals)
