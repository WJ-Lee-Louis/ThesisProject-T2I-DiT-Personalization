"""Use original UNO pipeline, reference preprocessing, LoRA processors and sampler."""
import os
from pathlib import Path


def task_prompt(row):
    return "single_image", row["prompt"]


class Engine:
    def __init__(self, config, paths, lock, work):
        import torch
        from .models import FILES, local_download_resolver
        if set(paths) != set(FILES):
            raise ValueError("Unexpected model paths")
        os.environ.update(FLUX_DEV=str(Path(paths["flux"]) / "flux1-dev.safetensors"),
                          AE=str(Path(paths["flux"]) / "ae.safetensors"),
                          T5=paths["t5"], CLIP=paths["clip"],
                          LORA=str(Path(paths["lora"]) / "dit_lora.safetensors"),
                          HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        # Environment paths must be set before official util creates its ModelSpec.
        from uno.flux import util
        from uno.flux.pipeline import UNOPipeline
        util.configs["flux-dev"].ckpt_path = os.environ["FLUX_DEV"]
        util.configs["flux-dev"].ae_path = os.environ["AE"]
        util.hf_hub_download = local_download_resolver(lock, paths)

        def strict_load_warning(missing, unexpected):
            if missing or unexpected:
                raise ValueError(f"UNO checkpoint mismatch: missing={missing}, unexpected={unexpected}")

        util.print_load_warning = strict_load_warning
        self.config = config
        self.pipe = UNOPipeline(config["model_type"], torch.device("cuda"), config["cpu_offload"],
                                only_lora=config["only_lora"], lora_rank=config["lora_rank"])
        for module in [self.pipe.model, self.pipe.clip, self.pipe.t5, self.pipe.ae]:
            if any(p.is_meta for p in module.parameters()):
                raise ValueError("Checkpoint left uninitialized meta tensors")
            module.eval().requires_grad_(False)
        # Retain each checkpoint's original dtype, including the FP32 autoencoder/LoRA if present.
        self.model_dtypes = {name: sorted({str(p.dtype) for p in module.parameters()}) for name, module in
                             [("transformer", self.pipe.model), ("clip", self.pipe.clip),
                              ("t5", self.pipe.t5), ("autoencoder", self.pipe.ae)]}

    def conditioning(self, reference, task):
        from PIL import Image
        from uno.flux.pipeline import preprocess_ref
        with Image.open(reference) as image:
            return preprocess_ref(image, self.config["reference_resolution"])

    def generate(self, row, conditioning):
        import torch
        from transformers import set_seed
        set_seed(row["seed"])
        cfg = self.config
        with torch.inference_mode():
            return self.pipe(prompt=row["prompt"], width=cfg["width"], height=cfg["height"],
                             guidance=cfg["guidance"], num_steps=cfg["steps"], seed=row["seed"],
                             ref_imgs=[conditioning], pe=cfg["pe"])
