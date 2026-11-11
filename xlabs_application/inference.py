"""CLIP projected-space edits before the unmodified XLabs adapter projection."""
from dataclasses import asdict, replace
from io import BytesIO
import time

import torch
from PIL import Image

from .config import SETTINGS
from .vendor.contrast_directions import build_contrast_directions, contrast_diagnostics
from .vendor.orthogonalization import orthogonalize_pooled_embedding
from .vendor.semantic_contrasts import ReferenceFingerprint, SemanticContrastConditioning


def validate_cache(row, payload, image_bytes):
    cached = SemanticContrastConditioning.from_dict(payload)
    cached.validate_reuse(identity_term=row["identity_term"], target_prompt=row["target_prompt"],
                          reference=ReferenceFingerprint.from_image(image_bytes), model="gpt-5.6-sol")
    return replace(cached, origin="source_target")


def edit_embedding(visual, conditioning, encode, strength=1.0):
    directions = build_contrast_directions(conditioning, encode)
    result = orthogonalize_pooled_embedding(visual, directions.values, strength=strength,
                    relative_tolerance=SETTINGS["relative_tolerance"], restore_norm=True,
                    preserve_dtype=False)
    # Avoid numerical round trips for a deliberately empty/no-op intervention.
    edited = visual if strength == 0 or result.basis.shape[1] == 0 else result.embedding.to(visual.dtype)
    diag = contrast_diagnostics(conditioning)
    diag.update(enabled=True, strength=strength, basis_rank=result.basis.shape[1],
                raw_direction_norms=directions.raw_norms.cpu().tolist(),
                singular_values=result.basis_diagnostics.singular_values.cpu().tolist(),
                projection={k: v.detach().cpu().tolist() if isinstance(v, torch.Tensor) else v
                            for k, v in asdict(result.projection_diagnostics).items()})
    return edited, diag


class Engine:
    def __init__(self, resolved):
        from diffusers import FluxPipeline
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer
        self.pipe = FluxPipeline.from_pretrained(resolved["base"], torch_dtype=torch.bfloat16).to("cuda")
        self.pipe.load_ip_adapter(resolved["adapter"], weight_name="ip_adapter.safetensors",
                                 image_encoder_pretrained_model_name_or_path=resolved["clip"],
                                 image_encoder_dtype=torch.float16)
        self.pipe.set_ip_adapter_scale(SETTINGS["ip_adapter_scale"])
        self.pipe.set_progress_bar_config(disable=True)
        self.text = CLIPTextModelWithProjection.from_pretrained(resolved["clip"], torch_dtype=torch.float16).to("cuda").eval()
        self.text.requires_grad_(False)
        self.tokenizer = CLIPTokenizer.from_pretrained(resolved["clip"])
        self.text_cache = {}

    @torch.inference_mode()
    def encode(self, texts):
        missing = list(dict.fromkeys(t for t in texts if t not in self.text_cache))
        for start in range(0, len(missing), 16):
            batch = missing[start:start + 16]
            lengths = self.tokenizer(batch, truncation=False, padding=False)["input_ids"]
            if any(len(x) > self.text.config.max_position_embeddings for x in lengths):
                raise ValueError("CLIP semantic input exceeds token limit; no truncation permitted")
            inputs = self.tokenizer(batch, padding=True, truncation=False, return_tensors="pt").to("cuda")
            embeds = self.text(**inputs).text_embeds.float().cpu()
            self.text_cache.update(zip(batch, embeds))
        return torch.stack([self.text_cache[t] for t in texts]).to("cuda")

    @torch.inference_mode()
    def generate(self, row, payload, image_bytes, condition, strength=None):
        cached = validate_cache(row, payload, image_bytes)
        if condition not in ("baseline", "source_target"):
            raise ValueError("Unknown condition")
        reference = Image.open(BytesIO(image_bytes)).convert("RGB").resize((1024, 1024))
        kwargs = dict(prompt=row["target_prompt"], width=1024, height=1024,
                      guidance_scale=SETTINGS["guidance_scale"], num_inference_steps=SETTINGS["steps"],
                      max_sequence_length=SETTINGS["max_sequence_length"], true_cfg_scale=1.0,
                      generator=torch.Generator("cuda").manual_seed(SETTINGS["seed"]))
        diag = dict(enabled=False)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        if condition == "baseline":
            kwargs["ip_adapter_image"] = reference
        else:
            visual = self.pipe.encode_image(reference, "cuda", 1)
            edited, diag = edit_embedding(visual, cached, self.encode,
                                         SETTINGS["strength"] if strength is None else strength)
            kwargs["ip_adapter_image_embeds"] = [edited.unsqueeze(0)]
        image = self.pipe(**kwargs).images[0]
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        free, total = torch.cuda.mem_get_info()
        return buffer.getvalue(), dict(orthogonalization=diag, mask_single_text_rows=False,
                seconds=elapsed, peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
                gpu_total_gib=total / 1024**3, gpu_free_gib=free / 1024**3,
                gpu_name=torch.cuda.get_device_name(), image_preprocess="RGB resize 1024x1024; CLIP processor",
                fallback=cached.fallback_trace is not None)
