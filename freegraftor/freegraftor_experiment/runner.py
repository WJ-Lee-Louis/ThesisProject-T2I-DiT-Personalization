"""Thin API adapter around the upstream FreeGraftor pipeline."""

from __future__ import annotations

from pathlib import Path
import sys

from .batch_config import BatchConfig


class FreeGraftorRunner:
    def __init__(self, upstream_src: str | Path, cache_root: str | Path) -> None:
        import torch

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        upstream_src = Path(upstream_src).resolve()
        if str(upstream_src) not in sys.path:
            sys.path.insert(0, str(upstream_src))

        # FLUX_DEV must already be set: flux.util resolves paths at import time.
        from pipeline import FreeGraftorPipeline

        cache_root = Path(cache_root)
        self.pipeline = FreeGraftorPipeline(
            device="cuda",
            image_cache_dir=str(cache_root / "images"),
            image_info_cache_dir=str(cache_root / "image_info"),
            requires_offload=True,
        )

    def generation_config(self, config: BatchConfig):
        from pipeline import GenerationConfig

        return GenerationConfig(
            seed=config.seed,
            num_steps=config.num_steps,
            guidance=config.guidance,
            width=config.width,
            height=config.height,
            start_inject_step=config.start_inject_step,
            end_inject_step=config.end_inject_step,
            inject_block_ids=config.inject_block_ids,
            sim_threshold=config.sim_threshold,
            cyc_threshold=config.cyc_threshold,
            inject_match_dropout=config.inject_match_dropout,
        )

    def generate_template(self, prompt: str, config: BatchConfig, destination: Path):
        import torch

        torch.manual_seed(config.seed)
        image, _ = self.pipeline.generate_template(
            prompt,
            self.generation_config(config),
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination, format="PNG")
        # Independent upstream CLI calls continue from exactly this RNG state
        # when they build the collage and run stochastic match dropout. Keeping
        # it lets multiple references reuse the deterministic template without
        # changing their per-job random sequence.
        rng_state = {
            "cpu": torch.get_rng_state().clone(),
            "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()],
        }
        return image, rng_state

    def generate(
        self,
        *,
        reference_path: Path,
        class_name: str,
        prompt: str,
        template_path: Path,
        config: BatchConfig,
        scratch_output: Path,
        post_template_rng_state: dict,
    ):
        import torch
        from pipeline import ConceptConfig

        # The stochastic match dropout must start from the same seed for every
        # reference/prompt pair, matching independent upstream CLI invocations.
        torch.set_rng_state(post_template_rng_state["cpu"])
        torch.cuda.set_rng_state_all(post_template_rng_state["cuda"])
        return self.pipeline(
            concept_configs=[
                ConceptConfig(
                    class_name=class_name,
                    image_path=str(reference_path),
                    alignment=config.alignment,
                )
            ],
            prompt=prompt,
            template_prompt=prompt,
            template_path=str(template_path),
            output_dir=str(scratch_output),
            config=self.generation_config(config),
            save_output=False,
        )
