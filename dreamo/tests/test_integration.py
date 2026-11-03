import copy
import json
from pathlib import Path
import tempfile
import unittest

from dreamo.protocol import BASE, selected_rows, validate_config, verify_upstream
from dreamo.runtime import task_prompt, local_download_resolver
from freecus.evaluate import prepare, project_evaluator
from freecus.manifest import PROJECT


class ProtocolTests(unittest.TestCase):
    def test_upstream_bytes_are_unchanged(self):
        self.assertEqual(verify_upstream()["commit"], "461e35ba6b068f42ca53b43f289b3d91b7f911cd")

    def test_generation_keeps_all_1350_cases_including_style(self):
        db, _ = selected_rows("dreambench")
        plus, _ = selected_rows("dreambench_plus")
        self.assertEqual(len(db), 750)
        self.assertEqual(len(plus), 1350)
        self.assertEqual(len({r["reference_id"] for r in plus}), 150)
        self.assertEqual(sum(r["category"] == "style" for r in plus), 180)
        self.assertEqual({r["seed"] for r in db + plus}, {42})
        original = json.loads((PROJECT / "Orthogonalization-InstantX-IP-Adapter/outputs/semantic_benchmark_gpt56_sol/generation_manifest.json").read_text(encoding="utf-8"))
        lookup = {r["case_id"]: r for r in original}
        for row in db + plus:
            self.assertEqual(row["prompt"], lookup[row["case_id"]]["target_prompt"])
            self.assertEqual(row["reference_id"], lookup[row["case_id"]]["reference_id"])

    def test_style_activation_changes_only_generation_prompt(self):
        rows, manifest = selected_rows("dreambench_plus")
        style = next(r for r in rows if r["category"] == "style")
        original = copy.deepcopy(style)
        task, prompt = task_prompt(style)
        self.assertEqual(task, "style")
        self.assertEqual(prompt, "generate a same style image. " + original["prompt"])
        self.assertEqual(style, original)
        plan = prepare(manifest, PROJECT, Path("unused"), project_evaluator(), "all")
        self.assertEqual(len(plan["records"]), 1350)
        record = next(r for r in plan["records"] if r["row"]["case_id"] == style["case_id"])
        self.assertEqual(record["row"]["target_prompt"], original["prompt"])
        subset = prepare(manifest, PROJECT, Path("unused"), project_evaluator(), "dbpp-no-style")
        self.assertEqual(len(subset["records"]), 1170)

    def test_config_rejects_silent_quantization_or_resolution_changes(self):
        cfg = json.loads((BASE / "config.json").read_text())
        validate_config(cfg)
        for key, value in [("quantization", "int8"), ("width", 512), ("steps", 30), ("reference_task", "id")]:
            changed = dict(cfg, **{key: value})
            with self.assertRaises(ValueError):
                validate_config(changed)

    def test_resolver_rejects_unlocked_download(self):
        resolver = local_download_resolver({"dreamo": dict(repo="ByteDance/DreamO")}, {"dreamo": "."})
        with self.assertRaises(ValueError):
            resolver("untrusted/model", "file.bin")
        with self.assertRaises(ValueError):
            resolver("ByteDance/DreamO", "../../file.bin")


class CPUCompatibilityTests(unittest.TestCase):
    def test_official_pipeline_can_load_fuse_and_unload_lora_with_pinned_peft(self):
        import torch
        from diffusers import FluxTransformer2DModel, FlowMatchEulerDiscreteScheduler
        from dreamo.dreamo_pipeline import DreamOPipeline
        model = FluxTransformer2DModel(in_channels=4, num_layers=1, num_single_layers=1,
                    attention_head_dim=16, num_attention_heads=1, joint_attention_dim=16,
                    pooled_projection_dim=16, axes_dims_rope=(4, 6, 6), guidance_embeds=True)
        pipe = DreamOPipeline(FlowMatchEulerDiscreteScheduler(), None, None, None, None, None, model)
        before = model.transformer_blocks[0].attn.to_q.weight.detach().clone()
        prefix = "transformer.transformer_blocks.0.attn.to_q"
        pipe.load_lora_weights({prefix + ".lora_A.weight": torch.ones(2, 16),
                                prefix + ".lora_B.weight": torch.ones(16, 2)}, adapter_name="probe")
        pipe.set_adapters(["probe"], [1])
        pipe.fuse_lora(adapter_names=["probe"], lora_scale=1)
        pipe.unload_lora_weights()
        after = model.transformer_blocks[0].attn.to_q.weight.detach()
        self.assertTrue(torch.isfinite(after).all())
        self.assertGreater((after - before).abs().max().item(), 0)
        self.assertFalse(hasattr(model.transformer_blocks[0].attn.to_q, "lora_A"))

    def test_official_transformer_forward_accepts_reference_embeddings(self):
        import torch
        import dreamo.dreamo_pipeline
        from diffusers import FluxTransformer2DModel
        previous = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            model = FluxTransformer2DModel(in_channels=4, num_layers=1, num_single_layers=1,
                        attention_head_dim=16, num_attention_heads=1, joint_attention_dim=16,
                        pooled_projection_dim=16, axes_dims_rope=(4, 6, 6), guidance_embeds=True).eval()
            with torch.inference_mode():
                output = model(hidden_states=torch.randn(1, 6, 4), encoder_hidden_states=torch.randn(1, 4, 16),
                               pooled_projections=torch.randn(1, 16), timestep=torch.ones(1), guidance=torch.ones(1),
                               img_ids=torch.zeros(6, 3), txt_ids=torch.zeros(4, 3), embeddings=torch.zeros(1, 6, 16)).sample
            self.assertEqual(tuple(output.shape), (1, 6, 4))
            self.assertTrue(torch.isfinite(output).all())
        finally:
            torch.set_num_threads(previous)

    def test_resize_preserves_aspect_and_multiple_of_sixteen(self):
        import numpy as np
        from dreamo.utils import resize_numpy_image_area
        image = np.zeros((512, 1024, 3), dtype=np.uint8)
        resized = resize_numpy_image_area(image, 512 * 512)
        self.assertEqual(resized.shape[0] % 16, 0)
        self.assertEqual(resized.shape[1] % 16, 0)
        self.assertLessEqual(abs(resized.shape[1] / resized.shape[0] - 2), 0.1)


if __name__ == "__main__":
    unittest.main()
