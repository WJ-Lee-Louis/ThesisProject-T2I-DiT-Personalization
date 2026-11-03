import ast
import json
from pathlib import Path
import tempfile
import unittest

from uno.protocol import BASE, selected_rows, validate_config, verify_upstream
from uno.runtime import task_prompt
from uno.models import local_download_resolver
from freecus.manifest import PROJECT
from freecus.evaluate import prepare, project_evaluator


class ProtocolTests(unittest.TestCase):
    def test_original_sources_and_official_defaults(self):
        self.assertEqual(verify_upstream()["commit"], "a563432dcfb4caf98df4368facb8102bf97d376c")
        tree = ast.parse((BASE / "upstream/inference.py").read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "InferenceArgs")
        defaults = {n.target.id: ast.literal_eval(n.value) for n in cls.body if isinstance(n, ast.AnnAssign)}
        config = json.loads((BASE / "config.json").read_text())
        validate_config(config)
        for local, upstream in [("steps", "num_steps"), ("guidance", "guidance"), ("lora_rank", "lora_rank"), ("pe", "pe"), ("only_lora", "only_lora")]:
            self.assertEqual(config[local], defaults[upstream])
        for key, changed in [("steps", 12), ("quantization", "fp8"), ("width", 704), ("pe", "h")]:
            with self.assertRaises(ValueError):
                validate_config(config | {key: changed})

    def test_same_cohort_and_native_single_image_style(self):
        db, _ = selected_rows("dreambench")
        plus, manifest = selected_rows("dreambench_plus")
        self.assertEqual((len(db), len(plus)), (750, 1350))
        self.assertEqual(len({r["reference_id"] for r in plus}), 150)
        styles = [r for r in plus if r["category"] == "style"]
        self.assertEqual(len(styles), 180)
        for row in db + plus:
            self.assertEqual(row["seed"], 42)
            self.assertEqual(task_prompt(row), ("single_image", row["prompt"]))
        full = prepare(manifest, PROJECT, BASE / "outputs/future", project_evaluator(), "all")
        subset = prepare(manifest, PROJECT, BASE / "outputs/future", project_evaluator(), "dbpp-no-style")
        self.assertEqual((len(full["records"]), len(subset["records"])), (1350, 1170))

    def test_resolver_rejects_unpinned_download(self):
        with tempfile.TemporaryDirectory() as folder:
            resolver = local_download_resolver({"lora":dict(repo="bytedance-research/UNO")}, {"lora":folder})
            with self.assertRaises(ValueError):
                resolver("another/repo", "dit_lora.safetensors")
            p = Path(folder) / "dit_lora.safetensors"
            p.write_bytes(b"fixture")
            self.assertEqual(resolver("bytedance-research/UNO", p.name), str(p))


class OfficialRuntimeTests(unittest.TestCase):
    def test_official_reference_resize(self):
        from PIL import Image
        from uno.flux.pipeline import preprocess_ref
        self.assertEqual(preprocess_ref(Image.new("RGB", (1200, 800)), 512).size, (512, 336))
        self.assertEqual(preprocess_ref(Image.new("RGB", (800, 1200)), 512).size, (336, 512))

    def test_official_seed_and_sampler_reference_path(self):
        import torch
        from uno.flux.model import Flux, FluxParams
        from uno.flux.util import set_lora
        from uno.flux.sampling import get_noise, get_schedule, prepare_multi_ip, denoise
        params = FluxParams(in_channels=64, vec_in_dim=8, context_in_dim=16, hidden_size=32,
                            mlp_ratio=2, num_heads=4, depth=1, depth_single_blocks=0,
                            axes_dim=[2, 2, 4], theta=10000, qkv_bias=True, guidance_embed=True)
        torch.manual_seed(42)
        model = set_lora(Flux(params), 2).eval()
        x = get_noise(1, 32, 32, torch.device("cpu"), torch.float32, 42)
        self.assertTrue(torch.equal(x, get_noise(1,32,32,torch.device("cpu"),torch.float32,42)))
        def t5(text):
            return torch.zeros(1, 3, 16)
        def clip(text):
            return torch.zeros(1, 8)
        inputs = prepare_multi_ip(t5, clip, x, "a subject", [torch.zeros(1, 16, 4, 4)], pe="d")
        ids = inputs["ref_img_ids"][0]
        self.assertEqual((ids[0, 0, 1].item(), ids[0, 0, 2].item()), (2, 2))
        with torch.inference_mode():
            result = denoise(model, **inputs, timesteps=get_schedule(2, 4), guidance=4.0)
        self.assertEqual(result.shape, (1, 4, 64))
        self.assertTrue(torch.isfinite(result).all())


if __name__ == "__main__":
    unittest.main()
