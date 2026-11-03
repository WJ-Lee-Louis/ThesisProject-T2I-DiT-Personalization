import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from freecus.manifest import dreambench, dreambench_plus, load, project_rows, PROJECT_MANIFEST, PROJECT
from freecus.run import artifact_paths, complete, validate_config, write_json
from freecus.runtime import module

BASE = Path(__file__).resolve().parents[1]


class ManifestTests(unittest.TestCase):
    def test_existing_dreambench_and_tamper_detection(self):
        root = BASE.parent / "dreambooth_reference_dataset"
        rows = list(dreambench(root))
        self.assertEqual(len(rows), 750)
        self.assertEqual(len({r["id"] for r in rows}), 750)
        self.assertEqual(len({r["subject"] for r in rows}), 30)
        self.assertEqual({r["seed"] for r in rows}, {42})
        self.assertEqual(rows[0]["prompt"], "a backpack in the jungle")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rows.jsonl"
            path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
            self.assertEqual(len(load(path, root)), 1)
            rows[0]["reference_sha256"] = "tampered"
            path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Reference changed"):
                load(path, root)

    def test_plus_missing_pairs_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "150"):
                list(dreambench_plus(Path(temp)))

    def test_project_manifest_preserves_all_exact_cases_and_references(self):
        original = json.loads(PROJECT_MANIFEST.read_text(encoding="utf-8"))
        by_id = {r["case_id"]: r for r in original}
        all_rows = []
        for benchmark, count in [("dreambench", 750), ("dreambench_plus", 1350)]:
            rows = list(project_rows(benchmark))
            self.assertEqual(len(rows), count)
            self.assertEqual({r["seed"] for r in rows}, {42})
            for row in rows:
                source = by_id[row["case_id"]]
                self.assertEqual(row["prompt"], source["target_prompt"])
                self.assertEqual(row["reference_id"], source["reference_id"])
                self.assertEqual(row["reference"], "Orthogonalization-InstantX-IP-Adapter/" + source["image_path"])
                self.assertEqual(row["evaluation_references"], [row["reference"]])
            all_rows += rows
        self.assertEqual(len({r["case_id"] for r in all_rows}), 2100)
        self.assertEqual(sum(r["category"] == "style" for r in all_rows), 180)
        from freecus.evaluate import project_evaluator
        evaluator = project_evaluator()
        self.assertEqual(evaluator.PROTOCOL["dino_model"], "dino_vits16")
        self.assertEqual(evaluator.PROTOCOL["score_scale"], 1)
        self.assertEqual(evaluator.PROTOCOL["clip_model"], "ViT-B/32")
        from freecus.manifest import sha256
        self.assertEqual(sha256(BASE / "project_metrics/benchmark_metrics.py"),
                         sha256(PROJECT / "Orthogonalization-InstantX-IP-Adapter/scripts/benchmark_metrics.py"))

    def test_targets_equal_official_benchmark_prompts(self):
        for benchmark, official in [
            ("dreambench", list(dreambench(PROJECT / "dreambooth_reference_dataset"))),
            ("dreambench_plus", list(dreambench_plus(BASE / "data/dreambench_plus/data")))]:
            lookup = {}
            for row in official:
                ref = row["subject"] if benchmark == "dreambench" else Path(row["reference"]).relative_to("images").with_suffix("").as_posix()
                lookup[ref, row["prompt_index"]] = row["prompt"]
            for row in project_rows(benchmark):
                ref = row["reference_id"].split("/", 1)[1]
                self.assertEqual(row["prompt"], lookup[ref, row["prompt_index"]])

    def test_resolution_enforced(self):
        cfg = json.loads((BASE / "config.json").read_text())
        validate_config(cfg)
        cfg["height"] = 512
        with self.assertRaises(ValueError):
            validate_config(cfg)

    def test_resume_detects_corrupted_artifact(self):
        from freecus.manifest import sha256
        with tempfile.TemporaryDirectory() as temp:
            paths = artifact_paths(Path(temp), dict(id="subject/0_0"))
            self.assertFalse(complete(paths))
            for p in paths.values():
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("data", encoding="utf-8")
            write_json(paths["metadata"], dict(artifact_sha256={k: sha256(p) for k, p in paths.items() if k != "metadata"}))
            self.assertTrue(complete(paths))
            paths["tgt_image"].write_text("corrupted")
            self.assertFalse(complete(paths))


class RuntimeTests(unittest.TestCase):
    def test_real_diffusers_transformer_rotary_and_57_processors(self):
        import torch
        from diffusers import FluxTransformer2DModel
        from freecus.runtime import VITAL_LAYERS
        threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            model = FluxTransformer2DModel(in_channels=64, num_layers=19, num_single_layers=38,
                        attention_head_dim=16, num_attention_heads=1, joint_attention_dim=16,
                        pooled_projection_dim=16, axes_dims_rope=(4, 6, 6)).eval()
            attention = module("sa_handler_for_flux_clean")
            pipe = SimpleNamespace(transformer=model)
            handler = attention.Handler()
            kwargs = dict(hidden_states=torch.randn(1, 4, 64), encoder_hidden_states=torch.randn(1, 512, 16),
                          pooled_projections=torch.randn(1, 16), timestep=torch.tensor([0.5]),
                          img_ids=torch.zeros(4, 3), txt_ids=torch.zeros(512, 3), return_dict=False)
            with torch.inference_mode():
                handler.register(pipe, "save", 1, 1.1, torch.tensor([0, 2]), VITAL_LAYERS)
                model(**kwargs)
                self.assertEqual(len(attention.all_store_kv["key"]), 57)
                self.assertEqual(sum(v is not None for v in attention.all_store_kv["key"]), 10)
                handler.register(pipe, "use", 1, 1.1, torch.tensor([0, 2]), VITAL_LAYERS)
                result = model(**kwargs)[0]
                self.assertEqual(result.shape, (1, 4, 64))
                self.assertTrue(torch.isfinite(result).all())
            handler.clear()
        finally:
            torch.set_num_threads(threads)

    def test_upstream_hashes(self):
        from freecus.manifest import sha256
        provenance = json.loads((BASE / "UPSTREAM.json").read_text())
        for name, expected in provenance["files"].items():
            self.assertEqual(sha256(BASE / "upstream" / name), expected)

    def test_noise_shift_uses_packed_token_count(self):
        import torch
        flux = module("flux_utils")
        class TorchProxy:
            def __getattr__(self, name):
                return getattr(torch, name)
            def tensor(self, data, **kwargs):
                kwargs.pop("device", None)
                return torch.tensor(data, **kwargs)
        flux.torch = TorchProxy()
        latent = torch.zeros(1, 16, 128, 128)
        class Preprocessed:
            def to(self, **kwargs):
                return latent
        class Scheduler:
            config = SimpleNamespace(base_image_seq_len=256, max_image_seq_len=4096,
                                     base_shift=0.5, max_shift=1.15)
            def set_timesteps(self, sigmas, device, mu):
                self.mu = mu
                self.timesteps = torch.tensor(sigmas)
            def scale_noise(self, image, timestep, noise):
                return image
        pipe = SimpleNamespace(image_processor=SimpleNamespace(preprocess=lambda *a, **k: Preprocessed()),
                               vae_scale_factor=8, transformer=SimpleNamespace(config=SimpleNamespace(in_channels=64)),
                               scheduler=Scheduler(), _pack_latents=lambda x, *a: x)
        flux._encode_vae_image = lambda **kwargs: latent
        flux.prepare_latent_image_ids = lambda *a: None
        flux.randn_tensor = lambda shape, **kwargs: torch.zeros(shape)
        flux.pil_to_latents_flux(None, pipe, 2, 1024, 1024, shift_type="negative")
        self.assertAlmostEqual(pipe.scheduler.mu, -1.15)
        flux.PAPER_SEQUENCE_LENGTH = False
        flux.pil_to_latents_flux(None, pipe, 2, 1024, 1024, shift_type="negative")
        self.assertAlmostEqual(pipe.scheduler.mu, -flux.calculate_shift(64))

    def test_attention_cache_preserves_outputs_and_resets(self):
        import torch
        torch.manual_seed(7)
        patched = module("sa_handler_for_flux_clean")
        spec = importlib.util.spec_from_file_location("original_attention", BASE / "upstream/sa_handler_for_flux_clean.py")
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        attn = SimpleNamespace(heads=2, to_q=torch.nn.Identity(), to_k=torch.nn.Identity(),
                               to_v=torch.nn.Identity(), norm_q=None, norm_k=None)
        x = torch.randn(1, 516, 8)
        for mod in [original, patched]:
            mod.dim_neg2_indices = torch.tensor([0, 2])
        with torch.inference_mode():
            for layer in range(57):
                for mod in [original, patched]:
                    proc = mod.FluxShareAttnProcessor2_0("save", layer, num_inference_steps=1, vital_layers=layer in {0, 56})
                    proc(attn, x.clone())
            for layer in range(57):
                results = [mod.FluxShareAttnProcessor2_0("use", layer, num_inference_steps=1,
                           vital_layers=layer in {0, 56})(attn, x.clone()) for mod in [original, patched]]
                torch.testing.assert_close(*results, rtol=0, atol=0)
        self.assertEqual(sum(v is not None for v in patched.all_store_kv["key"]), 2)
        patched.Handler().clear()
        self.assertEqual(patched.global_steps, 0)
        self.assertEqual(patched.inner_blocks, 0)
        self.assertEqual(patched.all_store_kv, {"key": [], "value": []})


if __name__ == "__main__":
    unittest.main()
