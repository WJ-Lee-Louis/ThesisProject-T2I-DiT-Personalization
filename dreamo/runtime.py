"""Invoke unchanged official DreamO modules with pinned local weights, without UI/face-ID extras."""
from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
import shutil


def task_prompt(row):
    if row["category"] == "style":
        return "style", "generate a same style image. " + row["prompt"]
    return "ip", row["prompt"]


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def local_download_resolver(lock, paths):
    from .models import FILES
    by_repo = {v["repo"]: key for key, v in lock.items()}

    def resolve(repo_id, filename, local_dir="models", **kwargs):
        key = by_repo.get(repo_id)
        if key not in FILES or filename not in FILES[key]:
            raise ValueError(f"Unpinned model file requested: {repo_id}/{filename}")
        source = (Path(paths[key]) / filename).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.resolve() != source:
                from freecus.manifest import sha256
                if sha256(target) != sha256(source):
                    raise ValueError("Staged DreamO adapter differs from locked snapshot")
        else:
            try:
                target.symlink_to(source)
            except OSError:
                shutil.copy2(source, target)
        return str(target)

    return resolve


class Engine:
    def __init__(self, config, paths, lock, work):
        import torch
        from dreamo import dreamo_pipeline as official
        base = Path(__file__).parent
        spec = importlib.util.spec_from_file_location("dreamo_official_ben2", base / "upstream/tools/BEN2.py")
        ben2 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ben2)
        self.ben = ben2.BEN_Base().eval().to("cuda")
        self.ben.loadcheckpoints(str(Path(paths["ben2"]) / "BEN2_Base.pth"))
        self.config = config
        self.pipe = official.DreamOPipeline.from_pretrained(paths["flux"], torch_dtype=torch.bfloat16,
                                                           local_files_only=True)
        # Official code uses fixed models/... paths. Resolve only allowlisted local files.
        official.hf_hub_download = local_download_resolver(lock, paths)
        with working_directory(work):
            self.pipe.load_dreamo_model("cuda", use_turbo=config["turbo"], version=config["version"])
        self.pipe = self.pipe.to("cuda")
        self.pipe.offload = False

    def conditioning(self, reference, task):
        import numpy as np
        from PIL import Image
        from dreamo.utils import resize_numpy_image_area
        with Image.open(reference) as opened:
            image = opened.convert("RGB")
        if task == "ip":
            image = self.ben.inference(image)
        return Image.fromarray(resize_numpy_image_area(np.array(image), self.config["reference_resolution"] ** 2))

    def generate(self, row, conditioning):
        import numpy as np
        import torch
        from transformers import set_seed
        from dreamo.utils import img2tensor
        task, prompt = task_prompt(row)
        # Upstream reference VAE.sample() uses global RNG, separate from the CPU noise generator.
        torch.cuda.empty_cache()  # Same cleanup as official Generator.pre_condition().
        set_seed(row["seed"])
        tensor = img2tensor(np.array(conditioning), bgr2rgb=False).unsqueeze(0) / 255.0
        cfg = self.config
        with torch.inference_mode():
            return self.pipe(prompt=prompt, width=cfg["width"], height=cfg["height"],
                             num_inference_steps=cfg["steps"], guidance_scale=cfg["guidance"],
                             ref_conds=[dict(img=2 * tensor - 1.0, task=task, idx=1)],
                             generator=torch.Generator(device="cpu").manual_seed(row["seed"]),
                             true_cfg_scale=cfg["true_cfg"], true_cfg_start_step=0, true_cfg_end_step=0,
                             negative_prompt=cfg["negative_prompt"], neg_guidance_scale=cfg["negative_guidance"],
                             first_step_guidance_scale=cfg["guidance"], max_sequence_length=512).images[0]
