# Project structure

The project keeps setup utilities separate from model-specific research code.

```text
setup/
  get_started.py
  modal_gpu_smoke.py
  requirements.txt

shared/
  modal_runtime.py

xlabs_flux_ip_adapter/
  ...

instantx_flux_ip_adapter/
  ...

flux_redux/
  ...

personalize_anything/
  ...
```

## Model folders

Each model family gets its own top-level folder because the conditioning path differs across implementations.

- `xlabs_flux_ip_adapter`: XLabs IP-Adapter checkpoint for FLUX.1-dev.
- `instantx_flux_ip_adapter`: InstantX IP-Adapter checkpoint for FLUX.1-dev.
- `flux_redux`: FLUX Redux image-prior adapter for `black-forest-labs/FLUX.1-dev`.
- `personalize_anything`: mask-guided Personalize Anything single-subject setup for FLUX.1-dev.
- `freecus`: pinned official FreeCus sources, isolated runtime, and 1024x1024 DreamBench/DreamBench++ benchmark preparation. See `freecus/README.md`.
- `dreamo`: pinned official DreamO v1.1 sources, BF16/Turbo generation, full DreamBench/DreamBench++ manifests, shared CLIP/DINO evaluation, and detached Modal preparation. See `dreamo/README.md`.
- `uno`: pinned official UNO sources, original single-reference pipeline and LoRA processors, 1024x1024 DreamBench/DreamBench++ comparison, shared CLIP/DINO evaluation, and autonomous Modal execution. See `uno/README.md`.
- `freegraftor`: original Modal integration and the preserved RunPod generation/recovery sources, manifests, incomplete-run records and pilot metrics. See `freegraftor/runpod/README.md`.

The `shared/` folder should only contain infrastructure that is genuinely common, such as Modal image, Secret, and Volume setup.
