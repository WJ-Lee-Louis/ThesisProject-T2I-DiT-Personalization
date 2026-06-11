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
  # planned

flux_redux/
  # planned
```

## Model folders

Each model family gets its own top-level folder because the conditioning path differs across implementations.

- `xlabs_flux_ip_adapter`: XLabs IP-Adapter checkpoint for FLUX.1-dev.
- `instantx_flux_ip_adapter`: planned folder for `InstantX/FLUX.1-dev-IP-Adapter`.
- `flux_redux`: planned folder for `black-forest-labs/FLUX.1-Redux-dev`.

The `shared/` folder should only contain infrastructure that is genuinely common, such as Modal image, Secret, and Volume setup.
