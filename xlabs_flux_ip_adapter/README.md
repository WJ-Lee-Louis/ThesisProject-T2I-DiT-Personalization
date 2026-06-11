# XLabs FLUX IP-Adapter

Modal-based inference scaffold for `XLabs-AI/flux-ip-adapter` on top of `black-forest-labs/FLUX.1-dev`.

The current priority is a reliable baseline inference path. Research hooks for training-free conditioning changes are intentionally separated from the Modal entrypoint:

```text
xlabs_flux_ip_adapter/
  modal_inference.py   # Modal remote function and local CLI entrypoint
  config.py            # model ids, default paths, default reference image
  model_loader.py      # FLUX.1-dev + XLabs IP-Adapter loading
  conditioning.py      # first place to add visual/text condition modulation
  inference.py         # generation flow
  LICENSE_NOTES.md
```

Shared Modal runtime settings live in:

```text
shared/modal_runtime.py
```

## Hugging Face access

`black-forest-labs/FLUX.1-dev` and XLabs adapter weights require Hugging Face access and license review. Before running inference:

1. Log in to Hugging Face in a browser.
2. Accept the relevant model licenses.
3. Create a Hugging Face access token.
4. Register it as a Modal Secret:

```powershell
python -m modal secret create huggingface-secret HUGGINGFACE_HUB_TOKEN=hf_your_token_here
```

Check that the secret exists:

```powershell
python -m modal secret list
```

## Run inference

The first run downloads large model files into the Modal Volume named `hf-model-cache`.

```powershell
$env:PYTHONUTF8='1'
python -m modal run xlabs_flux_ip_adapter/modal_inference.py
```

The default output is written locally to:

```text
outputs/xlabs_flux_ip_adapter.png
```

Override inference arguments from the CLI:

```powershell
python -m modal run xlabs_flux_ip_adapter/modal_inference.py `
  --prompt "a portrait photo of the same person wearing a white hoodie, outdoor daylight" `
  --reference-image-url "https://example.com/reference.png" `
  --ip-adapter-scale 0.8 `
  --num-inference-steps 20 `
  --seed 123
```

## Credit notes

Keeping Modal authenticated or storing code locally does not consume GPU credit.
Credit can be consumed when `modal run` starts a remote GPU container, including the short idle time before Modal shuts the container down.
