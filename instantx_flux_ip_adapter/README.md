# InstantX FLUX IP-Adapter

Modal-based inference scaffold for `InstantX/FLUX.1-dev-IP-Adapter` on top of `black-forest-labs/FLUX.1-dev`.

InstantX's model card says the code has not been integrated into Diffusers yet, so this scaffold downloads the repository's custom files at runtime:

```text
attention_processor.py
infer_flux_ipa_siglip.py
pipeline_flux_ipa.py
transformer_flux.py
ip-adapter.bin
```

## Files

```text
instantx_flux_ip_adapter/
  modal_inference.py   # Modal remote function and local CLI entrypoint
  config.py            # model ids and default paths
  model_loader.py      # InstantX custom code + FLUX.1-dev loading
  conditioning.py      # reference preprocessing and scale controls
  inference.py         # generation flow
  LICENSE_NOTES.md
  reference_images/
```

## Run inference

```powershell
$env:PYTHONUTF8='1'
python -m modal run instantx_flux_ip_adapter/modal_inference.py `
  --reference-image-path "dreambooth_reference_dataset/dog/00.jpg" `
  --prompt "a dog running on the beach"
```

Use a local image in this folder:

```powershell
python -m modal run instantx_flux_ip_adapter/modal_inference.py `
  --reference-image-path "instantx_flux_ip_adapter/reference_images/person_01.png" `
  --prompt "a person wearing sunglasses"
```

## Useful controls

InstantX's model card example uses `scale=0.7`, so this scaffold uses the same default.

```text
--ip-adapter-scale 0.7
--guidance-scale 3.5
--num-inference-steps 24
--width 960
--height 1280
```

Outputs are written to:

```text
outputs/instantx_flux_ip_adapter/<reference_name>_generated.png
```

If the filename already exists, `_001`, `_002`, and so on are appended.

## Notes

InstantX describes this as a regular IP-Adapter using `google/siglip-so400m-patch14-384` as the image encoder, with image tokens inserted into FLUX single and double blocks. The model card also notes limitations for fine-grained style transfer and character consistency.
