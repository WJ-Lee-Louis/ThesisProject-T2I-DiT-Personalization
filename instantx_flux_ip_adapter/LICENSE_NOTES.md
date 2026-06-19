# License notes

This folder is for local research scaffolding around `InstantX/FLUX.1-dev-IP-Adapter`.

## Checked sources

- InstantX model: https://huggingface.co/InstantX/FLUX.1-dev-IP-Adapter
- Base model: https://huggingface.co/black-forest-labs/FLUX.1-dev
- Image encoder: https://huggingface.co/google/siglip-so400m-patch14-384

## Practical distinction

The InstantX model card lists the license as `flux-1-dev-non-commercial-license` and states that the model is released under that license. Treat the adapter and FLUX.1-dev weights as non-commercial unless you have separate permission.

Do not redistribute downloaded model weights in this repository. Keep Hugging Face tokens in Modal Secrets or environment variables, not in source files.

This is a research note, not legal advice.
