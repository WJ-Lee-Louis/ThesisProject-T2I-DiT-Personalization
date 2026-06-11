# License notes

This folder is for local research scaffolding around `XLabs-AI/flux-ip-adapter`.

## Checked sources

- XLabs code repository: https://github.com/XLabs-AI/x-flux
- XLabs ComfyUI repository: https://github.com/XLabs-AI/x-flux-comfyui
- XLabs IP-Adapter weights: https://huggingface.co/XLabs-AI/flux-ip-adapter
- Base model: https://huggingface.co/black-forest-labs/FLUX.1-dev

## Practical distinction

XLabs states that its training and inference scripts are under Apache 2.0, while the released models fall under the FLUX.1 [dev] Non-Commercial License.

For this project, treat code and model weights separately:

- Local scaffolding code can be modified for research.
- `black-forest-labs/FLUX.1-dev` and `XLabs-AI/flux-ip-adapter` weights should be treated as non-commercial unless you have a separate commercial grant/license.
- Do not redistribute downloaded model weights in this repository.
- Keep Hugging Face tokens in Modal Secrets or environment variables, not in source files.

This is a research note, not legal advice.
