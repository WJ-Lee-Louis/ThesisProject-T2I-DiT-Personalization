# Orthogonalization InstantX FLUX IP-Adapter

Training-free research overlay for the three interventions described in
`0723_experimental_design_memo.md`, plus source-aware projection variants.

- SigLIP pooled-space content projection
- FLUX single-stream direct IP-to-text residual gate
- Their combination
- Reference-caption behavior/background projection, with or without the gate

The existing `instantx_flux_ip_adapter` package remains the untouched baseline.
This package shares the Hugging Face weight cache but uses a separate Modal app,
output directory, Python namespace, pinned InstantX/SigLIP revisions, and an
exact Modal dependency lock. Runtime versions are written to every sidecar JSON.

## Four required cells

| Projection | Text-row gate | Arguments |
|---|---|---|
| Off | Off | defaults |
| On | Off | `--orthogonalization-mode factorial` |
| Off | On | `--mask-single-text-rows` |
| On | On | both arguments |

## Factorial projection example

Run from `C:\Users\82107\Downloads\AI_PROJECT`:

```powershell
python -m modal run Orthogonalization-InstantX-IP-Adapter/modal_inference.py `
  --reference-image-path "dreambooth_reference_dataset/dog/00.jpg" `
  --prompt "a dog running on a beach" `
  --orthogonalization-mode "factorial" `
  --orthogonalization-strength 0.5 `
  --object-prompt "a dog" `
  --object-behavior-prompt "a dog running" `
  --object-background-prompt "a dog on a beach" `
  --object-behavior-background-prompt "a dog running on a beach"
```

Add `--mask-single-text-rows` for the combined condition.

For an action-only local contrast, supply the object and object+behavior prompts
and disable the background factor:

```powershell
python -m modal run Orthogonalization-InstantX-IP-Adapter/modal_inference.py `
  --reference-image-path "dreambooth_reference_dataset/dog/00.jpg" `
  --prompt "a dog running" `
  --orthogonalization-mode "local" `
  --object-prompt "a dog" `
  --object-behavior-prompt "a dog running" `
  --no-remove-background
```

Each generated PNG receives a sidecar JSON containing the retained basis rank,
edit norm, removed-component norm, input/output cosine, mask state, and the
actual FLUX text-row length.

## Verification

```powershell
python -m compileall Orthogonalization-InstantX-IP-Adapter
python -m pytest Orthogonalization-InstantX-IP-Adapter/tests
```

The first GPU check should use the Off/Off condition and compare it with the
existing InstantX baseline under the same reference, prompt, seed, step count,
dimensions, and scales.

## Resumable Modal batches

The animal dataset batch uses a separate deployed Modal class and named input
and result Volumes. It validates exactly 7 dog plus 3 cat references, expands
the 9 fixed prompts into 90 unique jobs per condition, and commits each PNG and
diagnostics file independently.

The original conditions remain available as `baseline`, `projection`, `gate`,
and `combined`. Source-aware experiments use separate condition names:

- `sa_projection`: construct behavior/background directions from each
  reference caption and remove the factor required by the target prompt type.
- `sa_combined`: apply the same source-aware projection plus the single-stream
  text-row gate.

The source-aware conditions require `source_captions.json`; the original four
conditions neither load nor modify it.

Start with a no-GPU preview:

```powershell
python Orthogonalization-InstantX-IP-Adapter\batch_client.py preview
```

Deployment, asynchronous submission, status polling, recovery, and download
commands are documented in [BATCH_MODAL_GUIDE.md](BATCH_MODAL_GUIDE.md).

The two concurrent source-aware runs (90 + 90 images) are documented in
[SOURCE_AWARE_MODAL_GUIDE.md](SOURCE_AWARE_MODAL_GUIDE.md).

## Six-method comparison grids

After all eight batch variants have been downloaded, create one 8-variant by
9-prompt comparison PNG for each reference:

```powershell
python Orthogonalization-InstantX-IP-Adapter\create_comparison_grids.py
```

The script validates the batch conditions, expected orthogonalization strengths,
common generation settings, reference and caption digests, all 720 expected
files, and their image dimensions before rendering. It writes ten
`3066x3494` PNGs and `comparison_manifest.json` under
`outputs/comparison_grids`. The final two rows compare target and source-aware
projection at lambda 1.0 with the single-stream gate. To intentionally
regenerate existing grids, add `--overwrite`.
