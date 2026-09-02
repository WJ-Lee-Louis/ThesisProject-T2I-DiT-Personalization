# Orthogonalization InstantX FLUX IP-Adapter

Training-free research overlay for the interventions described in
`0723_experimental_design_memo.md`, including target-, source-, and
source-target pooled-space projection variants.

- SigLIP pooled-space content projection
- FLUX single-stream direct IP-to-text residual gate
- Their combination
- Reference-caption behavior/background projection, with or without the gate
- One-SVD union of target-prompt and reference-caption directions, with or
  without the gate
- Absolute generic-class and source-contextual class-contrast projection,
  both with the single-stream gate

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
source/target direction cosine matrix, normalized-direction singular values,
pre-normalization direction norms, projection/edit norms, input/output cosine,
mask state, and the actual FLUX text-row length.

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
the structured 14-prompt catalog into 140 unique jobs per full condition, and
commits each PNG and diagnostics file independently.

The finalized API exposes exactly 22 unambiguous conditions: `baseline`,
SSG-only `ssg`, four nuisance-projection origins with an `_ssg` suffix, and a
4-by-4 matrix crossing those origins with `absolute_direct`,
`absolute_residual`, `no_object`, and `superclass` generic-class strategies.
Old `combined`, no-gate projection, class-only absolute, and compatibility
alias names are not accepted for new submissions.

Every intervention condition uses the single-stream gate. Absolute-direct
keeps overlap between the nuisance SVD basis and the normalized `a dog/cat`
anchor as a deliberate ablation. Absolute-residual, no-object, and superclass
remove this overlap before applying the independently scaled class projection.
Generic conditions accept class strengths 0.5 and 1.0. Source and ST conditions
require `source_captions.json`. See
[CLASS_PROJECTION_MODAL_GUIDE.md](CLASS_PROJECTION_MODAL_GUIDE.md) for the full
22-name matrix and Modal commands.

Each new condition generates a self-contained 140-image batch. ST-all does not
assume that a corresponding ST-matched batch exists, so its O+A+B jobs are not
automatically omitted.

Start with a no-GPU preview:

```powershell
python Orthogonalization-InstantX-IP-Adapter\batch_client.py preview
```

The final six-condition, lambda-1 experiment—including CMD deployment,
submission, status, recovery, and download commands—is documented in
[FINAL_16_PROMPT_MODAL_GUIDE.md](FINAL_16_PROMPT_MODAL_GUIDE.md).

The earlier batch workflow is retained in
[BATCH_MODAL_GUIDE.md](BATCH_MODAL_GUIDE.md) for historical reference.

The class-component condition matrix and its independent lambda sweep are
documented in [CLASS_PROJECTION_MODAL_GUIDE.md](CLASS_PROJECTION_MODAL_GUIDE.md).

The two concurrent source-aware runs (90 + 90 images) are documented in
[SOURCE_AWARE_MODAL_GUIDE.md](SOURCE_AWARE_MODAL_GUIDE.md).

## Final comparison grids and basis diagnostics

The existing compositor still renders the historical six-method experiment
using its original batch metadata names. New 22-condition submissions use the
SSG names documented above. Earlier comparison-grid outputs remain under
`outputs/comparison_grids`.

```powershell
python Orthogonalization-InstantX-IP-Adapter\create_comparison_grids.py
```

The script validates batch conditions, orthogonalization strength, common
generation settings, reference/caption digests, required files, and image
dimensions before rendering. To intentionally regenerate existing grids, add
`--overwrite`.

After downloading the batches, summarize the stored basis/SVD diagnostics
without loading a model or using a GPU:

```powershell
python Orthogonalization-InstantX-IP-Adapter\summarize_basis_diagnostics.py <batch-directory> [<batch-directory> ...] --output-dir Orthogonalization-InstantX-IP-Adapter\outputs\basis_analysis_v2
```

This writes a per-image CSV and grouped JSON summaries by condition, prompt
category, and semantic factors.
