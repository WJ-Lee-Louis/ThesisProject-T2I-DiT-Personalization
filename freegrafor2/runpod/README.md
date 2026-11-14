# FreeGraftor RunPod execution and recovery

This directory preserves the existing `/workspace/freegraftor` experiment
sources without replacing the method's original Modal code in `freegraftor/`.
The official upstream commit is
`b8e0f93e91753265e46e9d0b364daa38e25f1afa` (Nihukat/FreeGraftor).

The method uses the same 2,100 case IDs, target prompts and 180 selected
references as InstantX: 750 DreamBench and 1,350 DreamBench++ cases. It uses
seed 42, 512×512 output, 25 steps, guidance 3, BF16, and the unchanged CLIP/DINO
metric protocol recorded in `scripts/project_metrics.py`. InstantX images are
1024×1024, so this is a comparison at the methods' native generation settings.
See the original [experiment protocol](EXPERIMENT_PROTOCOL.md),
[provenance](PROVENANCE.json), and [results](../results/runpod/README.md).

## Method-folder layout

```text
freegraftor/
  runpod/                         # preserved execution environment and code
    scripts/                      # generation, metrics, conversion, recovery
    src_snapshot/                 # exact integrated source used by generation
    upstream/                     # pinned upstream source/configuration, no images
    env.sh, setup.sh, start.sh
    requirements-generation*.txt
    source-adaptations.patch, PROVENANCE.json
  manifests/runpod/               # original manifest, reference hashes, GPU assignment
  results/runpod/                 # final metrics, metadata, failures and recovery audits
  verification/runpod/            # existing setup/model-policy evidence
```

Generated images, selected reference images, templates, masks, model weights,
virtual environments and caches remain on the network volume. They are not
added to Git or Git LFS. Original JSON sidecars and detector/mask hashes are
preserved under `results/runpod/seed42_512/execution_records/` and
`results/runpod/recovery/`.

## Restoring the existing experiment on a later Pod

The scripts retain their original bytes and absolute paths. Restore this
directory's runtime files to `/workspace/freegraftor/`, with the following
additional mappings:

| Repository records | Original runtime location |
|---|---|
| `freegrafor2/manifests/runpod/*.json` | `/workspace/freegraftor/inputs/` |
| `freegrafor2/results/runpod/recovery/` | `/workspace/freegraftor/recovery/` |
| `freegrafor2/results/runpod/seed42_512/` | `/workspace/freegrafor2/results/seed42_512/` |
| `.../seed42_512/execution_records/<benchmark>/*.json` | `.../seed42_512/images/<benchmark>/` |

The existing network volume retains the image, selected-reference and mask
bytes needed by those records; reconnect that volume before resuming. Git
alone does not contain those binary assets. Preserve the existing runtime
directories when restoring records and do not overwrite valid results.

`setup.sh` recreates the missing `/opt/freegraftor-runtime` environment and
native FLUX weights using the original cached FLUX snapshot, Python 3.10,
uv and the pinned generation dependencies. The metric environment is reused
from `/workspace/instantx-0807/venvs/metric`. Its source is preserved here as
`scripts/project_metrics.py`; no metric environment is bundled in Git.

`../verification/runpod/conversion.json` was recovered from
`PROVENANCE.json` with the exact original SHA256 stored in
`../results/runpod/seed42_512/run.json`. It records the lossless conversion and
native-weight checksums. Rebuilt model artifacts and their conversion record
must match the historical record before existing job signatures are reused.

## Completed single-GPU recovery (2026-11-14)

The replacement Pod has one A100 80GB. The original source and native model
conversion signatures were verified against the historical run. The separate
`scripts/resume_single_gpu.py` adapter runs the two original logical case lanes
sequentially on physical GPU 0, retaining the existing completeness guard,
scoring implementation and run lock. On this Pod use:

```bash
cd /workspace/freegraftor
source env.sh
/opt/freegraftor-runtime/generation-local/bin/python -u scripts/resume_single_gpu.py
```

The final missing case, `156-04` (`dreambench_plus/object/61`), was recovered
using the equivalent detector label `film strip` after `roll of film` failed
on its watercolor template. Diffusion prompts, references, seed, generation
settings and detector thresholds were unchanged. The original failure is
archived under `results/runpod/recovery/resume-20261114/original/`; the repaired
image JSON includes signed detector/SAM audit metadata. Original native source
files were not changed.

Final evaluation contains all 750 DreamBench and 1,170 non-style DreamBench++
cases, with zero missing non-style cases. There are 36 audited mask-recovery
cases in total. All 180 style cases are excluded from primary metrics because
14 style generation attempts failed. See [final results](../results/runpod/README.md).

## Image export and reference-mask review

`scripts/package_download.py` exports exactly the 1,920 scored PNGs to
`/workspace/FreeGraftor_512/{dreamBench,dreamBench_plus,execution_records}`.
The two image folders contain PNGs only. Original JSON sidecars, metric
results, run settings, policy and a portable checksum manifest are kept in
`execution_records`. The ZIP defaults to
`/opt/freegraftor-downloads/FreeGraftor_512.zip`; generated images and archives
stay outside Git and Git LFS. Run this script with the generation Python.

`scripts/export_reference_masks.py` independently re-extracts masks for all
180 selected reference images, including 20 style references. Its gallery,
reference images and masks stay outside Git; JSON detection records are
preserved under `results/runpod/recovery/all-reference-masks-20261114/`.
These are new extraction results, not restored historical masks. Primary
non-style reference review covers 160 subjects. The can query returns zero
accepted boxes, whereas beer can returns one at the same thresholds.
