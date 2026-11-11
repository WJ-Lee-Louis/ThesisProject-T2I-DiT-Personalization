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
  results/runpod/                 # existing status, metadata, failures and pilot scores
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
| `freegraftor/manifests/runpod/*.json` | `/workspace/freegraftor/inputs/` |
| `freegraftor/results/runpod/recovery/` | `/workspace/freegraftor/recovery/` |
| `freegraftor/results/runpod/seed42_512/` | `/workspace/freegraftor/results/seed42_512/` |
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

The original controller assumes two GPU lanes and the generation worker
requires at least 75 GiB VRAM per visible GPU. Archived PID and pause records
refer to the old Pod. Do not launch `start.sh` or the completion guard blindly
on a new single-GPU Pod; inspect and adapt the launch/worker mapping first,
while retaining the frozen generation source and its recorded signatures.

The remaining failure is case `156-04`, reference
`dreambench_plus/object/61`: Grounding DINO found no `roll of film` instance in
the generated template. A different GPU does not establish that the mask
problem is resolved. The exact failure, attempted equivalent detector labels,
unchanged thresholds and recovery protocol are retained in
`../results/runpod/recovery/`.

No generation, recovery attempt, model conversion or metric evaluation was
started by this repository import.
