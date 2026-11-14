# FreeGraftor 512: completed benchmark export

This directory independently records the completed 2026-11-14 FreeGraftor
experiment. The existing `freegraftor/` directory is unchanged.

| Evaluation cohort | Images | CLIP-T | CLIP-I | DINO-I |
|---|---:|---:|---:|---:|
| DreamBench | 750 | 0.315605 | 0.782324 | 0.654891 |
| DreamBench++ non-style | 1170 | 0.342527 | 0.760082 | 0.581627 |

These reference-subject macro scores use the historical
`uno-selected-reference-v1` full-image CLIP ViT-B/32 and DINO v1 ViT-S/16
protocol. All 1,920 non-style cases are complete. All style cases are excluded:
14 of 180 attempted style cases failed. There are 36 audited detector-label
recovery cases; generation prompts, selected references and hyperparameters
were unchanged. See [results](results/runpod/README.md) for scope and provenance.

- [Final per-case scores](results/runpod/seed42_512/metrics/scores.json)
- [Aggregate CSV](results/runpod/seed42_512/metrics/summary.csv)
- [Matched InstantX comparison](results/runpod/seed42_512/metrics/comparison.json)
- [Execution and recovery code](runpod/README.md)
- [Image export manifest and checksum](results/runpod/downloads/)

The frozen native source and model-conversion signatures match the original
run. Code retains its signed runtime paths under `/workspace/freegraftor`;
`freegrafor2` is this repository's independent archival directory name.
The single-GPU adapter preserves the two original logical case assignments
and executes them sequentially on physical GPU 0.

Generated images, reference/mask images, model weights and ZIP archives are
excluded from Git and Git LFS. Original per-image JSON metadata are under
`results/runpod/seed42_512/execution_records/`. The separately downloadable ZIP
contains exactly:

```text
FreeGraftor_512/
  dreamBench/          # 750 PNGs only
  dreamBench_plus/     # 1170 PNGs only; no style cases
  execution_records/  # 1920 original JSON sidecars, metrics and run metadata
```

Original PNG and per-image JSON bytes are preserved; every member SHA256 and
the ZIP CRC were verified. The portable export manifest maps images to their
metadata. Original JSON source-machine paths and signed mask-audit references
are retained. Exported images and the source mask binary assets remain outside
this Git repository.
