# Completed FreeGraftor RunPod results

The 2026-11-14 recovery completed the missing non-style image and final metric
aggregation for the seed-42, 512x512 FreeGraftor run. Original 2026-11-06 records
and failed-attempt audits remain available alongside the completed results.

| Evaluation cohort | Images | CLIP-T | CLIP-I | DINO-I |
|---|---:|---:|---:|---:|
| DreamBench | 750 | 0.315605 | 0.782324 | 0.654891 |
| DreamBench++ non-style | 1170 | 0.342527 | 0.760082 | 0.581627 |

These are reference-subject macro means from `seed42_512/metrics/summary.csv`,
using the unchanged `uno-selected-reference-v1` full-image CLIP ViT-B/32 and
DINO v1 ViT-S/16 protocol. They are not the paper's segmented/DINOv2 metrics.
The matched InstantX comparison uses the same references and prompts;
FreeGraftor resolution is 512 while historical InstantX resolution is 1024.
It is a comparison at native method settings, not an equal-resolution test.

- Non-style: 1,920/1,920 generated and scored, zero missing.
- DreamBench++ style: 166/180 generated, 14 failed; all style cases excluded
  from primary metrics under the recorded whole-class policy.
- Total images generated across attempted cohorts: 2,086/2,100.
- Audited detector-label recovery: 36 non-style cases. The original diffusion
  prompts, references, seed, generation settings and thresholds are unchanged.
- Final case 156-04 used film strip after roll of film failed on the template.
  Original failures and the exact recovery source versions are retained.

| Record | Location |
|---|---|
| Final controller status and evaluation policy | `seed42_512/status.json`, `seed42_512/evaluation-policy.json` |
| Final per-case scores and aggregate results | `seed42_512/metrics/scores.json`, `summary.json`, `summary.csv` |
| Matched-cohort InstantX comparison | `seed42_512/metrics/comparison.json` |
| Original image JSON sidecars | `seed42_512/execution_records/<benchmark>/<case_id>.json` |
| Non-style completeness and detector aliases | `recovery/non-style-coverage.json`, `recovery/detector_aliases.json` |
| Preserved pre-resume failure records | `recovery/resume-20261114/original/` |
| Newly extracted reference-mask detection records | `recovery/all-reference-masks-20261114/` |
| Download export receipts and portable file manifest | `downloads/` |

Pilot scores and earlier startup attempts remain historical records. Archived
PIDs and pause records refer to earlier processes. The active final status is
complete_with_style_exclusions, with all non-style cases included.

Generated PNGs, reference images, templates, masks, ZIP archives, model weights
and environments are excluded from Git and Git LFS. The download archive has:

```text
FreeGraftor_512/
  dreamBench/          # 750 PNGs only
  dreamBench_plus/     # 1170 non-style PNGs only
  execution_records/  # original JSON metadata, metrics, policy and checksums
```

Original JSON absolute paths and signed audit hashes are preserved. Exported
image/metadata relative paths are mapped in execution_records/export_manifest.json.
Git metadata alone does not contain the image or mask binary assets.
