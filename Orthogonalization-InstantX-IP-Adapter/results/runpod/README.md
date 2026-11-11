# RunPod experiment records

These are the existing RunPod InstantX results from October 2026, imported
without regenerating images or recomputing metrics. Each condition has 750
DreamBench and 1,350 DreamBench++ cases, with seed 42, 24 steps, 1024×1024
output and guidance 3.5.

| Condition | Scale | Metric directory | External image ZIP |
|---|---:|---|---|
| Source + target | 0.7 | `full/metrics/` | `source_target_scale0.7.zip` |
| Source + target + SSG | 0.8 | `full/metrics/` | `source_target_ssg_scale0.8.zip` |
| Source + target + SSG | 0.6 | `scale_sweep_20261025/metrics/` | `source_target_ssg_scale0.6.zip` |
| Source + target + SSG | 0.9 | `scale_sweep_20261025/metrics/` | `source_target_ssg_scale0.9.zip` |
| Source + target + SSG | 1.0 | `scale_sweep_20261025/metrics/` | `source_target_ssg_scale1.0.zip` |
| SSG only | 0.7 | `ssg_only_scale_0.7_20261025/metrics/` | `ssg_only_scale0.7.zip` |
| Baseline | 0.8 | `ablation_20261026/baseline_0.8/metrics/` | `baseline_scale0.8.zip` |
| Baseline | 0.9 | `ablation_20261026/baseline_0.9/metrics/` | `baseline_scale0.9.zip` |
| Source only | 0.7 | `ablation_20261026/source_only/metrics/` | `source_only_scale0.7.zip` |
| Target only | 0.7 | `ablation_20261026/target_only/metrics/` | `target_only_scale0.7.zip` |

`scores.json` contains the original case scores and evaluation provenance;
`summary.json` and `summary.csv` contain the existing aggregates. Existing
evaluation pair lists and run/status records are retained where available.
`manifests/generation_manifest.json` is the original 2,100-case manifest.
`reports/` preserves existing table data and analysis. Historical screenshot
transcriptions in those reports are distinct from the RunPod score files.
Plot PNGs referenced by historical report documents are not included here.

The original execution and analysis scripts are in
[`../../runpod/experiments/`](../../runpod/experiments/README.md).
Original absolute server paths, timestamps and status fields are preserved:
an initial `run.json` can still say `running`, so consult the final status and
metric records when assessing completion.

Target-only evaluation was recovered after its original controller failed.
The original `ablation_20261026/target_only/metrics/recovery.json` records the
Modal recovery and the existing `target_conflict_no_projection_v1` fallback
for cases `092-08` and `154-07`; these are not silently relabelled as ordinary
target-only projection successes.

## Image storage

The ten ZIPs contain 21,000 final PNGs in total. They are stored outside Git
and Git LFS for download to the user's PC and subsequent manual Drive upload.
Each ZIP contains only:

```text
dreambench/<case_id>.png              # 750 images
dreambench_plus/<case_id>.png         # 1,350 images
execution_records/                   # existing manifest, sidecars and run logs
```

Reference images, masks, intermediates and metric files are excluded from the
image ZIPs. Metrics are preserved in this repository. JSON sidecars in the
ZIPs retain the exact prompts, settings, seed and generation diagnostics.

`image_archives/index.json` and the accompanying records document the exports
created on 2026-11-11. Their `/opt/instantx-exports-20261111` paths refer to the
temporary disk of the Pod that created them, not a permanent download service.
PC download and Drive upload completion have not been confirmed. At the user's
request, export checks were limited to counts, case-ID filenames and ZIP
directory structure; no full image/content verification was completed.

No generated images, image ZIPs, model weights, caches, virtual environments,
LFS objects or reference-image copies are added by this import.
