# Generated Image Storage and Native 512 Results

Generated PNGs and image ZIPs are distributed through the shared Google Drive,
not Git. Code, configuration, model revisions, original manifests, per-image
metrics, coverage and SHA-256/CRC verification records remain in this repository.
Source-code ZIPs are not generated-image archives and are preserved separately.
This policy supersedes the historical Git/LFS image-upload instructions in
METHOD_COMPARISON_TRANSFER.md.

Shared folder: https://drive.google.com/drive/folders/1TGZGEdoiG_2uvCHQ9PfAmMor9qfMOTmT

## Uploaded Native 512 Experiments

| Archive | DreamBench | DreamBench++ | Total PNGs | Bytes | SHA-256 |
|---|---:|---:|---:|---:|---|
| FreeCus_512.zip | 750 | 1170 non-style | 1920 | 652493842 | `55f5ebd017b820816f5b81a71a150926092ebfc88787198f553051529439cc9d` |
| UNO_512.zip | 750 | 1350 including style | 2100 | 696797490 | `a4d2d4a00faaa44f979e33b24afe78b6c59aaf920480abd1a16fd188f315d979` |

Both archives preserve native 512x512 PNG bytes. They are not downscaled 1024
outputs and were not recompressed into JPEG. Every target PNG was checked for
dimensions, byte identity with the source archive, SHA-256 and ZIP CRC.
Google Drive displays binary MiB for these files; the byte counts above are exact.

```text
FreeCus_512.zip / UNO_512.zip
  dreambench/<canonical-case-id>.png
  dreambench_plus/<canonical-case-id>.png
  execution_records/<benchmark>/<run_id>/...
  execution_records/upload_manifest.json
```

The records retain original prompts, reference IDs/hashes, metadata, manifests,
configurations, model/code provenance and evaluation results. Reference pixels
also remain in the original preservation bundles and the shared references.zip.

Local upload packages: `../AI_PROJECT_images/drive_upload_512/`.
Original preservation bundles: `../AI_PROJECT_images/freecus_512/` and
`../AI_PROJECT_images/uno_512/`. Both locations are outside this Git repository.
Copies of the upload verification records are stored in each run directory below.

## Canonical Run Records

| Method | Benchmark | Run directory |
|---|---|---|
| FreeCus 512 | DreamBench | [records](freecus/results_512/dreambench/freecus512-dreambench-s42-f86cadfd519ce417/) |
| FreeCus 512 | DreamBench++ non-style | [records](freecus/results_512/dreambench_plus/freecus512-dreambench_plus-s42-a63b054d17b9ab87/) |
| UNO 512 | DreamBench | [records](uno/results_512/dreambench/uno512-dreambench-s42-50396742e580597d/) |
| UNO 512 | DreamBench++ full | [records](uno/results_512/dreambench_plus/uno512-dreambench_plus-s42-ca39a70a1f5ebec3/) |

FreeCus keeps its 1024 paper settings except output dimensions: seed/caption_seed
42, 30 steps, guidance 3.5, scale 1.1. UNO keeps seed 42, 25 steps, guidance 4.0,
LoRA rank 512 and reference preprocessing 512; only output dimensions differ.
The original default config.json and models.lock.json files are unchanged.

## Metric Comparison

Common evaluator: CLIP ViT-B/32, DINOv1 ViT-S/16, one selected conditioning
reference per case and the original target prompt. Values below are subject macro
means, equal to case micro means for these balanced cohorts. Higher is better.

| Method | Benchmark | Resolution | CLIP-I | CLIP-T | DINO-I |
|---|---|---|---:|---:|---:|
| FreeCus | DreamBench | 1024 | 0.8184 | 0.2789 | 0.7156 |
| FreeCus | DreamBench | 512 | 0.8136 | 0.3019 | 0.6678 |
| FreeCus | DreamBench++ non-style | 1024 | 0.8210 | 0.2900 | 0.7036 |
| FreeCus | DreamBench++ non-style | 512 | 0.8104 | 0.3235 | 0.6513 |
| UNO | DreamBench | 1024 | 0.8069 | 0.3088 | 0.7178 |
| UNO | DreamBench | 512 | 0.8349 | 0.3023 | 0.7606 |
| UNO | DreamBench++ full | 1024 | 0.7768 | 0.3296 | 0.5901 |
| UNO | DreamBench++ full | 512 | 0.8101 | 0.3216 | 0.6573 |

FreeCus DreamBench++ excludes the 180 style cases from generation/evaluation;
the complete canonical 1350-case manifest and explicit exclusion list remain.
UNO includes all 180 style cases. No missing result is assigned a zero score.

## Image ZIP History Cleanup

Only generated-image ZIP paths for methods already present in the shared Drive
are selected for Git history removal. Other code, evaluation records, source
snapshots, datasets and unrelated experiment images are not selected.
Local, Modal and Drive image originals are not deleted. A history rewrite changes
commit IDs, so collaborators must not merge an old-history branch back into main.
Use a fresh clone or coordinate migration of local work after the rewrite.
