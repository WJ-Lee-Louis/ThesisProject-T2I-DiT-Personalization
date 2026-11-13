# FreeCus 512 experiment

Completed: 750 DreamBench + 1170 non-style DreamBench++ images, with full
metric coverage and verified exports. Shared Drive archive: `FreeCus_512.zip`.
Download: https://drive.google.com/drive/folders/1TGZGEdoiG_2uvCHQ9PfAmMor9qfMOTmT
See [shared storage and 512 results](../GENERATED_IMAGE_STORAGE.md) for hashes,
native-resolution validation and the Drive folder layout.

This is a fresh native 512x512 generation experiment, not a resized 1024 run.
The original config.json, model revisions, canonical manifests and evaluator are unchanged.
config.512.json sets 512x512, 30 steps, guidance 3.5, scale 1.1, paper implementation,
seed 42 and caption_seed 42. The revised user-approved cohort is all 750 DreamBench cases
and 1170 DreamBench++ non-style cases (1920 generated images total). The 180 style cases
are excluded before generation, matching the prior 1024 evaluation scope. The canonical
1350-case manifest is preserved unchanged and the exclusion list is recorded separately.
No replacement cases or segmentation fallbacks are used.

## Isolation

- Modal profile: second; workspace was checked before submission, not inferred from old guides.
- App: freecus-512-paper-seed42-20261112.
- Results Volume: freecus-512-paper-seed42-20261112-runs.
- Reused model cache: hf-model-cache; existing huggingface-secret is never printed.
- Client state: freecus/outputs/modal512-20261112/.
- Records and metrics: freecus/results_512/<benchmark>/<run_id>/.
- Image ZIPs: ../AI_PROJECT_images/freecus_512/<benchmark>/<run_id>/ (outside Git).

Use these commands from the repository root in PowerShell:

```powershell
freecus/.venv-modal512-py311/Scripts/python.exe -X utf8 -m freecus.experiment_512 prepare
freecus/.venv-modal512-py311/Scripts/python.exe -X utf8 -m freecus.experiment_512 deploy
freecus/.venv-modal512-py311/Scripts/python.exe -X utf8 -m freecus.experiment_512 submit
freecus/.venv-modal512-py311/Scripts/python.exe -X utf8 -m freecus.experiment_512 status
freecus/.venv-modal512-py311/Scripts/python.exe -X utf8 -m freecus.experiment_512 collect
```

Do not repeat prepare/deploy/submit after acceptance. A local receipt blocks duplicate submission;
the deterministic remote workflow ID also refuses an existing workflow. Use status instead.
Submission goes to a deployed function via spawn. After the receipt includes a call ID,
the PC/network can be turned off: staging, pilot, generation, evaluation and ZIP export
continue remotely. Local ZIP downloading requires the PC to be running.

The coordinator runs the two benchmarks sequentially, with up to two A100-80GB workers.
Each benchmark gets a 12-case pilot drawn only from its selected generation cohort.
Valid pilot images and captions are reused; recorded failed cases are not silently retried.
Measurements contain actual dimensions, image time, peak allocated/reserved GPU memory,
host peak memory, estimated generation cost and a 25% margin. Estimates are not the actual bill.
The user's Modal workspace spending limit is authoritative; no old $25/$29/$40 budget is reused.

After generation, a T4 uses the unchanged shared CLIP ViT-B/32 and DINOv1 vits16 evaluator,
one selected original reference, and the original target prompt. Output JSON/CSV records
per-case metrics and subject/class aggregations. An incomplete selected cohort is explicitly
labelled partial, never filled with zero scores. DreamBench++ means non-style (1170 cases)
throughout this experiment and is never reported as a complete 1350-case full benchmark.

The original artifact_bundle pack/verify path produces PNG-only lossless archives with
1 GiB part bounds, deduplicated reference links, per-member SHA-256 and ZIP CRC checks.
Source snapshots, model revisions, config, original manifest, case failures and coverage
are preserved. Collect repeats the ZIP/member checks locally and refuses different hashes.
No model weights, generated PNGs or image ZIPs are pushed to Git.
