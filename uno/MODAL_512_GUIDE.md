# UNO 512 after FreeCus

Completed on profile `third`: 750 DreamBench + 1350 DreamBench++ images,
including style, with full metric coverage and verified exports. Shared Drive
archive: `UNO_512.zip`.
Download: https://drive.google.com/drive/folders/1TGZGEdoiG_2uvCHQ9PfAmMor9qfMOTmT
See [shared storage and 512 results](../GENERATED_IMAGE_STORAGE.md).

Native 512x512 generation, not a resized 1024 result. The original UNO config.json,
upstream sources and models.lock.json are preserved. Only output width/height change:
seed 42, 25 steps, guidance 4.0, 512 reference preprocessing, LoRA rank 512, pe=d,
BF16 checkpoint handling, no quantization and no CPU offload remain unchanged.
DreamBench is 750 cases and DreamBench++ is all 1350 cases, including 180 style cases.

The detached CPU queue waits for the accepted FreeCus 512 recovery workflow to finish
both benchmarks with complete metric coverage and verified image ZIP exports. It does
not require the local PC or a Google Drive upload. If FreeCus fails or finishes with
missing cases, UNO is blocked instead of starting. Once ready, UNO runs DreamBench
first, then DreamBench++; each uses a small pilot and at most two A100-80GB workers.
The pilot images are reused by the full phase. No historical budget is reused:
the user's Modal workspace usage/spend limits are authoritative. Partial results,
worker logs and missing case IDs remain if credit limits interrupt the run.

Account migration: set `$env:UNO512_PROFILE='third'` before commands to use
workspace sangyunzzang1. This uses separate app/Volume names ending in `-third`
and receipts in `uno/outputs/modal512-20261112-third/`. The old account's UNO app
was stopped. A SHA-256-pinned verified FreeCus completion report is transferred
as prerequisite evidence; FreeCus is not rerun or accessed across workspaces.
The new account downloads its own pinned UNO model weights remotely.

Original profile: second / workspace sangyun07072323 (checked before preparation).
App: uno-512-seed42-after-freecus-20261112.
Separate Volume: uno-512-seed42-20261112-runs. Model cache hf-model-cache is reused.
No results from UNO 1024 are resumed, overwritten, or mixed into this run.

From the repository root:

```powershell
$env:UNO512_PROFILE='third'
freecus/.venv-modal512-py311/Scripts/python.exe -X utf8 -m uno.experiment_512 status
freecus/.venv-modal512-py311/Scripts/python.exe -X utf8 -m uno.experiment_512 collect
```

Status/receipt files: uno/outputs/modal512-20261112/.
Records: uno/results_512/<benchmark>/<run_id>/ (metrics.json, per_image.csv,
manifest.jsonl, provenance.json, bundle_index.json, verification.json).
Images: ../AI_PROJECT_images/uno_512/<benchmark>/<run_id>/images-000.zip (outside Git).
ZIPs preserve original PNGs, metadata, original prompts, conditioning references,
model revisions, generation config and source. Existing artifact_bundle format,
1 GiB part bounds, member SHA-256 and CRC verification are reused. Google Drive
upload is manual; add the share link to README only when one is provided.

The common CLIP ViT-B/32 / DINOv1 vits16 evaluator uses the original target prompt
and selected conditioning reference. It uses the verified existing DINO cache,
without disabling TLS validation. Incomplete cohorts are explicitly partial;
missing images are not scored as zero. Local collection is separate from remote
generation/evaluation/export and requires the PC to be on. Never repeat queue
submission or delete receipts to bypass duplicate guards.
