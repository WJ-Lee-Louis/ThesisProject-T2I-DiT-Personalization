# USO 1024 Canonical Comparison

Profile: `sangyun-lee0`. The wrapper explicitly selects this profile; no active
profile change is required. No other deployed experiment is stopped or modified.

Official source: https://github.com/bytedance/USO
Pinned commit: `6587514aa3adf8e8f46e5f7e804239651d30b32d`.
`upstream/` contains unchanged source files, enumerated by SHA-256 in UPSTREAM.json.
Model snapshots are pinned in models.lock.json and downloaded only in Modal.

Canonical cohorts: DreamBench 750; DreamBench++ 1350, including 180 style cases.
Original references, case IDs and target prompts are unchanged. One output per
case, seed 42, native 1024x1024, 25 steps, guidance 4.0, LoRA rank 128, BF16 FLUX.
No prompt augmentation. Official reference preprocessing uses long edge 512.
Animal/human/object references use the content route; style references use the
official style route with no content reference and the pinned SigLIP encoder.

The detached workflow stages models, runs a small category-covering pilot,
reuses successful pilot outputs, generates the full cohort on up to two A100-80GB
workers, then computes common selected-reference CLIP-I/CLIP-T/DINO-I and exports
verified bounded image ZIPs. It completes DreamBench before DreamBench++.
The user's Modal account spending limits apply. Failed/missing cases remain
explicitly missing and are never filled with alternate images or averaged as zeros.

```powershell
freecus/.venv-modal512-py311/Scripts/python.exe -X utf8 -m uso.experiment prepare
freecus/.venv-modal512-py311/Scripts/python.exe -X utf8 -m uso.experiment deploy
freecus/.venv-modal512-py311/Scripts/python.exe -X utf8 -m uso.experiment submit
freecus/.venv-modal512-py311/Scripts/python.exe -X utf8 -m uso.experiment status
freecus/.venv-modal512-py311/Scripts/python.exe -X utf8 -m uso.experiment collect
```

Only run submit once. After a call ID is saved, PC shutdown does not stop the
deployed cloud workflow. Status/collection can be run after reconnecting.
Receipts: `uso/outputs/modal1024-s42-sangyun-lee0-20261113/` (not committed).
Tracked records: `uso/results/<benchmark>/<run_id>/` contain config, manifests,
model/code provenance, metrics.json, per_image.csv, coverage, bundle index and
SHA-256/CRC verification. Generated PNGs and ZIPs stay outside Git at
`../AI_PROJECT_images/uso/<benchmark>/<run_id>/`. Image archive bounds are 1 GiB.

For Drive sharing, combine image bytes without resizing into `USO_1024.zip` with
`dreambench/<case_id>.png`, `dreambench_plus/<case_id>.png`, and
`execution_records/<benchmark>/<run_id>/`. Shared folder:
https://drive.google.com/drive/folders/1TGZGEdoiG_2uvCHQ9PfAmMor9qfMOTmT
Do not delete Modal originals before upload verification. No automatic git push.

## Drive Upload ZIP

The CPU-only export helper reuses the completed Modal preservation bundles and
the shared Drive layout builder. It does not regenerate or resize images.
Only the final combined ZIP is downloaded, avoiding duplicate local raw archives.

```powershell
freecus/.venv-modal512-py311/Scripts/python.exe -X utf8 -m uso.drive_bundle package
freecus/.venv-modal512-py311/Scripts/python.exe -X utf8 -m uso.drive_bundle status
freecus/.venv-modal512-py311/Scripts/python.exe -X utf8 -m uso.drive_bundle download
freecus/.venv-modal512-py311/Scripts/python.exe -X utf8 -m uso.drive_bundle stop
```

Upload `../AI_PROJECT_images/drive_upload_1024/USO_1024.zip` to the shared Drive.
The adjacent bundle index and verification JSONs record file hashes, CRC checks,
native 1024x1024 dimensions and exact coverage (750 + 1350, including style).
Local experiment records are collected from the verified ZIP without extracting
the image PNGs. Download authentication stays in ignored `uso/outputs/` only.
