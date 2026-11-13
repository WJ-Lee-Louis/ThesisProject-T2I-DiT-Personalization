# USO: Canonical Native 1024 Evaluation

Completed inference and common evaluation of the official pretrained USO model.
DreamBench: 750/750 cases. DreamBench++: 1350/1350 cases, including 180 style
references. No missing cases or replacements.

## Results

Subject-macro averages using one selected canonical reference per generated
image, the original target prompt, CLIP ViT-B/32 and DINOv1 ViT-S/16.

| Benchmark | Images | CLIP-I | CLIP-T | DINO-I |
|---|---:|---:|---:|---:|
| DreamBench | 750 | 0.8302 | 0.3198 | 0.7730 |
| DreamBench++ full | 1350 | 0.7750 | 0.3494 | 0.6201 |
| DreamBench++ non-style subset | 1170 | 0.7835 | 0.3494 | 0.6514 |

Machine-readable [summary JSON](results/reports/summary.json) and
[CSV](results/reports/summary.csv) retain full precision and evaluation scope.

## Images

[Download USO_1024.zip](https://drive.google.com/file/d/1kKre7W5IRzoij9pIaozEFLQ9NwmqSIcQ/view)
from the [shared Drive folder](https://drive.google.com/drive/folders/1TGZGEdoiG_2uvCHQ9PfAmMor9qfMOTmT).
Generated PNGs and image archives are not stored in Git.

Size: 2655155265 bytes. SHA-256:
`957d4244157749214063ad02a0911ae4d96767d15370280a87c88519c8e7eb6d`.

```text
USO_1024.zip
  dreambench/<canonical-case-id>.png
  dreambench_plus/<canonical-case-id>.png
  execution_records/<benchmark>/<run_id>/...
  execution_records/upload_manifest.json
```

All native 1024x1024 PNG bytes are unchanged from Modal generation. Original
prompts, reference IDs/hashes, metadata, model/code provenance and run records
are retained. Reference pixels remain in the original Modal preservation bundles
and shared reference collection.

[Local archive verification](results/reports/drive_upload_verification.json)
records whole-ZIP SHA-256, member SHA-256/CRC and actual PNG dimension checks.
[Drive verification](results/reports/google_drive_verification.json) separately
records exact remote length, complete entry names/sizes and upload-manifest
hash checks. Full remote-image SHA-256 was not recomputed.

## Reproduction

See [Modal guide](MODAL_GUIDE.md), [config](config.json),
[locked model revisions](models.lock.json) and [upstream source hashes](UPSTREAM.json).
The vendored [official source](https://github.com/bytedance/USO) is unchanged at
commit `6587514aa3adf8e8f46e5f7e804239651d30b32d`, with its [license](upstream/LICENSE).
The wrapper performs inference only: seed 42, native 1024x1024, 25 steps,
guidance 4.0, pretrained LoRA rank 128, and official content/style routes.
No prompt augmentation, per-subject training or post-generation resizing is used.

Completed run records:
- [DreamBench](results/dreambench/uso1024-dreambench-s42-59e9e8cf62e1b2c8/)
- [DreamBench++](results/dreambench_plus/uso1024-dreambench_plus-s42-736623e31bbbb95d/)

Local generation plans and the faithfully collected export/evaluation records are
kept separately. Source snapshots also remain inside the Drive ZIP. Credentials,
temporary receipts and local client state stay in ignored `outputs/`; tokens
must never be committed.
