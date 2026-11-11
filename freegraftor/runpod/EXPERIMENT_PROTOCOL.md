# FreeGraftor RunPod comparison

Output root: `results/seed42_512`. Code and experiment records persist under
`/workspace/freegraftor`. Generation packages and losslessly converted native
FLUX checkpoints are stored on the Pod's local `/opt/freegraftor-runtime`
disk to avoid the network-volume quota. `venvs/generation` and
`cache/flux-native` link there. After replacing the Pod, run `bash setup.sh`
to recreate this local runtime. Keep the existing InstantX FLUX cache available.
The metric environment and CLIP/DINO checkpoints reuse the historical
`/workspace/instantx-0807` installation without modifications.

## Protocol

- Official upstream: https://github.com/Nihukat/FreeGraftor, commit
  `b8e0f93e91753265e46e9d0b364daa38e25f1afa` (`upstream/`).
- Paper: https://arxiv.org/html/2504.15958v5 . Official CLI default is 512x512;
  the paper's efficiency comparison also uses 512x512. The pipeline dataclass's
  different default is overridden explicitly.
- Seed 42; 512x512; 25 steps each for template, FireFlow inversion and final
  generation; guidance 3; similarity 0.2; cycle 1.5; dropout 0.5; injection
  steps 0-25 and blocks 0-56; default center alignment.
- All models stay on each A100 80GB (`requires_offload=False`) to avoid CPU/GPU
  transfer. No quantization, prompt rewriting, tuning, or detection-threshold
  changes. One model instance per worker, one GPU per worker.
- Exact same 750 DreamBench and 1,350 DreamBench++ cases and 180 selected
  reference images as InstantX. Original manifest and reference checksums are
  retained in `inputs/`; each GPU gets 375 + 675 = 1,050 cases.
- `src_snapshot/src` preserves the previously integrated upstream source with
  environment-path correction, width/height fix and explicit detection errors.
  `upstream/` preserves the official source for comparison.
- Existing authorized FLUX Diffusers cache is converted by exact tensor
  rearrangement, with complete native-schema and round-trip equality checks.
  No numerical casts. `cache/flux-native/conversion.json` contains the audit.
- Metrics: unchanged `uno-selected-reference-v1`, CLIP ViT-B/32 and DINO v1
  ViT-S/16, full-image CLIP-T / CLIP-I / DINO-I, using the historical metric
  environment. This intentionally matches InstantX; it differs from the
  paper's segmented CLIP/DINOv2 and ImageReward protocol.
- Native resolutions differ: FreeGraftor 512, historical InstantX 1024.
  Report this as a comparison at method-native settings, not a controlled
  equal-resolution attribution of improvements.

## Execution and recovery

`bash start.sh` starts a detached controller. It tests all available subject
categories on both GPUs and scores those pilot images first. All five
non-style benchmark/category pairs must have a successful scored pilot case
before the full run starts. Successful pilot outputs are reused.

Every result PNG has a JSON sidecar with exact inputs, settings, code signature,
reference/PNG SHA256, runtime and package versions. Valid outputs and metric
checkpoints are reused on restart. Source/config changes reject reuse.
Do not start another controller while the run lock is held. Keep the Pod
running; closing SSH or this chat does not stop the detached controller.
There is no automatic Pod shutdown.

DreamBench++ includes 180 style cases. These are attempted with the original
subject-grafting pipeline and original identity term. Detection failures are
recorded rather than replaced by text-only images or arbitrary masks. If any
style case fails, all style cases are excluded from primary metric evaluation:
DreamBench++ then evaluates only its 1,170 non-style cases. The controller waits
for both generation lanes before fixing this shared evaluation policy and
evaluating on both GPUs. `evaluation-policy.json` records the decision. Other
unlocalizable subjects/templates can fail too. Reports show expected,
generated, scored and failed counts by category. An incomplete cohort is
never labelled a complete benchmark. `comparison.json` compares only exactly
matched successful cases against the original InstantX scores, checking
reference hashes and original prompts.

Useful files:

- `results/seed42_512/status.json`: controller stage and final status.
- `results/seed42_512/logs/lane-{0,1}.log`: generation/evaluation logs.
- `results/seed42_512/state/`: worker progress.
- `results/seed42_512/images/{dreambench,dreambench_plus}/`: PNG + JSON.
- `results/seed42_512/errors/`, `coverage.json`: explicit failed cases.
- `results/seed42_512/metrics/scores.json`, `summary.csv`, `summary.json`:
  successful image scores and subject/category summaries.
- `results/seed42_512/metrics/comparison.json`: matched-cohort InstantX deltas.

No existing InstantX results, sources or environments are overwritten.

## Required non-style completeness

The active controller is now `scripts/completion_guard.py`. It preserves the
running native inference workers, retries failed non-style subject masks with
audited equivalent detector noun phrases, and requires all 1,920 non-style
cases before final evaluation. `bash start.sh` resumes this guarded workflow.
See [recovery/PROTOCOL.md](recovery/PROTOCOL.md) for the exact retry protocol,
initial can-label diagnosis, source/mask audit records and failure handling.
Original detector failures are retained even after successful reprocessing.
