# Preserved RunPod execution scripts

These files preserve the existing scripts from
`/workspace/instantx-0807/analysis_tools/` and `run_pipeline.sh`.
They accompany the original [RunPod results](../../results/runpod/README.md).

- `run_scale_sweep.py` and `start_scale_sweep.py`: source/target/SSG scales.
- `run_ssg_only.py` and `queue_ssg_only.py`: SSG-only ablation.
- `run_ablation_20261026.py` and `start_ablation_20261026.py`: baselines and
  source-only/target-only ablations.
- `build_*.py`: existing comparison and table builders.
- `package_visual_results.py`: historical four-condition export.
- `package_experiments.py`: ten-condition export with full verification.
- `package_fast.py`: ten-condition export used on 2026-11-11 after the user
  requested skipping full verification; imports `package_experiments.py`.

The scripts retain their original source bytes and absolute server paths so
that source signatures and execution provenance remain interpretable. To use
the original commands, restore these Python files to the original
`/workspace/instantx-0807/analysis_tools/` directory. Several controller scripts
invoke sibling scripts there by absolute path. The original code clone path
is `/workspace/Ours/ThesisProject-T2I-DiT-Personalization`.

The repository's `scripts/benchmark_runpod.py` includes the existing server
change that orders jobs by experiment before partitioning workers. The other
generation modules match the repository versions used for this import.
Existing job signatures and recorded implementation hashes are not rewritten.

These are archival execution sources, not a new launch request. The original
generation workers enforce the recorded GPU requirements; do not interpret
their presence as verification that the current Pod can rerun generation.
