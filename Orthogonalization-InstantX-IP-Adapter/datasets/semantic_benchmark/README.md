# Full semantic benchmark preparation

180 references: DreamBench 30 × 25 targets (750), DreamBench++ 150 × 9 (1350).
`references.json` contains original identity labels, image paths/hashes and target text,
checked against the pinned DreamBench manifest and DreamBench++ caption files.
It contains no generated captions. Original source datasets remain unchanged.

Run `python scripts/run_full_semantic_benchmark.py --run` from the project root.
Model: `gpt-5.6-sol` for both stages; current runtime prompts are snapshotted once.
Credentials are read from SEMANTIC_LLM_API_KEY / SEMANTIC_LLM_ENDPOINT, never saved.
Existing valid successes resume without new calls. Add `--retry-errors` to retry failures.
Output: `outputs/semantic_benchmark_gpt56_sol/`:

- `run.json`, `prompts.json`: exact model/input/prompt provenance.
- `captions/`: 180 reference-bound A records, including preserve and raw response traces.
- `conditioning/`: one complete, validated runtime conditioning JSON per successful target.
- `generation_manifest.json`: original target/reference plus READY paths or explicit failures.
- `summary.json`, `progress.json`, `errors/`: execution counts and failures, not semantic accuracy.

The runner uses at most two output-validation retries per stage and two bounded transient
transport retries. Interrupted requests may have been billed without a saved result; trace
attempt totals are not billing totals. It never performs visual caption review, image generation,
or real SigLIP evaluation. Each B negative is stored directly for original-positive-minus-negative
local directions; generation must keep the manifest's original target text.

Intermediate pilots were moved outside the active project to
`../contrast_intermediate_archive_20261020` (relative to project root) after permanent deletion
was blocked by automatic approval review. Original baseline caption v1 and separate legacy
research data remain for provenance and existing research tools.

Offline full saved-output verification:
`python scripts/verify_full_semantic_benchmark.py`.
It checks reference/target/caption bindings, round trips, all three origin plans, exact negative
encoder inputs, normalized positive-minus-negative signs, and text deduplication using a mock.
`verification.json` is not a semantic or image-quality score.

For later generation, pass a READY row's `conditioning_path` to Modal
`--contrast-analysis-path`, its `image_path` to `--reference-image-path`, and its unchanged
`target_prompt` to `--prompt`, with `--llm-model gpt-5.6-sol`. Do not supply a new caption
or preserve alongside that full cache. In Python use `contrast_analysis_payload` with the
same original reference/target. Cache loading needs no additional LLM call.

## Completed run — 2026-10-20 KST

GPT-5.6 Sol A `caption_np_v6` / B `joint_contrast_v8`:
180/180 captions saved; all 2100 targets submitted; 2100 generation-ready conditionings:
2098 regular joint results and 2 explicitly authorized source-only fallbacks. These are structural outcomes, not semantic accuracy.
Local text direction rows: source 3657 + target 5170 = 8827; actual embedding rank is not measured.

Offline: 304 contract/integration/batch/fallback tests passed.
Full saved-output verification includes all 2100 conditionings. Source-only fallback
rejects target-only selection; source and source_target selections keep the source directions. No captions
were visually inspected, no real SigLIP evaluation was performed, and no images were generated.
Three initial A failures were exact-case preserve evidence mismatches; explicit retry feedback
resolved them. Failed paid traces remain in recovered_failures. A transient Windows progress-file
rename failure was fixed with bounded rename retries. Finalization was rerun with API requests
programmatically forbidden; all successes were reused.

### Recorded conflicts with authorized source-only fallback

These are observations of B's judgment, not independently established semantic contradictions.
The original targets and conflict traces are preserved. The original joint requests were not
retried to obtain a preferred answer. Instead, the user explicitly authorized one additional
source-only analysis per case; these two fallback API calls have completed.

- `092-08` / identity `man`: A depiction of a man weaving colorful dreams into the night sky, stars twinkling around her fingers
  Reported: IdentityConflictError: identity_conflict (target): The pronoun “her” refers to the man and contradicts the protected identity.
- `154-07` / identity `Eevee figurine`: An Eevee figurine transforming into a real Eevee under a magical, starry sky
  Reported: IdentityConflictError: identity_conflict (target): The transformation explicitly changes the protected Eevee figurine identity into a real Eevee.


Fallback artifacts use kind `semantic_source_fallback`, schema 4, policy
`target_conflict_source_only_v1`, and separate prompt/stage `source_only_fallback_v1`.
The normal A/B prompts and 2098 normal caches are unchanged. Both original conflict traces
and new fallback traces are bound to the same caption/target/identity/preserve/model.
`generation_manifest.json` marks the two READY rows with a non-null `fallback` field;
`summary.json` lists `fallback_cases`. Original `errors/B-092-08.json` and
`errors/B-154-07.json` are retained provenance, not outstanding generation failures.

- 092-08: source 0 / target 0 directions. The caption contains only identity and protected
  hair/clothing features, so no source removal was produced. The empty-list path makes no
  fabricated projection direction; this is not an unconditional disabling of source analysis.
- 154-07: source 1 / target 0 directions. The source negative removes `on a forest floor`
  while retaining the Eevee figurine and protected appearance.

Both receive the unchanged original target at generation. A full 2100-pair evaluation should
report these two source-only fallbacks. No images or real embedding/metric evaluation have run.
Reproduction (only explicitly listed unresolved target-conflict cases can invoke fallback):
`python scripts/run_full_semantic_benchmark.py --run --source-only-fallback 092-08 154-07`.
Existing prepared fallbacks are validated and reused without new API requests.
