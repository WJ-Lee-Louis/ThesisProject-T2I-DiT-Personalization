# Non-style subject-mask recovery

Primary evaluation requires complete valid non-style coverage: 750 DreamBench
and 1,170 DreamBench++ cases. Missing non-style cases must be repaired before
final metrics run. Style remains subject to the existing whole-class exclusion
policy if any style case fails.

The initial 25 failures all came from the one selected `dreambench/can`
reference. Its original detector query `can` gave no boxes at thresholds
0.35/0.25. The official upstream `configs/can.json` calls that subject
`beer can`. This equivalent noun phrase finds the actual foreground can with
approximately 0.95 confidence at the unchanged thresholds. Its SAM mask was
visually checked against the selected reference.

`mask_recovery.py` retries the original detector query first, then approved
equivalent subject labels from `detector_aliases.json`, then an article before
the original noun phrase. The original reference bytes, final/template
diffusion prompts, seed, resolution, sampler and all generation parameters
stay fixed. No full-frame masks, text-only replacement images, template seed
searches, or metric-driven selection are used. Empty or almost full-frame
retry masks are rejected for review.

Every repaired PNG retains its original case ID and generation job signature
and gets additional signed `mask_recovery` metadata: detector queries and
scores, unchanged thresholds, reference/template SAM masks and their hashes,
original failure archive, and recovery-source SHA256. Recovery source versions
are preserved under `source-snapshots/`. Thus the original job identity is
preserved while the changed subject-mask preprocessing is disclosed separately.
Final comparisons include the number and provenance of repaired cases rather
than treating them as unmodified original detector successes.

Original failed JSON records are preserved in `original_failures/`. A repaired
case leaves the active error directory only after PNG integrity, native input
signature and additional mask audit checks pass.

The first recovery worker's reference detector-call logging labelled PIL inputs
as templates because native `segment()` forwards both input types as PIL
images. `normalize_recovery_audit.py` corrects only those stage labels from the
ordered signed mask artifacts; it archives the original audit and records the
correction source. PNG bytes and detector results are unchanged. Later worker
versions record the segment stage directly.

`completion_guard.py` supervises the original live generation workers, runs
post-generation retries for any additional non-style failures, and validates
all 1,920 non-style PNG/JSON/mask audit records before invoking the existing
CLIP/DINO evaluator. If an equivalent-label retry cannot produce a valid mask,
it writes `needs_mask_review` and withholds final evaluation instead of silently
dropping that case or claiming a complete benchmark.

An immediate rescue temporarily suspends one native GPU worker while a
separate memory-capped recovery process uses that GPU. The other GPU's native
generation continues. The suspended worker resumes in `finally`, including
on recovery errors, with a supervisor timeout and a completion-guard watchdog.
Subsequent queued retries run after the native workers exit, on both GPUs.

`native-worker-pause.json` records the pause interval and interrupted native
case. Its raw elapsed time includes that pause; exclude it from throughput/ETA
estimates. Original elapsed times are retained rather than overwritten.
