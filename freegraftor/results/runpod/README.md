# Existing FreeGraftor RunPod results

These are the original records from the 2026-11-06 run. They are preserved
without generating images, changing settings, or recomputing metrics.

The final recorded status is `needs_mask_review`:

- DreamBench: 750 generated images out of 750.
- DreamBench++: 1,335 generated images out of 1,350.
- Non-style: 1,919 complete out of 1,920; case `156-04` remains unresolved.
- Style: 166 generated out of 180; 14 failures. Under the recorded policy,
  any style failure excludes the whole 180-case style cohort from primary
  evaluation.
- 35 non-style cases were reprocessed with audited detector-label retries.
- Final benchmark metrics are withheld until non-style coverage is complete.
  Files labelled `pilot` are pilot evaluations, not final benchmark scores.

Authoritative existing records:

| Record | Location |
|---|---|
| Final controller status | `seed42_512/status.json` |
| Run settings and source/model signatures | `seed42_512/run.json` |
| Generated-image metadata, prompts and diagnostics | `seed42_512/execution_records/` |
| Remaining generation failures | `seed42_512/errors/` |
| Original pilot evaluation | `seed42_512/metrics/*pilot*` |
| Non-style completeness | `recovery/non-style-coverage.json` |
| Exact remaining detector failure | `recovery/unresolved/dreambench_plus/156-04.json` |
| Original failures, detector/mask audits and recovery code versions | `recovery/` |
| Retry protocol | `recovery/PROTOCOL.md` |

Earlier startup attempts are retained in their original named result
directories. Historical PID, ETA, pause and `running` records describe the old
run; they do not indicate a currently running process. The final evaluation
policy and final scores were not produced, and are not synthesized here.

All PNG/JPEG images, selected-reference copies, templates, masks, caches and
weights are excluded from this repository import. The network volume retains
the original image/mask bytes; the metadata's absolute paths and SHA256 values
are unchanged. Restoring metadata alone is not a complete binary backup.
