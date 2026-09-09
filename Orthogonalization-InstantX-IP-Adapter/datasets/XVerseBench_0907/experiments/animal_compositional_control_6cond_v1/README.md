# Animal compositional control — six-condition evaluation v1

## Scope

- Official-derived subset: 29 animal reference/prompt records, four seeds,
  116 images per condition
- Custom benchmark: 15 animal references × six prompts, 90 images per condition
- Conditions: baseline, SSG, target projection + SSG, source projection + SSG,
  ST-matched projection + SSG, ST-all projection + SSG
- Total generated images: 1,236

This evaluation is named for its research question. Modal profile and remote run
identifiers are retained only as provenance:

- execution backend: Modal
- legacy Modal profile: `modal-research-2`
- legacy metric run ID: `m2-0908`
- official XVerse revision: `65e2581aac0faffe4607881a10cc486186f6499d`

## Where to look

```text
animal_compositional_control_6cond_v1/
├── generation/
│   ├── subset/
│   │   ├── generated_images_manifest.json
│   │   └── <condition>/images/...
│   └── custom/
│       ├── generated_images_manifest.json
│       └── <condition>/images/...
├── evaluation/
│   ├── scores/{subset,custom}/{dpg,ip,aes}.json
│   ├── summaries/{subset,custom}/
│   ├── ip/references/{crops,masks,segmentation_manifest.json}
│   ├── ip/gen/{subset,custom}/<condition>/...
│   └── run_metadata/
└── LEGACY_MODAL_EXECUTION_LOG.md
```

For routine analysis, start with:

1. `ANALYSIS_GUIDE.md` for interpretation and current headline results
2. `QUALITATIVE_CASES.md` for traceable top improvement/regression samples
3. `evaluation/summaries/<benchmark>/summary_by_condition.csv`
4. `evaluation/summaries/<benchmark>/paired_delta_vs_baseline.csv`
5. `evaluation/summaries/custom/summary_by_composition.csv`
6. `generation/<benchmark>/<condition>/images/` for qualitative inspection

## Metric validity

| Metric | Status | Notes |
|---|---|---|
| DPG | valid | Dependency-aware official-style scoring completed. |
| AES | valid | Per-image aesthetic scoring completed. |
| IP-Sim | valid | Corrected Florence2 loading; 1,236/1,236 generated images detected. |

The superseded all-zero IP-Sim files were failed measurements, not identity
scores. The canonical score files now contain the corrected full rerun:

- subset: 696/696 detected, mean IP-Sim 75.8187
- custom: 540/540 detected, mean IP-Sim 74.7478

See `IP_SIM_DIAGNOSIS.md` for the stage-by-stage evidence, root cause, pinned
checkpoint, and external references.

### IP-Sim correction

- XVerse's vendored Florence2 declared tied language weights but left
  `language_model.lm_head.weight` untied and uninitialized from the checkpoint.
- The evaluator now pins the Florence2 snapshot and applies the missing weight
  sharing used by Microsoft's corrected implementation before inference.
- A GPU smoke test changed both reference and generated-image outputs from empty
  tokens to parsed bboxes; direct SAM2 scores were 0.9880 and 0.9930.
- All 29 references are now detected by Florence2 and segmented by SAM2. No
  audited/manual bbox fallback remains in the code or artifacts.
- Full-run validation rejects any future all-zero Florence2 detection result.

## Local storage rule

Use an explicit, research-oriented evaluation ID:

```cmd
set EVALUATION_ID=animal_compositional_control_6cond_v1
```

Generation download and assembly:

```cmd
python eval_batch_client.py download --batch-id <remote-batch-id> --evaluation-id %EVALUATION_ID%
python eval_batch_client.py assemble --benchmark subset --batch-id <six remote batch IDs> --evaluation-id %EVALUATION_ID%
```

Metric download and aggregation:

```cmd
python metric_client.py download --run-id <remote-run-id> --evaluation-id %EVALUATION_ID%
python metric_client.py aggregate --run-id <remote-run-id> --evaluation-id %EVALUATION_ID%
```

The remote IDs may contain infrastructure-specific names; they no longer alter
the local folder layout.

Candidate paths inside the downloaded IP score JSON preserve XVerse's remote
`generated/...` notation. Locally, that redundant segment is omitted: resolve
`generated/<condition>/...` under `evaluation/ip/gen/<benchmark>/<condition>/...`.

## Historical execution notes

`LEGACY_MODAL_EXECUTION_LOG.md` preserves the original operational commands and
failure/retry history. Its old paths describe the pre-migration layout and are
not the canonical locations for analysis.
