# XLabs projection benchmark results

Protocol: `uno-selected-reference-v1`. Scores below are reference macro averages; raw per-image scores, case micro averages and class/style breakdowns are retained in JSON.

| Benchmark | Condition | Images | CLIP-T | CLIP-I | DINO-I |
|---|---|---:|---:|---:|---:|
| dreambench | baseline | 750 | 0.294142 | 0.751016 | 0.448522 |
| dreambench | source_target | 750 | 0.302210 | 0.746345 | 0.448915 |
| dreambench_plus | baseline | 1350 | 0.308226 | 0.773878 | 0.494367 |
| dreambench_plus | source_target | 1350 | 0.315363 | 0.766451 | 0.486250 |

Both conditions use 1024 PNG, seed 42, 24 steps, guidance 3.5 and IP-Adapter scale 1.0. `source_target` uses projection strength 1.0 with norm restoration and no single-stream gate.

The projection increases CLIP-T on both benchmarks. CLIP-I decreases on both; DINO-I changes little on DreamBench and decreases on DreamBench++. These observations do not establish improvements on all metrics.

- [DreamBench canonical bundle](../dreambench/598a7ef43356a8eb05e1314b/)
- [DreamBench++ canonical bundle](../dreambench_plus/598a7ef43356a8eb05e1314b/)
- [DreamBench scores](dreambench_scores.json)
- [DreamBench++ scores](dreambench_plus_scores.json)

Original PNGs, references, conditioning caches and metadata are stored in Git LFS ZIPs under `../../artifacts/`. See the project README for credential-free verification/restoration commands. No other method or paper table is modified.
