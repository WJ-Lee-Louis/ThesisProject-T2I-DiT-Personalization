# Benchmark comparison

Four versions, with separate benchmark and class summaries.

| Benchmark | Method | Scale | SSG | Images | CLIP-T ↑ | CLIP-I ↑ | DINO-I ↑ |
|---|---|---|---|---:|---:|---:|---:|
| dreambench | InstantX-IP-Adapter † | — | OFF | 750 | 0.2939 | 0.8319 | 0.6900 |
| dreambench | Ours † | 0.7 | ON | 750 | 0.3123 | 0.8022 | 0.6548 |
| dreambench | Ours | 0.8 | ON | 750 | 0.3078 | 0.8144 | 0.6726 |
| dreambench | Ours w/o SSG | 0.7 | OFF | 750 | 0.3017 | 0.8244 | 0.6806 |
| dreambench_plus | InstantX-IP-Adapter † | — | OFF | 1350 | 0.2994 | 0.8534 | 0.6820 |
| dreambench_plus | Ours † | 0.7 | ON | 1350 | 0.3280 | 0.8115 | 0.6154 |
| dreambench_plus | Ours | 0.8 | ON | 1350 | 0.3208 | 0.8273 | 0.6413 |
| dreambench_plus | Ours w/o SSG | 0.7 | OFF | 1350 | 0.3087 | 0.8414 | 0.6618 |

† User-provided historical screenshot: four-decimal rounded values; individual scores and runtime metadata are unavailable. Baseline scale is not shown in that screenshot.

Local source: `/workspace/instantx-0807/results/full/metrics/scores.json`; A100-SXM4-80GB. Protocol: `uno-selected-reference-v1` (CLIP ViT-B/32; DINO v1 ViT-S/16; one selected reference). Historical screenshot states the same reference policy and aggregation, but full protocol/runtime equivalence cannot be independently checked without its original JSON. No significance tests are possible for screenshot-only rows.

Ours scale 0.7 has the highest CLIP-T among the four versions on both benchmarks. Raising the gated method scale from 0.7 to 0.8 lowers CLIP-T and raises CLIP-I/DINO-I. Baseline retains the highest CLIP-I and DINO-I. The no-SSG row at scale 0.7 allows a same-scale comparison with gated scale 0.7; comparing scale 0.8 with no-SSG scale 0.7 changes both scale and gate.

Artifacts: `benchmark_tables.pdf` (three pages), `overall.png`, `overall.pdf`, `overall.svg`, `overall.tex` (requires booktabs), class tables, and combined CSV/JSON.

New scale sweep: `/workspace/instantx-0807/results/scale_sweep_20261025`, scales 0.6/0.9/1.0 with SSG ON, 2,100 images per scale, two GPUs. These pending results are not included in the completed comparison.
