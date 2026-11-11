# CLIP / DINO benchmark scores

Method = local source-target projection + SSG. Selected single reference; subject macro average; original cosine scale (no x100). Higher is better. Scores shown to 4 decimals.

**DreamBench overall**

| Group | Method | Images | CLIP-T | CLIP-I | DINO-I |
|---|---|---:|---:|---:|---:|
| Overall | Baseline | 750 | 0.2939 | 0.8319 | 0.6900 |
| Overall | Source-target + SSG | 750 | 0.3123 | 0.8022 | 0.6548 |

**DreamBench++ overall**

| Group | Method | Images | CLIP-T | CLIP-I | DINO-I |
|---|---|---:|---:|---:|---:|
| Overall | Baseline | 1350 | 0.2994 | 0.8534 | 0.6820 |
| Overall | Source-target + SSG | 1350 | 0.3280 | 0.8115 | 0.6154 |

**DreamBench by class**

| Group | Method | Images | CLIP-T | CLIP-I | DINO-I |
|---|---|---:|---:|---:|---:|
| Animal | Baseline | 225 | 0.2917 | 0.8901 | 0.7872 |
| Animal | Source-target + SSG | 225 | 0.3204 | 0.8561 | 0.7562 |
| Object | Baseline | 525 | 0.2949 | 0.8070 | 0.6483 |
| Object | Source-target + SSG | 525 | 0.3089 | 0.7791 | 0.6113 |

**DreamBench++ by class**

| Group | Method | Images | CLIP-T | CLIP-I | DINO-I |
|---|---|---:|---:|---:|---:|
| Animal | Baseline | 405 | 0.3117 | 0.8902 | 0.7609 |
| Animal | Source-target + SSG | 405 | 0.3352 | 0.8548 | 0.7226 |
| Human | Baseline | 180 | 0.2610 | 0.7941 | 0.6424 |
| Human | Source-target + SSG | 180 | 0.3010 | 0.7389 | 0.5466 |
| Object | Baseline | 585 | 0.3042 | 0.8492 | 0.6584 |
| Object | Source-target + SSG | 585 | 0.3299 | 0.8116 | 0.6004 |
| Style | Baseline | 180 | 0.2945 | 0.8436 | 0.6207 |
| Style | Source-target + SSG | 180 | 0.3325 | 0.7865 | 0.4916 |
| Non-style | Baseline | 1170 | 0.3001 | 0.8549 | 0.6914 |
| Non-style | Source-target + SSG | 1170 | 0.3273 | 0.8153 | 0.6345 |

DreamBench has no human/style references. Class is determined by the reference concept.
Overall scores reflect reference counts; Non-style excludes style.
