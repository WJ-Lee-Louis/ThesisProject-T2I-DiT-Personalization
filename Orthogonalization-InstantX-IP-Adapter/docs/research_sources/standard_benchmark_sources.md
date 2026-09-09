# Research source ledger

Date: 2026-09-07

Scope: De-facto standard benchmarks and metrics for single-subject text-to-image personalization, with emphasis on a budget-conscious preliminary evaluation of InstantX-IP-Adapter interventions.

Assumptions: The target claim is improved prompt controllability without material subject-fidelity loss. The immediate study is an internal novelty screen rather than a final cross-paper leaderboard comparison.

## Claim ledger

| Claim | Primary source | Date | URL | Access note |
|---|---|---:|---|---|
| DreamBench uses 30 subjects, 25 prompts, four samples per pair, DINO ViT-S/16, CLIP ViT-B/32, and CLIP-T | DreamBooth, Ruiz et al. | 2023 | https://arxiv.org/html/2208.12242 | Full HTML inspected, Sec. 4.1 |
| DreamBench++ contains 150 images and 1,350 prompts, with nine prompts per image split into four photorealistic, three stylized, and two complicated/imaginative prompts | DreamBench++, Peng et al. | 2025 | https://github.com/yuangpeng/dreambench_plus | Official README inspected |
| DreamBench++ evaluates concept preservation and prompt following and also reports DINO-I, CLIP-I, and CLIP-T | DreamBench++, Peng et al. | 2025 | https://arxiv.org/html/2406.16855 | Full HTML inspected |
| A recent FLUX training-free method applies SAM subject isolation and reports CLIP-I, DINO, and CLIP-T on DreamBench++ | FreeCus, Zhang et al. | 2025 | https://arxiv.org/html/2507.15249 | Full HTML inspected, evaluation section |
| A recent FLUX training-free method follows DreamBench's 3,000-image protocol and reports masked CLIP-I/DINOv2 plus CLIP-T/ImageReward | FreeGraftor, Yao et al. | 2026 revision | https://arxiv.org/html/2504.15958 | Full HTML inspected, Sec. IV-A3 |
| Personalize Anything uses DreamBench and CLIP-T, CLIP-I, DINO, DreamSim for single-subject DiT personalization | Personalize Anything, Feng et al. | 2025 | https://arxiv.org/html/2503.12590 | Full HTML inspected, Sec. 4.1 |
| ImageReward is a general text-to-image human-preference reward model, not an identity-specific metric | ImageReward, Xu et al. | 2023 | https://arxiv.org/html/2304.05977 | Full HTML inspected |
| CustomConcept101 contains 101 concepts and provides prompt-based image/text alignment evaluation | Custom Diffusion, Kumari et al. | 2023 | https://github.com/adobe-research/custom-diffusion/blob/main/customconcept101/README.md | Official dataset README inspected |

## Reconciliation notes

- `DINO`, `DINO-I`, and `DINOv2` are not interchangeable implementations. DreamBench's original DINO ViT-S/16 protocol should be retained for legacy comparability; DINOv2 must be reported separately.
- Global and subject-masked CLIP-I/DINO measure different quantities. Both should be retained because global similarity is the legacy protocol while masking is directly relevant to background leakage.
- `IR` is ambiguous across the broader literature. In FreeGraftor and the sources relevant here, it denotes ImageReward and belongs under text alignment/general preference, not subject identity.
- A task-selected DreamBench++ subset is not the official full benchmark and must not be compared numerically against the official leaderboard without rerunning baselines on the same subset.

