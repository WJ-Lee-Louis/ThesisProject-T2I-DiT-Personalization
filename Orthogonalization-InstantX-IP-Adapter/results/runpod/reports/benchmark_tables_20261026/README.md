# Benchmark tables — 2026-10-26

Best* 행은 사용자 지정에 따라 scale 0.8로 고정했습니다. 최적화로 선정한 값이 아닙니다.
집계는 기존 평가와 동일한 subject macro average이며, 열 순서는 CLIP-I / CLIP-T / DINO-I입니다.
baseline 및 Ours scale 0.7은 기존 스크린샷 전사값입니다. 나머지 로컬 실험은 scores.json의 집계를 사용합니다.
FreeCus / UNO / USO는 지정 PNG의 DreamBench 전체 수치만 사용합니다. class별 수치는 없어 —로 표시했습니다.
DreamBench++에는 FreeCus / UNO / USO를 넣지 않았습니다.
Non-style은 animal + human + object의 통합 부분집합입니다.
Ablation은 scale 0.7 결과이며 최종 scale 기준 재실험은 아직 수행하지 않았습니다.

## PNG tables

| Table | DreamBench | DreamBench++ |
|---|---|---|
| Method comparison | [Overall](dreambench_method_comparison.png) · [Classes](dreambench_method_comparison_classes.png) | [Overall](dreambench_plus_method_comparison.png) · [Classes](dreambench_plus_method_comparison_classes.png) |
| Scale hyperparameter sensitivity | [Overall](dreambench_scale_sensitivity.png) · [Classes](dreambench_scale_sensitivity_classes.png) | [Overall](dreambench_plus_scale_sensitivity.png) · [Classes](dreambench_plus_scale_sensitivity_classes.png) |
| Method ablation (scale 0.7) | [Overall](dreambench_method_ablation.png) · [Classes](dreambench_method_ablation_classes.png) | [Overall](dreambench_plus_method_ablation.png) · [Classes](dreambench_plus_method_ablation_classes.png) |
