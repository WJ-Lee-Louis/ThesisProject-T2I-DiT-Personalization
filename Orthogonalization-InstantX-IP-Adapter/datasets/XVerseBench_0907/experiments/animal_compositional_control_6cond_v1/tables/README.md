# 주요 결과 시각화

| 파일 | 내용 |
|---|---|
| `01_main_results_table.png` | subset/custom 조건별 DPG·IP-Sim·AES와 paired delta |
| `02_tradeoff_scatter.png` | ΔIP–ΔDPG trade-off, ΔAES 색상, 2D Pareto frontier |
| `03_composition_dpg_heatmap.png` | 조건·composition별 dependency-aware paired ΔDPG와 표본 수 |
| `04_qualitative_subset.png` | subset의 최대 DPG 향상·최대 IP 하락·중앙값 사례와 SAM2 산출물 |
| `05_qualitative_custom.png` | custom의 최대 DPG 향상·최대 IP 하락·중앙값 사례와 SAM2 산출물 |
| `figure_manifest.json` | 입력/출력 SHA-256, 이미지 크기, 정성 사례 선택 metadata |
| `comparison_grids/index.html` | subset 116개·custom 90개 reference–prompt–seed별 6-condition 비교 grid 색인·검색·정렬 |
| `comparison_grids/grid_manifest.csv` | 각 grid의 원점수와 동일 조합 baseline 대비 Δ, 입력 reference 및 결과 파일 경로 |

DPG는 두 benchmark 모두 `dpg_with_dependency`를 사용한다. Δ는 동일 record·seed의 baseline 대비 차이다.
정성 grid는 분석 가이드에 따라 ST-matched와 ST-all 후보에서 결정론적으로 선택한다.

재생성:

```powershell
python datasets/XVerseBench_0907/metrics/render_analysis_figures.py
```

전체 정성 비교 grid 재생성:

```powershell
python datasets/XVerseBench_0907/metrics/make_comparison_grids.py
```

`comparison_grids/subset/`은 29 reference–prompt 쌍 × 4 seed = 116장,
`comparison_grids/custom/`은 15 reference × 6 prompt × 1 seed = 90장이다.
한 grid는 reference와 여섯 condition 이미지를 포함하며 DPG(dependency-aware),
IP-Sim, AES의 이미지별 0–100 원점수 및 동일 record·seed baseline 대비 Δ를 표시한다.
