# 논문 표 통합 관리

논문용 PNG, 표 생성 코드, 수치 파일을 관리한다. 2026-11-14에 PNG 16개를 전수 검토하고 비교표 6개를 갱신했다. ablation/verification 6개와 XLabs 적용 전후 4개에는 이번 요청의 방법론 행이 없다.

## 현재 비교 방법과 평가 범위

| 방법론 | 생성 해상도 | DreamBench | DreamBench++ |
|---|---|---:|---:|
| FreeCus | 512×512 | 750장 / 30개 참조 | 1,170장 / 130개 참조, style 제외 |
| DreamO | 1024×1024 | 750장 / 30개 참조 | 1,350장 / 150개 참조 |
| UNO | 512×512 | 750장 / 30개 참조 | 1,350장 / 150개 참조 |
| XVerse | 768×768 | 750장 / 30개 참조 | 1,350장 / 150개 참조 |
| Diptych | 768×768 | 750장 / 30개 참조 | 1,350장 / 150개 참조 |
| MOSAIC | 512×512 | 750장 / 30개 참조 | 1,350장 / 150개 참조 |
| USO | 1024×1024 | 750장 / 30개 참조 | 1,350장 / 150개 참조 |
| FreeGraftor | 512×512 | 750장 / 30개 참조 | 1,170장 / 130개 참조, style 제외 |
| XLabs-IP-Adapter | 1024×1024 | 750장 / 30개 참조 | 1,350장 / 150개 참조 |

FreeCus·UNO는 `results_512/`의 원본 점수를 사용하며, 기존 1024 행과 수치는 활성 비교표・수치 파일에서 제거했다. USO는 `uso/results/`의 로컬 평가로 교체했고 `USO (external)` 행과 historical 수치를 삭제했다. FreeGraftor는 `freegrafor2/results/runpod/seed42_512/metrics/scores.json`을 사용한다. 원본 실험 기록은 삭제하지 않는다. 확정 bundle ID와 점수 경로는 `build_method_tables.py`의 `SOURCES`와 `score_path()`에 있다.

FreeCus와 FreeGraftor의 DreamBench++는 style 참조 20개 / 180장을 전부 제외한다. `†`, N=1,170, 각주로 범위를 표시하며 Style 행은 N=0, 점수 `—`다. FreeGraftor는 style 생성 180회 중 14회 실패해 전체 style class를 제외한 최종 정책을 따른다. 동일 범위 비교에는 Non-Style 블록을 사용한다. FreeGraftor의 detector-label 복구 36건은 원본 기록에 보존되어 있다.

## PNG와 집계

- `dreambench_method_comparison.png`, `dreambench_plus_method_comparison.png`: 전체 subject macro.
- 각각의 `_classes.png`: DreamBench Animal/Object, DreamBench++ Animal/Human/Object/Style/Non-Style.
- `dreambench_plus_method_comparison_balanced.png`: Original, Balanced, Balanced baseline 대비 절대 차이 Δ.
- `dreambench_plus_method_comparison_classes_balanced.png`: class 평균, 전체 balanced 점수 기여도 및 Non-Style Balanced.
- `_applicabiltiy.png` 4개: XLabs baseline/source_target 적용 전후. 기존 요청의 파일명 철자를 유지한다.
- ablation/verification 6개: 기존 실험 표를 유지한다.

모든 로컬 비교군은 `uno-selected-reference-v1`의 CLIP ViT-B/32, DINO v1 ViT-S/16, 단일 selected reference, 원래 target prompt, seed 42를 사용한다. PNG는 소수점 4자리다. 생성 해상도는 방법별로 다르며 동일 해상도 실험을 뜻하지 않는다.

Balanced는 `(Animal + Human + Object + Style) / 4`, Non-Style Balanced는 `(Animal + Human + Object) / 3`이다. class 내부는 참조별 평균을 유지한다. FreeCus/FreeGraftor의 4-class Balanced/Δ/기여도는 `—`이며 Non-Style Balanced만 계산한다. class별 원래 점수는 유지한다.

Baseline/Ours는 기존 PNG에서 보존한 소수점 4자리 수치를 유지한다. 이들의 balanced 수치와 해당 baseline을 사용하는 Δ는 근사값이다. `Ours (scale best*)`는 임시 scale 0.8이며 최적화 결과로 표현하지 않는다.

## 재생성 및 검증

프로젝트 루트에서 matplotlib/Pillow가 설치된 Python으로 실행한다.

```sh
python tables_for_paper/build_method_tables.py
# balanced 표만 갱신
python tables_for_paper/build_method_tables.py --balanced-only
```

기본 실행은 비교표 6개와 XLabs 적용 전후 4개를 재생성한다. GPU 추론, metric 모델 실행, 원격 작업은 수행하지 않는다. 17개 원본 score 파일의 20,640개 조건별 이미지 점수를 집계한다(FreeGraftor는 두 benchmark가 한 파일에 있다). XLabs baseline/source_target은 따로 검증한다.

원본 summary 재계산, 별도 subject macro 계산, canonical case/reference/prompt, class별 완전성, 생성 해상도/seed, score의 이미지 SHA256와 bundle index/export manifest를 확인한다. FreeGraftor는 1,920개 원본 JSON sidecar의 해시와 설정도 확인한다. Balanced는 이미지별 균등 class 가중합과 별도로 대조한다. 외부 보관 이미지 ZIP의 실제 바이트는 다시 읽지 않는다.

- `method_metrics.csv`, `.json`: 반올림 전 전체・분류별・Non-Style 수치, 평가 수, 참조 수, 해상도, 제외 여부.
- `class_balanced_metrics.csv`, `.json`: 반올림 전 balanced, 기여도, baseline Δ.
- `applicability_metrics.json`: XLabs 적용 전후와 반올림 전 Δ.
- `historical_rows.json`: 보존한 Baseline/Ours 수치.
- `aggregation_audit.json`, `class_balanced_audit.json`: 활성 원본 경로・해시・검증 범위・PNG 해시.
- `png_review_2026-11-14.json`: PNG 16개 전수 검토 기록과 실제 렌더링 수치의 OCR 대조 결과.

XLabs 두 조건은 1024×1024, seed 42, 24 steps, guidance 3.5, IP-Adapter scale 1.0, 동일 case/reference다. source_target은 projection strength 1.0, norm restoration ON, single-stream gate OFF다. Change는 반올림 전 `after − before`다.

## 과거 자료

`archive/before_refresh_2026-11-04.zip`과 `archive/cleanup_2026-11-04.json`은 과거 표와 정리 기록이다. 과거 실험・source snapshot・manifest・provenance는 원래 위치에 유지한다. 생성 이미지 ZIP은 저장소 밖에 보관하며 표 재생성에는 필요하지 않다.
