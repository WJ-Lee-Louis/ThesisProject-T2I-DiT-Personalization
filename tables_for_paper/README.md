# 논문 표 통합 관리

논문용 표 PNG와 표 재생성 코드·수치 파일은 이 폴더에서 관리한다. 방법론별 `results/`의 원본 score, manifest, source snapshot은 원래 위치에 보존한다. 생성 이미지 ZIP은 저장소 밖 `../AI_PROJECT_images/`에 보관하며, 표 재생성에는 필요하지 않다.

## 현재 표와 평가 범위

| 방법론 | 생성 해상도 | DreamBench | DreamBench++ |
|---|---|---:|---:|
| FreeCus | 1024×1024 | 750장 / 30개 참조 | **1,170장 / 130개 참조, style 제외** |
| DreamO | 1024×1024 | 750장 / 30개 참조 | 1,350장 / 150개 참조 |
| UNO | 1024×1024 | 750장 / 30개 참조 | 1,350장 / 150개 참조 |
| XVerse | 768×768 | 750장 / 30개 참조 | 1,350장 / 150개 참조 |
| Diptych | 768×768 | 750장 / 30개 참조 | 1,350장 / 150개 참조 |
| MOSAIC | 512×512 | 750장 / 30개 참조 | 1,350장 / 150개 참조 |
| XLabs-IP-Adapter | 1024×1024 | 750장 / 30개 참조 | 1,350장 / 150개 참조 |

`dreambench_method_comparison.png`와 `dreambench_plus_method_comparison.png`는 전체 집계, 각각의 `_classes.png`는 분류별 집계다. DreamBench는 Animal/Object, DreamBench++는 Animal/Human/Object/Style과 Non-Style을 보여준다. Non-Style은 앞의 세 분류를 합친 1,170장 범위다.

FreeCus의 DreamBench++는 style 참조 20개에 해당하는 180장을 모두 제외한다. 표의 `†`, N=1,170, 각주로 전체 1,350장 평가가 아님을 표시한다. Style 행은 **N=0, 점수 `—`**로 표시하며 제외 점수를 0으로 평균에 넣지 않는다. 동일 평가 범위로 비교하려면 `_classes.png`의 Non-Style 구간을 사용한다. 기존의 생성 보관 수 1,188장과 평가 수 1,170장도 구분한다.

## 재생성

프로젝트 루트에서 실행:

```powershell
& freecus/.venv/Scripts/python.exe -X utf8 tables_for_paper/build_method_tables.py
```

활성 표 생성 코드는 `build_method_tables.py` 하나다. 로컬 matplotlib/Pillow 환경을 사용하며 GPU 작업이나 원격 평가를 실행하지 않는다. 일곱 방법론의 확정된 bundle ID를 코드의 `SOURCES`에서 지정한다. 기본 실행은 기존 비교 PNG 6개(전체·분류별 4개, balanced 2개)를 갱신하고, XLabs 적용 전후 PNG 4개를 생성한다.

14개 원본 score JSON의 **16,620개 이미지별 점수**를 사용한다. XLabs 파일에는 baseline과 source_target 두 조건이 함께 들어 있으므로 각 조건을 분리한다. 저장된 개별 metric으로 전체·분류별·Non-Style subject macro를 다시 계산하며, PNG는 소수점 4자리로 표시한다. 별도 검증 기록 파일은 생성하거나 갱신하지 않는다. 기존 ablation/verification PNG 6개는 유지한다.

## 현재 기록

- `method_metrics.csv`, `method_metrics.json`: 일곱 비교 방법 및 XLabs projection의 전체·분류별·Non-Style 수치, 평가 수, 참조 수, 해상도, 제외 여부. 반올림 전 정밀도를 보존한다.
- `applicability_metrics.json`: XLabs baseline/source_target의 적용 전후 값, 반올림 전 차이와 실험 설정.
- `historical_rows.json`: 기존 PNG에서 옮긴 Baseline/Ours/USO의 4자리 수치.

Baseline/Ours는 현재 요청의 갱신 대상이 아니므로 기존 PNG의 4자리 정밀도 값을 유지했다. `Ours (scale best*)`의 기존 설명인 임시 scale 0.8도 유지하며 최적화 결과로 표현하지 않는다. USO는 기존 외부 overall 값만 보존하고 분류별 결과는 `—`로 둔다. UNO는 기존 외부 0.8356/0.3039/0.7604를 이번 로컬 측정으로 교체하고 과거 `[T]` 분류를 새 측정에 붙이지 않았다. FreeCus DreamBench의 오래된 PNG 값 0.8058/0.3004/0.6708도 확정 원본의 **0.8184/0.2789/0.7156**으로 교체했다 (순서: CLIP-I/CLIP-T/DINO-I).

## 정리한 과거 파일

`archive/before_refresh_2026-11-04.zip`에는 갱신 전 표 PNG 4개와 과거 FreeCus 표 생성 코드·PNG/PDF/SVG/CSV/JSON 등 15개 파일을 원래 상대 경로로 보존했다. ZIP 멤버의 SHA256을 검증한 뒤 해당 15개 구형 작업 파일과 재생성 가능한 pyc 캐시 1개를 기존 위치에서 제거했다. `archive/cleanup_2026-11-04.json`에 경로·크기·SHA256·처리를 기록했다. 정리 과정의 일회성 코드도 archive에 텍스트로 보존한다.

벤치마크 생성 이미지, 모델·평가 코드, 원본 score와 비교 실험 기록, 입력 manifest와 provenance/verification은 삭제하지 않았다. DreamO/UNO의 GPU 검증용 sample PNG도 연구 검증 기록으로 보존했다. 새 PDF/SVG나 방법론 폴더별 임시 표는 만들지 않는다.

## DreamBench++ class-balanced 표

기존 표를 유지하면서 추가로 생성한 파일:

- `dreambench_plus_method_comparison_balanced.png`: CLIP-I/CLIP-T/DINO-I 각각 기존 Original, Balanced, Balanced baseline 대비 절대 차이(Δ)를 표시한다. 네 class 전체와 Non-Style을 별도 구간으로 나눈다.
- `dreambench_plus_method_comparison_classes_balanced.png`: 각 class의 기존 평균과 `0.25 × class 평균`인 전체 balanced 점수 기여도를 함께 표시한다. 마지막에는 Non-Style Balanced 집계를 제공한다.

네 class의 균등 평균은 `(Animal + Human + Object + Style) / 4`, Non-Style 균등 평균은 `(Animal + Human + Object) / 3`이다. class 내부의 참조별 평균 방식은 그대로 유지한다. class 점수 자체가 변하는 것은 아니며, 각 class가 최종 점수에 기여하는 비중이 달라진다. 각 class의 평균에 참조 수 역수를 추가로 곱하지 않는다.

FreeCus는 style 결과가 없어 4-class Balanced/Δ/기여도를 `—`로 표시한다. Original은 기존 1,170장 non-style 점수임을 `†`와 각주로 명시한다. Non-Style Balanced에서는 다른 방법론과 동일한 세 class 및 동일한 1,170장 범위로 비교한다.

새 표 재생성:

```powershell
& freecus/.venv/Scripts/python.exe -X utf8 tables_for_paper/build_method_tables.py --balanced-only
```

`class_balanced_metrics.csv`와 `.json`에 반올림 전 집계, class 평균·가중치·기여도와 baseline 차이를 저장한다. 로컬 측정값은 각 class 평균의 균등 평균이며 이미지별 `1 / (class 수 × 해당 class 이미지 수)` 가중합과 같다.

Baseline/Ours의 balanced 값은 보존된 소수점 4자리 class 평균으로 계산하므로 근사값이다. 이 baseline을 사용하는 Δ도 근사값이며 PNG와 JSON에 이를 명시한다. 일곱 비교군의 Balanced 값은 반올림 전 이미지별 점수에서 계산한다. 원본 이미지를 생성하거나 metric 모델을 다시 실행하지 않는다.

## XLabs projection 적용 전후

요청한 파일명 접미사 `_applicabiltiy.png`를 그대로 사용한다.

| Benchmark | 전체 | 분류별 |
|---|---|---|
| DreamBench | [적용 전후](dreambench_method_comparison_applicabiltiy.png) | [분류별](dreambench_method_comparison_classes_applicabiltiy.png) |
| DreamBench++ | [적용 전후](dreambench_plus_method_comparison_applicabiltiy.png) | [분류별](dreambench_plus_method_comparison_classes_applicabiltiy.png) |

두 조건 모두 1024×1024, seed 42, 24 steps, guidance 3.5, IP-Adapter scale 1.0이며 동일 case/reference를 사용한다. `source_target`은 projection strength 1.0, norm restoration ON, single-stream gate OFF다. 각 표의 Change는 반올림 전 `after − before`다.
