# 논문 표 통합 관리

앞으로 논문용 표 PNG와 표를 위한 통합 집계·검토 기록은 이 폴더에서 관리한다. 방법론별 `results/`의 원본 score, 이미지 ZIP, manifest, source snapshot, 검증 기록은 연구 결과 원본이므로 원래 위치에 보존한다.

## 현재 표와 평가 범위

| 방법론 | 생성 해상도 | DreamBench | DreamBench++ |
|---|---|---:|---:|
| FreeCus | 1024×1024 | 750장 / 30개 참조 | **1,170장 / 130개 참조, style 제외** |
| DreamO | 1024×1024 | 750장 / 30개 참조 | 1,350장 / 150개 참조 |
| UNO | 1024×1024 | 750장 / 30개 참조 | 1,350장 / 150개 참조 |
| XVerse | 768×768 | 750장 / 30개 참조 | 1,350장 / 150개 참조 |

`dreambench_method_comparison.png`와 `dreambench_plus_method_comparison.png`는 전체 집계, 각각의 `_classes.png`는 분류별 집계다. DreamBench는 Animal/Object, DreamBench++는 Animal/Human/Object/Style과 Non-Style을 보여준다. Non-Style은 앞의 세 분류를 합친 1,170장 범위다.

FreeCus의 DreamBench++는 style 참조 20개에 해당하는 180장을 모두 제외한다. 표의 `†`, N=1,170, 각주로 전체 1,350장 평가가 아님을 표시한다. Style 행은 **N=0, 점수 `—`**로 표시하며 제외 점수를 0으로 평균에 넣지 않는다. 동일 평가 범위로 비교하려면 `_classes.png`의 Non-Style 구간을 사용한다. 기존의 생성 보관 수 1,188장과 평가 수 1,170장도 구분한다.

## 재생성 및 검토

프로젝트 루트에서 실행:

```powershell
& freecus/.venv/Scripts/python.exe -X utf8 tables_for_paper/build_method_tables.py
```

활성 표 생성 코드는 `build_method_tables.py` 하나다. 현재 로컬 matplotlib/Pillow 환경을 사용하며 GPU 작업이나 원격 평가를 실행하지 않는다. 네 방법론의 확정된 bundle ID를 코드의 `SOURCES`에서 지정하며, 새 실행 결과를 채택할 때에는 명시적으로 갱신한다.

이 명령은 8개 원본 score JSON의 **8,220개 점수**를 읽어 다음을 확인한다.

- 모든 방법론의 benchmark manifest bytes, case ID, 원래 target prompt, selected-one reference 및 SHA256, seed 42 일치.
- 평가 protocol, T4 장치와 라이브러리 버전 일치; 생성 run의 해상도·fingerprint 및 score provenance 일치.
- 이미지별 generated SHA256과 보관 bundle index의 이미지 SHA256 일치.
- 평가 범위의 누락·중복 없음; FreeCus의 style 제외; 참조별 25장 또는 9장 충족.
- 기존 evaluator로 전체 요약을 재집계하고, 별도의 `statistics.fmean` 구현으로 subject macro를 독립 확인. 전체·분류별·Non-Style 요약을 저장된 값과 절대 오차 1e-12 이내로 대조.
- 표의 텍스트가 캔버스 안에 있는지 확인하고, 다른 ablation/verification PNG 6개의 SHA256을 유지.

이번 검토는 **이미 저장된 이미지별 metric의 집계 검토**다. 모델을 다시 실행하거나 모든 이미지 ZIP의 전체 바이트를 다시 해시하지 않았다. 이미지 ZIP의 과거 무결성 검증은 각 방법론의 `verification.json`에 남아 있으며, 이번에는 score와 bundle index 사이의 hash 연결 및 ZIP 내 run 기록을 확인한다.

## 현재 기록

- `method_metrics.csv`, `method_metrics.json`: 네 방법론의 전체·분류별·Non-Style 수치, 평가 수, 참조 수, 해상도, 제외 여부. 반올림 전 정밀도를 보존한다.
- `aggregation_audit.json`: 사용한 원본 경로·SHA256, protocol·환경, 범위·검증 결과, 최종 PNG 해시, 변경하지 않은 6개 PNG의 해시.
- `historical_rows.json`: 갱신 전 PNG에서 그대로 옮긴 Baseline/Ours/USO의 기존 값. 네 방법론의 새 집계와 구분한다.

Baseline/Ours는 현재 요청의 갱신 대상이 아니므로 기존 PNG의 4자리 정밀도 값을 유지했다. `Ours (scale best*)`의 기존 설명인 임시 scale 0.8도 유지하며 최적화 결과로 표현하지 않는다. USO는 기존 외부 overall 값만 보존하고 분류별 결과는 `—`로 둔다. UNO는 기존 외부 0.8356/0.3039/0.7604를 이번 로컬 측정으로 교체하고 과거 `[T]` 분류를 새 측정에 붙이지 않았다. FreeCus DreamBench의 오래된 PNG 값 0.8058/0.3004/0.6708도 확정 원본의 **0.8184/0.2789/0.7156**으로 교체했다 (순서: CLIP-I/CLIP-T/DINO-I).

## 정리한 과거 파일

`archive/before_refresh_2026-11-04.zip`에는 갱신 전 표 PNG 4개와 과거 FreeCus 표 생성 코드·PNG/PDF/SVG/CSV/JSON 등 15개 파일을 원래 상대 경로로 보존했다. ZIP 멤버의 SHA256을 검증한 뒤 해당 15개 구형 작업 파일과 재생성 가능한 pyc 캐시 1개를 기존 위치에서 제거했다. `archive/cleanup_2026-11-04.json`에 경로·크기·SHA256·처리를 기록했다. 정리 과정의 일회성 코드도 archive에 텍스트로 보존한다.

벤치마크 생성 이미지, 모델·평가 코드, 원본 score와 비교 실험 기록, 입력 manifest와 provenance/verification은 삭제하지 않았다. DreamO/UNO의 GPU 검증용 sample PNG도 연구 검증 기록으로 보존했다. 새 PDF/SVG나 방법론 폴더별 임시 표는 만들지 않는다.
