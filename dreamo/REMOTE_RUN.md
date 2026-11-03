# DreamO remote execution — 2026-11-02

두 계정에 배포한 원격 coordinator가 모델 준비부터 GPU 검증과 전체 생성까지 이어서 처리한다. 접수 후 PC를 종료해도 진행된다. 기본 Modal profile은 변경하지 않았다.

A100과 L40S pilot이 모두 통과했고, 두 벤치마크 모두 L40S 2-worker 전체 생성 단계에 진입했다. 실측 peak reserved는 36.05 GiB다. 전체 생성 예상은 DreamBench 약 70분/$5.92, DreamBench++ 약 117분/$9.94이며 준비/검증 비용은 별도다. 25% 여유를 포함한 계산 예상은 각각 $7.40/$12.42다. `verification/gpu_comparison.json`에 비교와 호출 ID를 보존한다. 이후 완료/실패 상태는 원격 status가 기준이다.

| Benchmark | Profile / workspace | Images | Workflow ID |
|---|---|---:|---|
| DreamBench | method-comparison / louislee-85468 | 750 | b0cc75f6ccb948d88a436abc0971328b |
| DreamBench++ | method-comparison-dbpp / louislee | 1,350, style 포함 | 64597b472fce4931b954c12377de913f |

실행 순서는 모델 준비 → A100 80GB 2-worker pilot → 메모리 조건을 만족하면 L40S 48GB pilot → 비용 비교 → 선택 GPU 2개로 전체 생성이다. 모델 하나당 GPU 하나를 사용하며 두 벤치마크 동시 실행 시 GPU 총 4개다. 각 worker는 CPU 2 cores / RAM 64 GiB다. 양자화나 CPU offload는 켜지 않는다.

초기화와 이미지별 allocated/reserved peak를 모두 측정한다. A100 peak가 42 GiB 이하일 때 L40S를 시도하며, L40S도 같은 메모리 조건과 전체 계산비용 3% 이상 절감 조건을 만족해야 선택한다. Pilot은 각 lane/category에서 원본 reference 면적이 가장 큰 항목의 prompt index 0/4를 사용한다. DreamBench 8장, DreamBench++ 16장으로 범주별 경로를 검증한다.

CPU/RAM/GPU 합산 실측으로 전체 비용을 추정하고 25% 여유를 적용한다. 계정별 $40 중 계산 admission은 최대 $38이며 $2는 준비/저장 등 예비비다. 이 값은 앱 내부 추정 한도로, workspace 전체 실제 청구액을 읽는 하드 과금 상한은 아니다. 실패를 자동 재시도하지 않으며 로그와 완료 이미지를 보존한다.

기존 `freecus-method-comparison-models` Volume에서 고정 FLUX revision을 재사용한다. DreamO Secret과 `dreamo-runs` 결과 Volume은 별도다. A100/L40S 이미지는 GPU별 폴더에 분리한다.

```powershell
freecus/.venv-modal311/Scripts/python.exe -X utf8 -m dreamo.modal_client workflow-status --profile method-comparison --benchmark dreambench
freecus/.venv-modal311/Scripts/python.exe -X utf8 -m dreamo.modal_client workflow-status --profile method-comparison-dbpp --benchmark dreambench_plus
# 필요할 때 coordinator와 GPU 자식 호출을 함께 중지. 완료 이미지는 보존.
freecus/.venv-modal311/Scripts/python.exe -X utf8 -m dreamo.modal_client stop --profile method-comparison --benchmark dreambench
```

접수/상태는 `outputs/modal/<PROFILE>/<BENCHMARK>-workflow*.json`에 저장한다. 동일 receipt가 있으면 중복 접수하지 않는다. 실행 중 같은 앱을 다시 배포하지 않는다. 새 실행은 `prepare`, `bootstrap`, `deploy`, `workflow` 순서다. 수동 `stage/pilot/submit`과 동시에 사용하지 않는다.

결과는 `dreamo-runs` Volume의 `results/<BUNDLE_ID>/<SELECTED_GPU>/lane-{0,1}/`에 저장된다. 최종 경로는 status의 `output`에서 확인한다. 수집 시 `staged_adapters/`는 모델 링크/사본이므로 제외한다. 각 lane의 run.json/initialization.json 및 tgt_image/src_image/text/metadata/conditioning을 수집한 뒤 `<SELECTED_GPU>` 폴더를 `dreamo.assemble --source`에 지정한다.

DreamBench++는 style 포함 1,350장을 생성·보존하고 평가 기본값도 전체다. 나중에 평가 시에만 `--selection dbpp-no-style`로 1,170개를 선택할 수 있다. 현재 상태와 실측 결과는 workflow status 파일이 기준이다.
