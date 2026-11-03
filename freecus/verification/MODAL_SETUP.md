# Modal 설정 및 배포 검증

2026-10-30 (Asia/Seoul).

- 새 profile `method-comparison` 인증 완료. workspace `louislee-85468` 확인.
- 새 workspace에 HF Secret과 모델/결과용 v2 Volume 생성 완료. 인증 token 값은 코드·기록에 저장하지 않음.
- Modal SDK 1.6.3 및 Python 3.11 client 설치 완료. 새 workspace 이미지 빌더 `2026.10`의 지원 버전에 맞췄다.
- 원격 Python 3.11 / PyTorch 2.6.0+cu124 / diffusers 0.32.2 / transformers 4.49.0 이미지 빌드 성공.
- `freecus-method-comparison` 앱에 CPU stage, GPU lane, CPU workflow 함수 배포 완료.
- 원격 workflow 접수 및 `staging_dreambench` 상태를 확인했다. CPU stage에서 FLUX 다운로드 진행을 확인했다.
- DreamBench 입력 750개 / 2 lanes × 375개, DreamBench++ 1,350개 / 2 lanes × 675개.
- pilot 12개 / 2 lanes × 6개 검증. seed 42, 원문 target과 selected reference 유지.
- 로컬 기존 CPU suite 14개 통과. 새 Python 3.11 client에서 비용·예산 관련 4개 테스트 및 pip check 통과.
- 실제 GPU 시간/VRAM 및 비용은 pilot 결과로 확인한다. 이 문서의 배포 성공은 전체 생성 완료를 뜻하지 않는다.

원격 상태/실행 ID는 `freecus/outputs/modal/workflow.json`과 `workflow-status.json`에 저장된다. 예산 계산의 source/단가/가중치 크기는 `modal-cost-plan.json`, `model-size-metadata.json`에 기록했다. 명령과 비용 판정 방식은 `../MODAL_GUIDE.md`를 따른다.

## 별도 계정 및 실제 실행 갱신

- `method-comparison-dbpp` 인증 성공, workspace **louislee**. 기존 **louislee-85468**과 별도다. 기본 profile은 유지했다.
- DreamBench A100 80GB × 2, DreamBench++ H100 80GB × 2. 프로세스마다 GPU 1개, host RAM 64 GiB, CPU 2 cores. 두 계정의 staging은 CPU에서 독립 실행된다.
- 최초 DreamBench pilot 12장 성공: steady 약 32.08초/장, marginal $0.027662/장, 전체 예상 $21.153, 25% margin $26.441. Qwen peak 약 39.917 GiB, FLUX peak 약 34.692 GiB, host peak 약 37.60 GiB.
- 그러나 전체 caption 전처리에서 더 큰 참조 이미지의 Qwen dense vision attention이 80GB에서도 OOM. 이전 성공 pilot과 실패 시도 compute 합계 **$0.7311197273**. 전체 데이터 VRAM 최대치는 pilot 결과만으로 판단할 수 없었다.
- 입력을 축소하지 않고 이미지별 unmasked 4D SDPA로 동일한 attention block을 계산하도록 수정. 여러 이미지와 RoPE를 포함한 CPU 동등성 테스트 통과. CUDA fused kernel은 작은 수치 차이가 가능하다.
- 두 최종 벤치마크의 동일 source snapshot: `28dfe68174b3ec5f781b0d05652493a4d2cc4c5cdfe8fbba59d4617c9839b1ac`.
- DreamBench bundle `477fc2c07d3451850b37b400`, workflow `b3423c335690427bbd4fdb0d866ccd66`, call `fc-01M65XVTDHMJ4YDM0FFF3CR07W`. 새 compute admission $28. 원격 staging 완료 및 pilot 시작 확인.
- DreamBench++ bundle `0fa5460f44b4b8672480a187`, workflow `1ffa841fba594f07904490a4cf0c5bd7`, call `fc-01M65XVWT48E4KYWRVP4JAVY6Z`. compute admission $29. 원격 staging 접수 확인. 이후 H100 pilot 비용이 통과하면 자동 전체 생성, 초과하면 예산 검토 상태로 종료한다.
- 두 workflow 모두 배포된 원격 coordinator가 staging/pilot/예산판정/본 생성을 담당한다. 접수 후 로컬 PC 종료 가능. 결과와 invocation journal은 각 계정 Volume에 영속화한다. 재시도는 자동 수행하지 않는다.
- DreamBench++ receipt/status는 별도 `freecus/outputs/modal-method-comparison-dbpp/`. 명령은 `modal-dbpp.ps1`을 사용한다. 같은 계정의 기존 receipt를 삭제해 중복 실행하면 안 된다.
- 원문 target prompt, 선택 reference, seed 42, 1024×1024, FreeCus 30 steps/3.5 guidance/1.1 scale을 유지한다. 현재 작업은 생성이며 기존 evaluator의 지표 계산 완료를 의미하지 않는다.

## 사용자 예산 $40 반영

사용자가 $40까지 허용하여 기존 실행을 취소하지 않고 두 계정에 원격 budget supervisor를 추가 배포했다. 계정별 compute admission $39, CPU 등 예비비 $1. 이전 $0.7311도 DreamBench ledger에 포함한다. 정상 예산 gate에서 멈추면 기존 모델/출력/이미지별 checkpoint를 재사용하고, 10% 여유를 포함한 예상이 남은 예산에 들어갈 때 자동 전체 생성을 이어간다. 실험 source snapshot과 GPU 함수는 교체하지 않았다. GPU/데이터 오류의 무제한 재시도는 하지 않는다.

- DreamBench supervisor `f45d881bd3e04a3e8d90427c1f18594b`, call `fc-01M65Y6XGRVQG17T57AB5PN0AJ`.
- DreamBench++ supervisor `d1c370f2674645149a0840e61468eeb5`, call `fc-01M65Y713TT4MQH9YP28FVK8PG`.
- 기존 `workflow-status`/`stop`은 최신 supervisor receipt를 사용한다. `stop`은 부모 coordinator와 알려진 GPU/stage call도 취소하도록 보완했다.
- 수정본 pilot의 Qwen peak 약 30.56 GiB. 새로 생성한 cat2 PNG를 다운로드해 1024×1024 정상 출력을 시각적으로 확인했다. 전체 suite 16개 통과.
