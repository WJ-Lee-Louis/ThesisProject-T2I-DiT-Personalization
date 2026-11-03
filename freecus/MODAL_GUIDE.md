# FreeCus: Modal 원격 실행과 비용

DreamBench는 **method-comparison / louislee-85468**, DreamBench++는 **method-comparison-dbpp / louislee**로 분리한다. 기존 Modal 기본 profile은 변경하지 않는다. HF token은 값 출력 없이 각 workspace의 `freecus-huggingface` Secret으로 등록했다. 모델 cache 및 결과는 각 workspace의 별도 v2 Volume에 저장된다.

**현재 예산: 사용자의 추가 지시에 따라 계정별 $40.** 이미 실행 중인 작업은 유지하고, 별도의 원격 `freecus-budget-supervisor`가 이전 $28/$29 gate에서 멈춘 정상 작업을 이어서 실행한다. 총 compute admission은 계정별 $39이며 나머지 $1은 CPU 등 예비비다. DreamBench의 이전 $0.7311도 $39 ledger에 포함한다. $40 continuation은 실측 예상의 10% 여유로 판정한다. 기존 coordinator의 25% 판정 때문에 $30 근처에서 멈춰도 supervisor가 자동 인계한다. CUDA/데이터 오류는 유료 재시도를 반복하지 않고 기록한다. 아래 $30 및 $28/$29 설명은 최초 접수 이력이며 현재 최종 한도는 이 문단이 우선한다.

## 원격 실행 순서

1. CPU에서 입력 해시를 검증하고 고정된 모델 revision을 cache Volume에 다운로드한다. FLUX 저장소의 중복 원본 weight는 제외하고 diffusers 구성요소만 받는다.
2. DreamBench는 **A100-80GB worker 두 개**, DreamBench++는 별도 계정에서 **H100 worker 두 개**를 사용한다. 각 프로세스에 GPU 1개를 할당하고 case를 나눈다. 두 벤치마크가 함께 생성 중일 때 총 4개다. CPU 준비 중에는 GPU를 사용하지 않는다.
3. DreamBench의 backpack/cat2/dog/teapot에서 원문 prompt index 0/10/20을 사용해 **12장, worker당 6장**을 측정한다. pilot 두 worker 합산 compute admission은 $2.50이다.
4. 실제 이미지별 시간·caption 시간·초기 로딩·GPU peak VRAM·호스트 peak RAM을 기록하고, 전체 DreamBench 750장 비용에 **25% 여유**를 더해 계산한다.
5. 남은 compute admission 예산 안에 들어가면 같은 두 worker로 전체 DreamBench를 시작한다. pilot의 정상 결과는 재사용한다.
6. DreamBench++는 독립 workflow에서 모델 준비 → H100 pilot → 비용 판정 → 전체 1,350장 순서로 진행한다. pilot은 lane별 두 reference에서 prompt index 0/4/8을 선택한 12장이다. $30 안에 들어간다고 판정될 때만 전체 생성을 시작한다. GPU 개수 증가는 시간을 줄이지만 총 GPU초 비용을 줄이지 않는다. `--include-dbpp` 방식의 한 계정 순차 실행은 사용하지 않는다.

coordinator와 GPU worker 모두 **배포된 Modal 함수에 spawn**한다. 원격 실행 접수 ID가 저장된 후에는 로컬 PC를 종료해도 stage/pilot/예산판정/생성이 계속된다. 각 이미지와 caption을 Volume에 checkpoint하고 각 lane을 별도 디렉터리에 쓴다. 통신 실패나 GPU 오류 시 자동 재시도로 비용을 늘리지 않고 상태와 오류를 남긴다. 예산을 넘길 전망이면 `budget_review_required`로 종료하고 전체 생성을 시작하지 않는다.

## 명령

현재 workspace에서 생성한 receipt가 있으면 새 workflow를 중복 실행하지 않는다. 프로젝트 루트 PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File freecus/modal.ps1 workflow-status
powershell -NoProfile -ExecutionPolicy Bypass -File freecus/modal-dbpp.ps1 workflow-status
```

직접 Python으로 실행할 때:

```powershell
freecus/.venv-modal311/Scripts/python.exe -m freecus.modal_client workflow-status
```

새 환경 준비 및 최초 실행 순서:

```powershell
py -3.11 -m venv freecus/.venv-modal311
freecus/.venv-modal311/Scripts/python.exe -m pip install -r freecus/requirements-client.txt
./freecus/modal.ps1 prepare --benchmark dreambench
./freecus/modal.ps1 prepare --benchmark dreambench_plus
./freecus/modal.ps1 bootstrap
./freecus/modal.ps1 deploy
./freecus/modal.ps1 workflow --benchmark dreambench --budget-usd 28
# DreamBench++는 별도 계정 wrapper로 준비/배포 후 접수한다.
./freecus/modal-dbpp.ps1 prepare --benchmark dreambench_plus
./freecus/modal-dbpp.ps1 deploy --benchmark dreambench_plus
./freecus/modal-dbpp.ps1 workflow --benchmark dreambench_plus
```

`workflow`는 데이터 업로드와 원격 접수까지 마친 뒤 call ID를 출력한다. **접수 완료 전에는 로컬 PC를 종료하지 않는다.** DreamBench receipt/status는 `freecus/outputs/modal/`, DreamBench++는 `freecus/outputs/modal-method-comparison-dbpp/`에 분리한다. 두 작업 모두 접수가 완료되었다. 상태 확인 프로세스를 닫아도 원격 실행에는 영향이 없다. 실행 정책으로 ps1이 차단되면 위 상태 확인 명령처럼 `powershell -NoProfile -ExecutionPolicy Bypass -File`로 실행한다.

중단 및 결과 수집:

```powershell
./freecus/modal.ps1 stop
./freecus/modal.ps1 download --benchmark dreambench
```

`stop`은 coordinator와 알려진 stage/GPU call을 취소하고 checkpoint는 보존한다. 중단 후 재개는 기존 산출물·receipt·예산을 먼저 확인하여 별도 제출한다. 로컬 receipt를 임의로 지우고 같은 작업을 재제출하지 않는다. 다운로드 결과는 bundle/lane 디렉터리를 보존한다. 전체 지표 평가 전에 lane별 run fingerprint와 누락을 확인하고 하나의 평가 입력으로 병합해야 한다.

## VRAM 및 호스트 메모리

고정 HF safetensors index의 `total_size`로 확인한 weight 크기:

| 구성요소 | weight 크기 |
|---|---:|
| Qwen2-VL-7B | 15.44 GiB |
| Qwen2.5-7B | 14.19 GiB |
| FLUX transformer | 22.17 GiB |
| FLUX T5 | 8.87 GiB |

caption 단계에서 Qwen 두 모델을 로드하고 **완전히 해제한 뒤 FLUX와 BiRefNet을 로드**한다. 따라서 29.63 GiB의 Qwen과 약 32 GiB 이상의 FLUX 관련 weight를 동시에 합산할 필요는 없다. 활성값과 임시 텐서를 포함한 실측 전 사양으로 A100 80GB를 선택했다. 40/48GB가 충분하다고 아직 판정하지 않는다.

1024 reference 전체가 foreground라면 vital KV의 CPU cache 상한은 worker당 약 **14.06 GiB**다. GPU에는 현재 attention 계산에 필요한 KV를 옮긴다. 호스트 RAM은 worker당 **64 GiB**, CPU는 **2 physical cores**로 요청과 상한을 고정했다. VRAM 실측은 pilot에서 확인한다. 모델 크기 근거는 `verification/model-size-metadata.json`이다.

## $30 크레딧 판단

2026-10-30 [Modal 공식 요금](https://modal.com/pricing):

- A100 80GB: $0.000694/초 = **$2.4984/GPU·시간**.
- CPU: $0.0000131/core·초, RAM: $0.00000222/GiB·초.
- 2 cores + 64 GiB worker 한 개 합계: **약 $3.1042/시간**.
- worker 두 개가 한 시간 함께 동작하면 **약 $6.2084**다.

아래는 준비/초기로딩/재시도를 제외한 steady generation 계산 예시이며 **실측 속도가 아니다**.

| 이미지당 시간 | 이미지당 compute | DreamBench 750장 | 두 벤치마크 2,100장 |
|---|---:|---:|---:|
| 20초 | $0.01725 | $12.93 | $36.22 |
| 30초 | $0.02587 | $19.40 | $54.32 |
| 45초 | $0.03880 | $29.10 | $81.49 |
| 60초 | $0.05174 | $38.80 | $108.65 |

FreeCus는 이미지마다 참조 attention 추출 30 steps + 실제 생성 30 steps를 수행한다. 기존 InstantX의 24-step 한 번 실행 비용을 그대로 적용하지 않는다. Qwen 비용은 원문 prompt마다가 아니라 reference별 한 번 계산하고 caption을 재사용한다. 두 GPU는 시간을 줄이지만 같은 장수에 필요한 GPU초 총량을 절반으로 만들지는 않는다.

현재 compute admission은 DreamBench **$28**, DreamBench++ **$29**다. DreamBench의 이전 pilot·실패 시도 합계 약 **$0.7311**은 이전 receipt에 보존하며 새 $28 예산과 별도다. 새 pilot 비용은 각 admission에 포함한다. 남은 금액은 CPU stage/coordinator·빌드·시작/종료 등 예비비다. 초기 $25 + $5 예비비에서 실측 후 조정했으며, 전체 예상 비용의 **25% margin**은 유지한다. 부모 worker는 배정 시간을 넘기면 child를 종료한다. 이 추정 ledger는 Modal 실제 청구액이나 다른 작업의 비용을 읽지 않는다. 최종 과금 한도는 사용자가 관리하는 Workspace usage/spend limit을 따른다.

기존 A100 pilot은 약 32.08초/장, $0.02766/장, DreamBench 전체 약 $21.15, 25% 여유 포함 $26.44였다. DreamBench++는 A100에서 같은 속도라면 $30을 넘으므로 H100에서 별도로 측정한다. H100은 CPU/RAM 포함 worker당 약 $4.555/시간이며, 더 높은 시간당 요금을 감안한 실제 장당 비용으로 판단한다. **새 계정 $30으로 DreamBench++ 완료 가능 여부는 H100 pilot 전에는 확정하지 않는다.**

## 고해상도 reference의 Qwen attention 수정

최초 12장 pilot은 성공했지만 전체 DreamBench의 더 큰 reference에서 HF Qwen2-VL `VisionSdpaAttention`의 dense mask/attention이 OOM을 일으켰다. `runtime_src/qwen_for_subject.py`는 `cu_seqlens`별 동일 block attention을 4차원 unmasked SDPA로 계산하여 fused CUDA kernel을 사용할 수 있게 했다. 이미지 크기, visual token, 모델 정밀도, prompt는 그대로다. 여러 이미지 구간과 rotary embedding을 포함한 CPU 수치 동등성 테스트를 통과했다. fused kernel과 dense kernel 사이의 작은 부동소수점 차이는 가능하며 bitwise 동일성을 주장하지 않는다. 변경 전 pilot 출력은 별도 bundle에 보존하고, 최종 비교용 생성은 수정본으로 새 bundle에서 다시 시작한다. 최초 pilot의 Qwen peak 약 39.9 GiB와 FLUX peak 약 34.7 GiB는 전체 데이터의 최대 사용량을 보증하지 않는다.

워크스페이스의 결제수단, 크레딧 잔액, Workspace usage limit $30 / Spend limit $0 설정은 사용자가 직접 처리한다. [Modal GPU 안내](https://modal.com/docs/guide/gpu)는 GPU 사용에 결제수단 등록이 필요하다고 명시한다. [Budget 안내](https://modal.com/docs/guide/budgets)는 usage cap과 net spend cap을 구분한다.

## 환경 및 재현

새 workspace의 `2026.10` 이미지 빌더는 Python 3.10을 지원하지 않아 **원격 Python 3.11 + Modal SDK 1.6.3**을 사용한다. 기존 Python 3.10 CPU 검증 환경은 유지한다. 로컬 Python 3.11 client에는 Modal/HF SDK만 설치하며 CUDA 모델은 원격에서만 로드한다. PyTorch/diffusers/transformers 등 모델 의존성의 고정 버전, 1024 해상도, seed 42, 30 steps, guidance 3.5, selected reference와 원문 target은 유지한다.

FreeCus의 `paper` 모드는 논문에 맞춘 해석적 보정이다. 이 차이와 주 비교 기준은 `COMPARISON_PROTOCOL.md`를 따른다. 배포된 코드 snapshot과 모델 revision은 bundle에 고정한다. 코드가 바뀌면 기존 bundle에 섞지 못하도록 재준비를 요구한다.
