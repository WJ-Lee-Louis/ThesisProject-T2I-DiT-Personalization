# Modal 원격 이미지 생성 — 오류 복구 후 두 조건만 진행

**기존 A/B 작업의 복구는 완료했다. 지금은 2절의 A/B `resume` 명령부터 실행한다. 두 계정 모두 `SUBMITTED`가 출력된 뒤 VS Code와 노트북을 종료할 수 있다.** 기존 접수증·Volume·입력을 재사용하므로 새 `submit`이나 전체 재업로드는 필요 없다.

## 1. 이번 실행 범위와 복구 내용

| 조건 | single stream gate | projection | A / shard 0 | B / shard 1 | 전체 |
|---|---|---|---:|---:|---:|
| `baseline` | OFF | OFF — 최초 InstantX-IP-Adapter 기준 | 1,050장 | 1,050장 | 2,100장 |
| `source_target_ssg` | ON | source·target 모두 `local` 방식 | 1,050장 | 1,050장 | 2,100장 |
| **합계** | | | **2,100장** | **2,100장** | **4,200장** |

**SSG 없는 projection 조건 `source_target`은 보류했다.** 기존 pilot 4장 등 해당 조건의 파일은 삭제하지 않고 보관하며 이번 작업에서는 생성하거나 합산하지 않는다.

1024×1024, 24 steps, seed 42, A100-80GB 설정을 유지한다. 모듈 A/B의 기존 caption·보존 표현·차분 캐시를 재사용하며 추가 LLM 호출은 없다. 본 method는 캐시의 source/target local 차분을 인코딩해 SVD projection·norm restoration을 적용하고 single stream gate를 켠다.

기존 승인된 `092-08`, `154-07`의 source-only fallback도 그대로 유지하며 임의로 제외하지 않는다.

복구에서 처리한 사항:

- 이전 A/B app의 모든 호출이 종료된 것을 확인했다.
- 생성 함수를 원격에 전달하기 전에 Modal 동기화 wrapper를 제거했다. 기존 생성 함수와 inference 코드 자체는 바꾸지 않아 이미지 signature를 유지한다.
- 기존 접수증의 계획을 두 조건으로 줄이고 수정된 transport 코드에 연결했다. 원래 계획은 로컬과 Volume의 history에 보관했다.
- 기존 PNG·JSON과 업로드 입력은 보존했다. 정상 완료 결과는 재개 시 건너뛴다.
- Synchronizer 오류는 즉시, 기타 생성 오류는 3회 연속 발생 시 해당 lane을 중단한다. 전체 traceback도 저장한다. 무한 재시도는 하지 않는다.

기존 `outputs/remote_receipt_A.json`, `outputs/remote_receipt_B.json`을 그대로 사용한다. `outputs/remote_benchmark_runs/`와 Modal Volume은 삭제하지 않는다. 보류된 `source_target`을 다시 넣기 위해 이전 세 조건 명령을 사용하지 않는다.

## 2. A/B 본 작업 원격 재개 — 여기서 시작

각 계정의 worker 수는 기존 요청대로 **3개**를 유지한다. 계정당 2,100개 작업을 세 lane에 나눠 **lane당 700장(조건마다 350장)**을 맡는다. 두 계정 합계 최대 GPU 6개이며, 실제 할당은 각 workspace의 quota와 가용 GPU에 따른다. 이미 완료된 이미지 수만큼 신규 생성 수는 줄어든다.

각 lane은 baseline을 먼저 처리한 뒤 source_target_ssg를 처리한다. 모든 조건이 원격 계획에 들어 있으므로, 노트북을 꺼도 다음 조건으로 진행한다. 사용량·잔액은 각 workspace 대시보드에서 확인한다.

### A — wj-lee-louis / shard 0

```powershell
Set-Location 'C:\Users\82107\Downloads\AI_PROJECT\Orthogonalization-InstantX-IP-Adapter'
$modalPython = 'C:\Users\82107\.venvs\orthogonalization-modal311\Scripts\python.exe'
$env:PYTHONUTF8 = '1'
$env:MODAL_PROFILE = 'wj-lee-louis'
Remove-Item Env:MODAL_TOKEN_ID,Env:MODAL_TOKEN_SECRET -ErrorAction SilentlyContinue
& $modalPython scripts/benchmark_remote.py resume --receipt outputs/remote_receipt_A.json
if ($LASTEXITCODE -ne 0) { throw 'A resume failed; inspect the error before retrying' }
```

### B — modal-research-2 / shard 1

```powershell
Set-Location 'C:\Users\82107\Downloads\AI_PROJECT\Orthogonalization-InstantX-IP-Adapter'
$modalPython = 'C:\Users\82107\.venvs\orthogonalization-modal311\Scripts\python.exe'
$env:PYTHONUTF8 = '1'
$env:MODAL_PROFILE = 'modal-research-2'
Remove-Item Env:MODAL_TOKEN_ID,Env:MODAL_TOKEN_SECRET -ErrorAction SilentlyContinue
& $modalPython scripts/benchmark_remote.py resume --receipt outputs/remote_receipt_B.json
if ($LASTEXITCODE -ne 0) { throw 'B resume failed; inspect the error before retrying' }
```

두 블록은 별도 PowerShell 창에서 실행하거나, 한 창에서 A → B 순서로 실행해도 된다.

**각 블록에서 `ACCEPTED lane=0/1/2`와 마지막 `SUBMITTED: ...remote_receipt_A.json` / `...remote_receipt_B.json`을 확인한다.** 두 제출 모두 SUBMITTED가 나올 때까지 인터넷과 터미널을 유지한다.

A/B 모두 `SUBMITTED`가 나온 뒤에는 VS Code·터미널·노트북을 종료해도 된다. 계산과 PNG/JSON 저장은 Modal에서 계속 진행한다. 로컬 다운로드는 나중에 4절을 실행한다.

`SUBMITTED`는 접수 완료를 뜻하며 이미지 생성 완료는 아니다. 최초 모델 로딩은 수분 걸릴 수 있다. 별도의 CPU 점검에서는 제출한 Python 프로세스를 종료한 뒤 새 Python 프로세스에서 원격 완료 결과와 Volume 저장을 확인했다.

## 3. 상태 확인 — 새 GPU 작업을 시작하지 않음

```powershell
Set-Location 'C:\Users\82107\Downloads\AI_PROJECT\Orthogonalization-InstantX-IP-Adapter'
$modalPython = 'C:\Users\82107\.venvs\orthogonalization-modal311\Scripts\python.exe'
$env:PYTHONUTF8 = '1'
Remove-Item Env:MODAL_TOKEN_ID,Env:MODAL_TOKEN_SECRET -ErrorAction SilentlyContinue
$env:MODAL_PROFILE = 'wj-lee-louis'
& $modalPython scripts/benchmark_remote.py status --receipt outputs/remote_receipt_A.json
$env:MODAL_PROFILE = 'modal-research-2'
& $modalPython scripts/benchmark_remote.py status --receipt outputs/remote_receipt_B.json
```

각 계정의 두 조건이 각각 `expected=1050`, `committed_pairs=1050`이고 세 lane이 `completed`이면 원격 생성 완료다. `calls.status=active`는 원격 호출 실행 중, `partial` 또는 `interrupted`와 실패 기록은 확인이 필요한 상태다. 강제 종료 시 lane의 과거 `running` 기록이 남을 수 있으므로 `calls`와 대시보드도 함께 확인한다.

진행률의 pair 수는 파일 쌍 기준이다. 다운로드에서 PNG hash·signature·설정을 검증한다. 복구 직후에는 `phase=repaired`, lane은 `ready`이며 이전 종료된 call 기록이 보일 수 있다. 2절을 실행하면 새 app/call 정보로 갱신된다.

## 4. 생성 결과 다운로드 — 기존 로컬 결과에 합치기

3절과 같은 프로젝트 폴더에서 실행한다. 이미 로컬에 있는 정상 파일은 건너뛴다.

```powershell
$modalPython = 'C:\Users\82107\.venvs\orthogonalization-modal311\Scripts\python.exe'
$env:MODAL_PROFILE = 'wj-lee-louis'
& $modalPython scripts/benchmark_remote.py download --receipt outputs/remote_receipt_A.json
if ($LASTEXITCODE -ne 0) { throw 'A download incomplete; check status and retry later' }
$env:MODAL_PROFILE = 'modal-research-2'
& $modalPython scripts/benchmark_remote.py download --receipt outputs/remote_receipt_B.json
if ($LASTEXITCODE -ne 0) { throw 'B download incomplete; check status and retry later' }
```

저장 위치는 `outputs/benchmark_images_v1/baseline/`과 `outputs/benchmark_images_v1/source_target_ssg/`다. 다운로드는 PNG hash·이미지 형식·job signature·manifest 행·설정을 검증한다. 생성 중에 실행하면 아직 미완료인 이미지는 `pending`으로 남는다. 생성 완료 후 같은 명령을 다시 실행하면 된다. 다운로드 후에도 원격 결과는 Volume에 남는다.

## 5. 두 조건 전체 완료 검증 — GPU 사용 없음

양쪽 다운로드가 끝나면 실행한다. `--run`이 없으므로 GPU를 시작하지 않는다.

```powershell
Set-Location 'C:\Users\82107\Downloads\AI_PROJECT\Orthogonalization-InstantX-IP-Adapter'
$modalPython = 'C:\Users\82107\.venvs\orthogonalization-modal311\Scripts\python.exe'
$env:PYTHONUTF8 = '1'
foreach ($benchmarkCondition in @('baseline', 'source_target_ssg')) {
  & $modalPython scripts/benchmark_modal.py --condition $benchmarkCondition --num-shards 1 --shard-index 0
  if ($LASTEXITCODE -ne 0) { throw "Validation failed: $benchmarkCondition" }
}
```

두 조건 각각 **`selected=2100`, `complete=2100`, `pending=0`**이면 총 4,200장의 생성과 로컬 저장이 완료됐다. metric 평가는 추후 별도 요청 시 진행한다.

## 6. 오류·시간 초과로 멈췄을 때

같은 오류가 반복되면 새 receipt로 중복 제출하지 않는다. 상태와 traceback을 먼저 확인한다. 문제가 해결된 뒤 모든 lane이 종료된 상태에서만 2절의 같은 `resume` 명령을 실행한다. 이미 commit된 이미지 쌍은 재사용한다. 실행 중인 call이 있으면 중복 실행 방지를 위해 재개가 거부된다.

원격 lane은 호출당 최대 24시간이며, 그 전에 commit한 결과는 남는다. 노트북 종료는 detached 원격 작업을 중단하지 않는다. 원격 실행을 중단하려면 각 Modal 대시보드에서 해당 app을 `Stop app`으로 중단한다. 로컬 Ctrl+C만으로 detached 작업이 중단된다고 가정하지 않는다.

이번 복구 검증: 로컬 테스트 24개가 통과했다. A의 `018-08`, B의 `017-12`에서 각각 baseline 1장과 본 method 1장을 **실제 A100-80GB GPU로 생성**했다. 로컬 `app.run` 연결 종료 후 원격에서 완료됐으며 PNG·JSON과 hash·1024×1024 크기를 검증했다. baseline은 gate/projection OFF, 본 method는 gate ON·`semantic_local`·source_target origin이며 basis rank는 A 3, B 2다. 이 4장은 로컬과 기존 본 작업 Volume에 보존해 재개 시 다시 생성하지 않는다. 검증용 GPU app은 종료하고 전체 미완료 작업은 아직 제출하지 않았다.
