# Modal 배치 생성 실행 가이드

이 가이드는 아래의 10개 reference image와 이미지당 9개 prompt를 사용해, 한 실험 조건당 총 90장을 Modal GPU에서 생성하는 절차를 설명한다.

- Reference image: dog 7장, cat 3장
- Reference 경로: `C:\Users\82107\Downloads\AI_PROJECT\dreambooth_reference_dataset\InstantX-IP-Adapter-DreamBench_filtered`
- 로컬 결과 경로: `C:\Users\82107\Downloads\AI_PROJECT\Orthogonalization-InstantX-IP-Adapter\outputs`
- Modal 앱 이름: `orthogonalization-instantx-flux-batch`
- 원격 결과 Volume: `orthogonalization-instantx-batch-results`

`submit`이 성공해 `fc-...` 형태의 Function Call ID가 출력된 뒤에는 생성 작업이 Modal 서버에서 독립적으로 실행된다. 따라서 로컬 터미널을 닫거나 인터넷 연결을 끊고 PC를 종료해도 작업은 계속되며, `tmux`는 필요하지 않다. 다만 제출 도중에는 reference 업로드가 진행되므로, 반드시 Call ID가 출력될 때까지 터미널과 인터넷 연결을 유지한다.

## 1. 최초 1회 환경 준비

PowerShell을 열고 프로젝트 루트로 이동한다.

```powershell
cd C:\Users\82107\Downloads\AI_PROJECT
```

이 코드에서 검증한 Modal CLI 버전을 설치한다.

```powershell
python -m pip install "modal==1.4.3"
```

Modal 계정 인증을 진행한다. 명령 실행 후 브라우저에 표시되는 인증 절차를 마친다.

```powershell
python -m modal setup
```

이 프로젝트는 gated model인 `black-forest-labs/FLUX.1-dev`를 사용하므로, 먼저 Hugging Face에서 해당 모델의 이용 조건에 동의하고 읽기 권한이 있는 토큰을 준비해야 한다. 토큰을 PowerShell 환경 변수에 임시로 넣은 뒤 Modal Secret을 만든다.

```powershell
$env:HF_TOKEN="hf_여기에_본인의_토큰"
python -m modal secret create huggingface-secret "HF_TOKEN=$env:HF_TOKEN"
Remove-Item Env:HF_TOKEN
```

이미 같은 이름의 Secret이 있고 토큰만 교체하려면 다음처럼 `--force`를 사용한다.

```powershell
$env:HF_TOKEN="hf_새로운_토큰"
python -m modal secret create --force huggingface-secret "HF_TOKEN=$env:HF_TOKEN"
Remove-Item Env:HF_TOKEN
```

토큰을 명령 파일이나 Git 저장소에 기록하지 않는다.

## 2. Reference와 90개 작업을 먼저 검증하기

`preview`는 GPU를 시작하거나 이미지를 생성하지 않는다. 로컬 reference 10장, 종 분류, prompt 조합, 출력 파일명과 manifest를 사전 검증하는 단계다.

```powershell
python Orthogonalization-InstantX-IP-Adapter\batch_client.py preview
```

다음 항목을 확인한 뒤 진행한다.

- reference가 dog 7장, cat 3장으로 인식되는가
- 총 작업 수가 90개인가
- 서로 다른 출력 경로도 90개인가
- manifest digest가 정상적으로 출력되는가
- 예시 경로가 `cat_03/cat_03_running.png` 형태인가

이 검증이 실패하면 Modal에 제출하지 말고 reference 파일의 수, 확장자, 파일명을 먼저 확인한다.

## 3. Modal 배치 앱 배포하기

다음 명령은 배치 worker 코드를 Modal에 배포한다. 배포만으로 90장 생성 작업이 제출되는 것은 아니다.

```powershell
python -m modal deploy Orthogonalization-InstantX-IP-Adapter\batch_modal.py
```

코드나 Modal image 의존성을 변경한 경우에는 같은 명령으로 다시 배포한 뒤 새로운 배치를 제출한다.

## 4. 실험 조건별 90장 제출하기

지원하는 조건은 다음 네 가지다.

| 조건 | Visual embedding projection | Single-stream text-row mask |
|---|---:|---:|
| `baseline` | 적용 안 함 | 적용 안 함 |
| `projection` | 적용 | 적용 안 함 |
| `gate` | 적용 안 함 | 적용 |
| `combined` | 적용 | 적용 |

각 명령은 독립적으로 90장을 생성한다. 네 조건을 모두 실행하면 총 360장이므로, 비용과 출력 상태를 확인하기 쉽도록 우선 `baseline` 한 배치를 끝까지 실행한 뒤 나머지 조건을 순차적으로 제출하는 것을 권장한다.

`batch-id`는 원격 결과 폴더명이 되므로 조건마다 고유하게 지정한다. 영문 소문자, 숫자, 하이픈을 이용한 한 개의 안전한 이름을 권장한다. 아래 날짜 부분은 실제 실행일이나 실험 버전에 맞게 바꾼다.

### Baseline

```powershell
python Orthogonalization-InstantX-IP-Adapter\batch_client.py submit --condition baseline --batch-id dreambench-baseline-0723 --yes
```

### Projection only

```powershell
python Orthogonalization-InstantX-IP-Adapter\batch_client.py submit --condition projection --batch-id dreambench-projection-0723 --orthogonalization-strength 0.5 --yes
```

### Text-row gate only

```powershell
python Orthogonalization-InstantX-IP-Adapter\batch_client.py submit --condition gate --batch-id dreambench-gate-0723 --yes
```

### Combined

```powershell
python Orthogonalization-InstantX-IP-Adapter\batch_client.py submit --condition combined --batch-id dreambench-combined-0723 --orthogonalization-strength 0.5 --yes
```

`--yes`는 reference 10장과 90개 작업의 요약을 확인했다는 의미로 실제 비동기 제출을 승인한다. 이 옵션을 생략하면 비용이 발생하는 배치를 제출하지 않는다.

기본 생성 설정은 모든 조건에서 동일하게 유지되므로 조건 간 비교가 가능하다. projection 강도를 비교하려면 한 번의 배치 안에서 값을 바꾸지 말고, `batch-id`를 새로 만들어 별도의 90장 배치로 제출한다.

## 5. 안전하게 PC를 종료할 수 있는 시점

제출이 성공하면 터미널에 다음 두 식별자가 출력된다.

- Batch ID: 사용자가 지정한 `dreambench-...`
- Function Call ID: Modal이 발급한 `fc-...`

이때 reference 업로드와 원격 비동기 호출 등록이 모두 끝난 것이다. `fc-...`가 출력되고 명령 프롬프트가 돌아온 것을 확인한 뒤에는 다음 동작을 해도 생성 작업에 영향을 주지 않는다.

- PowerShell 종료
- 인터넷 연결 해제
- 로컬 PC 종료

해당 Batch가 끝나기 전에는 같은 앱을 다시 `modal deploy`하거나
`modal app stop`하지 않는다. 이는 로컬 연결 종료와 달리 원격 앱의 실행
상태를 직접 변경하는 명령이다.

로컬의 `.modal_jobs`에는 Batch ID와 Call ID의 대응 관계가 저장된다. 그래도 복구를 쉽게 하려면 터미널에 출력된 두 ID를 별도로 메모해 두는 것이 좋다.

PC가 꺼져 있는 동안에는 원격 결과가 Modal Volume에 저장될 뿐 로컬 `outputs`로 자동 복사되지는 않는다. PC를 다시 켠 뒤 7절의 다운로드 명령을 실행해야 한다.

## 6. 진행 상태와 로그 확인하기

PC를 다시 켜고 프로젝트 루트에서 Batch ID로 상태를 확인한다.

```powershell
cd C:\Users\82107\Downloads\AI_PROJECT
python Orthogonalization-InstantX-IP-Adapter\batch_client.py status --batch-id dreambench-baseline-0723
```

로컬 job 기록이 없어졌지만 `fc-...`를 알고 있다면 Call ID로도 확인할 수 있다.

```powershell
python Orthogonalization-InstantX-IP-Adapter\batch_client.py status --call-id fc-여기에_Call_ID
```

상세 로그는 Modal CLI로 확인한다.

```powershell
python -m modal app logs orthogonalization-instantx-flux-batch --function-call fc-여기에_Call_ID --timestamps
```

실시간으로 이어서 보려면 `-f`를 추가한다.

```powershell
python -m modal app logs orthogonalization-instantx-flux-batch --function-call fc-여기에_Call_ID --timestamps -f
```

로그 보기에서 `Ctrl+C`를 누르는 것은 로컬의 로그 스트리밍만 종료하며, 이미 제출된 원격 생성 작업을 취소하지 않는다.

원격 Volume에 어느 정도 저장되었는지 파일 목록만 확인하려면 다음 명령을 사용할 수 있다.

```powershell
python -m modal volume ls orthogonalization-instantx-batch-results dreambench-baseline-0723
```

## 7. 완료 결과 다운로드하기

상태가 완료된 것을 확인한 뒤 Batch ID로 다운로드한다.

```powershell
python Orthogonalization-InstantX-IP-Adapter\batch_client.py download --batch-id dreambench-baseline-0723
```

기본적으로 조건별 결과는 다음과 같이 Batch ID 아래에 분리된다.

```text
Orthogonalization-InstantX-IP-Adapter/
└─ outputs/
   └─ dreambench-baseline-0723/
      ├─ cat_03/
      │  ├─ cat_03_running.png
      │  └─ ... 총 9장
      ├─ dog_02/
      │  └─ ... 총 9장
      ├─ ... reference별 결과 폴더
      └─ _metadata/
         ├─ manifest.json
         ├─ config.json
         └─ status.json
```

동일 Batch ID를 다시 다운로드하면서 기존 로컬 결과를 의도적으로 덮어쓰려면 `--force`를 사용한다.

```powershell
python Orthogonalization-InstantX-IP-Adapter\batch_client.py download --batch-id dreambench-baseline-0723 --force
```

## 8. 결과가 정확히 90장인지 검증하기

다운로드 후 해당 Batch ID 폴더의 PNG 개수를 센다.

```powershell
$batchDir = "C:\Users\82107\Downloads\AI_PROJECT\Orthogonalization-InstantX-IP-Adapter\outputs\dreambench-baseline-0723"
(Get-ChildItem -LiteralPath $batchDir -Recurse -File -Filter *.png | Measure-Object).Count
```

정상 완료라면 출력은 `90`이다. Reference별 9장도 확인할 수 있다.

```powershell
Get-ChildItem -LiteralPath $batchDir -Directory |
    Where-Object { $_.Name -ne "_metadata" } |
    ForEach-Object {
        [PSCustomObject]@{
            Reference = $_.Name
            PNGCount  = (Get-ChildItem -LiteralPath $_.FullName -File -Filter *.png | Measure-Object).Count
        }
    }
```

10개 reference 폴더가 모두 표시되고 각 `PNGCount`가 `9`여야 한다.

## 9. 실패한 배치 이어서 실행하기

Worker는 생성된 각 PNG와 상태를 원격 Volume에 순차적으로 저장한다. 일부 이미지 생성 후 작업이 실패했다면, 먼저 이전 Call이 실제로 실패 또는 종료되었는지 상태와 로그로 확인한다. 아직 실행 중인 Call과 같은 Batch ID를 동시에 제출하면 안 된다.

실패가 확인되면 처음 사용한 것과 동일한 `batch-id`, 조건, projection 강도 및 생성 설정으로 다시 제출한다.

```powershell
python Orthogonalization-InstantX-IP-Adapter\batch_client.py submit --condition combined --batch-id dreambench-combined-0723 --orthogonalization-strength 0.5 --yes
```

기본 동작에서는 원격에 이미 존재하는 완성 PNG를 건너뛰고 나머지 작업만 이어서 생성한다. 재개할 때 `--overwrite`를 추가하지 않는다. 전체 90장을 처음부터 다시 생성하려는 경우에만 새 Batch ID를 사용하는 것을 권장하며, 기존 결과를 의도적으로 재생성해야 할 때만 `--overwrite`를 사용한다.

Function Call 결과 조회에는 보존 기간이 있지만, 생성된 이미지와 metadata는 별도의 Modal Volume에 남는다. 오래된 Call ID의 상태를 더 이상 조회할 수 없는 경우에도 Batch ID로 원격 Volume을 확인하고 결과를 다운로드할 수 있다.

## 빠른 실행 순서

최초 설정을 마친 뒤 일상적인 실행은 아래 여섯 단계로 요약된다.

1. `batch_client.py preview`로 10개 reference와 90개 작업을 확인한다.
2. 코드가 변경되었다면 `modal deploy ...\batch_modal.py`로 배포한다.
3. `batch_client.py submit ... --yes`로 한 조건을 제출한다.
4. `fc-...` 출력과 프롬프트 복귀를 확인한 뒤 PC를 종료해도 된다.
5. 나중에 `status` 또는 `modal app logs`로 완료 여부를 확인한다.
6. `download` 후 PowerShell로 PNG가 정확히 90장인지 검증한다.
