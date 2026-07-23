# Source-aware 실험: Modal 2-GPU 실행 가이드

이 문서는 VS Code의 PowerShell 터미널에서 다음 두 실험을 동시에 실행하는 절차를 설명한다.

- `sa_projection`: reference caption에서 만든 source behavior/background 방향을 visual embedding에서 약화
- `sa_combined`: 동일한 source-aware projection과 single-stream text-row gate를 함께 적용

각 조건은 reference 10장 × target prompt 9개로 90장을 생성한다. 두 조건의 총 생성량은 180장이다.

기존 `baseline`, `projection`, `gate`, `combined` 결과는 수정하지 않는다. 아래에서 새로운 batch ID와 결과 폴더를 사용하므로 기존 360장과 충돌하지 않는다.

## 0. 실행 전 확인 사항

VS Code에서 `터미널 → 새 터미널`을 선택하고, 터미널 종류가 PowerShell인지 확인한다.

프로젝트 최상위 폴더로 이동한다.

```powershell
cd C:\Users\82107\Downloads\AI_PROJECT
```

현재 위치를 확인한다.

```powershell
Get-Location
```

다음 경로가 출력되어야 한다.

```text
C:\Users\82107\Downloads\AI_PROJECT
```

이번 실험은 다음 caption 파일을 사용한다.

```powershell
$captionFile = "Orthogonalization-InstantX-IP-Adapter\source_captions.json"
Test-Path -LiteralPath $captionFile
```

출력이 `True`인지 확인한다.

## 1. 확정된 source caption 확인

[source_captions.json](C:/Users/82107/Downloads/AI_PROJECT/Orthogonalization-InstantX-IP-Adapter/source_captions.json)은 reference별 외형·품종·털 색상은 제외하고, source behavior와 source background만 기록한다.

현재 파일에는 연구자가 제공한 다음 10개 caption이 최종값으로 반영되어 있다.

| Reference stem | Source behavior | Source background |
|---|---|---|
| `dog_02` | `sitting` | `against an orange wall` |
| `dog2_04` | `lying` | `on a street` |
| `dog3_04` | `running` | `on a rocky beach` |
| `dog5_02` | `lying` | `on a grey sofa` |
| `dog6_02` | `sitting` | `against a solid orange background` |
| `dog7_03` | `running` | `on the beach` |
| `dog8_04` | `lying` | `on a blanket` |
| `cat_03` | `sitting` | `on dry leaves` |
| `cat2_00` | `sitting` | `on a wood floor` |
| `PA_white_cat_background` | `walking` | `through green grass` |

`PA_white_cat_background`는 전달받은 caption 목록에서는 `.jpg`로 표기되었지만, 실제 데이터셋 파일은 `PA_white_cat_background.png`이다. 코드는 확장자를 제외한 정확한 stem과 이미지 content hash로 이를 연결한다.

예를 들어 `dog_02`의 레코드는 다음과 같다.

```json
"dog_02": {
  "source_behavior": "sitting",
  "source_background": "against an orange wall"
}
```

코드는 이 두 표현으로 다음 네 prompt를 자동 구성한다.

```text
a dog
a dog sitting
a dog against an orange wall
a dog sitting against an orange wall
```

위 caption은 더 이상 자동 추정값이 아니라 실험의 확정 입력이다. 이후 caption을 수정하려면 두 실험을 제출하기 전에 완료한다. `sa_projection` 제출과 `sa_combined` 제출 사이에는 파일을 수정하지 않는다. 두 조건은 동일한 caption digest를 사용해야 공정하게 비교할 수 있다.

## 2. Reference와 caption 사전 검증

이 단계에서는 GPU를 사용하거나 과금하지 않는다.

```powershell
python Orthogonalization-InstantX-IP-Adapter\batch_client.py preview --source-captions $captionFile
```

정상적인 핵심 출력은 다음과 같다.

```text
references: 10 (dog 7, cat 3)
prompts/reference: 9
total jobs: 90
Source caption 검증 완료
references: 10
semantic digest: 587f3d0c02cabf1f79d13cbfb1ac585ed4f21ed45fa5f54d0a9e001c85b91cb7
```

누락된 reference, 잘못된 species, 비어 있는 behavior/background가 있으면 여기서 중단되므로 오류를 수정한 후 다시 실행한다. 위 digest는 현재 확정 caption과 정확한 10개 reference 이미지 content hash에 대응한다.

## 3. 수정된 Modal worker 배포

새로운 `sa_projection`, `sa_combined` 코드가 원격 worker에 반영되려면 한 번 재배포해야 한다.

```powershell
python -m modal deploy Orthogonalization-InstantX-IP-Adapter\batch_modal.py
```

배포 완료 메시지가 출력되고 PowerShell 입력 프롬프트가 돌아올 때까지 기다린다.

배포는 기존 로컬 결과 이미지를 삭제하지 않는다. 기존 Modal Volume 결과도 그대로 유지된다.

> Modal 앱을 다시 배포하면 동적 autoscaler 설정이 코드의 기본값인 `max_containers=1`로 초기화된다. 따라서 반드시 먼저 배포하고, 다음 단계에서 2개로 변경한다.

## 4. 최대 동시 GPU container 수를 2개로 설정

다음 명령은 배포된 `BatchGenerator`의 최대 container 수를 2개로 변경한다.

```powershell
python -c "import modal; C=modal.Cls.from_name('orthogonalization-instantx-flux-batch','BatchGenerator'); worker=C(); worker.update_autoscaler(max_containers=2); print('max_containers=2 applied')"
```

정상 출력:

```text
max_containers=2 applied
```

이 설정만으로 GPU 두 대가 즉시 과금되는 것은 아니다. 다음 단계에서 서로 독립적인 두 batch call을 제출하면 수요에 따라 최대 두 개의 A100-80GB container가 생성된다. Modal GPU 재고나 계정 한도가 부족하면 한 작업이 먼저 실행되고 다른 작업은 대기할 수 있다.

## 5. `sa_projection` 90장 제출

이전 실험과 동일한 생성 설정을 명시적으로 사용한다.

```powershell
python Orthogonalization-InstantX-IP-Adapter\batch_client.py submit --condition sa_projection --source-captions $captionFile --batch-id dreambench-sa-projection-0724 --orthogonalization-strength 0.5 --ip-adapter-scale 0.7 --guidance-scale 3.5 --steps 24 --seed 42 --width 960 --height 1280 --yes
```

다음 두 값을 확인하고 기록한다.

```text
Batch ID: dreambench-sa-projection-0724
Function Call ID: fc-...
```

`fc-...`가 출력되고 PowerShell 입력 프롬프트가 돌아오면 첫 작업은 Modal에 비동기로 제출된 상태다.

## 6. `sa_combined` 90장 제출

같은 VS Code 터미널에서 바로 다음 명령을 실행한다. 별도의 터미널 창을 열 필요는 없다.

```powershell
python Orthogonalization-InstantX-IP-Adapter\batch_client.py submit --condition sa_combined --source-captions $captionFile --batch-id dreambench-sa-combined-0724 --orthogonalization-strength 0.5 --ip-adapter-scale 0.7 --guidance-scale 3.5 --steps 24 --seed 42 --width 960 --height 1280 --yes
```

두 번째 Function Call ID도 확인한다.

```text
Batch ID: dreambench-sa-combined-0724
Function Call ID: fc-...
```

두 Function Call ID가 모두 출력됐다면 다음 작업이 최대 두 GPU에서 동시에 진행된다.

```text
GPU container 1 → sa_projection 90장
GPU container 2 → sa_combined 90장
```

이 시점부터 VS Code를 닫거나 로컬 PC를 종료해도 원격 작업은 계속된다. `tmux`는 필요하지 않다.

실행 중에는 다음 명령을 사용하지 않는다.

```text
modal deploy
modal app stop
```

재배포하면 autoscaler 설정과 배포 버전이 바뀔 수 있고, app stop은 진행 중인 작업을 중단할 수 있다.

## 7. 진행 상태 확인

PC를 다시 켠 경우 먼저 프로젝트 폴더로 이동한다.

```powershell
cd C:\Users\82107\Downloads\AI_PROJECT
```

`sa_projection` 상태:

```powershell
python Orthogonalization-InstantX-IP-Adapter\batch_client.py status --batch-id dreambench-sa-projection-0724
```

`sa_combined` 상태:

```powershell
python Orthogonalization-InstantX-IP-Adapter\batch_client.py status --batch-id dreambench-sa-combined-0724
```

진행 중이면 다음과 유사하게 출력된다.

```text
Modal call state: pending 또는 running
```

완료되면 다음 핵심 값이 확인되어야 한다.

```text
Modal call state: completed
completed_count: 90
state: completed
```

## 8. 상세 로그 확인

Function Call ID를 직접 복사하지 않아도 로컬 job 기록에서 가져올 수 있다.

`sa_projection` 로그:

```powershell
$projectionCallId = (Get-Content -Raw "Orthogonalization-InstantX-IP-Adapter\.modal_jobs\dreambench-sa-projection-0724.json" | ConvertFrom-Json).call_id
python -m modal app logs orthogonalization-instantx-flux-batch --function-call $projectionCallId --timestamps -f
```

로그 화면을 닫으려면 `Ctrl+C`를 누른다. 이는 로그 보기만 종료하며 원격 생성 작업은 계속된다.

`sa_combined` 로그:

```powershell
$combinedCallId = (Get-Content -Raw "Orthogonalization-InstantX-IP-Adapter\.modal_jobs\dreambench-sa-combined-0724.json" | ConvertFrom-Json).call_id
python -m modal app logs orthogonalization-instantx-flux-batch --function-call $combinedCallId --timestamps -f
```

## 9. 완료 결과 다운로드

각 batch가 `completed_count: 90`으로 완료된 후 다운로드한다.

`sa_projection`:

```powershell
python Orthogonalization-InstantX-IP-Adapter\batch_client.py download --batch-id dreambench-sa-projection-0724
```

`sa_combined`:

```powershell
python Orthogonalization-InstantX-IP-Adapter\batch_client.py download --batch-id dreambench-sa-combined-0724
```

결과 경로:

```text
Orthogonalization-InstantX-IP-Adapter\outputs\
├─ dreambench-sa-projection-0724\
└─ dreambench-sa-combined-0724\
```

각 batch 폴더에는 reference별 하위 폴더 10개와 `_metadata`가 생성된다. 각 reference 폴더에는 target prompt 9개에 해당하는 PNG 9장이 들어간다.

이미 로컬에 같은 batch의 일부 파일이 있어 다운로드 충돌이 발생한다면, 원격 작업이 정확히 90장 완료됐는지 확인한 다음 `--force`를 추가한다.

```powershell
python Orthogonalization-InstantX-IP-Adapter\batch_client.py download --batch-id dreambench-sa-projection-0724 --force
```

## 10. 90장 + 90장 검증

다음 PowerShell 코드는 새 두 batch만 계산하며 기존 360장은 포함하지 않는다.

```powershell
$batchIds = @(
    "dreambench-sa-projection-0724",
    "dreambench-sa-combined-0724"
)

$total = 0
foreach ($id in $batchIds) {
    $path = "Orthogonalization-InstantX-IP-Adapter\outputs\$id"
    $count = (Get-ChildItem -LiteralPath $path -Recurse -File -Filter *.png | Measure-Object).Count
    $total += $count
    Write-Output "$id -> $count PNG"
}
Write-Output "source-aware total -> $total PNG"
```

정상 결과:

```text
dreambench-sa-projection-0724 -> 90 PNG
dreambench-sa-combined-0724 -> 90 PNG
source-aware total -> 180 PNG
```

기존 네 조건 360장까지 합친 전체 누적 생성량은 540장이다.

## 11. 모든 작업 완료 후 autoscaler 복원

두 작업의 완료와 다운로드를 확인한 뒤 최대 container 수를 다시 1개로 복원한다.

```powershell
python -c "import modal; C=modal.Cls.from_name('orthogonalization-instantx-flux-batch','BatchGenerator'); worker=C(); worker.update_autoscaler(max_containers=1); print('max_containers=1 restored')"
```

정상 출력:

```text
max_containers=1 restored
```

Modal은 입력이 없으면 GPU container를 자동으로 scale-to-zero한다. 따라서 `modal app stop`이나 결과 Volume 삭제는 필요하지 않다. 원격 Volume은 재다운로드와 실험 재현을 위해 보존하는 것을 권장한다.

## 실패 시 재개

한 batch가 실패한 경우 caption 파일과 모든 설정을 변경하지 말고 동일한 batch ID로 다시 제출한다. 기본적으로 이미 완료된 PNG는 건너뛰고 남은 작업부터 재개한다.

예를 들어 `sa_projection` 재개:

```powershell
python Orthogonalization-InstantX-IP-Adapter\batch_client.py submit --condition sa_projection --source-captions $captionFile --batch-id dreambench-sa-projection-0724 --orthogonalization-strength 0.5 --ip-adapter-scale 0.7 --guidance-scale 3.5 --steps 24 --seed 42 --width 960 --height 1280 --yes
```

재개 시 `--overwrite`는 사용하지 않는다. `--overwrite`를 추가하면 완료된 결과까지 다시 생성한다.
