# Modal CLIP-T / CLIP-I / DINO-I 실행 가이드

**현재 이 프로젝트의 공통 입력 준비는 완료되어 있으므로 2절·3절부터 실행하면 된다.** 이전 안내의 `$evalPython`, 로컬 `pip install`, `benchmark_metrics.py evaluate` 명령은 이번 Modal 제출에 사용하지 않는다. 이 문서에 해당 명령이 보이면 디스크의 최신 파일을 다시 연다.

**1절부터 순서대로 진행한다.** 1절은 한 터미널에서 한 번, 2·3절은 각각 터미널 A/B에서 실행한다. 로컬 CUDA나 평가 패키지 설치는 필요 없다. 기존 Python 3.11 venv와 두 Modal profile을 사용한다. 원본 생성 이미지와 생성용 Volume은 보존한다.

| 계정 | profile | 평가 대상 | baseline | source_target_ssg | 합계 | GPU |
|---|---|---|---:|---:|---:|---|
| A | wj-lee-louis | DreamBench++ | 1,350 | 1,350 | 2,700 | T4 1개 |
| B | modal-research-2 | DreamBench | 750 | 750 | 1,500 | T4 1개 |

projection만 적용한 `source_target`은 보류한다. 각 이미지에서 세 지표를 계산하며 class마다 GPU 작업을 추가하지 않는다. 생성 때 사용한 A100 worker 3개 설정은 평가에 사용하지 않는다.

## 1. 공통 준비 — 한 터미널에서 한 번

이 명령이 A/B 생성 Volume의 결과를 **자동으로 로컬에 수집**한다. 사용자가 먼저 전체 이미지를 수동 다운로드했을 필요는 없다. 이미 있는 정상 PNG+JSON은 재사용한다. 누락 이미지가 있으면 준비가 실패하고 GPU 평가는 시작하지 않는다.

```powershell
Set-Location 'C:\Users\82107\Downloads\AI_PROJECT\Orthogonalization-InstantX-IP-Adapter'
$modalPython = 'C:\Users\82107\.venvs\orthogonalization-modal311\Scripts\python.exe'
$env:PYTHONUTF8 = '1'
Remove-Item Env:MODAL_TOKEN_ID,Env:MODAL_TOKEN_SECRET -ErrorAction SilentlyContinue
& $modalPython scripts/benchmark_metric_remote.py prepare
if ($LASTEXITCODE -ne 0) { throw 'Metric input preparation failed; do not start GPU evaluation' }
```

profile은 내부에서 별도 Python process로 A/B를 각각 선택한다. 생성 receipt `outputs/remote_receipt_A.json`, `outputs/remote_receipt_B.json`을 사용한다. 표준 생성 폴더 `outputs/benchmark_images_v1`에 결과를 보존하며 기존 파일의 job signature·PNG hash를 확인한다.

전체 4,200건의 reference/target/설정, 1024×1024 크기, baseline의 projection/SSG OFF 및 본 method의 projection/SSG ON을 검증한다. 다음 파일들을 만든다.

- `outputs/benchmark_metrics_v1/pairs.json`: 지정 reference 한 장과 생성 PNG의 전체 평가 목록
- `outputs/metric_receipt_A.json`, `outputs/metric_receipt_B.json`: 평가 실행·재개 정보
- `outputs/benchmark_metrics_v1/runs/<run_id>/inputs_A.zip`, `inputs_B.zip`: 계정별 입력 묶음

`PREPARED`가 나온 뒤 2·3절로 이동한다. 다시 실행해도 준비된 입력은 재사용한다. CPU FETCH/검증/패키징은 이미지 파일 수와 용량에 따라 시간이 걸린다. 이 단계는 GPU를 사용하지 않으며 완료 전까지 터미널을 열어 둔다.

## 2. 터미널 A — DreamBench++ 2,700장

```powershell
Set-Location 'C:\Users\82107\Downloads\AI_PROJECT\Orthogonalization-InstantX-IP-Adapter'
$modalPython = 'C:\Users\82107\.venvs\orthogonalization-modal311\Scripts\python.exe'
$env:PYTHONUTF8 = '1'
$env:MODAL_PROFILE = 'wj-lee-louis'
Remove-Item Env:MODAL_TOKEN_ID,Env:MODAL_TOKEN_SECRET -ErrorAction SilentlyContinue
& $modalPython scripts/benchmark_metric_remote.py start --account A
if ($LASTEXITCODE -ne 0) { throw 'Account A evaluation submission failed; inspect status before retrying' }
```

명령은 다음 순서로 진행한다.

1. A workspace의 **평가 전용 Volume**에 입력을 업로드한다. `CPU UPLOAD` 동안 기다린다.
2. 원격 CPU에서 archive hash와 2,700개 입력을 재검증한다. 성공하면 `INPUTS_VALIDATED 2700`이 나온다.
3. T4에서 animal/human/object/style 각 1 case의 두 조건, 총 **8장 pilot**을 실행한다. `PILOT_COMPLETE`까지 기다린다. 첫 실행은 image build와 모델 다운로드로 시간이 걸린다.
4. pilot이 통과하면 본 평가를 **자동으로 detached 제출**한다. pilot 점수 8개는 재사용한다.

**`SUBMITTED A`와 `DETACHED`가 나온 뒤** A 작업은 로컬 터미널 없이 계속된다. 업로드·CPU 검증·pilot 대기 중에는 노트북을 종료하지 않는다.

## 3. 터미널 B — DreamBench 1,500장

1절이 끝났으면 터미널 A와 병행해 실행할 수 있다.

```powershell
Set-Location 'C:\Users\82107\Downloads\AI_PROJECT\Orthogonalization-InstantX-IP-Adapter'
$modalPython = 'C:\Users\82107\.venvs\orthogonalization-modal311\Scripts\python.exe'
$env:PYTHONUTF8 = '1'
$env:MODAL_PROFILE = 'modal-research-2'
Remove-Item Env:MODAL_TOKEN_ID,Env:MODAL_TOKEN_SECRET -ErrorAction SilentlyContinue
& $modalPython scripts/benchmark_metric_remote.py start --account B
if ($LASTEXITCODE -ne 0) { throw 'Account B evaluation submission failed; inspect status before retrying' }
```

B는 `INPUTS_VALIDATED 1500` 이후 animal/object 각 1 case의 두 조건, 총 **4장 pilot**을 실행한다. 통과하면 나머지를 자동 제출한다. B profile의 실제 workspace 이름은 `woojin716`이다.

**양쪽 모두 `SUBMITTED`와 `DETACHED`가 확인되면 VSCode와 노트북을 종료해도 된다.** 먼저 끝난 계정의 GPU 함수는 종료되며 최소 상시 worker 수는 0이다.

## 4. 진행 상태 조회 — 언제든 새 PowerShell에서

```powershell
Set-Location 'C:\Users\82107\Downloads\AI_PROJECT\Orthogonalization-InstantX-IP-Adapter'
$modalPython = 'C:\Users\82107\.venvs\orthogonalization-modal311\Scripts\python.exe'
$env:PYTHONUTF8 = '1'
Remove-Item Env:MODAL_TOKEN_ID,Env:MODAL_TOKEN_SECRET -ErrorAction SilentlyContinue
$env:MODAL_PROFILE = 'wj-lee-louis'
& $modalPython scripts/benchmark_metric_remote.py status --account A
if ($LASTEXITCODE -ne 0) { throw 'Account A status lookup failed' }
$env:MODAL_PROFILE = 'modal-research-2'
& $modalPython scripts/benchmark_metric_remote.py status --account B
if ($LASTEXITCODE -ne 0) { throw 'Account B status lookup failed' }
```

`state.status`는 `pilot_running` → `pilot_completed` → `running` → `completed` 순서다. `state.scored`는 **원격 Volume에 저장한 case 점수 수**다. A는 2,700, B는 1,500이 최종 수다. receipt의 `phase=full_submitted`가 완료 후 남아 있어도 원격 `state.status=completed`와 count를 기준으로 본다. 최신 call 상태와 오류 traceback도 함께 출력한다.

처음에는 `STAGING_INPUTS`에서 원격 ZIP을 worker의 임시 디스크로 한 번 복사·해제한다. `INPUTS_STAGED` 이후 입력 검사와 모델 계산은 worker의 로컬 파일을 사용하므로 원격 Volume의 수천 개 파일을 하나씩 읽는 지연을 피한다. 모델 다운로드나 첫 batch 계산 중에는 count 갱신이 늦을 수 있다. 본 평가는 batch size 32이며 batch마다 점수와 상태를 원격 Volume에 commit한다. GPU 오류가 발생하면 실패 상태를 저장하고 즉시 함수를 종료하며, 무한 재시도하지 않는다.

## 5. 중단·오류 후 재개

2026-10-21의 `cannot mount volume on non-empty path: "/cache"` 오류는 평가 이미지 빌드 cache와 모델 Volume의 경로 충돌이었다. 수정본은 설치가 끝난 뒤 모델 경로를 `/metric_models`로 지정하고, 빌드 때 해당 경로가 비어 있는지 검사한다. A/B의 기존 업로드 입력은 보존해 복구했다. 이 오류로 새로 다운로드하거나 대용량 ZIP을 다시 업로드할 필요는 없다.

복구 후 실제 T4에서 A 8장·B 4장 pilot이 모두 통과했고 본 평가도 detached 재개했다. 첫 본 평가 checkpoint에서 A 296건·B 324건의 세 지표가 유한값이고 reference hash·원래 target prompt가 계획과 일치하는 것을 확인했다. 이는 복구 확인 당시 snapshot이며 최신 진행 수는 4절로 조회한다. 복구·중간 저장 관련 로컬 테스트는 12개 통과했다.

해당 계정의 2절 또는 3절 **동일한 `start` 명령**을 다시 실행한다.

- 기존 call이 실행 중이면 `ALREADY_ACTIVE`를 출력하고 중복 제출하지 않는다.
- 이전 call이 끝났고 일부 점수가 저장되어 있으면 완료 case를 건너뛴다.
- pilot 대기 중 로컬이 종료되어도 원격 pilot은 계속된다. pilot 완료 후 `start`를 다시 실행하면 본 평가로 넘어간다.
- 이미 평가가 완료됐으면 `EVALUATION_COMPLETE`를 출력하며 새 GPU 작업을 만들지 않는다.
- profile·평가 코드·전처리·GPU/패키지 환경·입력 해시가 다르면 기존 점수를 섞지 않고 실패한다. 실행 중 평가 관련 Python 파일을 수정하지 않는다.

`Uncertain interrupted submission`은 spawn 직후 receipt 저장이 끊겨 call ID를 확인할 수 없는 경우다. 출력된 app ID를 Modal에서 확인한 뒤 조치해야 하며, receipt를 지우고 새로 제출하지 않는다. detached 제출 뒤 Ctrl+C는 원격 평가를 중단하지 않는다. 중단하려면 해당 Modal app을 명시적으로 stop한다.

## 6. 두 계정 완료 후 점수 다운로드 및 CPU 집계

4절에서 양쪽 `state.status=completed`와 A=2,700/B=1,500을 확인한 다음 실행한다. 이미지 전체를 다시 다운로드하지 않고 **점수 JSON만** 가져온다.

```powershell
Set-Location 'C:\Users\82107\Downloads\AI_PROJECT\Orthogonalization-InstantX-IP-Adapter'
$modalPython = 'C:\Users\82107\.venvs\orthogonalization-modal311\Scripts\python.exe'
$env:PYTHONUTF8 = '1'
Remove-Item Env:MODAL_TOKEN_ID,Env:MODAL_TOKEN_SECRET -ErrorAction SilentlyContinue
$env:MODAL_PROFILE = 'wj-lee-louis'
& $modalPython scripts/benchmark_metric_remote.py download --account A
if ($LASTEXITCODE -ne 0) { throw 'Account A score download failed' }
$env:MODAL_PROFILE = 'modal-research-2'
& $modalPython scripts/benchmark_metric_remote.py download --account B
if ($LASTEXITCODE -ne 0) { throw 'Account B score download failed' }
& $modalPython scripts/benchmark_metrics.py merge --parts outputs/benchmark_metrics_v1/scores_A.json outputs/benchmark_metrics_v1/scores_B.json --scores outputs/benchmark_metrics_v1/summary.json
if ($LASTEXITCODE -ne 0) { throw 'Score merge failed; inspect missing/duplicate/environment mismatch' }
```

최종 결과는 `outputs/benchmark_metrics_v1/summary.json`이다. 기존 결과 파일이 있으면 덮어쓰지 않으므로 재집계 시 `--scores`에 새 경로를 준다. 각 case의 reference hash·원래 target prompt·category·점수, benchmark/condition별 전체 및 class별 점수, `method_minus_baseline` 차이를 보존한다. 누락·중복·다른 환경의 점수가 있으면 최종 집계를 거부한다.

## 7. 고정 평가 정의와 class 집계

| 지표 | 정의 |
|---|---|
| CLIP-T | OpenAI CLIP ViT-B/32; 생성 PNG와 원래 target_prompt의 cosine; max(0, score), weight=1, prefix 없음, truncate=True |
| CLIP-I | 동일 CLIP; 생성 PNG와 case manifest의 image_path **한 장** 비교 |
| DINO-I | DINO v1 dino_vits16 (ViT-S/16); 동일 지정 reference **한 장** 비교 |

CLIP은 짧은 변 224 bicubic resize → 224 center crop → CLIP 정규화, DINO는 짧은 변 256 bicubic resize → 224 center crop → ImageNet 정규화다. RGB 원본 PNG를 사용하며 JPEG 변환·segmentation을 적용하지 않는다. 점수는 ×100 하지 않는다. CLIP-I/DINO-I는 음수 clipping을 하지 않는다. 모델 코드 commit과 라이브러리 버전은 고정·기록한다. 양쪽 모두 T4/CUDA, batch size 32를 사용한다.

UNO 공개 evaluator와 같은 백본·전처리를 쓰되, 공개 코드의 폴더 전체 reference 평균을 **사용자 지정 한 장 비교로 변경**한 `uno-selected-reference-v1`이다. 공개 all-reference 결과·논문 표와 동일한 평가라고 단정하지 않는다. DreamBench++도 이 공통 설정으로 평가하며 공식 ++ DINO-v1-S/8과 구분한다. [UNO 코드](https://github.com/bytedance/UNO/blob/a563432dcfb4caf98df4368facb8102bf97d376c/eval/evaluate_clip_dino_score_single_subject.py)

| benchmark | class | reference 수 | 조건당 이미지 | 두 조건 합계 |
|---|---|---:|---:|---:|
| DreamBench | animal | 9 | 225 | 450 |
| DreamBench | object | 21 | 525 | 1,050 |
| DreamBench++ | animal | 45 | 405 | 810 |
| DreamBench++ | human | 20 | 180 | 360 |
| DreamBench++ | object | 65 | 585 | 1,170 |
| DreamBench++ | style | 20 | 180 | 360 |

class는 생성 prompt에 등장하는 단어가 아닌 **reference의 분류**를 따른다. DreamBench의 animal/object는 명시적 보고용 분류이며 plushie·toy·backpack_dog는 object다. DreamBench++는 reference collection category를 따른다. 없는 class는 N/A이며 0점으로 채우지 않는다.

`subject_macro`는 reference별 prompt 평균 후 reference 평균, `case_micro`는 이미지별 평균이다. `by_class`에 class별 같은 집계를 저장한다. `class_macro`는 존재하는 class를 동일 가중치로 평균한 별도 값이며 전체 평균 대신 사용하지 않는다. DreamBench++의 `non_style` 1,170건/조건도 별도 출력한다. 두 benchmark나 두 계정의 평균을 단순 평균하지 않는다. class 재집계에는 GPU가 필요 없다.

평가 전용 Volume은 `orthogonalization-dreambench-metrics-v1`, 모델 cache Volume은 이름 뒤 `-models`다. 생성용 Volume과 분리되어 있으며 입력·checkpoint·최종 점수는 원격에 보존된다. batch 저장의 내구성은 [Modal Volume commit](https://modal.com/docs/guide/volumes), 원격 분리 실행은 [Modal App.run detach](https://modal.com/docs/reference/modal.App#run)를 사용한다. CP/PF는 이번 실행 대상에 포함하지 않는다.
