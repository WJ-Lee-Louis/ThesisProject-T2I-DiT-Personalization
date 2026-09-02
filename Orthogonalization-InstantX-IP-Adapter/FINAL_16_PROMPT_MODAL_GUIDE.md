# Final 16-Prompt Modal Inference Guide

> Historical document only. The active catalog now has 14 prompts after
> removing `smiling` and `crying`, and the legacy condition names below are no
> longer accepted for new submissions. Use `CLASS_PROJECTION_MODAL_GUIDE.md`.

이 문서는 VS Code의 **CMD 터미널**에서 최종 실험 6개를 실행하는 절차다.
PowerShell 변수나 줄 연결 문법을 사용하지 않으며, 모든 명령은 한 줄씩 실행한다.

## 1. 최종 실험 구성

| 표시 이름 | 배치 condition | Projection | Gate | 이미지 수 |
|---|---|---|---|---:|
| Baseline | `baseline` | 없음 | Off | 160 |
| Gate only | `gate` | 없음 | On | 160 |
| Target + Gate | `combined` | Target, lambda 1 | On | 160 |
| Source + Gate | `sa_combined` | Source matched, lambda 1 | On | 160 |
| Matched Union + Gate | `st_combined` | Target + matched source, single SVD, lambda 1 | On | 160 |
| All-Source Union + Gate | `st_all_combined` | Target + source A/B, single SVD, lambda 1 | On | 100 |

전체 GPU 생성량은 900장이다. `st_all_combined`에서는 matched union과 수학적으로
같은 6개 `O+A+B` prompt를 다시 생성하지 않아 60장을 절약한다. 이 조건은
`O+A` 8개와 `O+B` 2개만 생성한다. `jumping over a wooden fence`는
`jumping` + `over a wooden fence`의 `O+A+B`로 분해한다.

모든 조건은 seed, step, guidance scale, IP scale, 해상도를 동일하게 유지한다.
Projection 조건은 모두 `--orthogonalization-strength 1.0`을 사용한다.
Baseline과 gate-only 명령에도 설정 비교를 위해 같은 값이 기록되지만,
projection이 비활성이라 실제 embedding에는 적용되지 않는다.

## 2. CMD 터미널 위치 이동

VS Code에서 `Terminal > New Terminal`을 열고 터미널 종류가 CMD인지 확인한 뒤
다음을 실행한다.

```bat
cd /d C:\Users\82107\Downloads\AI_PROJECT
```

Modal 로그인 profile을 확인한다.

```bat
python -m modal profile current
```

기존에 `huggingface-secret`을 등록했다면 다시 만들 필요가 없다. 이름만 확인하려면:

```bat
python -m modal secret list
```

## 3. GPU를 사용하지 않는 사전검증

다음 명령은 reference 10장, prompt 16개, 총 160개 manifest와 source caption
10개를 검증한다. GPU 비용은 발생하지 않는다.

```bat
python Orthogonalization-InstantX-IP-Adapter\batch_client.py preview --reference-root dreambooth_reference_dataset\InstantX-IP-Adapter-DreamBench_filtered --source-captions Orthogonalization-InstantX-IP-Adapter\source_captions.json
```

정상 출력에서 다음 값이 보여야 한다.

- references: 10
- prompts/reference: 16
- total jobs: 160
- `st_all_combined optimized jobs`: 100
- source captions: 10

## 4. Modal worker 재배포

prompt catalog와 worker 구현 digest가 바뀌었으므로 반드시 한 번 재배포한다.

```bat
python -m modal deploy Orthogonalization-InstantX-IP-Adapter\batch_modal.py
```

현재 worker 설정은 다음과 같다.

- GPU: A100-80GB
- 동시 GPU container 상한: 3
- 한 container 내부에서는 inference를 동시에 실행하지 않음
- idle container는 최대 300초 동안 유지되어 다음 queued batch에서 model을 재사용할 수 있음

따라서 아래 6개 명령을 연속 제출하면 최대 3개 GPU가 병렬 실행되고, 남은
호출은 queue에서 기다린다. 로컬 PC나 인터넷 연결이 종료되어도 제출 완료된
Modal 작업은 계속 진행된다. tmux는 필요하지 않다.

## 5. 여섯 배치 제출

### 5.1 Baseline — 160장

```bat
python Orthogonalization-InstantX-IP-Adapter\batch_client.py submit --condition baseline --reference-root dreambooth_reference_dataset\InstantX-IP-Adapter-DreamBench_filtered --batch-id dreambench-v2-baseline-0825 --orthogonalization-strength 1.0 --ip-adapter-scale 0.7 --guidance-scale 3.5 --steps 24 --seed 42 --width 960 --height 1280 --yes
```

### 5.2 Gate only — 160장

```bat
python Orthogonalization-InstantX-IP-Adapter\batch_client.py submit --condition gate --reference-root dreambooth_reference_dataset\InstantX-IP-Adapter-DreamBench_filtered --batch-id dreambench-v2-gate-0825 --orthogonalization-strength 1.0 --ip-adapter-scale 0.7 --guidance-scale 3.5 --steps 24 --seed 42 --width 960 --height 1280 --yes
```

### 5.3 Target projection + Gate — 160장

```bat
python Orthogonalization-InstantX-IP-Adapter\batch_client.py submit --condition combined --reference-root dreambooth_reference_dataset\InstantX-IP-Adapter-DreamBench_filtered --batch-id dreambench-v2-target-gate-l1-0825 --orthogonalization-strength 1.0 --ip-adapter-scale 0.7 --guidance-scale 3.5 --steps 24 --seed 42 --width 960 --height 1280 --yes
```

### 5.4 Source-aware projection + Gate — 160장

```bat
python Orthogonalization-InstantX-IP-Adapter\batch_client.py submit --condition sa_combined --reference-root dreambooth_reference_dataset\InstantX-IP-Adapter-DreamBench_filtered --source-captions Orthogonalization-InstantX-IP-Adapter\source_captions.json --batch-id dreambench-v2-source-gate-l1-0825 --orthogonalization-strength 1.0 --ip-adapter-scale 0.7 --guidance-scale 3.5 --steps 24 --seed 42 --width 960 --height 1280 --yes
```

### 5.5 Target + matched source union SVD + Gate — 160장

```bat
python Orthogonalization-InstantX-IP-Adapter\batch_client.py submit --condition st_combined --reference-root dreambooth_reference_dataset\InstantX-IP-Adapter-DreamBench_filtered --source-captions Orthogonalization-InstantX-IP-Adapter\source_captions.json --batch-id dreambench-v2-matched-union-gate-l1-0825 --orthogonalization-strength 1.0 --ip-adapter-scale 0.7 --guidance-scale 3.5 --steps 24 --seed 42 --width 960 --height 1280 --yes
```

### 5.6 Target + all-source union SVD + Gate — 100장

```bat
python Orthogonalization-InstantX-IP-Adapter\batch_client.py submit --condition st_all_combined --reference-root dreambooth_reference_dataset\InstantX-IP-Adapter-DreamBench_filtered --source-captions Orthogonalization-InstantX-IP-Adapter\source_captions.json --batch-id dreambench-v2-all-source-union-gate-l1-0825 --orthogonalization-strength 1.0 --ip-adapter-scale 0.7 --guidance-scale 3.5 --steps 24 --seed 42 --width 960 --height 1280 --yes
```

각 submit 명령이 `Function Call ID`와 Dashboard URL을 출력한 뒤 다음 명령을
실행한다. 여섯 명령을 모두 제출하는 동안 같은 batch ID를 반복 사용하지 않는다.

## 6. 상태 확인

여섯 batch 상태를 한 번씩 확인하는 CMD 명령이다.

```bat
for %B in (dreambench-v2-baseline-0825 dreambench-v2-gate-0825 dreambench-v2-target-gate-l1-0825 dreambench-v2-source-gate-l1-0825 dreambench-v2-matched-union-gate-l1-0825 dreambench-v2-all-source-union-gate-l1-0825) do @python Orthogonalization-InstantX-IP-Adapter\batch_client.py status --batch-id %B
```

전체 app 로그를 실시간으로 보고 싶다면:

```bat
python -m modal app logs orthogonalization-instantx-flux-batch -f --timestamps --show-function-call-id --show-container-id
```

로그 보기를 종료하는 `Ctrl+C`는 로컬 로그 스트리밍만 끝내며 원격 batch는
중단하지 않는다.

완료 기준은 각 status의 `state`가 `completed`이고 다음 count가 일치하는 것이다.

- 앞의 다섯 batch: `completed_count = 160`
- `st_all_combined`: `completed_count = 100`

## 7. 결과 다운로드

완료 후 여섯 batch를 차례로 다운로드한다.

```bat
for %B in (dreambench-v2-baseline-0825 dreambench-v2-gate-0825 dreambench-v2-target-gate-l1-0825 dreambench-v2-source-gate-l1-0825 dreambench-v2-matched-union-gate-l1-0825 dreambench-v2-all-source-union-gate-l1-0825) do @python Orthogonalization-InstantX-IP-Adapter\batch_client.py download --batch-id %B
```

다운로드 위치는 다음과 같다.

```text
C:\Users\82107\Downloads\AI_PROJECT\Orthogonalization-InstantX-IP-Adapter\outputs\<batch-id>
```

Downloader는 원격 `status.total_count`를 읽어 조건별로 160장 또는 100장을
자동 검증한다. 정상 완료 전에는 `--allow-partial`을 사용하지 않는다. 같은 파일이
이미 존재하는 경우 원격 완료 상태를 확인한 뒤에만 `--force`를 사용한다.

## 8. 재개와 충돌 방지

- 실패한 batch를 재개할 때 같은 설정과 같은 batch ID로 submit 명령을 다시 실행한다.
- 정상적으로 저장된 PNG와 JSON sidecar는 건너뛰며 나머지만 생성한다.
- 재개 시 `--overwrite`를 추가하지 않는다.
- prompt, caption, seed 또는 모델 설정을 바꾸려면 반드시 새 batch ID를 사용한다.
- 이번 v2 manifest는 기존 9-prompt 실험과 digest가 다르므로 기존 batch ID를
  재사용하면 안 된다.

## 9. 저장되는 분석 metadata

각 PNG의 `_metadata/images/...json` sidecar에는 공통 conditioning 정보가
저장되며, projection 조건의 sidecar에는 다음 basis/SVD 값도 저장된다.
Baseline과 gate-only는 projection이 꺼져 있으므로
`orthogonalization_result.enabled=false`만 기록되고 SVD 값은 생성되지 않는다.

- prompt category와 behavior/background 분해
- target factors와 source factors
- matched/all source policy
- textual direction 이름과 정규화 전 norm
- direction cosine matrix
- SVD singular values와 relative singular values
- retained/dropped rank와 SVD threshold
- unit visual embedding의 projection/removed component norm
- edit norm, input cosine, degeneracy 여부
- gate 상태와 실제 FLUX text sequence 길이

Batch 단위 metadata에는 다음 파일이 추가된다.

- `_metadata/manifest.json`: 16개 PromptSpec과 160-job 전체 manifest
- `_metadata/job_selection.json`: 실제 생성 prompt와 중복 생략 내역
- `_metadata/execution.json`: 요청/실제 GPU와 container 설정
- `_metadata/status.json`: 조건별 동적 total/completed count

이 값들은 작은 벡터·행렬에서 이미 계산된 결과를 저장하는 것이므로 diffusion
추론 횟수나 SigLIP encoding 횟수를 추가하지 않는다.

## 10. 다운로드 후 진단 요약과 비교 그리드

여섯 batch의 sidecar를 한 번에 검증하고 CSV/JSON으로 요약하려면 다음 한 줄을
실행한다. 모델과 GPU는 사용하지 않는다.

```bat
python Orthogonalization-InstantX-IP-Adapter\summarize_basis_diagnostics.py Orthogonalization-InstantX-IP-Adapter\outputs\dreambench-v2-baseline-0825 Orthogonalization-InstantX-IP-Adapter\outputs\dreambench-v2-gate-0825 Orthogonalization-InstantX-IP-Adapter\outputs\dreambench-v2-target-gate-l1-0825 Orthogonalization-InstantX-IP-Adapter\outputs\dreambench-v2-source-gate-l1-0825 Orthogonalization-InstantX-IP-Adapter\outputs\dreambench-v2-matched-union-gate-l1-0825 Orthogonalization-InstantX-IP-Adapter\outputs\dreambench-v2-all-source-union-gate-l1-0825 --output-dir Orthogonalization-InstantX-IP-Adapter\outputs\basis_analysis_v2
```

최종 6행 × 16열 비교 그리드는 기존 결과와 충돌하지 않는 새 폴더에 생성한다.
`st_all_combined`에서 생략한 6개 `O+A+B` 셀은 검증된
`st_combined` 결과를 자동으로 재사용한다.

```bat
python Orthogonalization-InstantX-IP-Adapter\create_comparison_grids.py --destination Orthogonalization-InstantX-IP-Adapter\outputs\comparison_grids_v2
```
