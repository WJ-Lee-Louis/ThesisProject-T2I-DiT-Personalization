# XVerseBench 평가 이미지 생성용 Modal 가이드

이 문서는 VS Code의 **CMD 터미널**을 기준으로 한다. PowerShell 문법인 `$변수`는 사용하지 않는다.

## 1. 이번 생성 범위

평가 condition은 다음 여섯 가지로 고정한다.

1. `baseline`
2. `ssg`
3. `target_projection_ssg`
4. `source_projection_ssg`
5. `st_matched_projection_ssg`
6. `st_all_projection_ssg`

공통 생성 설정은 다음과 같다.

- Resolution: 1024 x 1024
- Denoising steps: 24
- IP-Adapter scale: 0.7
- Guidance scale: 3.5
- Projection lambda: 1.0
- `baseline`을 제외한 모든 condition: SSG 활성화

이미지 수는 다음과 같다.

- `subset`: 29 reference-prompt records x 4 seeds = 116장/condition, 총 696장
- `custom`: 15 references x 6 prompts x 1 deterministic seed = 90장/condition, 총 540장
- 전체: 1,236장

Subset은 각 record에 seed 42, 43, 44, 45를 적용한다. Custom은 record index에 따라 seed 42--131을 한 개씩 사용한다. 같은 record는 모든 condition에서 동일한 seed를 공유한다.

공식 subset prompt는 원문을 그대로 생성에 사용한다. Projection 계산에만 `subset_bench/target_factor_annotations.json`의 고정된 action/background/attribute 분해를 사용한다. `O+attribute` 5개 record에서는 공식 prompt와 core object prompt의 SigLIP 차이인 `e(O+attribute) - e(O)`를 독립 attribute direction으로 사용한다. 대응되는 source attribute가 없으므로 이 5개에서는 `st_matched_projection_ssg`가 `target_projection_ssg`와 동일하고, `st_all_projection_ssg`는 target attribute direction과 source caption의 behavior/background direction을 union-SVD로 함께 처리한다. `source_projection_ssg`는 이 5개에서 제거할 matched source factor가 없어 SSG-only와 같다.

## 2. 생성과 metric 평가는 분리한다

다음 순서를 따른다.

1. Modal에서 이미지 생성
2. 생성 완료 상태 확인
3. 모든 결과 다운로드
4. condition manifest 결합
5. 이미지 수와 seed 완전성 검증
6. 그 후 DPG, IP-Sim, AES 평가

Metric을 이미지 생성과 동시에 실행할 필요가 없다. 완성되지 않은 manifest를 평가하면 condition별 누락으로 paired comparison이 깨질 수 있으므로 이미지 생성·다운로드·검증을 먼저 끝내는 것이 안전하다.

## 3. Modal 계정/profile 확인

프로젝트 폴더로 이동한다.

```cmd
cd /d C:\Users\82107\Downloads\AI_PROJECT
```

등록된 profile과 현재 profile을 확인한다.

```cmd
modal profile list
modal profile current
modal token info
modal config show
```

새 Modal 계정이 아직 profile로 등록되지 않았다면 브라우저에서 사용할 계정으로 로그인한 뒤 다음을 실행한다.

```cmd
modal token new
modal profile list
```

원하는 profile로 전환한다.

```cmd
modal profile activate 새_PROFILE_이름
modal config set-environment main
modal profile current
modal token info
```

CMD 환경변수에 별도 token이 남아 있으면 active profile보다 우선할 수 있으므로 현재 터미널에서 제거한다.

```cmd
set MODAL_TOKEN_ID=
set MODAL_TOKEN_SECRET=
```

그 후 `modal profile current`와 `modal token info`를 다시 확인한다.

Modal Workspace와 Environment마다 App, Volume, Secret은 분리된다. 새 계정/profile에서는 `huggingface-secret`과 evaluation App을 다시 준비해야 한다. 기존 계정의 profile이나 리소스를 삭제할 필요는 없다.

## 4. Hugging Face secret과 App 배포

현재 profile의 secret을 확인한다.

```cmd
modal secret list
```

`huggingface-secret`이 없다면 FLUX.1-dev 접근 권한이 있는 Hugging Face read token을 사용해 생성한다.

```cmd
modal secret create huggingface-secret HF_TOKEN=hf_여기에_토큰
```

동일 이름이 있지만 token을 교체해야 한다면 다음을 사용한다.

```cmd
modal secret create huggingface-secret HF_TOKEN=hf_여기에_토큰 --force
```

토큰 문자열을 화면이나 문서에 저장하지 않는다. 필요하면 Modal Dashboard의 Secrets 화면에서 직접 입력해도 된다.

평가 전용 App을 현재 profile에 배포한다.

```cmd
modal deploy Orthogonalization-InstantX-IP-Adapter\eval_batch_modal.py
```

CPU access check로 세 repository 권한을 확인한다.

```cmd
python Orthogonalization-InstantX-IP-Adapter\eval_batch_client.py check-access
```

결과에 `"access": "OK"`가 표시되어야 한다. 새 계정은 자체 HF cache Volume을 사용하므로 첫 GPU 실행에서는 모델 다운로드와 cold start가 다시 발생할 수 있다.

## 5. 로컬 manifest 사전 검증

두 benchmark를 모두 확인한다. 이 단계는 GPU를 사용하지 않는다.

```cmd
python Orthogonalization-InstantX-IP-Adapter\eval_batch_client.py preview --benchmark subset
python Orthogonalization-InstantX-IP-Adapter\eval_batch_client.py preview --benchmark custom
```

다음 값이 나와야 한다.

- subset: references 29, images per condition 116
- custom: references 15, images per condition 90
- source captions: 각각 29, 15 validated

## 6. Batch ID 접두사 설정

계정과 실행을 구별할 짧은 이름을 지정한다. 예시는 두 번째 Modal 계정의 9월 8일 실행이다.

```cmd
set RUN=m2-0908
```

다른 계정에서 다시 실행할 때는 `m1-0908`, `m3-0908`처럼 다른 값을 사용한다.

## 7. Stage 1: baseline과 SSG 제출

Subset 두 condition을 제출한다.

```cmd
for %C in (baseline ssg) do @python Orthogonalization-InstantX-IP-Adapter\eval_batch_client.py submit --benchmark subset --condition %C --batch-id xverse-%RUN%-subset-%C --orthogonalization-strength 1.0 --yes
```

Custom 두 condition을 제출한다.

```cmd
for %C in (baseline ssg) do @python Orthogonalization-InstantX-IP-Adapter\eval_batch_client.py submit --benchmark custom --condition %C --batch-id xverse-%RUN%-custom-%C --orthogonalization-strength 1.0 --yes
```

`max_containers=3`이므로 네 호출 중 일부가 pending이어도 정상이다. 제출이 끝나면 VS Code나 PC를 종료해도 원격 작업은 계속된다.

## 8. Stage 2: projection condition 제출

Stage 1이 정상적으로 시작된 것을 확인한 후 나머지 네 condition을 제출한다.

Subset:

```cmd
for %C in (target_projection_ssg source_projection_ssg st_matched_projection_ssg st_all_projection_ssg) do @python Orthogonalization-InstantX-IP-Adapter\eval_batch_client.py submit --benchmark subset --condition %C --batch-id xverse-%RUN%-subset-%C --orthogonalization-strength 1.0 --yes
```

Custom:

```cmd
for %C in (target_projection_ssg source_projection_ssg st_matched_projection_ssg st_all_projection_ssg) do @python Orthogonalization-InstantX-IP-Adapter\eval_batch_client.py submit --benchmark custom --condition %C --batch-id xverse-%RUN%-custom-%C --orthogonalization-strength 1.0 --yes
```

한 번에 모두 제출하면 최대 세 container가 실행되고 나머지는 pending으로 대기한다. 비용을 더 세밀하게 통제하려면 subset 네 개가 끝난 뒤 custom 네 개를 제출해도 결과에는 차이가 없다.

## 9. 진행 상태 확인

Subset 전체:

```cmd
for %C in (baseline ssg target_projection_ssg source_projection_ssg st_matched_projection_ssg st_all_projection_ssg) do @python Orthogonalization-InstantX-IP-Adapter\eval_batch_client.py status --batch-id xverse-%RUN%-subset-%C
```

Custom 전체:

```cmd
for %C in (baseline ssg target_projection_ssg source_projection_ssg st_matched_projection_ssg st_all_projection_ssg) do @python Orthogonalization-InstantX-IP-Adapter\eval_batch_client.py status --batch-id xverse-%RUN%-custom-%C
```

각 batch의 `state`가 `completed`이고 다음 수치가 일치해야 한다.

- subset: `completed_count=116`
- custom: `completed_count=90`

## 10. 다운로드

작업을 제출했던 profile이 active인지 먼저 확인한다.

```cmd
modal profile current
```

Subset:

```cmd
for %C in (baseline ssg target_projection_ssg source_projection_ssg st_matched_projection_ssg st_all_projection_ssg) do @python Orthogonalization-InstantX-IP-Adapter\eval_batch_client.py download --batch-id xverse-%RUN%-subset-%C
```

Custom:

```cmd
for %C in (baseline ssg target_projection_ssg source_projection_ssg st_matched_projection_ssg st_all_projection_ssg) do @python Orthogonalization-InstantX-IP-Adapter\eval_batch_client.py download --batch-id xverse-%RUN%-custom-%C
```

다운로드가 끊기면 같은 명령을 다시 실행한다. 기존 PNG는 재사용하고 최신 status와 manifest는 다시 받으며, 각 파일 다운로드는 기본 5회 재시도한다. Windows의 긴 metadata 경로에서도 동작하도록 임시파일명은 짧게 생성된다. 손상 가능성이 있어 전부 다시 받아야 할 때만 `--force`를 추가한다.

로컬 저장 구조는 다음과 같다.

```text
datasets/XVerseBench_0907/evaluation_outputs/
└─ <Modal-profile>/
   ├─ <batch-id>/
   │  ├─ images/<record-id>/seed_0042.png
   │  ├─ generated_images_manifest.json
   │  └─ _metadata/
   ├─ subset/generated_images_manifest.json
   └─ custom/generated_images_manifest.json
```

Windows 경로 길이를 줄이기 위해 개별 batch 폴더에는 benchmark/condition 디렉터리를 중복하지 않는다. 해당 정보는 batch metadata와 결합 manifest에 기록된다. Modal profile별 상위 폴더를 분리했기 때문에 계정을 바꾸어도 로컬 결과와 call record가 충돌하지 않는다.

## 11. 여섯 condition manifest 결합

Subset:

```cmd
python Orthogonalization-InstantX-IP-Adapter\eval_batch_client.py assemble --benchmark subset --batch-id xverse-%RUN%-subset-baseline --batch-id xverse-%RUN%-subset-ssg --batch-id xverse-%RUN%-subset-target_projection_ssg --batch-id xverse-%RUN%-subset-source_projection_ssg --batch-id xverse-%RUN%-subset-st_matched_projection_ssg --batch-id xverse-%RUN%-subset-st_all_projection_ssg
```

Custom:

```cmd
python Orthogonalization-InstantX-IP-Adapter\eval_batch_client.py assemble --benchmark custom --batch-id xverse-%RUN%-custom-baseline --batch-id xverse-%RUN%-custom-ssg --batch-id xverse-%RUN%-custom-target_projection_ssg --batch-id xverse-%RUN%-custom-source_projection_ssg --batch-id xverse-%RUN%-custom-st_matched_projection_ssg --batch-id xverse-%RUN%-custom-st_all_projection_ssg
```

현재 profile 이름을 CMD 변수로 얻는다.

```cmd
for /f "delims=" %P in ('modal profile current') do @set MPROFILE=%P
```

결합 결과는 다음 위치에 생성된다.

```text
Orthogonalization-InstantX-IP-Adapter\datasets\XVerseBench_0907\evaluation_outputs\%MPROFILE%\subset\generated_images_manifest.json
Orthogonalization-InstantX-IP-Adapter\datasets\XVerseBench_0907\evaluation_outputs\%MPROFILE%\custom\generated_images_manifest.json
```

## 12. 최종 완전성 검사

Subset:

```cmd
python Orthogonalization-InstantX-IP-Adapter\datasets\XVerseBench_0907\metrics\validate_generated_manifest.py --benchmark-root Orthogonalization-InstantX-IP-Adapter\datasets\XVerseBench_0907 --benchmark subset --generated-manifest Orthogonalization-InstantX-IP-Adapter\datasets\XVerseBench_0907\evaluation_outputs\%MPROFILE%\subset\generated_images_manifest.json
```

Custom:

```cmd
python Orthogonalization-InstantX-IP-Adapter\datasets\XVerseBench_0907\metrics\validate_generated_manifest.py --benchmark-root Orthogonalization-InstantX-IP-Adapter\datasets\XVerseBench_0907 --benchmark custom --generated-manifest Orthogonalization-InstantX-IP-Adapter\datasets\XVerseBench_0907\evaluation_outputs\%MPROFILE%\custom\generated_images_manifest.json
```

기대 결과는 다음과 같다.

- subset: 6 conditions, 696 images
- custom: 6 conditions, 540 images

이 검사가 통과한 후 DPG, IP-Sim, AES를 실행한다.

## 13. 원래 Modal 계정으로 돌아갈 때

실행 중인 원격 job은 profile을 바꿔도 중단되지 않는다. 원래 계정으로 돌아갈 때는 다음처럼 profile만 전환한다.

```cmd
modal profile list
modal profile activate 원래_PROFILE_이름
modal config set-environment main
modal profile current
modal token info
```

해당 계정에 evaluation App을 아직 배포하지 않았거나 코드가 변경되었다면 다시 배포한다.

```cmd
modal deploy Orthogonalization-InstantX-IP-Adapter\eval_batch_modal.py
```

각 계정의 App, Secret, HF cache, 입력 Volume, 결과 Volume은 독립적이다. 다른 계정에서 만든 결과를 조회하거나 다운로드하려면 반드시 그 작업을 제출한 profile을 다시 활성화해야 한다.

## 14. 공식 Modal 참고 문서

- Profiles: https://modal.com/docs/cli/latest/profile
- Tokens: https://modal.com/docs/cli/latest/token
- Workspaces: https://modal.com/docs/guide/workspaces
- Environments: https://modal.com/docs/guide/environments
- Secrets: https://modal.com/docs/cli/latest/secret

## 15. Metric 평가 구현 최종 점검 결과

2026-09-09 기준으로 로컬 결합 manifest와 생성 결과가 다음과 같이 검증되었다.

- subset: 6 conditions x 116 images = 696 images
- custom: 6 conditions x 90 images = 540 images
- Modal 생성 결과 Volume의 12개 batch도 모두 `completed`이며 각 batch의 PNG 수가 manifest와 일치한다.
- 총 1,236장의 생성 이미지는 다시 업로드하지 않고 기존 `orthogonalization-instantx-xverse-eval-results` Volume에서 읽는다.
- metric 결과는 별도 `orthogonalization-instantx-xverse-metric-results` Volume에 저장한다.
- DPG, IP-Sim, AES를 서로 다른 GPU job으로 분리한다. 각 job은 모델을 한 번만 로드하고 subset 다음 custom을 연속 평가한다.
- 최대 동시 container 수는 3개이고 기본 GPU는 A10G이다.
- 공식 XVerse 코드는 commit `65e2581aac0faffe4607881a10cc486186f6499d`로 고정했다.

Metric별 동작은 다음과 같다.

1. DPG
   - subset은 XVerse가 제공한 공식 DSG 질문과 dependency를 그대로 사용한다.
   - custom은 직접 설계한 O+A, O+B, O+A+B DSG를 사용하며 factor/component 점수도 저장한다.
   - 공식 subset DSG에는 `qid2factor`가 없으므로 전체 DPG만 공식 값으로 계산한다. custom에서만 action/background별 및 O+A+B joint 점수를 추가 계산한다.
2. IP-Sim
   - Florence-2 + SAM2로 reference와 생성 이미지의 foreground를 분리한다.
   - XVerse 공개 코드와 동일하게 `facebook/dino-vits16`의 class-token cosine similarity를 사용한다.
   - 29개 reference crop/mask는 한 번 생성한 뒤 subset과 custom에서 공동 재사용한다.
   - 생성 이미지의 후보 crop과 binary mask도 결과 Volume에 보존한다.
3. AES
   - `aesthetic-predictor-v2.5` SigLIP 모델로 생성 이미지 한 장씩 평가한다.
   - predictor weight URL과 SHA-256을 결과 JSON에 남긴다.
   - 2x2 grid 자체를 평가하지 않고 seed별 개별 이미지를 평가한 후 집계한다.

중요: subset의 DPG는 공식 prompt/DSG를 사용하는 재현 subset이지만, custom DPG와 seed별 AES 집계는 연구 목적에 맞춘 확장 평가다. 공식 XVerse 전체 leaderboard 점수와 동일한 명칭으로 혼합해 보고하지 않는다.

## 16. 평가에 사용할 Modal 계정 확인

아래 명령은 모두 VS Code의 **CMD 터미널**에서, `AI_PROJECT` 경로를 현재 위치로 둔 상태를 기준으로 한다.

```cmd
cd /d C:\Users\82107\Downloads\AI_PROJECT
modal profile current
modal token info
```

현재 생성 결과는 `modal-research-2` profile의 Volume에 있다. 출력이 다른 profile이면 먼저 전환한다.

```cmd
modal profile activate modal-research-2
modal config set-environment main
modal profile current
```

CMD 환경변수가 profile 인증을 덮어쓰지 않게 정리한다.

```cmd
set MODAL_TOKEN_ID=
set MODAL_TOKEN_SECRET=
```

다른 Modal 계정에서는 같은 이름의 Volume이라도 별개의 저장소이므로 현재 1,236장을 찾을 수 없다. 평가가 끝날 때까지 이미지를 생성한 `modal-research-2` profile을 사용해야 한다.

## 17. 평가 전용 Modal App 배포

이미지 생성용 app이 아니라 새 평가 app을 배포한다.

```cmd
modal deploy Orthogonalization-InstantX-IP-Adapter\metric_modal.py
```

최초 배포는 pinned XVerse repository와 metric 의존성을 포함하는 image를 빌드하므로 시간이 걸릴 수 있다. 이 명령은 metric GPU 추론을 시작하지 않는다.

이전 배포에서 `ModuleNotFoundError: orthogonalization_instantx_ip_adapter`가 발생한 call은 자동으로 정상 상태로 바뀌지 않는다. 패키지가 평가 image에 포함되도록 수정되어 있으므로 다시 배포한 뒤 새 `check-access` call을 실행한다. 과거 failed call이나 실패한 image layer는 따로 삭제할 필요가 없다.

배포 후 모델 접근성을 CPU 함수로 검사한다.

```cmd
python Orthogonalization-InstantX-IP-Adapter\metric_client.py check-access
```

마지막에 `"access": "OK"`가 출력되어야 한다. 기존 `huggingface-secret`이 현재 Modal 계정에도 존재해야 한다.

## 18. 제출 전 dry-run

이번 생성 batch의 공통 run 이름을 그대로 사용한다.

```cmd
set METRIC_RUN=m2-0908
python Orthogonalization-InstantX-IP-Adapter\metric_client.py preview --run-id %METRIC_RUN%
```

기대 출력의 핵심은 다음과 같다.

- active profile: `modal-research-2`
- subset: 116 images/condition, 6 conditions
- custom: 90 images/condition, 6 conditions
- 마지막 줄: `No GPU job was submitted.`

`preview`는 로컬 이미지와 결합 manifest를 검사한다. 실제 worker는 제출 후 원격 Volume의 batch status, condition, benchmark ID, 이미지 수도 다시 검사한다.

## 19. DPG, IP-Sim, AES 동시 제출

세 metric을 각각 한 번씩 제출한다.

```cmd
for %M in (dpg ip aes) do @python Orthogonalization-InstantX-IP-Adapter\metric_client.py submit --run-id %METRIC_RUN% --metric %M --yes
```

정상 제출되면 metric마다 Function Call ID와 Dashboard URL이 출력된다. 최대 세 container가 동시에 동작한다. 제출이 끝난 뒤 VS Code나 PC를 종료해도 원격 평가는 계속된다.

비용을 단계적으로 확인하고 싶다면 위 `for` 문 대신 아래 세 명령을 하나씩 실행해도 최종 결과는 같다.

```cmd
python Orthogonalization-InstantX-IP-Adapter\metric_client.py submit --run-id %METRIC_RUN% --metric aes --yes
python Orthogonalization-InstantX-IP-Adapter\metric_client.py submit --run-id %METRIC_RUN% --metric dpg --yes
python Orthogonalization-InstantX-IP-Adapter\metric_client.py submit --run-id %METRIC_RUN% --metric ip --yes
```

IP-Sim은 Florence-2/SAM2 검출과 artifact 저장을 1,236장에 수행하므로 세 작업 중 가장 오래 걸릴 가능성이 높다.

## 20. 진행 상태 확인과 안전한 재실행

```cmd
for %M in (dpg ip aes) do @python Orthogonalization-InstantX-IP-Adapter\metric_client.py status --run-id %METRIC_RUN% --metric %M
```

각 metric의 원격 status가 다음 상태가 되면 완료다.

```text
"state": "completed"
"completed_benchmarks": ["subset", "custom"]
```

일시적인 네트워크 문제로 로컬 status 조회가 실패해도 원격 job은 중단되지 않는다. 실제 metric job이 실패했다면 원인을 수정하고 동일한 `run-id`와 metric으로 다시 submit할 수 있다. 이미 완전하게 저장된 benchmark score는 재사용한다. 단, 같은 `run-id`에 서로 다른 코드 digest나 batch 구성을 섞는 것은 거부된다.

## 21. Metric 결과 다운로드

세 metric이 모두 완료된 뒤 전체 score와 IP foreground artifact를 받는다.

```cmd
python Orthogonalization-InstantX-IP-Adapter\metric_client.py download --run-id %METRIC_RUN%
```

네트워크가 끊기면 같은 명령을 다시 실행한다. 이미 받은 파일은 재사용하고 나머지만 받으며, 각 파일은 짧은 임시 파일명과 최대 5회 retry를 사용한다.

점수 JSON과 metadata만 먼저 받고 crop/mask는 나중에 받고 싶을 때만 다음 옵션을 사용한다.

```cmd
python Orthogonalization-InstantX-IP-Adapter\metric_client.py download --run-id %METRIC_RUN% --scores-only
```

전체 다운로드의 기본 저장 구조는 다음과 같다.

```text
datasets/XVerseBench_0907/metric_outputs/modal-research-2/m2-0908/
├─ subset/
│  ├─ dpg.json
│  ├─ ip.json
│  └─ aes.json
├─ custom/
│  ├─ dpg.json
│  ├─ ip.json
│  └─ aes.json
├─ r/
│  ├─ crops/
│  ├─ masks/
│  └─ segmentation_manifest.json
├─ a/
│  ├─ subset/generated/...candidate crop와 mask...
│  └─ custom/generated/...candidate crop와 mask...
└─ _metadata/
```

## 22. 로컬 최종 집계

세 metric JSON 다운로드가 끝나면 GPU 없이 로컬 CPU에서 집계한다.

```cmd
python Orthogonalization-InstantX-IP-Adapter\metric_client.py aggregate --run-id %METRIC_RUN%
```

다음 파일이 subset과 custom 각각에 생성된다.

```text
summary/<benchmark>/per_image_scores.csv
summary/<benchmark>/summary_by_condition.csv
summary/<benchmark>/summary_by_composition.csv
summary/<benchmark>/summary_by_prompt.csv
summary/<benchmark>/paired_delta_vs_baseline.csv
summary/<benchmark>/summary.json
```

우선 확인할 파일은 `summary_by_condition.csv`와 `paired_delta_vs_baseline.csv`다. custom에서는 이어서 `summary_by_composition.csv`를 통해 O+A, O+B, O+A+B별 controllability 차이를 비교한다.

## 23. 2026-09-09 실패 작업 수정 후 재실행

최초 실행의 최종 상태는 DPG와 IP-Sim이 `failed`, AES가 `completed`였다.

- DPG: 의존성 및 checkpoint 문제를 고친 뒤의 재시도에서는 mPLUG 입력만 CUDA이고 모델 weight는 CPU에 남아 있어 실패했다. 직접 로드한 공식 mPLUG 모델을 `cuda:0`으로 명시적으로 이동한 뒤 평가 모드로 전환하도록 수정했다. XVerse의 질문, 전처리, dependency 계산은 그대로 유지한다.
- IP-Sim: Florence-2가 `13_polar bear.jpg`에 이어 `18_sea turtle.jpg`도 exact/head-noun query로 검출하지 못했다. 희귀 class마다 작업이 순차적으로 중단되는 것을 막기 위해 고정된 29개 subset reference 전부에 사람이 확인한 normalized bounding box를 준비했다. 자동 Florence 검출을 항상 먼저 사용하고, 검출 결과가 0개일 때만 해당 box를 공식 SAM2의 prompt로 넣는다. `animal` 같은 넓은 query나 원본 전체 이미지 대체는 사용하지 않는다.
- AES: subset과 custom 결과가 모두 정상 완료되었으므로 재실행하지 않는다.

IP-Sim reference 산출물의 `r/segmentation_manifest.json`에는 각 이미지의 실제 query 시도 목록, 자동 검출 또는 bbox fallback 여부, 사용한 box, 원본 SHA-256이 기록된다. 따라서 어떤 reference가 fallback을 사용했는지 사후 감사할 수 있다.

아래 명령은 VS Code의 **CMD 터미널**에서 실행한다.

```cmd
cd /d C:\Users\82107\Downloads\AI_PROJECT
modal profile activate modal-research-2
modal config set-environment main
set MODAL_TOKEN_ID=
set MODAL_TOKEN_SECRET=
set PYTHONUTF8=1
modal profile current
```

수정된 image를 다시 배포한다.

```cmd
modal deploy Orthogonalization-InstantX-IP-Adapter\metric_modal.py
```

DPG의 필수 import까지 CPU에서 검사한다. 결과에 `"access": "OK"`와 `"loader": "direct-modelscope-mplug"`가 모두 있어야 한다.

```cmd
python Orthogonalization-InstantX-IP-Adapter\metric_client.py check-access
```

동일한 생성 이미지와 run-id를 유지하면서 실패한 두 metric만 재제출한다.

```cmd
set METRIC_RUN=m2-0908
for %M in (dpg ip) do @python Orthogonalization-InstantX-IP-Adapter\metric_client.py submit --run-id %METRIC_RUN% --metric %M --yes
```

이번 구현은 이전 status가 실제로 `failed`이고 생성 batch 등 평가 의미가 같은 경우에만 새 code digest로 동일 run-id 재시도를 허용한다. `running`/`completed` 결과를 덮어쓰거나 다른 batch 구성을 같은 run-id에 섞는 것은 계속 차단한다.

세 metric 상태를 확인한다. AES의 기존 `completed` 결과가 그대로 유지되는 것이 정상이다.

```cmd
for %M in (dpg ip aes) do @python Orthogonalization-InstantX-IP-Adapter\metric_client.py status --run-id %METRIC_RUN% --metric %M
```

세 항목이 모두 `completed`가 되면 다운로드하고 로컬 집계를 수행한다.

```cmd
python Orthogonalization-InstantX-IP-Adapter\metric_client.py download --run-id %METRIC_RUN%
python Orthogonalization-InstantX-IP-Adapter\metric_client.py aggregate --run-id %METRIC_RUN%
```
