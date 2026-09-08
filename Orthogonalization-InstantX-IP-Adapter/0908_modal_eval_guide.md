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
