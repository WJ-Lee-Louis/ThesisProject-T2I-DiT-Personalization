# 0903 Modal Experiment Guide

## 1. 실행 범위

이 가이드는 이미 생성된 `baseline`과 `ssg`를 제외하고, 다음 두 nuisance
설정에 generic class 설계 네 가지를 적용한다.

| Nuisance 설정 | Generic 설계 | Condition |
|---|---|---|
| Target | Absolute direct | `class_target_absolute_direct_ssg` |
| Target | Absolute residual | `class_target_absolute_residual_ssg` |
| Target | No-object | `class_target_no_object_ssg` |
| Target | Superclass | `class_target_superclass_ssg` |
| Target + all source | Absolute direct | `class_st_all_absolute_direct_ssg` |
| Target + all source | Absolute residual | `class_st_all_absolute_residual_ssg` |
| Target + all source | No-object | `class_st_all_no_object_ssg` |
| Target + all source | Superclass | `class_st_all_superclass_ssg` |

각 condition은 generic lambda 0.5와 1.0으로 각각 실행한다.

- 방법론: 8개
- Modal batch: 8 × 2 = 16개
- batch당 이미지: 10 references × 14 prompts = 140장
- 총 신규 이미지: 2,240장
- 모든 condition에 single-stream gate 적용
- Action/background lambda는 1.0으로 고정

이 가이드에서는 grid를 생성하지 않는다.

## 2. GPU budget 운영 권장사항

현재 Modal 설정은 A100 80GB 컨테이너를 최대 3개까지 동시에 활성화할 수
있다. 16개를 한꺼번에 제출해도 나머지는 pending 상태로 대기하지만, 최대
3개 GPU 비용이 동시에 발생할 수 있다.

비용을 쉽게 통제하려면 아래 명령을 두 개씩 제출하고, 두 batch가 완료된
후 다음 두 개를 제출하는 방식을 권장한다. 로컬 VS Code나 PC를 종료해도
제출이 끝난 Modal 작업은 계속 실행된다.

## 3. VS Code CMD 사전 확인 및 배포

VS Code에서 터미널 프로필을 `Command Prompt`로 연 뒤 다음을 실행한다.

```cmd
cd /d C:\Users\82107\Downloads\AI_PROJECT
python Orthogonalization-InstantX-IP-Adapter\batch_client.py preview --source-captions Orthogonalization-InstantX-IP-Adapter\source_captions.json
python -m modal deploy Orthogonalization-InstantX-IP-Adapter\batch_modal.py
```

Preview에서 다음 값을 확인한다.

- references: 10
- prompts/reference: 14
- total jobs: 140
- source captions: 10

배포를 다시 수행해야 이번에 수정한 14-prompt catalog와 22-condition 코드가
Modal worker에 반영된다.

## 4. Target generic 조건 제출

### 4.1 Absolute direct

```cmd
python Orthogonalization-InstantX-IP-Adapter\batch_client.py submit --condition class_target_absolute_direct_ssg --batch-id exp0903-target-abs-direct-c05 --orthogonalization-strength 1.0 --class-projection-strength 0.5 --yes
python Orthogonalization-InstantX-IP-Adapter\batch_client.py submit --condition class_target_absolute_direct_ssg --batch-id exp0903-target-abs-direct-c10 --orthogonalization-strength 1.0 --class-projection-strength 1.0 --yes
```

### 4.2 Absolute residual

```cmd
python Orthogonalization-InstantX-IP-Adapter\batch_client.py submit --condition class_target_absolute_residual_ssg --batch-id exp0903-target-abs-residual-c05 --orthogonalization-strength 1.0 --class-projection-strength 0.5 --yes
python Orthogonalization-InstantX-IP-Adapter\batch_client.py submit --condition class_target_absolute_residual_ssg --batch-id exp0903-target-abs-residual-c10 --orthogonalization-strength 1.0 --class-projection-strength 1.0 --yes
```

### 4.3 No-object

```cmd
python Orthogonalization-InstantX-IP-Adapter\batch_client.py submit --condition class_target_no_object_ssg --batch-id exp0903-target-no-object-c05 --orthogonalization-strength 1.0 --class-projection-strength 0.5 --yes
python Orthogonalization-InstantX-IP-Adapter\batch_client.py submit --condition class_target_no_object_ssg --batch-id exp0903-target-no-object-c10 --orthogonalization-strength 1.0 --class-projection-strength 1.0 --yes
```

### 4.4 Superclass

```cmd
python Orthogonalization-InstantX-IP-Adapter\batch_client.py submit --condition class_target_superclass_ssg --batch-id exp0903-target-superclass-c05 --orthogonalization-strength 1.0 --class-projection-strength 0.5 --class-superclass-prompt "an animal" --yes
python Orthogonalization-InstantX-IP-Adapter\batch_client.py submit --condition class_target_superclass_ssg --batch-id exp0903-target-superclass-c10 --orthogonalization-strength 1.0 --class-projection-strength 1.0 --class-superclass-prompt "an animal" --yes
```

## 5. Target + all source 조건 제출

이 네 condition은 reference별 source caption을 사용한다.

### 5.1 Absolute direct

```cmd
python Orthogonalization-InstantX-IP-Adapter\batch_client.py submit --condition class_st_all_absolute_direct_ssg --batch-id exp0903-sta-abs-direct-c05 --source-captions Orthogonalization-InstantX-IP-Adapter\source_captions.json --orthogonalization-strength 1.0 --class-projection-strength 0.5 --yes
python Orthogonalization-InstantX-IP-Adapter\batch_client.py submit --condition class_st_all_absolute_direct_ssg --batch-id exp0903-sta-abs-direct-c10 --source-captions Orthogonalization-InstantX-IP-Adapter\source_captions.json --orthogonalization-strength 1.0 --class-projection-strength 1.0 --yes
```

### 5.2 Absolute residual

```cmd
python Orthogonalization-InstantX-IP-Adapter\batch_client.py submit --condition class_st_all_absolute_residual_ssg --batch-id exp0903-sta-abs-residual-c05 --source-captions Orthogonalization-InstantX-IP-Adapter\source_captions.json --orthogonalization-strength 1.0 --class-projection-strength 0.5 --yes
python Orthogonalization-InstantX-IP-Adapter\batch_client.py submit --condition class_st_all_absolute_residual_ssg --batch-id exp0903-sta-abs-residual-c10 --source-captions Orthogonalization-InstantX-IP-Adapter\source_captions.json --orthogonalization-strength 1.0 --class-projection-strength 1.0 --yes
```

### 5.3 No-object

```cmd
python Orthogonalization-InstantX-IP-Adapter\batch_client.py submit --condition class_st_all_no_object_ssg --batch-id exp0903-sta-no-object-c05 --source-captions Orthogonalization-InstantX-IP-Adapter\source_captions.json --orthogonalization-strength 1.0 --class-projection-strength 0.5 --yes
python Orthogonalization-InstantX-IP-Adapter\batch_client.py submit --condition class_st_all_no_object_ssg --batch-id exp0903-sta-no-object-c10 --source-captions Orthogonalization-InstantX-IP-Adapter\source_captions.json --orthogonalization-strength 1.0 --class-projection-strength 1.0 --yes
```

### 5.4 Superclass

```cmd
python Orthogonalization-InstantX-IP-Adapter\batch_client.py submit --condition class_st_all_superclass_ssg --batch-id exp0903-sta-superclass-c05 --source-captions Orthogonalization-InstantX-IP-Adapter\source_captions.json --orthogonalization-strength 1.0 --class-projection-strength 0.5 --class-superclass-prompt "an animal" --yes
python Orthogonalization-InstantX-IP-Adapter\batch_client.py submit --condition class_st_all_superclass_ssg --batch-id exp0903-sta-superclass-c10 --source-captions Orthogonalization-InstantX-IP-Adapter\source_captions.json --orthogonalization-strength 1.0 --class-projection-strength 1.0 --class-superclass-prompt "an animal" --yes
```

## 6. 상태 확인

한 batch는 다음처럼 확인한다.

```cmd
python Orthogonalization-InstantX-IP-Adapter\batch_client.py status --batch-id exp0903-target-abs-direct-c05
```

16개 전체 상태는 CMD의 `for` 문으로 확인할 수 있다.

```cmd
for %B in (exp0903-target-abs-direct-c05 exp0903-target-abs-direct-c10 exp0903-target-abs-residual-c05 exp0903-target-abs-residual-c10 exp0903-target-no-object-c05 exp0903-target-no-object-c10 exp0903-target-superclass-c05 exp0903-target-superclass-c10 exp0903-sta-abs-direct-c05 exp0903-sta-abs-direct-c10 exp0903-sta-abs-residual-c05 exp0903-sta-abs-residual-c10 exp0903-sta-no-object-c05 exp0903-sta-no-object-c10 exp0903-sta-superclass-c05 exp0903-sta-superclass-c10) do @python Orthogonalization-InstantX-IP-Adapter\batch_client.py status --batch-id %B
```

`completed_count = 140`, `failed_count = 0`을 확인한다. 로컬 터미널을
종료했더라도 같은 batch ID로 다시 상태를 조회할 수 있다.

## 7. 다운로드

한 batch 다운로드:

```cmd
python Orthogonalization-InstantX-IP-Adapter\batch_client.py download --batch-id exp0903-target-abs-direct-c05
```

완료된 16개 batch 전체 다운로드:

```cmd
for %B in (exp0903-target-abs-direct-c05 exp0903-target-abs-direct-c10 exp0903-target-abs-residual-c05 exp0903-target-abs-residual-c10 exp0903-target-no-object-c05 exp0903-target-no-object-c10 exp0903-target-superclass-c05 exp0903-target-superclass-c10 exp0903-sta-abs-direct-c05 exp0903-sta-abs-direct-c10 exp0903-sta-abs-residual-c05 exp0903-sta-abs-residual-c10 exp0903-sta-no-object-c05 exp0903-sta-no-object-c10 exp0903-sta-superclass-c05 exp0903-sta-superclass-c10) do @python Orthogonalization-InstantX-IP-Adapter\batch_client.py download --batch-id %B
```

결과는 다음 위치에 저장된다.

```text
C:\Users\82107\Downloads\AI_PROJECT\Orthogonalization-InstantX-IP-Adapter\outputs\<batch-id>
```

다운로드가 네트워크 오류로 중단되면 같은 명령을 다시 실행한다. 이미 정상
저장된 파일은 건너뛰고 나머지만 이어서 받는다. `No remote results found`가
표시되면 먼저 `status`로 batch ID와 완료 상태를 확인한다.
