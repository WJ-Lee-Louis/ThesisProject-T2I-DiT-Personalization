# FreeGraftor 동물 personalization baseline 실행 가이드

## 구축된 실험

- 공식 FreeGraftor commit: `b8e0f93e91753265e46e9d0b364daa38e25f1afa`
- reference: 기존 동물 이미지 10장 `(dog 7, cat 3)`
- prompt: reference당 16개
- 최종 생성량: `10 × 16 = 160장`
- 출력 경로: `freegraftor\outputs\<batch-id>\<reference-stem>\`
- 파일명 예: `dog_02_flying.png`
- GPU: Modal `A100-80GB`, 기본 2개 shard
- 모델은 컨테이너당 한 번만 로드하며 개별 이미지 실패는 다른 이미지 생성을 중단시키지 않는다.

논문 재현값을 명시적으로 고정했다.

| 설정 | 값 |
|---|---:|
| resolution | 512 × 512 |
| template generation | 25 steps |
| FireFlow inversion | 25 steps |
| final generation | 25 steps |
| guidance | 3.0 |
| similarity threshold | 0.2 |
| cycle threshold | 1.5 |
| dynamic match dropout | 0.5 |
| injection steps | 0–25 |
| injection blocks | 0–56 |
| seed | 42 |
| concept alignment | center |
| CPU offload | enabled |

`generate.py`의 CLI 기본값과 논문 값에는 차이가 있다. 특히 dropout은 CLI `0.4`, dataclass `0.2`, 논문 `0.5`이므로 이 배치에서는 논문 값 `0.5`를 사용한다.

`center`는 single-concept `ConceptConfig`의 기본값이다. 공식 cat+dog 다중 subject 예제에는 수동 `bottom` 정렬이 들어가지만, flying/jumping 같은 behavior 평가에 cutout 위치라는 추가 변수를 만들 수 있어 primary baseline에는 적용하지 않았다. 필요하면 별도 batch에서 `--alignment bottom`으로 ablation할 수 있다.

## 최초 1회 준비

아래 명령은 모두 VS Code의 **CMD 터미널**에서 프로젝트 루트로 이동한 뒤 실행한다.

```cmd
cd /d C:\Users\82107\Downloads\AI_PROJECT
python -m pip install -r freegraftor\requirements-client.txt
python -m modal token info
```

Hugging Face에서 `black-forest-labs/FLUX.1-dev` 사용 조건에 동의되어 있어야 한다. 이미 `huggingface-secret`에 유효한 `HF_TOKEN`을 등록했다면 secret 생성 명령은 건너뛴다. 등록하지 않은 경우에만 다음을 실행한다.

```cmd
python -m modal secret create huggingface-secret HF_TOKEN=hf_YOUR_TOKEN
```

## 1. 로컬 manifest 검증

GPU를 사용하지 않는다. reference 10장과 prompt 16개, 총 160개 output 경로를 검증한다.

```cmd
python freegraftor\batch_client.py preview
```

## 2. Modal worker 배포

```cmd
python -m modal deploy freegraftor\batch_modal.py
```

코드를 변경했다면 반드시 다시 배포한다. 로컬 코드와 배포 코드의 digest가 다르면 worker가 실행을 거부하도록 되어 있다.

## 3. 모델 warm-up

먼저 GPU를 쓰지 않는 access check를 실행한다.

```cmd
python freegraftor\batch_client.py check-access
```

`access: FAILED (GatedRepoError)`가 나오면, 해당 토큰을 만든 Hugging Face 계정으로 [FLUX.1-dev 페이지](https://huggingface.co/black-forest-labs/FLUX.1-dev)의 사용 조건에 먼저 동의하고 read 권한 토큰을 발급한다. 기존 Modal secret은 다음처럼 교체한다.

```cmd
python -m modal secret create huggingface-secret HF_TOKEN=hf_NEW_TOKEN --force
python freegraftor\batch_client.py check-access
```

토큰 문자열은 코드나 메모 파일에 저장하지 않는다. `access: OK`가 확인된 후에만 warm-up을 실행한다.

최초 한 번 실행을 권장한다. FLUX.1-dev, Grounding DINO, SAM ViT-H, ModelScope LaMa를 persistent cache에 준비하고 실제 pipeline load까지 확인한다. 이 명령이 끝날 때까지 터미널 연결을 유지한다.

```cmd
python freegraftor\batch_client.py warmup --yes
```

warm-up을 생략해도 submit 시 자동 준비되지만, 2개 컨테이너가 동시에 대형 checkpoint를 준비할 가능성이 있어 최초 실행은 warm-up 후 submit하는 편이 안전하다.

## 4. 160장 비동기 제출

원하는 고유 batch ID를 지정한다.

```cmd
python freegraftor\batch_client.py submit --batch-id freegraftor-baseline-0826 --shards 2 --yes
```

`Submission complete`와 두 개의 Function Call ID가 출력된 뒤에는 VS Code, 인터넷 연결, 로컬 PC를 종료해도 Modal 작업은 계속된다. 별도의 `tmux`는 필요하지 않다.

한 GPU로만 순차 실행하려면 다음처럼 지정한다.

```cmd
python freegraftor\batch_client.py submit --batch-id freegraftor-baseline-0826-1gpu --shards 1 --yes
```

## 5. 상태와 로그 확인

```cmd
python freegraftor\batch_client.py status --batch-id freegraftor-baseline-0826
python -m modal app logs freegraftor-dreambench-batch
```

두 shard의 aggregate processed 수가 `160/160`, failed 수가 `0`이면 정상 완료다. `completed_with_errors`인 경우 `_metadata\errors`에 Grounding DINO/SAM 검출 실패 등의 원인이 기록된다.

## 6. 결과 다운로드

```cmd
python freegraftor\batch_client.py download --batch-id freegraftor-baseline-0826
```

다운로드는 이미 크기가 일치하는 파일을 건너뛰며 파일별 재시도를 수행한다. 네트워크가 끊기면 동일 명령을 다시 실행하면 된다. 작업 중 일부만 먼저 받고 싶을 때만 다음 옵션을 사용한다.

```cmd
python freegraftor\batch_client.py download --batch-id freegraftor-baseline-0826 --allow-partial
```

## 실패 job 재실행

먼저 status와 remote error JSON을 확인한다. 같은 설정에서 누락된 PNG만 재실행하려면 기존 제출 명령에 `--resume`을 추가한다. 이미 PNG와 sidecar JSON이 모두 있는 job은 자동으로 skip한다.

```cmd
python freegraftor\batch_client.py submit --batch-id freegraftor-baseline-0826 --shards 2 --resume --yes
```

완료된 이미지까지 의도적으로 다시 만들 때만 `--overwrite`를 사용한다. 기존 batch와 설정 충돌을 피하려면 overwrite 실험은 새 batch ID를 권장한다.

## 구현상 재현성 보정

- 저자 로컬의 `/workspace/yzb/...` 경로로 환경변수를 덮어쓰던 코드를 제거했다.
- 비정방 출력에서 `width/height`를 반대로 넘기던 `unpack` 호출을 수정했다.
- 자동 reference/template 검출 실패를 명시적 job error로 바꿨다.
- upstream의 content-hash 파일명 대신 deterministic 파일명을 별도로 저장한다.
- prompt와 species, reference SHA-256, seed, 모든 FreeGraftor hyperparameter, 실제 GPU와 package version을 이미지별 sidecar에 기록한다.
- 동일 prompt/species/seed의 template은 한 shard 안에서 재사용한다. 독립 CLI 실행에서도 매번 같은 template이 생성되므로 결과 의미는 같고, 중복 template diffusion만 줄인다.
- template 생성 직후의 CPU/CUDA RNG state를 저장해 각 reference 전에 복원하므로 stochastic match-dropout 순서도 독립 CLI 실행과 맞춘다.

## 비교 해석 시 주의

이 160장은 한 seed `(42)`의 정성 비교용 baseline이다. 논문 수준의 통계적 주장을 하려면 여러 seed 반복이 별도로 필요하다. 또한 기존 InstantX 실험의 `960×1280 / 24 steps / guidance 3.5`와 이번 방법-native FreeGraftor 설정 `512×512 / 25 steps / guidance 3.0`은 해상도와 sampler 설정이 다르다. 현재 결과는 각 방법의 권장 환경에서 보는 정성 비교로 해석하고, 통제된 정량 비교가 필요하면 InstantX도 FreeGraftor가 안정적으로 지원하는 공통 `512×512` 설정으로 다시 생성하는 편이 안전하다.

이미지별 기록의 `generation_seconds_excluding_template`는 공유 template 시간을 제외한다. 따라서 upstream 논문의 end-to-end runtime 표와 직접 비교하지 않는다.

## 라이선스 주의

FreeGraftor 공식 저장소에는 현재 명시적인 코드 `LICENSE`가 없다. 논문의 CC BY 4.0과 코드 라이선스는 동일한 것으로 간주할 수 없다. 이 폴더는 내부 연구 재현용 upstream snapshot으로 취급하고, 외부 재배포 전에는 저자에게 코드 사용 조건을 확인해야 한다. FLUX.1-dev의 별도 비상업 라이선스도 적용된다.
