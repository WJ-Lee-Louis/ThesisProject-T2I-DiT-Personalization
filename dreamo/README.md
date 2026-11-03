# DreamO 벤치마크 환경

로컬 결과 통합을 위한 metric과 실행 기록은 `results/`, 생성 이미지 ZIP은 Git LFS로 추적하는 `artifacts/`에 보관한다. Git 추적 구조와 이후 결과 통합 순서는 [METHOD_COMPARISON_TRANSFER.md](../METHOD_COMPARISON_TRANSFER.md)를 따른다.

2026-11-02. 공식 [bytedance/DreamO](https://github.com/bytedance/DreamO)의 최신 확인 commit `461e35ba6b068f42ca53b43f289b3d91b7f911cd`에서 추론·전처리 코드와 LICENSE/설명/원래 requirements를 `upstream/`에 보존했다. 예제 이미지, 모델 weight, 전체 Git 이력, Gradio 실행 환경은 포함하지 않는다. 파일별 Git blob SHA-256은 `UPSTREAM.json`에 기록하며 실행 전 검사한다. 원본 코드는 수정하지 않는다.

## 생성 및 평가 범위

| 항목 | 설정 |
|---|---|
| DreamBench 생성 | 750개 case / 참조 30개 |
| DreamBench++ 생성 | **전체 1,350개 case / 참조 150개**, style 180개 포함 |
| 출력 | 실제 1024×1024 PNG, seed 42, case당 1장 |
| 입력 | FreeCus와 같은 case ID·참조 원본·reference ID·원문 target |
| DreamO 버전 | v1.1, Turbo 활성화, 12 steps, guidance 4.5, true CFG 1 |
| 정밀도 | FLUX BF16, 양자화·CPU offload 없음 |
| 일반 참조 처리 | 공식 IP task: BEN2로 배경 제거 후 공식 면적 기반 resize |
| style 참조 처리 | 공식 Style task: 배경 제거 없이 resize, 공식 활성화 문구 추가 |
| 참조 내부 해상도 | 공식 기본 면적 512×512, aspect 유지, 가로·세로 16배수 |
| 주 평가 | 기존 `uno-selected-reference-v1` CLIP-T / CLIP-I / DINO-I |
| DreamBench++ 평가 기본값 | 전체 1,350개. style 제외는 **평가 시에만 선택 가능** |

사용자 수정 지시에 따라 style 제외를 DreamO 생성에 적용하지 않는다. FreeCus의 기존 non-style 1,170개 결과는 그대로 두고, DreamO의 전체/비스타일/스타일 비교 범위는 나중에 결정한다. 기존 baseline·우리 방법의 원래 1,350개 raw score도 보존되어 있다. 서로 다른 case 수의 평균을 직접 비교하지 않도록 최종 비교표의 cohort를 명시한다.

일반 human도 IP task를 사용한다. ID task는 얼굴만 별도로 crop/align하는 다른 조건이므로 자동 전환하지 않는다. 현재 필요한 IP/Style 경로에 없는 facexlib·얼굴 검출기·Gradio·Nunchaku·Quanto는 설치하지 않는다.

공식 v1.1의 12-step Turbo/guidance 4.5를 사용하며 FreeCus의 30-step/guidance 3.5로 강제 변경하지 않는다. `DreamO v1.1 + Turbo`로 보고하고 원논문 v1 점수의 직접 재현이라고 부르지 않는다. [공식 v1.1 설명](https://github.com/bytedance/DreamO/blob/main/dreamo_v1.1.md)은 기본 참조 해상도를 512로 설명한다. 참조 내부 해상도와 최종 출력 1024는 서로 다른 설정이다.

## 모델과 메모리

모델 revision은 `models.lock.json`에 고정했다. FLUX는 기존 FreeCus와 동일한 `3de623fc3c33e44ffbe2bad470d0f45bccf2eb21`을 사용한다. 나머지 DreamO/CFG/SFT/DPO LoRA, Turbo LoRA, BEN2도 각각 revision을 고정했다. 원격 두 계정에서 모델 준비와 실제 A100 추론 pilot이 완료됐다.

HF 파일 metadata 기반 용량은 다음과 같다. 파일 크기는 실제 runtime peak VRAM과 다르다.

| 구성 | 파일 크기 | 실행 중 처리 |
|---|---:|---|
| FLUX transformer + T5/CLIP text encoders + VAE | 약 31.42 GiB | BF16, GPU 상주 |
| BEN2 | 약 1.06 GiB | 공식 FP32 weight, GPU 상주 |
| DreamO/CFG/SFT/DPO LoRA 합계 | 약 2.00 GiB | CPU에서 fuse 후 adapter unload |
| Turbo LoRA | 약 0.65 GiB | CPU에서 fuse 후 adapter unload |

GPU 상주 weight 근사 하한은 **약 32.48 GiB**다. LoRA 파일 전부를 상주 weight에 다시 더하지 않는다. 여기에 임시 텐서·activation·attention·CUDA workspace가 필요하므로 **40–48 GiB는 계획용 추정 범위**이며 **실측이 아니다**.

- 첫 검증 권장: **A100 80GB 1개 / 프로세스**, host RAM 64 GiB, CPU 2 cores.
- 속도 비교 후보: **H100 80GB 1개 / 프로세스**. 시간당 요금만으로 장당 비용 우위를 판단하지 않고 pilot을 측정한다.
- 비용 절감 후보: **L40S 48GB**, 최대 사용량과 여유를 실제 pilot으로 확인한 뒤 선택한다. 48GB 구동을 검증했다고 주장하지 않는다.
- 24GB 이하에서는 현재 BF16 전 모델 상주 구성이 weight 하한만으로 초과한다. 공식 문서의 저메모리 구성은 offload/양자화 경로다. 주 비교에 조용히 양자화를 켜지 않는다.
- 벤치마크당 worker 2개면 GPU 총 2개다. 모델 하나를 두 GPU에 분산하는 것이 아니라 case를 나눈다. 두 벤치마크를 동시에 실행하려면 총 4개다.

DreamO에는 FreeCus의 Qwen 두 모델 같은 추가 caption LLM이 없다. T5와 CLIP text encoder는 FLUX 기본 구성으로 이미 위 용량에 포함했다. 참조 512 면적은 약 1,024개 visual tokens, 출력 1024는 4,096개에 해당하며 text token도 추가된다. A100 pilot에서 초기화 peak reserved는 31.82 GiB, 생성 peak는 36.05 GiB였다. 장당 생성 시간 추정은 DreamBench 10.30초, DreamBench++ 9.47초다. 실측은 `verification/a100_pilot.json`, 원래 metadata 추정은 `verification/model_sizes.json`, `verification/memory_plan.json`에 보존한다. 전체 실행과 다른 GPU의 peak를 보장하는 값은 아니다.

## 로컬 검증 환경

독립 Python 3.11 CPU 환경 `dreamo/.venv`를 구성한다. 기존 FreeCus 환경이나 기존 결과를 바꾸지 않는다.

설치는 완료했고 `pip check` 및 테스트 **10개**가 통과했다. 실제 축소 DreamOPipeline의 LoRA load/fuse/unload, 참조 embedding을 더한 transformer forward, 입력 cohort·style prefix·원본 hash, 비용 기반 GPU 선택과 메모리 guard를 검사했다. dry-run과 로컬 ZIP 검증도 완료했다. 기록은 `verification/SETUP.json`, CPU 설치 목록은 `verification/requirements-local-cpu.txt`, 공통 의존성 제약은 `constraints.txt`에 있다. 전체 모델의 A100 검증은 별도 원격 pilot으로 완료했다.

```powershell
py -3.11 -m venv dreamo/.venv
dreamo/.venv/Scripts/python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
dreamo/.venv/Scripts/python.exe -m pip install -r dreamo/requirements.txt
dreamo/.venv/Scripts/python.exe -m pip check
dreamo/.venv/Scripts/python.exe -X utf8 -m unittest discover -s dreamo/tests -q
```

필요 generation 버전은 torch 2.6 / torchvision 0.21 / diffusers 0.33.1 / transformers 4.45.2 / PEFT 0.15.2 / accelerate 0.32다. upstream의 느슨한 PEFT 버전을 고정하여 오래된 Transformers와의 호환성을 검증한다. CPU torch 환경으로 이미지 생성은 하지 않는다. CUDA 설치에는 `requirements-gpu.txt`를 사용한다.

원본 보존·입력 해시·dry run:

```powershell
freecus/.venv-modal311/Scripts/python.exe -X utf8 -m dreamo.protocol
freecus/.venv-modal311/Scripts/python.exe -X utf8 -m dreamo.models
freecus/.venv-modal311/Scripts/python.exe -X utf8 -m dreamo.run --benchmark dreambench --output dreamo/outputs/db-dry --dry-run
freecus/.venv-modal311/Scripts/python.exe -X utf8 -m dreamo.run --benchmark dreambench_plus --output dreamo/outputs/dbpp-dry --dry-run
```

## 추론 시 동작과 원본과의 차이

`runtime.py`는 공식 DreamOPipeline, transformer forward, BEN2, resize를 그대로 호출한다. Gradio/얼굴용 Generator 전체를 띄우지 않는 IP/Style 전용 headless wrapper다. 공식 generator의 BF16 경로에서 `version` 인자를 전달하지 않는 부분을 우회하고 `load_dreamo_model(version='v1.1')`을 명시한다.

공식 로더의 고정 `models/...` 경로는 per-worker 작업 폴더의 고정 snapshot symlink로 연결한다. 허용된 repo/파일만 읽고 HF `main`으로 되돌아가지 않는다. LoRA 종류·fuse weight·전처리 수식은 수정하지 않는다. 공식 pipeline의 reference VAE `.sample()`은 별도 global RNG를 사용하므로 case마다 global seed와 CPU noise generator를 모두 42로 고정한다. 이를 통해 shard 순서와 resume에 따른 참조 latent의 우연한 변화를 막는다. 기존 방법과 동일한 seed 정수라도 noise tensor 자체가 같다는 의미는 아니다.

Style task는 공식 예제대로 실제 생성 입력에 `generate a same style image. `를 붙인다. metadata에는 `generation_prompt`로 남기고, `text/` 및 CLIP-T에는 기존 **원문 target만** 사용한다. 원본 reference를 평가하며 BEN2 전처리 이미지를 reference 점수에 대신 사용하지 않는다.

`run.json`에 source/config/model revision/manifest hash와 library/GPU 정보를 남기고, 매 이미지 artifact hash·case/reference ID·timing·GPU peak를 metadata에 기록한다. `initialization.json`에 초기화 VRAM peak를 별도로 보존한다. 중단 후 해시가 맞는 결과만 재사용하며 source/config/library가 바뀐 결과를 섞지 않는다. `conditioning/`에는 전처리된 참조 PNG를 저장한다.

## Modal remote execution

See [REMOTE_RUN.md](REMOTE_RUN.md) for the deployed autonomous A100/L40S workflow, account mapping, status commands, budget controls, and collection paths. This supersedes the previous preparation-only instructions.

## Metric 평가

4-workspace 원격 평가에는 `freecus.operations.evaluate_comparison --method dreamo --benchmark dreambench` 또는 `--benchmark dreambench_plus`를 사용한다. 이미 접수된 평가는 `--status`를 추가해 진행 상황과 완료 점수를 수집한다. 평가 기본값은 전체 750/1,350장이고 로컬 수집 경로는 `dreamo/outputs/metrics/*_scores.json`이다.

`dreamo.evaluate`는 FreeCus와 같은 해시 보존 evaluator를 사용한다. 지표 구현을 새로 만들거나 변경하지 않는다. 출력 condition만 `dreamo`로 구분한다. 원본 FreeCus 결과는 덮어쓰지 않는다. 평가용 패키지는 생성 환경과 별도인 `requirements-eval.txt`에 있다. 기존 T4 평가 환경(torch 2.7.1, torchvision 0.22.1, Pillow 11.3.0)을 재사용하는 것이 좋다.

기본값은 DreamBench 750 / DreamBench++ **전체 1,350**이다. `--selection dbpp-no-style`을 명시할 때만 같은 생성 이미지에서 1,170장을 선택한다. style 제외 때문에 다시 생성할 필요가 없다.

```powershell
# 아직 이미지가 없어도 입력 계획은 확인할 수 있다.
freecus/.venv-modal311/Scripts/python.exe -X utf8 -m dreamo.evaluate --benchmark dreambench_plus --run dreamo/outputs/future-dbpp-run --output dreamo/outputs/metrics/dbpp-all.json --dry-run

# 추후 두 lane을 검증·병합한 결과 경로에서 별도 평가 환경으로 실행한다.
python -m dreamo.evaluate --benchmark dreambench --run <MERGED_DB_RUN> --output dreamo/outputs/metrics/dreambench.json --device cuda
python -m dreamo.evaluate --benchmark dreambench_plus --run <MERGED_DBPP_RUN> --output dreamo/outputs/metrics/dbpp-all.json --device cuda
# 나중에 제외하기로 결정한 경우에만:
python -m dreamo.evaluate --benchmark dreambench_plus --run <MERGED_DBPP_RUN> --output dreamo/outputs/metrics/dbpp-no-style.json --selection dbpp-no-style --device cuda
```

CLIP-T는 원문 target, CLIP-I/DINO-I는 생성 전체 이미지와 원래 선택 참조 1장, CLIP ViT-B/32 및 DINOv1 vits16과 동일 전처리를 사용한다. source/model/case hash, 1024 해상도, seed 42, 누락·중복·파일 손상을 검증하고 원본 case score·reference macro·case micro·category 집계 및 batch checkpoint를 남긴다. DreamO pilot 이미지는 생성됐으며 전체 생성 후 실제 metric 계산을 진행한다.
