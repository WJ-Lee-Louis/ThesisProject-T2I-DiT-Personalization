# FreeCus: 프로젝트 공통 1024 벤치마크

로컬 결과 통합을 위한 metric과 실행 기록은 `results/`, 생성 이미지 ZIP은 Git LFS로 추적하는 `artifacts/`에 보관한다. Git 추적 구조와 이후 결과 통합 순서는 [METHOD_COMPARISON_TRANSFER.md](../METHOD_COMPARISON_TRANSFER.md)를 따른다.

Modal 새 계정 연결·원격 실행·VRAM/비용 계산은 [MODAL_GUIDE.md](MODAL_GUIDE.md)를 따른다. 연결 profile은 `method-comparison`이며 Python 3.11 client를 별도로 사용한다. 아래 초기 로컬 검증 설명과 구분한다.

기존 InstantX 기반 연구와 직접 비교하도록 **1024×1024, seed 42, pair당 1장**으로 구성했다. DreamBench **750장**, DreamBench++ **1,350장**, 총 **2,100장/방법**이다. 이전 4-seed 목록은 기본 설정에서 제외하고 `manifests/previous-four-seed/`에 보존했다. 실제 생성·평가는 아직 실행하지 않았다.

## 기준 입력과 평가

`Orthogonalization-InstantX-IP-Adapter/outputs/semantic_benchmark_gpt56_sol/generation_manifest.json`의 case ID, 선택 reference, 원문 target을 그대로 사용한다. reference를 다시 고르지 않고, 기존 연구의 conditioning cache를 FreeCus에 주입하지 않는다. 공식 문장과 대조해 DreamBench 750/750, DreamBench++ 1,350/1,350 target이 일치함을 확인했다.

기본 평가는 기존 `uno-selected-reference-v1`을 그대로 재사용한다. CLIP ViT-B/32, DINOv1 **vits16**, 선택 reference 한 장, full-image cosine, 원문 target의 CLIP-T, 기존 clipping/전처리/집계를 유지한다. 소스와 해시는 `project_metrics/`에 보존했다. FreeCus 논문의 SAM 지표와 별개다.

해상도·프롬프트·SFC·논문 재현 및 추가 통제 실험은 [COMPARISON_PROTOCOL.md](COMPARISON_PROTOCOL.md)를 참조한다. 확인한 로컬 실행 설정과 prompt 대조 결과는 `verification/project-protocol.json`에 기록했다.

## 로컬 확인

프로젝트 루트 `AI_PROJECT`에서 실행한다. `--data-root .`은 기존 연구의 정확한 reference 경로를 재사용하기 위해 프로젝트 루트를 의미한다.

```powershell
freecus/.venv/Scripts/python.exe -m pip check
freecus/.venv/Scripts/python.exe -m freecus.verify
freecus/.venv/Scripts/python.exe -m unittest discover -s freecus/tests -v
freecus/.venv/Scripts/python.exe -m freecus.run --manifest freecus/manifests/dreambench_project_seed42.jsonl --data-root . --output freecus/outputs/db-paper-1024-seed42 --dry-run
freecus/.venv/Scripts/python.exe -m freecus.run --manifest freecus/manifests/dreambench_plus_project_seed42.jsonl --data-root . --output freecus/outputs/dbpp-paper-1024-seed42 --dry-run
freecus/.venv/Scripts/python.exe -m freecus.evaluate --manifest freecus/manifests/dreambench_plus_project_seed42.jsonl --data-root . --run freecus/outputs/dbpp-paper-1024-seed42 --output freecus/outputs/dbpp-paper-1024-seed42/metrics.json --dry-run
```

dry-run은 모델을 다운로드하지 않는다. 생성 전 평가 dry-run은 `generated_present=0`을 보고한다. 실제 평가는 전체 이미지와 provenance가 정상이어야 진행한다.

새 체크아웃에서 manifest 준비:

```powershell
freecus/.venv/Scripts/python.exe -m freecus.manifest dreambench --data-root . --output freecus/manifests/dreambench_project_seed42.jsonl
freecus/.venv/Scripts/python.exe -m freecus.manifest dreambench_plus --data-root . --output freecus/manifests/dreambench_plus_project_seed42.jsonl
```

기존 파일은 덮어쓰지 않는다. 기본 source는 `project`, seeds는 `[42]`다. `--source standalone`은 공식 데이터 폴더에서 별도 목록을 만드는 보조 경로이며 주 비교에 사용하지 않는다. `fetch_data.py`로 받은 공식 DB++ 데이터는 target 대조 자료로 보존한다.

## Python 및 CUDA 환경

현재 `freecus/.venv`는 Python 3.10 + CPU PyTorch의 독립 검증 환경이다. 생성·평가 모델 가중치는 다운로드하지 않았다. 재설치:

```powershell
py -3.10 -m venv freecus/.venv
freecus/.venv/Scripts/python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
freecus/.venv/Scripts/python.exe -m pip install -r freecus/requirements-eval.txt
```

Linux GPU:

```bash
python -m pip install -r freecus/requirements-gpu.txt
python -m pip install -r freecus/requirements-eval.txt
python -m freecus.verify --require-gpu
```

`requirements.txt`/`constraints.txt`는 생성 의존성과 전이 의존성을 고정한다. `requirements-eval.txt`는 기존 evaluator와 동일한 OpenAI CLIP Git revision을 추가한다. CUDA wheels는 torch 2.6.0+cu124 / torchvision 0.21.0+cu124다. CPU 설치 기록 `verification/requirements-local-cpu.txt`를 CUDA requirements로 직접 사용하지 않는다.

프로젝트 루트에서 `docker build -f freecus/Dockerfile -t freecus:1024 .`로 생성 환경을 구성할 수도 있다. Docker/CUDA 검증은 아직 하지 않았다. 첫 GPU smoke test는 A100/H100 80GB급과 여유 있는 호스트 RAM을 기준으로 한다. 실측 최소 사양은 아니다. 전체 foreground 1024 기준 vital KV cache는 호스트 RAM 약 14GiB를 사용한다. Qwen 모델은 caption 준비 후 해제하고 FLUX를 로드한다.

## 모델 고정과 첫 GPU 실행

```bash
python -m freecus.models --lock freecus/models.lock.json
python -m freecus.models --lock freecus/models.lock.json --download --resolved-config freecus/config.resolved.json
python -m freecus.run --config freecus/config.resolved.json --manifest freecus/manifests/dreambench_project_seed42.jsonl --data-root . --output freecus/outputs/db-paper-1024-seed42-smoke --limit 1
```

FLUX.1-dev 사용 조건에 동의한 HF 계정 접근은 추후 준비한다. `models.lock.json`을 보존하고 실행 환경 안에서 resolved snapshot 경로를 생성한다. BiRefNet은 공식 코드처럼 `trust_remote_code=True`를 사용한다. 실제 생성에서는 `--dry-run`을 빼고, 전체 실행에서는 `--limit`도 제거한다.

기존 InstantX는 24 steps, FreeCus는 논문 권장 **30 steps**, guidance 3.5, λr=λp=1.1, negative shifting, vital layer 10개를 유지한다. caption seed와 생성 seed는 모두 42다. 각 모델의 고정된 generation config를 보존하고 step 수와 전처리 비용을 함께 보고한다.

Modal 새 계정의 profile·앱·Secret·Volume과 원격 coordinator를 구성했다. 실제 실행 및 원격 상태는 `MODAL_GUIDE.md`를 따른다. 입력 묶음에는 기존 reference 경로와 같은 모델 revision을 포함한다. attention 전역 상태 때문에 engine은 순차 실행하고 출력 폴더는 writer 하나만 사용한다.

## 원본과 보정 모드

[공식 FreeCus](https://github.com/Monalissaa/FreeCus)의 추론 Python 파일 5개와 README/requirements/LICENSE를 `upstream/`에 보존했다. 모델·데모 이미지·Git 이력은 복제하지 않았다. `UPSTREAM.json`의 commit과 SHA-256은 원본 Git blob 기준이다. 수정본은 `runtime_src/`에 분리했다.

| 항목 | 공개 코드 동작 `upstream` | 기본 `paper` 보정 |
|---|---|---|
| SFC | target을 Qwen2.5로 필터링하며 참조 설명 호출 누락 | 참조 → Qwen2-VL → Qwen2.5 필터 → target 뒤에 추가 |
| 이미지/마스크 | VAE는 resize, BiRefNet은 center crop | 동일 center crop + resize |
| ANS μ | latent height/width를 VAE factor로 다시 축소 | 실제 packed token 수; 1024에서 4096 tokens, 참조 μ=-1.15 |

`paper`는 저자 승인 패치가 아닌 논문에 맞춘 해석적 보정이다. 필요하면 config를 복사해 `implementation=upstream`으로 별도 민감도 분석을 한다. `upstream` 모드에도 seed·CPU cache·상태 초기화 등 batch 실행 보완은 적용된다.

공통 보완은 vital KV만 CPU에 저장, sample마다 seed 설정으로 VAE sampling 고정, 종료/예외 시 cache·counter 초기화와 processor 복원, 상대 rotary import 수정이다. CPU에서 원본 attention 출력 일치와 실제 축소 FLUX 57개 processor의 save/use + rotary를 검증했다. CUDA 동등성은 미검증이다.

## 산출물과 평가

2026-11-01 후속 수집·평가 경로는 [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md)를 따른다. DreamBench는 전체 750장, DreamBench++는 style 참조 20개/180개 case를 모두 제외한 non-style 1,170장이다. 기존 방법에도 같은 subset을 적용한다. 두 계정의 원본 생성 결과 수집 및 T4 평가를 진행했으며 실제 상태는 `outputs/modal*/evaluation.json`과 점수 파일로 확인한다.

```text
outputs/<run>/
  run.json
  captions/<subject>.json
  src_image/<subject>/<prompt>_<sample>.png
  tgt_image/<subject>/<prompt>_<sample>.png
  text/<subject>/<prompt>_<sample>.txt
  metadata/<subject>/<prompt>_<sample>.json
```

metadata에 기존 case/reference ID, 원문 target, 실제 확장 prompt, seed, 입력/출력 해시를 남긴다. `text/`에는 원문 target만 저장한다. 정상 결과는 재시작 시 건너뛰고 손상된 결과는 다시 만든다. config/code/manifest 또는 주요 library version이 바뀌면 기존 run 폴더 사용을 거부한다.

생성 완료 후 `freecus.evaluate`의 dry-run을 제거한다. 기존 evaluator를 호출하므로 CLIP/DINO weights가 필요하다. 기존과 같은 case별 raw score, reference macro/case micro/category 집계를 남기며 불완전한 결과를 최종 점수로 출력하지 않는다.

이전 HF vits8/foreground 구현은 `python -m freecus.evaluate_diagnostic`로 이동했다. 주 비교 점수와 섞지 않는다. 논문의 SAM 선택·crop/background·metric variant 세부가 공개 코드에 없어 엄밀한 재현은 미완성이다. GPT judge도 호출하지 않았다.

논문은 **512×512, pair당 4장, SAM subject 유사도**를 사용한다. 이번 1024·1장·공통 selected-reference 결과는 방법 간 비교다. 논문 Table 1의 CLIP-T 0.308 / CLIP-I 0.853 / DINO 0.696과 직접 동등 비교하지 않는다. 원조건 재현은 별도 실험이다.

검증 범위는 `verification/RESULTS.md`, 비교 설계는 [COMPARISON_PROTOCOL.md](COMPARISON_PROTOCOL.md)를 참조한다. 데이터 저작권은 기존 출처 기록을 따르며 FreeCus MIT LICENSE와 Apache 헤더를 보존한다.
