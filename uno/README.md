# UNO benchmark reproduction and controlled comparison

로컬 결과 통합을 위한 metric과 실행 기록은 `results/`, 생성 이미지 ZIP은 Git LFS로 추적하는 `artifacts/`에 보관한다. Git 추적 구조와 이후 결과 통합 순서는 [METHOD_COMPARISON_TRANSFER.md](../METHOD_COMPARISON_TRANSFER.md)를 따른다.

공식 [UNO 저장소](https://github.com/bytedance/UNO), [논문](https://arxiv.org/abs/2504.02160)을 기반으로 구축했다. 원본 commit은 `a563432dcfb4caf98df4368facb8102bf97d376c`이며, 기존 프로젝트에서 참조 이미지 선정에 사용한 UNO revision과 같다. 필요한 파일 15개만 `upstream/`에 Git blob 그대로 복제했다. 파일별 SHA256은 `UPSTREAM.json`에 보존하고 매 실행 시 검증한다. 학습 코드, 학습 데이터, Gradio, 예제 이미지, 가중치는 소스에 포함하지 않는다.

## 비교 조건과 논문 재현의 구분

이번 작업의 목적은 기존 InstantX/FreeCus/DreamO와 입력 조건을 맞춘 비교다. `freecus/manifests/*_project_seed42.jsonl`을 그대로 사용한다. DreamBench는 참조 30개 × 25 prompt = 750장, DreamBench++는 참조 150개 × 9 prompt = 1,350장이다. 모든 case는 같은 선택 참조 1장, 원문 target prompt, seed 42, 출력 1024×1024를 사용한다. GPU별 RNG·모델·방법론이 다르므로 같은 seed가 같은 noise tensor를 의미하지는 않는다.

공식 `inference.py` 기본값은 출력 512×512/seed 3407이고 README 예제는 704×704다. README는 512 bucket 학습과 더 높은 해상도의 일반화를 설명한다. 따라서 **이번 1024×1024/seed 42 결과를 논문 원래 조건의 직접 재현이라고 표시하지 않는다**. 논문 Table 1의 UNO 수치(DINO 0.760 / CLIP-I 0.835 / CLIP-T 0.304)와 비교할 때 이 차이를 명시한다. 공개 코드 기본값을 유지한 것과 논문에 명시되지 않은 상세 설정을 추정한 것을 혼동하지 않는다.

논문의 single-subject 평가와 multi-subject 30조합/4,500개 평가는 별개다. 이번 작업은 단일 참조 비교이며 multi-subject 벤치마크를 실행하지 않는다. UNO는 사전 학습된 LoRA를 사용하는 inference-time tuning-free 방법이다. 참조마다 LoRA를 학습하지 않는다.

## 추론 설정

| 항목 | 이번 설정 | 출처/이유 |
|---|---|---|
| Base | FLUX.1-dev, 원본 FLUX-format weight | 공식 `UNOPipeline` |
| Steps / guidance | 25 / 4.0 | 공식 inference 기본값 |
| UNO LoRA | rank 512, 공식 processor 유지 | 공식 inference 기본값 |
| UnoPE | `d`, 가로·세로 동시 offset | 공식 기본값 |
| 출력 | 1024×1024 | 프로젝트 공통 비교 조건 |
| Seed | 각 case 42 | 프로젝트 공통 비교 조건 |
| 참조 | 긴 변 512, 비율 유지 후 16배수 중앙 crop | 원본 `preprocess_ref` |
| 정밀도 | 공식 체크포인트 원래 dtype, 주 DiT/T5/CLIP BF16 | FP8·offload 사용 안 함 |
| Style | 일반 단일 참조 입력, prompt prefix 추가 없음 | UNO에는 별도 Style task 분기가 없음 |

원본 `get_schedule`의 sequence-length 계산, CPU/GPU noise 생성 방식, sampler, VAE, LoRA 연산을 수정하지 않는다. 공식 `UNOPipeline`을 직접 호출한다. 실행 wrapper는 고정 로컬 snapshot 경로를 연결하고 checkpoint key 누락/추가나 meta tensor가 남으면 중단한다. T5와 CLIP은 공식 코드에서 지정한 `xlabs-ai/xflux_text_encoders`, `openai/clip-vit-large-patch14`를 사용한다. 모델 revision은 `models.lock.json`에 고정했다. 임의 caption 생성/배경 제거/face crop을 추가하지 않는다.

DreamBench++는 style 180장을 포함한 전체 1,350장을 생성·보존한다. metric 기본값도 전체이며 추후 선택한 경우에만 평가 시 `--selection dbpp-no-style`로 1,170장을 선택할 수 있다.

## 환경과 검증

Modal 실행은 Python 3.11 / torch 2.6.0+cu124 / torchvision 0.21.0+cu124 / transformers 4.45.2 / einops 0.8.1을 고정한다. 이는 공식 pyproject의 허용 범위 안에서 기존 검증 환경을 사용한 것이다. 원본 requirements의 torch 2.4.0 환경과 동일하다고 주장하지 않는다. 추가 생성 의존성은 `requirements.txt`, 공통 제약은 `constraints.txt`, GPU wheel은 `requirements-gpu.txt`에 있다. 사용하지 않는 Gradio·학습 패키지는 설치하지 않는다.

CPU 호환성 검증에는 기존 `dreamo/.venv`를 재사용하며 해당 환경을 수정하지 않았다. UNO GPU 환경은 별도 Modal 이미지다. 실제 축소 UNO transformer/LoRA/reference embedding/sampler 경로, seed, 전처리, 입력 cohort, style 보존, 원본 hash, 비용 선택 및 초기화 VRAM guard를 포함한 7개 테스트가 통과했다.

```powershell
dreamo/.venv/Scripts/python.exe -X utf8 -m unittest discover -s uno/tests -q
freecus/.venv-modal311/Scripts/python.exe -X utf8 -m uno.protocol
freecus/.venv-modal311/Scripts/python.exe -X utf8 -m uno.run --benchmark dreambench --output uno/outputs/dry-db --dry-run
```

## Modal 원격 실행

| 작업 | Profile / workspace | Case 수 | Worker |
|---|---|---:|---:|
| DreamBench | `wj-lee-louis` / `wj-lee-louis` | 750 | 375 × 2 |
| DreamBench++ | `woojin716` / `woojin716` | 1,350 | 675 × 2 |

앱 `uno-method-comparison`, 모델 Volume `uno-models`, 결과 Volume `uno-runs`, Secret `uno-huggingface`를 사용한다. 기존 앱과 기본 profile은 변경하지 않는다. `modal-research-2`와 `woojin716`은 같은 workspace이므로 별도 계정으로 중복 계산하지 않는다.

접수된 workflow ID는 DreamBench `c961a26586b5470da259eb79d1b8c531`, DreamBench++ `563e9df070984928896410ca0d6133bf`다. 배포 앱은 [DreamBench](https://modal.com/apps/wj-lee-louis/main/deployed/uno-method-comparison), [DreamBench++](https://modal.com/apps/woojin716/main/deployed/uno-method-comparison)에서 확인한다. 실제 상태는 아래 `workflow-status`를 기준으로 한다.

두 A100 pilot 모두 성공했고, 두 작업 모두 A100 80GB 2-worker 전체 생성 단계에 진입했다. 초기화 peak reserved는 33.57 GiB, 생성 peak allocated는 37.42 GiB, peak reserved는 42.26 GiB였다. 공식 dtype은 transformer/LoRA/T5/CLIP BF16, autoencoder FP32임을 실제로 확인했다. 42 GiB admission 기준을 넘어서 L40S pilot은 실행하지 않았다. 이는 L40S에서 물리적으로 실행 불가능하다는 판정이 아니라 메모리 여유를 확보하기 위한 이번 작업의 선택이다.

| A100 2-worker 추정 | 전체 생성 시간 | 전체 계산비 | 검증비 포함 | 25% 여유+검증비 |
|---|---:|---:|---:|---:|
| DreamBench | 약 128분 | $13.19 | $13.52 | $16.82 |
| DreamBench++ | 약 216분 | $22.38 | $22.85 | $28.44 |

표의 시간은 전체 생성 시작 기준이며 모델 준비와 pilot 시간은 별도다. 비용에는 GPU+CPU+RAM을 포함하며 모델 준비 CPU/저장/기타 기존 앱 청구액은 별도다. 실측 raw timing과 호출 ID는 `verification/gpu_pilots.json`에 보존한다. 전체 완료를 의미하지 않으며 진행/실패 여부는 원격 status를 확인한다. 전체 생성 후 metric은 별도 평가한다.

원격 coordinator는 CPU 모델 준비 → A100 80GB 2-worker pilot → 메모리 조건을 만족하면 L40S pilot → 비용 선택 → 전체 생성을 연결한다. receipt 저장 이후 PC를 종료해도 다음 단계까지 진행한다. 각 GPU worker는 CPU 2 physical cores / RAM 64 GiB, GPU 1장이다. 두 벤치마크 동시 실행 시 GPU 총 4개다. Pilot은 각 lane/category에서 원본 크기가 가장 큰 참조의 prompt index 0/4를 사용한다. DreamBench 8장 / DreamBench++ 16장이다.

초기화와 생성 peak allocated/reserved가 42 GiB 이하인 경우에만 L40S를 검토한다. CPU/RAM/GPU 합산 비용이 A100보다 3% 이상 낮아야 선택한다. 전체 생성 추정에 25% 여유를 적용하고 계정별 계산 admission은 최대 $38, 나머지 $2는 준비/저장 등의 예비비다. 이는 기존 $40 작업 예산에 맞춘 앱 내부 추정치이며 계정의 실제 남은 크레딧을 확인하거나 하드 청구 상한을 설정한 값은 아니다. 과거 작업 비용은 이 ledger에 포함되지 않는다. 초기 pilot 실패 또는 예상 초과 시 중단하고 로그를 보존한다. 유료 자동 재시도는 하지 않는다.

Worker timeout은 6시간, 내부 시간 제한은 최대 21,000초다. 주기적으로 Volume에 commit한다. source/config/model/manifest/library가 다른 결과는 섞지 않는다. A100과 L40S 결과도 GPU별 폴더로 분리한다.

```powershell
# 최초 배포/접수. 이미 receipt가 있으면 workflow를 중복 실행하지 않는다.
freecus/.venv-modal311/Scripts/python.exe -X utf8 -m uno.modal_client prepare --profile wj-lee-louis --benchmark dreambench
freecus/.venv-modal311/Scripts/python.exe -X utf8 -m uno.modal_client bootstrap --profile wj-lee-louis
freecus/.venv-modal311/Scripts/python.exe -X utf8 -m uno.modal_client deploy --profile wj-lee-louis --benchmark dreambench
freecus/.venv-modal311/Scripts/python.exe -X utf8 -m uno.modal_client workflow --profile wj-lee-louis --benchmark dreambench
# 상태 확인
freecus/.venv-modal311/Scripts/python.exe -X utf8 -m uno.modal_client workflow-status --profile wj-lee-louis --benchmark dreambench
freecus/.venv-modal311/Scripts/python.exe -X utf8 -m uno.modal_client workflow-status --profile woojin716 --benchmark dreambench_plus
# 필요할 때 coordinator와 자식 작업을 함께 중지. 완료 이미지는 보존.
freecus/.venv-modal311/Scripts/python.exe -X utf8 -m uno.modal_client stop --profile wj-lee-louis --benchmark dreambench
```

접수/상태 파일은 `outputs/modal/<PROFILE>/<BENCHMARK>-workflow*.json`에 저장된다. 현재 작업 중 동일 앱을 다시 배포하지 않는다. 수동 stage/pilot/submit 명령과 workflow를 동시에 사용하지 않는다. 최종 output은 `uno-runs`의 `results/<BUNDLE_ID>/<SELECTED_GPU>/lane-{0,1}/`이다. 상태의 `output`이 최종 경로다. 소스에 모델을 복사하거나 결과 폴더에 모델 symlink를 만들지 않는다.

## 결과 수집과 metric

4-workspace 원격 평가에는 `freecus.operations.evaluate_comparison`을 사용한다. 생성 완료 workflow를 확인하고 기존 FreeCus와 같은 T4/torch 2.7.1/CLIP/DINO 환경에서 전체 파일 검증 후 계산한다. receipt가 있으면 `--status`로 진행 상황과 완료 점수를 수집한다. 기본 평가 cohort는 DreamBench 750장, DreamBench++ style 포함 1,350장이다.

```powershell
freecus/.venv-modal311/Scripts/python.exe -X utf8 -m freecus.operations.evaluate_comparison --method uno --benchmark dreambench --status
freecus/.venv-modal311/Scripts/python.exe -X utf8 -m freecus.operations.evaluate_comparison --method uno --benchmark dreambench_plus --status
```

이미지와 run.json/metadata/text/src_image 등을 두 lane 모두 수집한 뒤 `uno.assemble`로 병합한다. FreeCus의 artifact hash/seed/prompt/reference/해상도 검사와 동일한 검증을 수행한다. `uno.evaluate`는 기존 evaluator의 byte-identical 사본을 재사용하며 output condition만 `uno`로 구분한다. 평가 환경은 생성과 분리하고 기존 T4 평가 패키지 `requirements-eval.txt`를 사용한다.

```powershell
freecus/.venv-modal311/Scripts/python.exe -X utf8 -m modal volume get uno-runs results/<BUNDLE_ID>/<SELECTED_GPU> uno/outputs/collected --profile wj-lee-louis
freecus/.venv-modal311/Scripts/python.exe -X utf8 -m uno.assemble --benchmark dreambench --source <FOLDER_WITH_TWO_LANES> --output uno/outputs/evaluation_inputs/dreambench
python -m uno.evaluate --benchmark dreambench --run uno/outputs/evaluation_inputs/dreambench --output uno/outputs/metrics/dreambench.json --device cuda
python -m uno.evaluate --benchmark dreambench_plus --run <MERGED_DBPP_RUN> --output uno/outputs/metrics/dbpp-all.json --device cuda
```

CLIP-T는 원문 target prompt, CLIP-I/DINO-I는 원래 선택한 참조 1장과 전체 생성 이미지를 사용한다. CLIP ViT-B/32와 DINOv1 vits16 및 subject macro 집계를 그대로 유지한다. 생성용 CLIP ViT-L/14와 평가용 CLIP ViT-B/32를 혼동하지 않는다. 최종 metric은 전체 생성 완료 후 별도로 산출한다.
