# Diptych / XVerse: 준비 초안 및 해상도 검토

최신 XVerse DreamBench++ 실행: [세 workspace 분할 안내](../xverse/DBPP_SPLIT.md). wj-lee-louis 459장·woojin716 216장·louislee-85468 675장으로 제출하며, 각 예산 상한은 사용자 승인된 $5 여유를 포함해 $21/$13/$28이다. 기존 louislee DreamBench는 별도 유지한다. 아래 이전 예산 검토보다 이 추가 지시가 우선한다.

2026-11-03: 사용자가 해상도 검토 후 **Diptych와 XVerse의 Modal 실행을 재개하도록 요청**했다. 두 방법 모두 최종 출력 768×768로 설정했으며, FreeCus·DreamO·UNO 재실행은 보류한다. 기존 1024 초안은 실행하지 않았고 source snapshot만 이력으로 보관한다. 원본 코드는 `upstream/`, 파일별 checksum은 각 `UPSTREAM.json`, 새 실행 설정은 각 `config.json`에 기록한다.

Diptych는 `method-comparison` profile의 `louislee-85468`, XVerse는 `method-comparison-dbpp` profile의 `louislee`에서 실행한다. 신규 실행 총예산은 각 $40다. Diptych는 H100 시범 완료 후 예산 검토로 전체 생성 대기, XVerse는 A100-80GB 최대 4개 worker로 DreamBench 실행을 제출했다. 원격 workflow가 CPU 모델 준비 → category pilot → 예산 검토 → DreamBench 750장 → metric → DreamBench++ 1350장 → metric 순서로 진행한다. 장당 실측으로 계산한 예상 잔여 비용이 한도를 넘거나 pilot이 실패하면 상태를 기록하고 종료한다. 로컬 PC 종료와 무관하게 배포된 workflow 및 spawn된 작업은 Modal에서 계속 실행된다. 세부 실행 상태는 각 `outputs/workflow.json`, `outputs/workflow-status.json`과 원격 volume의 `workflows/`에 기록한다. 예산은 신규 작업의 추정 compute 상한이며 계정의 기존 사용액이나 workspace 결제 한도를 변경하지 않는다.

A100 시범은 Diptych 80.86초/장·peak reserved 44.64 GiB, XVerse 31.81초/장·48.64 GiB였다. H100 시범은 각각 42.41초/장·44.66 GiB, 31.16초/장·48.67 GiB였다. XVerse는 속도 개선이 작아 장당 비용이 낮은 A100으로 복귀했다. 현재 XVerse admission은 이전 시범 몫 $2.5를 제외한 $37.5다. Diptych DreamBench의 안전 여유분 포함 추정은 약 $49로 현재 한도를 넘는다. 전체 DB+DB++ 추정은 Diptych 약 $140, XVerse 약 $75이며, 추가 예산은 사용자에게 질의 중이다. 응답 전에는 기존 각 $40를 유지한다.

## 확인한 출력 해상도

| 방법 | 확인 근거 | 주 비교용 제안 | 기존 로컬 결과 |
|---|---|---|---|
| FreeCus | 논문 §4.1: 512×512 추론 | 512×512 재생성 | 1024×1024 |
| DreamO v1.1 + Turbo | 공식 app.py 출력 기본값 1024×1024; 최신 논문 §4.1도 1024 결과 설명 | 1024×1024 유지 | 1024×1024 |
| UNO | 공식 inference.py 기본값 512×512, README 예제 704×704; 논문 벤치마크 출력 크기로 단정할 근거는 불충분 | 512×512 기본값 비교를 추가하고 근거 표시 | 1024×1024 |
| Diptych Prompting | 논문 §4.1: 1536×768 캔버스 / 768×768 출력 패널. §I는 2048×1024 생성 불가를 명시 | 768×768 출력 패널 | 미실행 |
| XVerse | 공식 inference_single_sample.py: 768×768 기본값; 확인한 논문 본문에는 평가 출력 크기 명시 없음 | 768×768 공식 CLI 기본값을 우선 후보로 사용 | 미실행 |

DreamO의 참조 이미지 기본 크기 512와 출력 1024는 서로 다른 설정이다. UNO README의 512 bucket 학습 크기 역시 그 자체로 논문 평가 출력 크기를 증명하지 않는다. XVerse config의 val_target_size 512를 CLI 출력 기본값 768과 혼동하지 않는다.

Diptych 논문은 1024 패널 문제를 FLUX의 생성 가능한 해상도 한계로 기술한다. 이를 단순 GPU VRAM 문제라고 한정하거나, 더 큰 GPU면 해결된다고 보장할 근거는 없다. 공개 코드가 크기를 인자로 받는다는 사실과 논문에서 해당 크기의 결과를 검증했다는 사실은 구분한다. 공식 코드의 8-pixel border를 유지하면 768 패널의 내부 canvas는 1568×784이며, 최종 오른쪽 패널과 border crop 후 768×768이다.

## 평가 원칙

방법별 공식 권장/기본 해상도의 비교를 주 결과로 하고, 기존 공통 1024 결과는 별도 통제 실험으로 보존한다. FreeCus와 UNO는 새 해상도로 재생성이 필요하고, DreamO는 해상도 변경을 위한 재생성이 필요하지 않다. 기준점인 InstantX 및 Ours의 1024 결과도 별도 근거 없이 축소 재생성하지 않는다.

참조 선정, 원본 target prompt, seed42, case당 1장, CLIP ViT-B/32, DINOv1 vits16, 전처리와 subject macro 집계는 동일하게 유지한다. DreamBench는 750 cases, DreamBench++는 style을 포함한 1350 cases 생성을 기본으로 하며, FreeCus의 이전 no-style 집계는 그 선택 범위를 명시한다. 완전한 공통 집계에는 동일한 case/reference subset이 필요하다.

CLIP은 shorter-edge 224 후 center crop 224, DINO는 shorter-edge 256 후 center crop 224를 사용하므로 서로 다른 원본 출력 크기의 지표 계산은 가능하다. 하지만 생성 해상도가 구성·디테일을 바꾸는 효과는 남는다. 기존 PNG를 리사이즈하는 것으로 다른 해상도에서의 재생성을 대체하지 않는다. 초해상도 보정도 별도 방법이 추가되는 것이므로 기본 평가에는 포함하지 않는다.

기존 shared validator는 1024×1024를 강제하므로 `comparison_candidates/evaluation.py`에 768×768 전용 검증 경로를 분리했다. 기존 evaluator 원본과 점수는 변경하지 않는다. 해시, case·seed·prompt·reference, 완전성 검사를 유지한다. 생성 config에는 출력 크기와 Diptych 내부 캔버스를 별도로 기록한다. 기존 평가 모델/전처리/집계 코드를 그대로 사용하되 `generation_resolution`을 추가 기록한다.

권장 해상도만 맞춰도 논문의 보고 수치를 직접 재현하는 것은 아니다. 예를 들어 FreeCus는 논문에서 pair당 4장과 SAM 기반 subject 평가를 사용하며, 현재 프로젝트는 1장·선택 참조 1장·전체 이미지 지표다. DreamO v1.1과 원논문 모델 버전의 차이도 별도로 보고한다. 이번 비교는 공통 평가 조건의 방법별 기본 설정 비교로 명명하는 것이 적절하다.

Diptych subject에는 공식 Grounding DINO+SAM 배경 제거, attention 1.3, controlnet 0.95, 30 steps, guidance 3.5를 사용한다. style에는 부록 G대로 배경 제거와 attention enhancement를 사용하지 않고, 원본 subject_word를 왼쪽 style 설명으로 사용한다. XVerse는 공식 single-sample 함수와 pretrained modulation adapter/LoRA를 사용하며 28 steps, conditioning 256, IP weight 5, human ID weight 3을 유지한다. XVerse CLI의 기본 int8-quanto 대신 공식 지원 unquantized BF16 branch를 선택했다. 두 방법 모두 최종 출력은 768이며, 원본 target prompt로 평가한다.

## InstantX 해상도 근거 정정

InstantX 모델 카드에는 1024×1024가 유일한 권장 출력이라는 명시는 없다. 다만 공식 `pipeline_flux_ipa.py`에서 `default_sample_size=64`, VAE block 4개에 따른 `vae_scale_factor=16`, 기본 height/width 계산을 확인했으므로 **정사각형 기본 출력은 1024×1024**이다. 공식 예제의 명시적 출력은 width=960, height=1280이다. 따라서 프로젝트의 1024 정사각형 실험은 파이프라인 기본 출력에 부합한다. `resize_img`의 min_side=1024/max_side=1280은 참조 이미지 전처리 설정이며 출력 권장 크기의 근거로 사용하지 않는다.

- 모델 카드 및 공식 예제: https://huggingface.co/InstantX/FLUX.1-dev-IP-Adapter
- 참조 전처리와 추론 함수: https://huggingface.co/InstantX/FLUX.1-dev-IP-Adapter/blob/main/infer_flux_ipa_siglip.py
- 프로젝트가 고정한 revision의 파이프라인 기본값: https://huggingface.co/InstantX/FLUX.1-dev-IP-Adapter/blob/e44c6d889c951cac03ac806991e8d46c9ce1ddba/pipeline_flux_ipa.py

## 일차 출처

- Diptych 논문 §4.1 및 §I: https://arxiv.org/html/2411.15466v2
- FreeCus 논문 §4.1: https://arxiv.org/html/2507.15249
- DreamO 논문 §4.1: https://arxiv.org/html/2504.16915
- DreamO 공식 출력 기본값: https://github.com/bytedance/DreamO/blob/main/app.py
- DreamO 참조 크기 설명: https://github.com/bytedance/DreamO/blob/main/dreamo_v1.1.md
- UNO 공식 기본값: https://github.com/bytedance/UNO/blob/main/inference.py
- UNO README 예제 및 Notes: https://github.com/bytedance/UNO/blob/main/README.md
- XVerse 공식 기본값: https://github.com/bytedance/XVerse/blob/main/inference_single_sample.py
- XVerse 논문: https://arxiv.org/html/2506.21416v1

복제한 원본 commit과 파일별 SHA256은 각 방법의 `UPSTREAM.json`에 보관했다. 이 초안과 문서는 아직 Git에 stage/commit/push하지 않았다.


