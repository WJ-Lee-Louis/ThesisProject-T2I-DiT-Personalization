# Single-subject 비교군 9개 연구의 DreamBench / DreamBench++ 평가 검토

확인일: 2026-10-21 KST. 범위는 사용자 지정 arXiv 논문·공개 부록 및 연결된 공식 저장소다. 논문의 기존 metric 수치를 인용해 순위를 비교하지 않고, 모든 방법의 출력을 직접 생성해 재평가할 때 필요한 차이를 정리했다. 실제 모델 실행과 metric 계산은 수행하지 않았다.

## 1. 결론과 근거 수준

이 연구들은 동일한 평가 프로토콜을 공유하지 않는다. 백본 외에도 **평가 데이터의 범위, full image와 subject 영역, 평가 reference 집합, seed 반복 수, MLLM judge와 rubric, 평균의 분모**가 다르다. 같은 이미지에 같은 이름의 metric을 적용해도 결과가 달라질 수 있다.

이 문서의 ‘논문 명시’와 ‘공개 코드 확인’을 구분한다. 저장소에 생성 코드만 있고 해당 benchmark evaluator가 없으면, 논문 설명만으로 내부 reference 평균·crop·누락 처리를 확정하지 않는다. ‘평가를 확인하지 못함’은 해당 방법이 그 데이터셋에서 실행 불가능하다는 뜻이 아니다. 다른 후속 논문이 재평가한 결과도 원 논문의 평가 프로토콜과 구분한다.

UNO에 관해서는 이전 가이드 설명을 정정한다. 사용자가 개별 확인한 논문 점수 산출 방식은 selected reference 한 장 비교다. 공개 코드는 subject 폴더 전체와의 평균이므로 두 동작을 동일시하면 안 된다. 이번 확인으로 독립 검증한 것은 **공개 코드의 동작**이며, 논문의 실제 점수 계산 실행 로그까지 확보한 것은 아니다. 논문 일반 설명에서 입력 reference가 한 장이라는 사실만으로 평가 reference도 한 장이라고 역추론하지 않는다. [UNO 공개 evaluator](https://github.com/bytedance/UNO/blob/a563432dcfb4caf98df4368facb8102bf97d376c/eval/evaluate_clip_dino_score_single_subject.py), [UNO 논문](https://arxiv.org/html/2504.02160v1).

## 2. 연구별 비교표

‘DB++ 없음’은 검토한 원 논문에서 DreamBench++ 직접 평가를 확인하지 못했다는 의미다. XVerseBench·USO-Bench를 DB++로 간주하지 않는다.

| 연구·검토 버전 | DreamBench single-subject | DreamBench++ single-reference 평가 | 직접 재평가 시 고려할 차이 |
|---|---|---|---|
| **DreamO v5** | 사용. test sample마다 seed 4개 | 원 논문 DB++ 직접 평가 확인 못함 | 생성 결과 배경 제거 후 subject 유사도. Turbo 사용 및 입력 전처리 여부 기록. [§4.1–4.3](https://arxiv.org/html/2504.16915v5#S4) |
| **XVerse v1** | 원 논문 DB 직접 평가 확인 못함 | DB++ 기반 **별도 XVerseBench** | 별도 reference·prompt와 DPG/ID/IP/AES. 공개 IP evaluator는 crop 후보 최대값이며 논문과 기본 DINO 버전도 다름. [§4](https://arxiv.org/html/2506.21416v1#S4) |
| **USO v1** | 부록 평가. prompt당 **6개**, 총 **4,500개** | DB++ 직접 평가 확인 못함 | 주 평가 USO-Bench와 DB 부록을 구분. style·subject 입력 역할이 별도. [부록 F.2.1](https://arxiv.org/html/2508.18966v1) |
| **MOSAIC v1** | DB에서 CLIP-I/T·DINO 보고 | DB++ 직접 평가 확인 못함. 별도 XVerseBench 사용 | reference 선택·seed·crop 세부를 확정할 evaluator 미확인. [§5.1.2, Table 1–2](https://arxiv.org/pdf/2509.01977v1) |
| **DynaIP v2** | DB 직접 평가 확인 못함 | **1,350개**. CP/PF 및 그 곱 | judge는 **InternVL3-78B + Qwen3-VL-32B 평균**. 원 DB++ GPT-4o와 다르며 보수 채점 문구도 있음. [§5.1, §7.1](https://arxiv.org/html/2512.09814v2) |
| **Squeezing v1** | CLIP-I/T·DINO-I 사용 | Table 4의 **subject categories**, **7개 MLLM**의 0~4 subject-consistency 점수 | CP 성격의 평가이며 CP/PF 전체 쌍이 아님. 정확한 subset·분모 확인 필요. [§4.2, Appendix I/J](https://arxiv.org/html/2605.26111v1) |
| **Personalize Anything v1** | **750 test cases**. CLIP-I/T·DINO 외 DreamSim/FID | DB++ 직접 평가 확인 못함 | DINO 인용은 DINOv2. single-subject와 multi-subject SegCLIP-I를 혼동하지 않음. 입력 mask 의존성 있음. [§4.1–4.2](https://arxiv.org/html/2503.12590v1#S4) |
| **FreeGraftor v5** | **30×25×4=3,000개** | DB++ 직접 평가 확인 못함 | GroundedSAM으로 reference·생성 subject를 분리하고 CLIP 및 **DINOv2** 유사도. ImageReward도 추가. [§IV-A3](https://arxiv.org/html/2504.15958v5) |
| **FreeCus v1** | 원 논문 DB 직접 평가 확인 못함 | **animal/human/object 세 범주 평균**, style 포함 전체 평가와 구분 | SAM subject 영역 평가. 기본 생성 512², 30 steps. caption 생성·필터링도 방법 구성요소. [§4.1–4.2, 부록](https://arxiv.org/html/2507.15249v1#S4) |

### 표에서 바로 도출되는 구분

- DB 750은 subject–prompt pair 수다. 같은 750개라도 seed 1/4/6개면 생성 수는 750/3,000/4,500개다. 여러 seed를 평균하는 것과 여러 reference를 평균하는 것은 다르다.
- DB++ 1,350개는 style까지 포함한 혼합 concept 평가다. ‘single-subject’라는 제목 아래에서도 subject-only subset 또는 style 포함 평가가 공존한다.
- FreeCus가 보고한 세 범주와 현재 전체 매니페스트를 대응시키면 **130 reference×9=1,170개**이며 style은 20×9=180개다. 이는 **현재 로컬 데이터 기준의 대응 수**다. 논문의 실제 실행에서 모든 사례가 누락 없이 평가됐다는 추가 증거는 아니다.
- ‘DINO’는 단일 metric 구현명이 아니다. DINOv1/v2, patch 크기, CLS/평균 feature, full image/crop을 별도 필드로 고정해야 한다.

## 3. 논문과 공개 코드에서 추가로 확인한 지점

### 3.1 XVerse: 데이터셋·DINO·검출 후보 선택을 모두 구분

논문은 DB++를 확장한 XVerseBench를 사용한다. 공개 single 매니페스트는 **90행**이며, 이는 DB 750행이나 DB++ 1,350행과 동일하지 않다. `idip_sam-dino_score.py`는 human을 IP 평가에서 제외하고, reference 한 장과 생성 이미지에서 검출한 **최대 3개 후보 중 최고 유사도**를 사용한다. 검출 실패 시 0으로 남긴다. 후보 선택의 최대값은 여러 reference에 대한 평균과 다른 연산이다.

논문은 DINOv2를 기술하지만 `dino.py`의 기본값은 **`facebook/dino-vits16`**, CLS cosine×100이다. `DINO_MODEL_PATH` 환경변수로 바꿀 수 있으므로, 논문 실제 실행에서 무엇을 사용했는지는 환경·가중치 기록 없이 확정할 수 없다. 이 코드를 DB++ evaluator로 이름만 바꿔 재사용하지 않는다.

근거: [single manifest](https://github.com/bytedance/XVerse/blob/65e2581aac0faffe4607881a10cc486186f6499d/eval/tools/XVerseBench_single.json), [IP 평가 경로](https://github.com/bytedance/XVerse/blob/65e2581aac0faffe4607881a10cc486186f6499d/eval/tools/idip_sam-dino_score.py), [DINO loader](https://github.com/bytedance/XVerse/blob/65e2581aac0faffe4607881a10cc486186f6499d/eval/tools/dino.py), [실행 설정](https://github.com/bytedance/XVerse/blob/65e2581aac0faffe4607881a10cc486186f6499d/eval/eval_scripts/run_eval_single.sh).

### 3.2 MLLM 평가: DB++ rubric을 인용해도 동일한 metric은 아님

DynaIP의 Fig. 9–10은 0~4 rubric을 사용하지만, 불확실할 때 낮은 점수를 주도록 하는 보수 채점 지시가 포함되어 있다. 2개 open MLLM의 평균까지 포함하므로 원 DB++ GPT-4o evaluator와 같은 이름·같은 수치 척도로 취급하지 않는다. 공개 PDF p.23의 rubric도 시각적으로 확인했다. [DynaIP §5.1/§7.1 및 Fig. 9–10](https://arxiv.org/pdf/2512.09814v2#page=23).

Squeezing의 7개 judge 평균은 GPT-4o 단독 점수가 아니다. Table 4는 subject consistency이며 PF를 함께 측정한 전체 DB++ 평가로 해석하지 않는다. Appendix J의 추가 benchmark에서는 human face/identity 제외를 명시한다. 이 사실만으로 Table 4의 정확한 category 목록까지 확정하지는 않는다. 재현 시 실제 평가 case 목록을 확보해야 한다. [Squeezing Table 4, Appendix I/J](https://arxiv.org/html/2605.26111v1).

DreamO의 **0~5는 별도 인간 사용자 평가**다. DB++의 CP/PF 0~4를 바꾼 GPT rubric이라는 근거가 아니다. [DreamO §4.4](https://arxiv.org/html/2504.16915v5#S4.SS4).

### 3.3 평가 전처리와 생성 전처리를 분리

평가 시 foreground를 분리하는 것과, 모델 입력의 배경을 제거하거나 source caption을 만드는 것은 구분한다. 전자는 metric 정의이고 후자는 방법 자체의 조건 입력이다. USO의 style reference는 subject reference와 입력 역할이 다르고, Personalize Anything의 mask 및 FreeCus의 caption 처리처럼 고유 전처리를 제거하면 원래 방법을 평가한 것인지 불명확해진다.

공통 비교는 같은 **원본** reference를 제공한 뒤, 각 방법의 명시된 전처리를 기록하는 방식이 적절하다. 평가에는 공통 원본 target을 사용하고, 내부 caption·prompt 확장·identity binding 입력을 별도 보존한다. 근거: [USO 입력 예제](https://github.com/bytedance/USO/blob/6587514aa3adf8e8f46e5f7e804239651d30b32d/README.md), [Personalize Anything mask 안내](https://github.com/fenghora/personalize-anything/blob/30e3b773a7723e525551bba65b77e49f7328b61b/README.md), [FreeCus caption 구현](https://github.com/Monalissaa/FreeCus/blob/1f33f7111c1536b72de8775fd2b478f3d31d19ac/qwen_for_subject.py).

## 4. 공개 artifact가 확인해 주는 범위

다음은 확인 시점의 commit이다. 해당 tree와 README를 확인했으며 XVerse는 위 evaluator 내부까지 대조했다. 다른 6개 저장소에서는 해당 DB/DB++ metric evaluator를 확인하지 못했다. 학습·생성 코드 공개와 논문 metric 구현 공개는 다르다.

| 방법 | 공식 저장소 | 확인 commit | 평가 재현 시 남는 불확실성 |
|---|---|---|---|
| DreamO | [bytedance/DreamO](https://github.com/bytedance/DreamO/tree/461e35ba6b068f42ca53b43f289b3d91b7f911cd) | `461e35ba6b06` | reference 집합·배경 제거 구현 및 seed 집계 세부. README Turbo 기본값과 버전 기록 |
| XVerse | [bytedance/XVerse](https://github.com/bytedance/XVerse/tree/65e2581aac0faffe4607881a10cc486186f6499d) | `65e2581aac0f` | 별도 benchmark. 실제 DINO 환경 override·가중치 |
| USO | [bytedance/USO](https://github.com/bytedance/USO/tree/6587514aa3adf8e8f46e5f7e804239651d30b32d) | `6587514aa3ad` | DB 부록의 metric 구현·평가 reference 집합 |
| MOSAIC | [bytedance-fanqie-ai/MOSAIC](https://github.com/bytedance-fanqie-ai/MOSAIC/tree/b3c90086d28778fb8bba2f0e372d2f99ceca0913) | `b3c90086d287` | evaluator 미확인. README 공개 checkpoint는 512², 1024²는 미완료로 표시 |
| Personalize Anything | [fenghora/personalize-anything](https://github.com/fenghora/personalize-anything/tree/30e3b773a7723e525551bba65b77e49f7328b61b) | `30e3b773a772` | 평가 reference 집합·seed 반복·single-subject crop 세부 |
| FreeGraftor | [Nihukat/FreeGraftor](https://github.com/Nihukat/FreeGraftor/tree/b8e0f93e91753265e46e9d0b364daa38e25f1afa) | `b8e0f93e9175` | GroundedSAM 후처리·실패 처리·DINOv2 checkpoint 세부 |
| FreeCus | [Monalissaa/FreeCus](https://github.com/Monalissaa/FreeCus/tree/1f33f7111c1536b72de8775fd2b478f3d31d19ac) | `1f33f7111c15` | category macro/image micro 평균 여부, SAM 후처리·reference 평균 세부 |

DynaIP와 Squeezing은 논문·공식 연결 페이지에서 재현에 사용할 공식 코드/가중치 artifact를 이번 조사로 확보하지 못했다. 이를 공개되지 않았다고 단정하지 않는다. 동명이인 inertial-pose DynaIP 저장소는 이 연구와 무관하다. Squeezing의 [공식 프로젝트 페이지](https://zsh2000.github.io/squeeze-mllm-subject-gen/)는 논문 링크와 결과를 제공한다.

MOSAIC는 사용자 지정 **arXiv v1**을 검토했다. ICLR 최종본 OpenReview PDF는 접근 확인 절차로 열람되지 않아 최종본 변경까지 확인했다고 주장하지 않는다. DreamO/FreeGraftor/DynaIP는 각각 현재 arXiv v5/v5/v2를 사용했다. DynaIP v1도 judge 2개 평균 설명을 교차 확인했다. 이후 실험에는 논문 version과 코드 commit을 함께 고정한다.

## 5. 이번 프로젝트에 권고하는 공통 평가 규약

아래는 논문 전체가 합의한 표준이라는 주장이 아니라, 이번 직접 비교를 위한 **제안**이다. 기존 source-caption/차분 cache와 생성 설정을 변경하지 않는다.

### 5.1 비교할 데이터와 reference

1. **DreamBench 주 평가:** 기존 selected reference 30장·원문 target 750개를 전 방법에 동일하게 제공한다. CLIP-I/DINO-I는 각 생성 결과와 해당 입력 reference 한 장을 비교한다. 평가 파일명에는 `ref_scope=selected_one`을 기록한다.
2. **DreamBench 보조 평가:** 필요할 때만 전체 DreamBooth reference 집합 평균을 추가한다. `ref_scope=all_subject_refs`로 이름을 나누고 같은 열에 혼합하지 않는다. 이미 생성된 결과로 다시 계산하므로 이 변경 자체는 재생성이 필요 없다.
3. **DreamBench++:** 전체 150×9=1,350을 mixed concept track으로 유지한다. 동시에 non-style 130×9=1,170과 style 20×9=180을 분리한다. animal/human/object도 별도 보고한다. 해당 방법이 style task를 지원하지 않으면 그 한계를 명시하고, 누락된 style을 제외한 평균을 전체 1,350 평균으로 표시하지 않는다.
4. **Squeezing 등 제한된 대상 범위:** 전체 평가에 없는 범주를 자동 삭제하지 않는다. 필요한 경우 모든 방법이 공유하는 subset을 추가로 정의하고 exact case ID 목록과 분모를 남긴다.

평가 reference 수는 생성 입력 이미지 수와 독립된 필드다. foreground crop 수도 별도다. 세 축을 한 가지 ‘reference 수’로 축약하지 않는다.

### 5.2 metric과 영역

| 항목 | 공통 주 평가 제안 | 추가 분석 |
|---|---|---|
| CLIP-I | 동일 checkpoint/processor, selected reference와 생성 full image cosine | 공통 foreground 처리 후 별도 `FG-CLIP-I` |
| CLIP-T | **원문 target**과 생성 full image. clipping·scale·긴 prompt 처리 고정 | caption/negative/확장 prompt를 평가 target으로 대체하지 않음 |
| DINO-I | v1/v2 모델·patch·feature·resize를 명시해 모든 방법에 동일 적용 | 두 버전을 모두 보고할 경우 열 분리 |
| CP/PF | 전 방법에 **동일 judge snapshot + 정확히 동일 rubric + image encoding** | 대체 judge는 별도 평가 열 |
| 집계 | case별 raw score를 저장하고 reference별 평균 후 전체 평균 | category macro와 image micro를 구분 |

foreground 지표를 추가할 때는 segmentation 모델뿐 아니라 배경 채우기, bbox crop·padding, 후보 선택, 실패 시 처리까지 고정한다. 예를 들어 생성 subject 검출 실패는 0점으로 두고 detection coverage를 함께 보고하는 규칙을 사전에 정할 수 있다. 실패 이미지를 조용히 제외하거나 방법마다 유리한 후보 선택을 달리하지 않는다. Style reference에는 subject foreground 분리가 부적절할 수 있으므로 이 지표의 적용 범위를 non-style로 제한한다.

이번 연구가 배경·행동 방향 제거를 다루므로 full-image 유사도와 foreground 유사도를 함께 보는 것이 유용하다. 단, foreground 점수만 올랐다는 이유로 원래 target의 배경·행동 요구를 만족했다고 해석하지 않는다. PF/CLIP-T와 대응시킨다.

CP/PF는 0~4 raw, raw 평균/4, 필요 시 그 둘의 곱을 구분해서 저장한다. 원 DB++ 집계 관례를 따를 때는 **집계 CP × 집계 PF**이며 이미지별 곱의 평균으로 바꾸지 않는다. 2/7개 judge 평균도 GPT-4o 단독과 구분한다. [원 DB++ 집계](https://github.com/yuangpeng/dreambench_plus/blob/66ca27e52b5e7daa350bdb92ac8c60d4a9339c5e/dreambench_plus/benchmarking.py).

### 5.3 생성 비교의 공정성과 분석

처음에는 seed 1개로 전 방법을 동일 조건에서 pilot하고, 본 평가는 가능한 공통 seed 집합으로 반복한다. seed 정수만 같아도 서로 다른 pipeline이 같은 noise tensor를 사용한다는 보장은 없다. best-of-N 선택 없이 모든 seed를 같은 방식으로 평균한다. DB seed 4개면 방법당 3,000장, DB++ seed 4개면 5,400장이다.

모든 방법을 24 steps·동일 guidance로 강제하는 것만으로 공정해지지는 않는다. Turbo처럼 권장 sampler·step 수가 다른 경우가 있다. 공통 해상도/입력/평가기를 고정한 비교와 각 방법 권장 설정 비교를 구분하고 설정을 명시한다. MOSAIC 512² checkpoint를 1024² 생성과 같은 조건으로 간주하지 않는다. caption·mask·추가 MLLM 및 전처리 비용도 추론 비용에 포함한다.

결과에는 `method`, `checkpoint`, `case_id`, 원본 reference hash, 평가 reference 집합, 원문 target, 내부 확장 prompt/caption, mask 설정, seed, 원본 이미지 크기, evaluator revision, raw score, 실패 유형을 남긴다. 이미지별 paired 차이와 reference 단위 bootstrap을 사용하면 reference 하나의 여러 prompt를 독립 표본처럼 세는 문제를 줄일 수 있다.

기존 `092-08`, `154-07` fallback은 전체 평가에서 플래그를 유지하고, 별도 민감도 분석에서만 모든 방법에 같은 두 case 제외 규칙을 적용한다. 평균에서 조용히 누락하지 않는다.

## 6. 이번 문서 작성으로 변경한 것

- 이 비교 검토 문서를 추가했다.
- 상위 [Modal 실행 가이드](../DREAMBENCH_EVALUATION_MODAL_GUIDE.md)의 UNO 설명과 추천 주 평가를 정정했다. 기존 UNO 명령은 공개 코드 all-reference **보조 평가**로 명명했다.
- 생성 코드·차분 cache·reference·target·Modal 인증 설정은 변경하지 않았다. 공통 selected-reference evaluator나 9개 방법별 생성 wrapper를 구현·실행한 상태는 아니다.

실험 착수 전 남는 작업은 공통 evaluator의 위 필드 고정, 방법별 artifact 확보, style/human 지원 범위 확인, 동일 pilot manifest에 대한 실제 인퍼런스 검증이다. 공식 코드가 없어서 알 수 없는 논문 내부 세부를 ‘같다’고 가정하는 대신, 직접 비교에서는 공통 규약으로 명시해 통제한다.
