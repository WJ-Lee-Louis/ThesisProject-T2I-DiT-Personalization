# FreeCus와 기존 연구의 비교 기준

## 확인된 기존 설정

활성 입력은 `Orthogonalization-InstantX-IP-Adapter/outputs/semantic_benchmark_gpt56_sol/generation_manifest.json`이다. 2,100개 case가 READY 상태이며 DreamBench 750개, DreamBench++ 1,350개다. 실제 A/B 원격 실행 plan의 각 2,100 jobs에서도 전부 seed 42, 1024×1024, 24 steps, guidance 3.5를 확인했다. 계정별 job 수는 조건 수와 shard 분할을 포함하므로 benchmark pair 수와 혼동하지 않는다.

FreeCus 입력은 이 manifest의 **case ID·선택 참조 이미지·reference ID·원문 target을 그대로 가져온다**. 첫 파일을 새로 고르지 않는다. conditioning cache나 기존 GPT 응답은 FreeCus에 주입하지 않으며, FreeCus 고유의 Qwen 기반 SFC를 사용한다. 참조를 전달하는 방법 내부 crop/resize와 평가 입력의 reference 원본은 구분한다.

별도 공식 자료와 원문 target을 대조한 결과:

| 벤치마크 | 동일한 문장 | 다른 문장 |
|---|---:|---:|
| DreamBench: 공식 prompts/classes와 비교 | 750 | 0 |
| DreamBench++: 공식 배포 captions와 비교 | 1,350 | 0 |

따라서 현재 준비된 공통 데이터에서 **방법별 benchmark target이 서로 다른 문제는 없다**. FreeCus 내부에서 caption을 덧붙여 실제 생성 입력이 달라지는 문제와 구별해야 한다. 근거와 입력 해시는 `verification/project-protocol.json`에 기록했다.

## 생성 입력과 평가 target

공통 요청을 T, 참조 이미지를 I라고 하면:

- 기존 연구: I, T를 받아 기존 source/target 분석과 투영 과정을 수행한다.
- FreeCus 전체 방법: I에서 Qwen2-VL 설명 C(I)를 얻고 Qwen2.5로 필터링한 뒤, `T + '. ' + C(I)`를 생성 prompt로 사용한다.
- 평가: **두 결과 모두 동일한 T와 동일한 원본 I**를 기준으로 평가한다. FreeCus의 확장 prompt나 우리 방법의 내부 caption을 평가 target으로 바꾸지 않는다.

방법마다 입력을 활용하는 계산과 내부 표현이 다른 것은 비교 대상의 일부다. 모든 생성 문자열을 강제로 동일하게 만들기 위해 FreeCus의 SFC를 끄면 전체 FreeCus와의 비교가 되지 않는다. 반대로 FreeCus 결과를 확장 prompt에 대해 평가하면 참조 설명까지 평가 기준으로 들어가므로 우리의 결과와 같은 과제를 평가하지 못한다.

예를 들어 원문 T가 “빨간 모자를 쓴 개”, 참조 설명이 “갈색 털과 검은 코를 가진 개”라면 이 설명은 원문 요구를 보충한다. 원문 T가 “빨간 가방”, 참조 설명이 “파란 가방”이라면 충돌할 수 있다. 그 충돌을 해결하는 능력 역시 전체 방법의 편집 수행 성능이다. 다만 target이 바꾸라고 한 색을 reference similarity가 보존하라고 보상하는 한계가 있으므로 identity 점수와 instruction 점수를 함께 읽어야 한다.

## 주 비교와 원인 분석

| 실험 | 목적 | 보고할 수 있는 결론 |
|---|---|---|
| 기존 전체 방법 vs FreeCus 전체 방법, 공통 1024·seed 42·전체 원문 T | 실제 사용 조건에서 전체 시스템 비교 | 이 공통 조건에서 identity/instruction/cost 측면의 우위 또는 trade-off |
| FreeCus 전체 vs FreeCus w/o SFC | 참조 caption의 이득과 원문 충돌 확인 | SFC가 각 유형에 미치는 영향. w/o SFC는 주 baseline 대체 불가 |
| 우리 방법의 기존 ablation과 동일 case 평가 | 자체 모듈의 기여 확인 | 투영/SSG 등 자체 구성요소 기여 |
| 동일한 참조 caption/동일 MLLM 예산을 사용하는 추가 통제 실험 | language front-end와 주입 메커니즘의 효과 분리 | 조건이 통제된 범위에서 알고리즘 기여. 원본 전체 방법 성능과 별도 표기 |

아래 세 통제 실험은 계획이며 아직 구현/실행하지 않았다. 특히 공통 caption을 기존 연구에 적용하는 실험은 기존 contrast 분석 구조를 바꾸는 별도 구현이 필요하며, 문자열만 바꿔 동일 조건이라고 간주하지 않는다.

참조 caption과 충돌하기 쉬운 property/color 변경, action/pose, background, style, accessory 유형을 **결과를 보기 전에** 정해 층별로 분석한다. 모든 원문을 전 방법에 동일하게 제공하고, 특정 방법이 잘 나온 문장만 고르거나 baseline 프롬프트만 별도 수정하지 않는다. 1,350개 전체와 non-style 1,170개/style 180개를 구분한다. 분류가 없으면 사전 정의된 rubric으로 별도 annotation하고 구현 내부의 source/target 차분을 곧바로 평가 정답으로 쓰지 않는다.

CLIP-T만 높거나 reference similarity만 높다고 전체 우위를 주장하지 않는다. 동일 case별 차이, reference별 평균, category별 성능과 실패율을 함께 보고한다. 신뢰구간은 reference 단위 paired bootstrap으로 산출하는 것이 적합하며, seed 1개 결과는 seed 42 조건의 비교로 해석한다. 같은 seed 정수도 CPU/GPU generator나 pipeline의 난수 소비 순서가 다르면 같은 noise tensor를 뜻하지 않는다. 본 문서 작성에서 bootstrap이나 새 judge를 실행하지 않았다.

기존 방법은 GPT-5.6-sol 기반 분석 cache를 쓰고 FreeCus는 Qwen 7B 모델 두 개를 쓰므로, 전체 시스템 비교에서는 이러한 front-end 차이와 전처리 비용도 함께 공개한다. 전체 시스템 차이를 투영 알고리즘만의 효과라고 해석하지 않는다.

## 해상도 처리

이번 주 비교는 **두 방법 모두 실제 1024×1024 생성**으로 고정한다. FreeCus도 FLUX.1-dev 기반이므로 1024 비교를 준비할 수 있다. 이를 `FreeCus, adapted to 1024`로 명시하고, GPU smoke test로 실제 구동을 확인한다. 512 생성 후 1024 resize는 동일한 조건이 아니다. CLIP/DINO가 각각 224 입력 등으로 전처리하더라도 생성 시 해상도와 잡음 trajectory의 차이가 사라지지는 않는다.

원 해상도에서만 유효한 checkpoint가 있는 다른 방법은 이를 감추지 않고 native-resolution 비교로 따로 보고한다. 모든 방법을 무조건 1024로 바꾸는 것이 항상 공정한 것은 아니다. 본 연구의 공통 1024 track과 각 논문의 원조건 재현 track은 다른 질문에 답한다.

FreeCus 논문의 원조건 재현은 별도로 512, pair당 4장, SAM subject 영역, 논문과 동일한 reference·metric·집계 정책을 요구한다. 이 경로를 이번 주 비교에 섞지 않으며 현재 실행기는 1024만 허용한다. 기존 방법은 24 steps, FreeCus는 원래 30 steps를 유지하고 시간/비용을 함께 보고한다. step 수를 맞추는 통제 실험은 필요할 때 추가한다.

## 기존 evaluator 재사용

주 지표는 기존 `scripts/benchmark_metrics.py`의 `uno-selected-reference-v1`을 파일 해시까지 보존하여 재사용한다.

- CLIP: OpenAI `ViT-B/32`, 고정 revision.
- DINO: DINOv1 **vits16**, 고정 torch.hub revision. 이전 FreeCus 보조 구현의 vits8과 구별한다.
- CLIP-T: 원문 T, 음수 0 clipping, `truncate=True`, scale 1.
- CLIP-I/DINO-I: 생성 full image와 정확히 선택된 reference 한 장, cosine, clipping 없음.
- 원본 생성 PNG, 동일 전처리, reference macro/case micro/category 집계.

FreeCus 논문의 SAM 평가와 다르므로 이것을 논문 점수의 직접 재현이라고 부르지 않는다. 기본 `freecus.evaluate`는 기존 지표를 사용한다. 이전 HF vits8/SAM-view 입력용 구현은 `freecus.evaluate_diagnostic`로 이동해 보조 분석으로만 남겼다.

FreeCus 공개 코드와 논문 사이의 SFC/마스크/ANS 차이를 보정한 현재 `paper` 모드는 저자 승인 구현이 아니다. 논문 방식에 맞춘 해석적 보정이라는 점을 결과에 명시하고, 필요하면 공개 코드 동작 모드 `upstream`과의 민감도 분석으로 결론이 그 보정에만 의존하는지 확인한다. Modal에서 `paper` 생성 작업을 접수했으며 민감도 실험은 수행하지 않았다. Qwen의 고해상도 비전 attention은 dense mask 대신 동일한 이미지별 attention을 fused SDPA로 계산하도록 수정했다. 입력 해상도·토큰·정밀도는 유지하고 수치 동등성 테스트를 통과했지만 kernel 변경에 따른 미세한 부동소수점 차이는 가능하다. 최종 DreamBench와 DreamBench++는 동일한 수정본 source snapshot을 사용하며 GPU 종류는 각각 A100/H100이다. GPU 종류 차이에 따른 bitwise 동일성은 가정하지 않는다.
