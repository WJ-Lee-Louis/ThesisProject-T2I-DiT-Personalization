# Standard Benchmark and Metric Plan for Preliminary Novelty Evaluation

Date: 2026-09-07  
Scope: InstantX-IP-Adapter 기반 SSG 및 embedding projection 방법의 초기 정량적 novelty 검증

## 1. 결론

현재 단계의 중심 평가 세트는 다음과 같이 구성하는 것이 적절하다.

1. Benchmark: **DreamBench++에서 action/background 중심으로 고정한 목적 지향 subset**
2. Text controllability: **CLIP-T**
3. Identity preservation: **DINO-I와 CLIP-I**
4. Leakage에 의한 identity score 과대평가 방지: **masked DINO-I와 masked CLIP-I를 함께 보고**
5. 선택적 보조 지표: **ImageReward (IR)**

DreamBench 원 논문은 30개 subject와 25개 prompt에 대해 subject-prompt당 4장을 생성하고, DINO ViT-S/16과 CLIP ViT-B/32 기반 DINO-I, CLIP-I, CLIP-T를 사용했다. 이 조합은 이후 personalization 연구에서 가장 반복적으로 사용된 비교 가능한 최소 평가 세트다. [DreamBooth](https://arxiv.org/html/2208.12242)

DreamBench++는 150개 reference와 1,350개 prompt를 포함하며, 각 reference에 photorealistic 4개, non-photorealistic 3개, complicated/imaginative 2개의 prompt가 연결된다. 공식 benchmark는 GPT 기반 concept preservation과 prompt following 평가를 중심으로 하지만, DINO-I, CLIP-I, CLIP-T도 함께 제공한다. [DreamBench++ paper](https://arxiv.org/html/2406.16855) [official repository](https://github.com/yuangpeng/dreambench_plus)

최근 FLUX 기반 FreeCus는 DreamBench++에서 SAM으로 subject를 분리한 뒤 CLIP-I와 DINO를 계산하고, 전체 생성 이미지와 prompt 사이의 CLIP-T를 사용했다. 이는 현재 연구와 backbone 및 문제 설정이 가장 가까운 선례다. [FreeCus](https://arxiv.org/html/2507.15249)

## 2. 지표의 정확한 의미

### CLIP-I

생성 이미지와 reference 이미지의 CLIP image embedding 사이 cosine similarity다. 전역적인 의미·색상·구성을 반영하므로 subject identity뿐 아니라 동일한 배경과 pose도 점수를 올릴 수 있다.

### DINO-I

생성 이미지와 reference 이미지의 DINO visual embedding 사이 cosine similarity다. DreamBench 원형은 self-supervised DINO ViT-S/16이 같은 class 안의 instance 차이를 비교적 잘 보존한다는 이유로 이를 선호했다. DINO는 vision-only encoder이므로 표준 personalization 설정에 **DINO-T라는 지표는 없다**.

### CLIP-T

생성 이미지의 CLIP image embedding과 target prompt의 CLIP text embedding 사이 cosine similarity다. 현재 연구에서 표준적인 text controllability 주 지표로 사용한다.

### Masked CLIP-I / Masked DINO-I

Reference와 generated image에서 subject 영역을 동일한 segmentation pipeline으로 추출한 뒤 CLIP-I와 DINO-I를 계산한다. 배경 복제가 identity 점수를 올리는 문제를 줄인다. FreeCus는 SAM, FreeGraftor는 GroundedSAM을 활용한 subject-level image alignment를 보고했다. [FreeCus](https://arxiv.org/html/2507.15249) [FreeGraftor](https://arxiv.org/html/2504.15958)

### ImageReward (IR)

Personalization 표에서 `IR`은 대체로 **ImageReward**를 뜻한다. 이는 text-image pair의 인간 선호를 학습한 general T2I reward이므로 identity 지표가 아니라 prompt alignment와 전반적 품질의 보조 지표다. FreeGraftor도 CLIP-T와 함께 text-alignment 항목에 ImageReward를 배치한다. [ImageReward](https://arxiv.org/html/2304.05977) [FreeGraftor](https://arxiv.org/html/2504.15958)

## 3. 구현 시 고정해야 할 evaluator protocol

같은 지표명이라도 논문마다 구현이 다르므로 다음 내용을 manifest에 명시한다.

- CLIP backbone: DreamBench 호환성을 위해 우선 `CLIP ViT-B/32`
- DINO backbone: 문헌 호환용 `DINO ViT-S/16`
- 추가 분석을 위해 DINOv2를 사용하면 별도 열로 보고하고 DINO-I와 혼합하지 않음
- Global score와 masked score를 별도 열로 유지
- Mask pipeline과 crop/padding/background 처리 방식을 모든 condition에서 동일하게 유지
- Reference가 여러 장이면 generated-reference 모든 pair의 평균, 한 장이면 그 한 장과 비교
- CLIP-T에는 identity token을 별도로 발명하지 않고 실제 생성에 사용한 전체 natural-language target prompt를 사용
- Method가 SigLIP 공간을 직접 조정하므로 SigLIP similarity를 주 evaluator로 사용하지 않음
- 모든 condition에서 동일한 prompt, seed, resolution, denoising step, guidance 및 IP scale 사용

권장 결과 열은 다음과 같다.

| Condition | CLIP-T | CLIP-I global | DINO-I global | CLIP-I masked | DINO-I masked | IR optional |
|---|---:|---:|---:|---:|---:|---:|

Global 점수는 기존 문헌과의 호환성, masked 점수는 본 연구의 background leakage 문제에 대한 타당성을 담당한다. 하나를 다른 하나로 대체하지 않는다.

## 4. DreamBench++ subset 구축 절차

### 4.1 데이터만 먼저 검토

공식 DreamBench++ metadata와 preview를 내려받고 이미지 생성 전에 reference-prompt pair를 전수 목록화한다. GPU는 이 과정에 필요하지 않다.

### 4.2 Annotation manifest 작성

각 reference-prompt pair에 다음 필드를 부여한다.

```text
sample_id
reference_id
subject_category
class_noun
source_action
source_background
target_prompt
prompt_structure        # OA / OB / OAB
target_action
target_background
source_action_conflict
source_background_conflict
prompt_style            # photorealistic / stylized / imaginative
eligible
exclusion_reason
```

### 4.3 포함 기준

- 주 subject가 하나이고 identity가 육안으로 구분 가능할 것
- 동일한 class noun을 모든 method에 안정적으로 사용할 수 있을 것
- action 또는 background가 target prompt에 명시적으로 표현될 것
- source-aware 계산을 위해 source action/background를 짧고 일관되게 caption할 수 있을 것
- target action/background가 reference에 이미 존재하는 요소의 단순 복사가 되지 않도록 우선적으로 충돌 pair를 선택할 것
- segmentation이 가능한 크기의 subject일 것

### 4.4 제외 기준

- style 변화만 요구하고 action/background가 없는 prompt
- subject의 종이나 핵심 외형 자체를 바꾸는 property modification
- action의 주체가 불분명한 문장
- 여러 subject가 있어 identity score의 대상이 모호한 이미지
- target 조건이 reference에 이미 완전히 충족되어 method 간 차이를 보기 어려운 pair
- 심하게 가려졌거나 지나치게 작은 subject

### 4.5 균형 구성

초기 novelty 검증에는 총 **60개 reference-prompt pair**를 권장한다.

- O+A: 20 pairs
- O+B: 20 pairs
- O+A+B: 20 pairs
- 최소 20개 서로 다른 reference 사용
- 한 reference가 최대 3개 pair를 넘지 않도록 제한
- 가능하면 animal과 object를 함께 포함하되, 행동 평가는 자연스럽게 행동 가능한 living subject 중심으로 구성

DreamBench++의 9개 prompt를 subject마다 전부 사용할 필요는 없다. 본 subset은 공식 full benchmark가 아니므로 결과에서는 `DreamBench++ Action/Background Subset`처럼 명시하고, 공식 leaderboard 숫자와 직접 동일선상에서 비교하지 않는다.

### 4.6 생성 수와 seed

DreamBench 계열의 표준 관행을 유지하기 위해 pair당 4개 seed를 사용한다.

```text
60 pairs x 4 seeds = 240 images / condition
```

GPU 예산을 줄일 때 seed 수보다 pair 수를 줄이는 편이 낫다. 4-seed 구조를 유지하면 기존 benchmark 관행을 따르면서 stochastic variation을 평균낼 수 있다.

## 5. 현재 3단계 평가 프로세스

### Stage 1: SSG 효과

비교 condition:

1. InstantX baseline
2. InstantX + SSG

판단 기준:

- CLIP-T가 상승하는가
- global/masked CLIP-I와 DINO-I가 유지되는가
- 모든 값은 동일한 reference-prompt-seed pair끼리 paired comparison

필요 생성량: `2 x 240 = 480 images`

### Stage 2: Projection 효과

SSG를 공통 적용하고 다음을 비교한다.

1. SSG only
2. Target projection + SSG
3. Source-aware projection + SSG
4. Target+source matched projection + SSG
5. Target+source all projection + SSG

Stage 1의 SSG 이미지를 재사용하므로 추가 생성량은 `4 x 240 = 960 images`다.

주 결과는 CLIP-T이며, identity score를 함께 보아 단순한 prompt gain이 identity 손실과 교환된 것인지 확인한다. O+A, O+B, O+A+B별 macro average도 반드시 별도로 보고한다. Matched와 all의 차이는 세 prompt 구조가 균형을 이뤄야 해석할 수 있다.

### Stage 3: Generic component removal

Stage 2의 best projection+SSG를 기준으로 소수 generic variant만 비교한다. 시각적 차이가 작았으므로 네 방식을 모두 반복하기보다 서로 다른 가정을 대표하는 두 방식을 우선한다.

권장 후보:

1. `absolute residual`: nuisance 방향과 중복되는 class component를 제거한 뒤 남은 generic residual만 차감
2. `contextual without superclass`: target context 안에서 직접 추정한 object/class 방향을 사용하며 `animal` 같은 추가 superclass 가정이 없음

`absolute direct`는 nuisance와 중복된 방향을 이중으로 약화할 가능성이 있고, `contextual superclass`는 animal-only semantic assumption을 추가하므로 일반 personalization의 초기 대표 variant로는 우선순위가 낮다. 다만 subset을 동물로만 제한한다면 contextual superclass를 두 번째 후보로 교체할 수 있다.

두 variant를 사용하면 추가 생성량은 `2 x 240 = 480 images`다.

### 총 preliminary 생성량

```text
Stage 1:  480
Stage 2:  960 additional
Stage 3:  480 additional
Total:   1,920 images
```

## 6. 집계와 판단 규칙

- 전체 image를 한 번에 평균하지 않고 먼저 subject-prompt별 seed 평균을 계산
- 이후 subject별 macro average와 OA/OB/OAB category별 macro average를 계산
- 동일 pair와 seed에 대한 paired difference를 사용
- CLIP-T, CLIP-I, DINO-I는 각각 따로 보고하며 임의의 단일 평균 점수로 합치지 않음
- best variant는 `CLIP-T 최대`만으로 고르지 않고 masked identity가 baseline 대비 명확히 무너지지 않는 범위에서 선택
- Stage 3 결과는 generic removal의 추가 gain만 평가하며 Stage 1·2의 주 효과와 섞지 않음

현재 프로세스는 preliminary novelty screening으로 논리적으로 타당하다. 다만 Stage 2에서 고른 best method와 Stage 3 결과는 같은 subset에서 선택된 exploratory 결과이므로 최종 논문의 확증적 성능 수치와는 구분해 기록한다.

## 7. 근거 문헌 요약

- DreamBooth: DreamBench, 30 subjects, 25 prompts, 4 samples, DINO/CLIP-I/CLIP-T의 원형 프로토콜. [Paper](https://arxiv.org/html/2208.12242)
- Custom Diffusion: CustomConcept101과 CLIP image/text alignment를 사용한 넓은 concept customization 평가. [Paper](https://arxiv.org/abs/2212.04488) [Dataset](https://github.com/adobe-research/custom-diffusion/blob/main/customconcept101/README.md)
- DreamBench++: 150 references, 1,350 prompts 및 concept preservation/prompt following 평가. [Paper](https://arxiv.org/html/2406.16855) [Code and data guide](https://github.com/yuangpeng/dreambench_plus)
- Personalize Anything: DreamBench에서 CLIP-T, CLIP-I, DINO를 포함하는 DiT personalization 평가. [Paper](https://arxiv.org/html/2503.12590)
- FreeCus: DreamBench++와 masked subject evaluation을 사용하는 FLUX training-free customization 선례. [Paper](https://arxiv.org/html/2507.15249)
- FreeGraftor: DreamBench 3,000-image protocol, masked CLIP-I/DINOv2, CLIP-T/ImageReward 사용. [Paper](https://arxiv.org/html/2504.15958)

