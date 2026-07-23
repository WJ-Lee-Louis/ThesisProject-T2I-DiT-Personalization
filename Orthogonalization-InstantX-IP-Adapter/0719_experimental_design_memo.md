# InstantX FLUX IP-Adapter Textual Subspace 실험 설계 메모

## 1. 문서 목적

이 문서는 InstantX FLUX IP-Adapter의 reference-image leakage를 줄이기 위해, SigLIP 공간에서 textual subspace를 구성하고 reference visual pooled embedding에서 해당 성분을 제거하는 다섯 가지 실험 방법을 구체적으로 설명한다.

다섯 방법은 다음과 같다.

1. **T1 — Raw token hidden-state basis**
2. **T2 — Text-head projected token basis**
3. **P1 — Absolute pooled phrase basis**
4. **P2 — Pooled span-difference basis**
5. **P3 — Factorial action/background basis**

최종 목표는 다음 세 성능 사이의 균형을 찾는 것이다.

- reference animal의 **개체 정체성(identity)** 보존
- target prompt의 **동작·자세·배경** 준수
- reference의 원래 동작·자세·배경이 생성물로 복사되는 **source leakage** 감소

본 메모의 실험은 모두 model parameter를 업데이트하지 않는 training-free/tuning-free inference intervention을 전제로 한다.

---

## 2. Harmony와 본 실험의 차이

Harmony의 BLIP-D 경로와 ELITE global stream에서는 visual pseudo-token과 text token을 동일한 CLIP text transformer에 함께 넣은 뒤, 최종 contextual token sequence에서 visual row와 text row를 선택한다. 논문식은 text rows가 형성하는 subspace와 겹치는 성분을 visual rows에서 제거하지만, 공개 구현은 여러 projection의 평균만 빼므로 rank가 2 이상이면 해당 성분을 완전히 제거하지 않고 감쇠한다. ELITE의 별도 local visual K/V branch는 이 orchestration의 대상이 아니다.

즉 Harmony의 개념적 연산은 다음과 같다.

```text
reference image → pretrained mapper/Q-Former → visual pseudo-tokens
text prompt                                → text tokens

[visual pseudo-tokens ; text tokens]
                ↓ shared CLIP text transformer
[contextualized visual rows ; contextualized text rows]
                ↓
contextualized text rows로 basis 구성
                ↓
contextualized visual rows에서 해당 성분 제거
```

반면 InstantX는 reference image의 SigLIP vision `pooler_output` 하나를 기존 MLP에 넣어 128개의 IP token으로 확장한다.

```text
reference image
    ↓ SigLIP vision tower
visual pooler_output [B, 1152]
    ↓ 본 실험의 subspace projection 위치
edited visual pooler_output [B, 1152]
    ↓ InstantX image projection MLP
IP tokens [B, 128, 4096]
```

따라서 본 실험은 Harmony의 contextual token orchestration을 그대로 복제하는 것이 아니라, **SigLIP visual-pooler 단계의 candidate nuisance filtering**으로 보는 것이 정확하다. T1과 T2는 Harmony의 token 단위 구성을 형태적으로 모방하지만 cross-modal 정렬은 검증해야 하는 실험이고, P1~P3는 SigLIP이 실제 image–text contrastive alignment에 사용하는 pooled ambient space를 활용하는 실험이다.

참고 자료:

- [Harmony 논문](https://arxiv.org/abs/2403.14155)
- [Harmony 공식 구현](https://github.com/ldynx/harmony-zero-t2i)
- [InstantX FLUX IP-Adapter 구현](https://huggingface.co/InstantX/FLUX.1-dev-IP-Adapter/tree/main)
- [SigLIP 구현](https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/modeling_siglip.py)

---

## 3. 공통 표기와 projection 방식

예시는 다음 target prompt를 기준으로 한다.

> **“a spotted dog running in a snowy field”**

의미 요소를 다음처럼 나눈다.

- Identity/class: `a spotted dog`
- Action/behavior: `running`
- Background: `in a snowy field`

SigLIP text pooled embedding을 다음과 같이 정의한다.

$$
e(p)=\mathrm{normalize}\big(E_T^{\text{pool}}(p)\big)
$$

Reference image의 raw SigLIP vision pooled embedding은 다음과 같다.

$$
v=E_I^{\text{pool}}(I_{\text{ref}}),
\qquad
\hat v=\frac{v}{\|v\|}
$$

여기서 `raw`는 vision tower 내부의 post-LayerNorm과 attention pooling은 거쳤지만, full `SiglipModel.forward()`의 contrastive L2 normalization은 거치지 않은 `SiglipVisionModel.pooler_output`을 뜻한다. InstantX image projection MLP에는 input normalization이 없고, 생성된 IP token 쪽에 output LayerNorm이 있다.

각 방법으로 textual direction들을 만든 뒤 열 방향으로 쌓은 행렬을 $D$라 한다. Near-collinear direction에서 불필요한 basis가 생기지 않도록 FP32 SVD 또는 rank-revealing QR을 사용하고, singular-value threshold를 통과한 유효 방향만 유지한다.

$$
D=U\Sigma V^\top,
\qquad
B=U[:,\,\sigma_i/\sigma_1>\tau]
$$

기본 soft projection은 다음과 같다.

$$
\hat v' = \hat v-\lambda B(B^\top \hat v)
$$

InstantX MLP는 L2-normalized vector가 아니라 raw vision `pooler_output` 분포를 입력으로 학습되었으므로, 기본 설정에서는 원래 norm을 복원한다.

$$
v'=\|v\|\frac{\hat v'}{\|\hat v'\|+\epsilon}
$$

### 공통 하이퍼파라미터

| 항목 | 초기 권장값 | 설명 |
|---|---:|---|
| Projection strength $\lambda$ | 0.25, 0.5, 0.75, 1.0 | identity–editability Pareto curve 확인 |
| 최대 basis rank | 1, 2, 4 | threshold 이후 유효 singular direction을 작은 rank부터 truncation |
| Relative SVD threshold $\tau$ | $10^{-3},10^{-2}$ | development에서 고정하고 test에서는 변경하지 않음 |
| QR/SVD dtype | FP32 | 수치 안정성 확보 |
| MLP 입력 dtype | 기존 BF16 | projection 후 원래 dtype으로 복귀 |
| Norm restoration | On/Off 모두 | MLP input distribution 영향 확인 |
| Projection aggregation | Sum/Mean 모두 | 논문식 full projection과 공개 코드식 soft projection 비교 |

Harmony 논문의 수식은 basis projection의 합을 제거하지만, 공개 코드는 여러 projection의 평균을 제거한다. 따라서 basis가 $r$개일 때 공개 코드식 연산은 대략 full projection의 $1/r$ 강도에 해당한다. 본 실험에서는 이 차이를 `projection_reduce=sum`과 `projection_reduce=mean`으로 분리해야 한다.

단, `mean`은 새로운 geometry가 아니라 정규직교 basis에서 사실상 $\lambda/r$로 강도를 낮춘 연산이다. 그러므로 `sum`과 `mean`의 우열을 주장할 때는 같은 $\lambda$만 비교하지 말고, 실제 edit magnitude $\|v'-v\|$ 또는 edit angle을 맞춘 control을 함께 둔다.

---

## 4. T1 — Raw token hidden-state basis

### 4.1 쉬운 설명

Target prompt를 SigLIP text encoder에 넣고, `running` 또는 `snowy field`에 해당하는 최종 token hidden state를 직접 textual basis로 사용하는 방법이다.

이 문서에서 `raw token`은 **text projection head 이전**이라는 뜻이다. 실제로 사용하는 public `last_hidden_state`는 final LayerNorm까지 지난 값이며, encoder 내부의 pre-norm raw activation을 뜻하지 않는다.

```text
“a spotted dog running in a snowy field”
               └─────┘       └─────────┘
               action token   background token span
```

Harmony가 contextualized text token을 basis로 사용했다는 형태를 가장 직접적으로 모방한다.

### 4.2 구체적인 구성

SigLIP text encoder의 final-LayerNorm 이후, text head 이전 token hidden states를 다음과 같이 둔다. 이 checkpoint의 기본 text shape은 `[K, 64, 1152]`이다.

$$
H=[h_1,h_2,\ldots,h_T],
\qquad h_i\in\mathbb R^{1152}
$$

Action token span이 $S_A$, background token span이 $S_B$라면 다음 방향을 만든다.

$$
d_A^{\text{raw}}
=\mathrm{normalize}
\left(\frac{1}{|S_A|}\sum_{i\in S_A}h_i\right)
$$

$$
d_B^{\text{raw}}
=\mathrm{normalize}
\left(\frac{1}{|S_B|}\sum_{i\in S_B}h_i\right)
$$

Action만 제거할 때는 $D=[d_A^{\text{raw}}]$, action과 background를 함께 제거할 때는 $D=[d_A^{\text{raw}},d_B^{\text{raw}}]$로 구성한다.

### 4.3 이 방법이 의미하는 것

`running` token row가 표현하는 방향과 reference visual pooled embedding이 겹치는 부분을 action-related component라고 가정한다. Background도 같은 방식으로 해석한다.

### 4.4 장점

- 구현이 단순하다.
- action 또는 background를 token/span 단위로 개별 조절할 수 있다.
- 짧은 `identity + one action` prompt에서 Harmony와 비슷한 granularity를 시험할 수 있다.

### 4.5 한계

- SigLIP image pooled embedding과 직접 대조학습된 것은 개별 token state가 아니라 text pooled output이다.
- 같은 1152차원이라는 이유만으로 raw token state와 image pooled vector가 완전히 같은 metric coordinate라고 단정할 수 없다.
- SigLIP text encoder는 bidirectional이므로 `running` token도 전체 identity/background 문맥을 포함한다. 한 token이라고 해서 순수 action direction인 것은 아니다.
- SentencePiece tokenizer에서 한 단어가 여러 piece로 나뉠 수 있다.

### 4.6 구현 시 필수 조건

- `tokenizer.is_fast`이고 offset mapping을 지원하면 offset으로 실제 token span을 찾는다. 기본 slow `SiglipTokenizer`에서는 unpadded prefix tokenization 또는 SentencePiece proto의 begin/end offset을 이용하고, 최종 token/piece/index를 사람이 확인한다.
- 동일 단어를 구성하는 모든 SentencePiece row를 평균해야 한다.
- SigLIP에서는 `pad_token == eos_token`일 수 있으므로 padding 전 실제 token 길이를 별도로 기록하고, semantic EOS 뒤에 반복되는 padding row를 basis에서 제외해야 한다.
- `output_hidden_states=True`를 사용할 경우 final LayerNorm 적용 여부를 확인해야 한다. 첫 구현에서는 public output인 final `last_hidden_state`를 사용한다.

### 4.7 실험상의 위치

T1은 **Harmony 형태를 모방하는 exploratory baseline**으로 사용한다. 주 방법으로 채택하기보다, token-level geometry가 실제로 도움이 되는지 확인하는 비교군으로 두는 것이 적절하다.

---

## 5. T2 — Text-head projected token basis

### 5.1 쉬운 설명

T1과 동일하게 action/background token을 선택하되, raw token hidden state를 그대로 쓰지 않고 SigLIP text pooled output을 만드는 text projection head에 먼저 통과시킨다.

```text
selected action token hidden state
              ↓ SigLIP text head
pooled text head와 같은 output coordinate의 token-derived direction
```

### 5.2 구체적인 구성

SigLIP text affine head를 $g_T(h)=W_T h+b_T$라고 하면 action basis는 다음과 같다.

$$
d_A^{\text{head}}
=\mathrm{normalize}
\left(
g_T\left(
\frac{1}{|S_A|}\sum_{i\in S_A}h_i
\right)
\right)
$$

Background도 동일하게 구성한다.

$$
d_B^{\text{head}}
=\mathrm{normalize}
\left(
g_T\left(
\frac{1}{|S_B|}\sum_{i\in S_B}h_i
\right)
\right)
$$

기본 실험은 실제 pooled coordinate를 모방하도록 head의 bias까지 포함한다. `weight-only`인 $W_T h$는 필요할 때 별도 ablation으로 둔다. Transformers 버전에 따라 head attribute path가 달라질 수 있으므로 repository와 Transformers revision을 고정한 뒤 실제 모듈 경로를 확인한다.

### 5.3 T1과의 차이

- T1: text head 이전의 raw contextual token coordinate 사용
- T2: 각 selected token/span에 text head를 적용한 coordinate 사용

SigLIP에서 vision pooler와 비교되는 text pooled embedding도 같은 text head를 통과한다. 따라서 T2는 head output coordinate에 놓이지만, 임의 token에서 cross-modal alignment가 T1보다 실제로 좋아지는지는 실험으로 검증해야 한다.

### 5.4 장점

- Token/span 단위의 제어 가능성을 유지한다.
- T1보다 image pooled embedding과의 좌표 정합성이 나을 가능성이 있다.
- T1과 P2 사이의 중간 실험으로 해석하기 쉽다.

### 5.5 한계

- SigLIP 학습 loss는 text sequence의 pooled position에 적용된 head output을 사용한다. 임의의 action token에 text head를 적용하도록 직접 학습된 것은 아니다.
- 따라서 T2도 완전히 이론적으로 보장된 방법이 아니라 empirical heuristic이다.
- Contextual token 자체가 identity/background 의미를 포함할 수 있다는 문제는 남는다.

### 5.6 실험상의 위치

T2는 **권장 token-level baseline**이다. Token 접근을 평가할 때 T1만 사용하지 말고 T1/T2를 반드시 함께 비교해야 한다.

다음 결과가 나오면 해석이 가능하다.

- T2 > T1: text head가 cross-modal alignment에 유효
- T1 ≈ T2: token context 자체가 주요 요인
- P2 > T1/T2: pooled contribution 방식이 더 안정적

---

## 6. P1 — Absolute pooled phrase basis

### 6.1 쉬운 설명

동작이나 배경을 표현한 하나의 완성된 문장/구를 SigLIP text encoder에 넣고, 그 pooled embedding 자체를 textual basis로 사용하는 방법이다.

예:

- Action phrase: `a photo of an animal running`
- Background phrase: `a photo of an animal in a snowy field`
- Joint phrase: `a photo of an animal running in a snowy field`

### 6.2 구체적인 구성

Action basis는 다음처럼 정의한다.

$$
d_A^{\text{abs}}
=e(\text{“a photo of an animal running”})
$$

Background basis는 다음과 같다.

$$
d_B^{\text{abs}}
=e(\text{“a photo of an animal in a snowy field”})
$$

Action과 background를 한 번에 나타내려면 다음 joint direction을 사용할 수 있다.

$$
d_{AB}^{\text{abs}}
=e(\text{“a photo of an animal running in a snowy field”})
$$

### 6.3 이 방법이 의미하는 것

SigLIP이 전체 문장을 이미지와 대응시키도록 학습되었다는 점을 직접 활용한다. Text와 image의 pooled output이 실제 similarity 계산에 사용되므로 T1/T2보다 공간적 근거가 강하다.

### 6.4 장점

- 가장 구현하기 쉬운 pooled-space baseline이다.
- Action과 background를 하나의 joint concept로 처리하기 쉽다.
- 여러 단어로 된 event phrase에도 자연스럽게 적용할 수 있다.

### 6.5 한계

- `animal`, `photo` 같은 공통 의미까지 basis에 포함된다.
- Action phrase에 class/identity 성분이 섞여 reference identity를 함께 약화시킬 수 있다.
- `running`처럼 짧은 단어만 단독 입력하면 SigLIP이 학습한 caption 형태와 달라질 수 있다.
- Action과 background를 joint phrase 하나로 표현하면 어느 요소가 실제 개선을 만들었는지 분리하기 어렵다.

### 6.6 권장 변형

최소한 다음 두 종류를 함께 비교한다.

1. **Absolute non-identity phrase**

   ```text
   “running in a snowy field”
   ```

2. **Caption-style phrase**

   ```text
   “a photo of an animal running in a snowy field”
   ```

둘의 차이를 통해 문법적으로 완성된 caption template의 효과와 identity/class contamination을 확인할 수 있다.

### 6.7 실험상의 위치

P1은 **가장 단순한 pooled baseline**이다. 최종 방법으로 추천하기보다는 P2/P3가 단순한 phrase embedding보다 실제로 factor-specific한지 비교하기 위한 기준으로 사용한다.

---

## 7. P2 — Pooled span-difference basis

### 7.1 쉬운 설명

전체 prompt와 그중 특정 의미 요소만 제거한 prompt의 pooled embedding 차이를 사용한다.

이 방식은 `running`이라는 token 자체를 사용하지 않고도, `running`이 전체 문장의 의미를 얼마나 바꿨는지를 pooled space에서 측정한다.

```text
full prompt               : “a spotted dog running in a snowy field”
prompt without action     : “a spotted dog in a snowy field”
두 pooled embedding의 차이: action contribution
```

### 7.2 구체적인 구성

Action contribution은 다음과 같다.

$$
d_A^{\Delta}
=\mathrm{normalize}
\left[
e(\text{“a spotted dog running in a snowy field”})
-e(\text{“a spotted dog in a snowy field”})
\right]
$$

Background contribution은 다음과 같다.

$$
d_B^{\Delta}
=\mathrm{normalize}
\left[
e(\text{“a spotted dog running in a snowy field”})
-e(\text{“a spotted dog running”})
\right]
$$

Action과 background를 한 번에 제거하고 싶다면 joint difference를 사용할 수 있다.

$$
d_{AB}^{\Delta}
=\mathrm{normalize}
\left[
e(\text{“a spotted dog running in a snowy field”})
-e(\text{“a spotted dog”})
\right]
$$

### 7.3 이 방법이 의미하는 것

Identity/class와 문장 template를 최대한 유지하면서 action 또는 background만 바꾼 **counterfactual secant contrast**이다. 즉 해당 factor의 순수하거나 인과적인 방향이라고 가정하는 것이 아니라, confound를 줄이도록 구성한 두 pooled point 사이의 방향이며 factor-specificity는 intrinsic test로 검증한다.

단일 action을 다루더라도 이 방법은 의미상 token-specific하다. Representation은 pooled이지만, `running` span이 문장 전체에 만든 변화만 방향으로 사용하기 때문이다.

### 7.4 장점

- 모든 방향이 image와 실제로 비교되는 SigLIP pooled space에 존재한다.
- Absolute phrase보다 identity/class/template 공통 성분을 부분적으로 상쇄할 가능성이 있다.
- Action과 background를 개별 또는 joint로 다룰 수 있다.
- 단순 prompt와 복합 prompt 모두에 적용하기 쉽다.

### 7.5 한계

- Transformer가 nonlinear이므로 두 pooled embedding의 차이가 순수한 causal factor라고 보장되지는 않는다.
- 단어를 삭제하면 문법과 token position이 함께 변한다.
- `running`을 제거한 문장이 부자연스러우면 difference에 문법 변화가 포함될 수 있다.

### 7.6 반드시 비교할 counterfactual 방식

Action factor를 추정할 때 다음 세 방법을 비교한다.

1. **Deletion**

   ```text
   “a dog running in snow” − “a dog in snow”
   ```

2. **Neutral replacement**

   ```text
   “a dog running in snow” − “a dog doing an action in snow”
   ```

3. **Alternative action replacement**

   ```text
   “a dog running in snow” − “a dog standing in snow”
   ```

Deletion과 neutral replacement가 비슷한 결과를 보이면 direction의 안정성이 높다고 볼 수 있다. Alternative replacement는 target action과 source/neutral action 사이의 contrast direction으로 해석한다.

### 7.7 실험상의 위치

P2는 **가장 권장하는 MVP**이다. 첫 generation 실험에서 주 방법으로 사용하고, T1/T2/P1을 비교군으로 둔다.

---

## 8. P3 — Factorial action/background basis

### 8.1 쉬운 설명

Action과 background가 동시에 들어간 prompt에서는 두 의미가 서로 영향을 줄 수 있다. P3는 네 가지 prompt 조합에서 action의 평균 contrast, background의 평균 contrast, 둘의 비가산 residual을 각각 계산한다.

예를 들어 다음 네 prompt를 사용한다.

| 기호 | Action | Background | Prompt |
|---|---|---|---|
| $p_{00}$ | 없음 | 없음 | `a spotted dog` |
| $p_{10}$ | running | 없음 | `a spotted dog running` |
| $p_{01}$ | 없음 | snowy field | `a spotted dog in a snowy field` |
| $p_{11}$ | running | snowy field | `a spotted dog running in a snowy field` |

### 8.2 Action main effect

선택한 두 background level에서의 action difference를 평균한다.

$$
d_A=
\frac{1}{2}
\left[
e(p_{10})-e(p_{00})
+e(p_{11})-e(p_{01})
\right]
$$

이 방향은 두 background level에 걸쳐 평균된 `running` contrast이다. 다른 background에도 일반화되는지는 추가 prompt bank에서 검증해야 한다.

### 8.3 Background main effect

선택한 두 action level에서의 background difference를 평균한다.

$$
d_B=
\frac{1}{2}
\left[
e(p_{01})-e(p_{00})
+e(p_{11})-e(p_{10})
\right]
$$

이 방향은 두 action level에 걸쳐 평균된 `snowy field` contrast이다. 다른 action에도 일반화되는지는 별도로 검증한다.

### 8.4 Action–background 비가산 residual(interaction diagnostic)

네 pooled point가 단순히 더해지는 모델에서 벗어나는 비가산 residual은 다음과 같이 계산한다.

$$
d_{AB}
=e(p_{11})-e(p_{10})-e(p_{01})+e(p_{00})
$$

이 residual에는 semantic interaction뿐 아니라 tokenization, 문법, normalization, encoder nonlinearity도 포함될 수 있다. 따라서 네 counterfactual cell이 모두 자연스러울 때에만 제한적으로 action–background interaction으로 해석한다. `swimming in water`, `flying in the sky`처럼 나머지 조합이 부자연스러운 사건에서는 큰 값이 잘못 구성된 counterfactual에서 생길 수 있다.

### 8.5 세 가지 P3 변형

1. **Joint rank-1 basis**

   \[
   D=[e(p_{11})-e(p_{00})]
   \]

   Action과 background를 한 번에 약화한다. 위 정의에서는 $d_A+d_B=e(p_{11})-e(p_{00})$이므로, 이는 두 main-effect vector 합의 한 방향만 사용하는 rank-1 설정이다.

2. **Separate main-effect basis**

   \[
   D=[d_A,d_B]
   \]

   Action과 background가 만드는 공동 span을 제거한다. 단순히 $[d_A,d_B]$의 SVD basis에 하나의 $\lambda$를 적용하면 basis column은 원래 factor label을 보존하지 않으므로, factor별 strength 조절에는 $d_A$, $d_B$ projector를 따로 적용하거나 weighted projector를 추가 설계해야 한다.

3. **Interaction-aware basis**

   \[
   D=[d_A,d_B,d_{AB}]
   \]

   Action–background 결합까지 제거한다. 가장 표현력이 높지만 identity damage와 과도한 projection 위험도 커진다.

### 8.6 장점

- 선택한 prompt level에서 action/background contrast와 비가산 residual을 구조적으로 비교할 수 있다.
- 동물 종류, background, 문장 template를 추가로 평균하면 factor-specific한 global basis로 확장할 수 있다.
- 논문 수준의 체계적인 ablation을 설계하기 좋다.

### 8.7 한계

- Prompt 조합 수와 계산량이 증가한다.
- 모든 action과 background가 독립적으로 조합 가능한 것은 아니다.
- Global basis를 너무 넓게 만들면 동물의 형상이나 identity와 관련된 방향까지 제거할 수 있다.
- Interaction direction까지 포함하면 rank가 커지므로 작은 $\lambda$가 필요할 수 있다.

### 8.8 동물 데이터셋에서의 특별한 처리

동물 action은 형태와 환경에 강하게 종속될 수 있으므로 두 그룹으로 나누는 것이 좋다.

1. **상대적으로 독립적인 action–background 조합**

   - running × grass/snow/beach
   - sitting × indoor/forest/grass
   - jumping × grass/snow/indoor

   이 그룹은 main effect 분리에 적합하다.

2. **본질적으로 결합된 action–background 조합**

   - swimming × water
   - flying × sky
   - perching × branch

   이 그룹에는 2×2 factorial interaction을 적용하지 않는다. 대신 `swimming in water` 같은 **joint event phrase basis**를 만들고, 교차 가능한 독립-factor benchmark와 별도 결과로 보고한다.

### 8.9 실험상의 위치

P3는 **가장 강한 확장 연구 설정**이다. P2에서 projection 가설이 유효함을 확인한 뒤 action+background 복합 prompt와 비가산 residual 분석에 적용한다.

---

## 9. 다섯 방법의 핵심 비교

| 방법 | Basis가 만들어지는 위치 | 설계상 factor granularity | 공간적 정당성 | 권장 역할 |
|---|---|---:|---:|---|
| T1 Raw token | SigLIP text final token hidden | Token/span, 검증 필요 | 낮음 | Exploratory/negative baseline |
| T2 Token + text head | Selected token을 text head에 투영 | Token/span, 검증 필요 | T1보다 높을 가능성 | 권장 token baseline |
| P1 Absolute phrase | SigLIP text pooled output | Phrase/joint concept | 높음 | 단순 pooled baseline |
| P2 Pooled difference | 두 pooled prompt의 차이 | Counterfactual contrast, 검증 필요 | 높음 | 권장 MVP |
| P3 Factorial basis | 여러 pooled contrast의 main/residual 구성 | Action/background 구조화, 검증 필요 | 높음 | 주 확장 방법 |

### 단일 action/background prompt에서의 해석

`a dog running`처럼 non-identity factor가 하나뿐이라면 다음 세 방식이 직접 경쟁한다.

- T1/T2: `running` token/span 자체를 basis로 사용
- P1: `a dog running` 전체 pooled vector 사용
- P2: `e(a dog running)-e(a dog)`를 사용

이 경우 P2는 semantic granularity는 action 단일 factor이면서도 representation은 pooled aligned space라는 장점을 가진다.

### Action과 background가 모두 있는 prompt에서의 해석

`a dog running in snow`에서는 다음 세 설정을 비교한다.

- Joint phrase: action+background를 하나의 concept로 처리
- Separate basis: action과 background를 독립 방향으로 처리
- Interaction-aware basis: action, background, 결합 의미까지 처리

성능만 확인하려면 joint phrase가 단순하지만, leakage 원인과 factor별 효과를 분석하려면 separate/factorial basis가 더 적합하다.

---

## 10. Textual subspace를 어느 prompt에서 만들 것인가

Basis construction 방식과 별개로, 어떤 의미 source를 기준으로 할지도 독립 실험 축이다.

### 10.1 Target-prompt basis

목표 생성 prompt의 non-identity 요소를 basis로 사용한다.

```text
target: “a dog running in snow”
basis: running / snow
```

Harmony의 기본 취지와 가장 가깝다. Visual branch에서 target textual direction과 평행한 성분을 약화하면 text control이 상대적으로 우세해질 수 있다는 가설이다. 다만 InstantX의 IP attention은 main text attention과 별도 softmax/residual로 계산되므로, 이 projection이 source nuisance만 선택적으로 제거하거나 Harmony와 같은 token competition을 만든다는 보장은 없다.

### 10.2 Reference-caption basis

Reference image를 설명하는 source action/background를 basis로 사용한다.

```text
reference caption: “a dog sitting on grass”
basis: sitting / grass
```

Source leakage를 직접 제거한다는 해석은 쉽지만, caption 오류와 identity–nuisance entanglement 문제가 있다. Harmony의 특정 BLIP-D/U-Net caption-based ablation에서는 target-prompt basis보다 target compliance가 낮았지만, 그 결과를 SigLIP pooled 설정에 일반화하지 말고 본 실험에서 다시 검증한다.

### 10.3 Conflict-aware basis

Reference nuisance 중 target과 호환되지 않는 방향만 제거한다.

```text
reference: sitting on grass
target   : running in snow

remove  : source-specific sitting/grass component
preserve: target과 호환되거나 identity에 필요한 component
```

이 방식은 가장 직접적이지만 source caption 또는 reference attribute annotation이 필요하므로 P2/P3 이후 확장 실험으로 둔다.

---

## 11. 동물 중심 데이터셋 설계 원칙

사용할 데이터셋은 본 프로젝트에서 별도로 선별할 동물 중심 데이터셋을 전제로 한다. 특정 기존 데이터셋의 label 체계에 실험을 종속시키기보다, 다음 속성을 만족하도록 subset을 구성하는 것이 중요하다.

### 11.1 필수 annotation

각 reference image에 최소한 다음 정보가 필요하다.

- Animal species/class
- 가능하면 개체 identity 또는 동일 개체 그룹 ID
- Source action/pose
- Source background/environment
- Subject segmentation mask 또는 bounding box
- Action의 species compatibility

### 11.2 동물 identity 구성

- 동일 species 안에서 외형이 다른 여러 개체를 포함해야 한다.
- 색·무늬·귀/꼬리 형태·체형 등 개체 구분 단서가 보이는 reference를 우선한다.
- 핵심 identity benchmark에서는 같은 개체의 다른 pose/background 이미지가 반드시 있어야 한다. 이것이 없으면 측정값은 instance identity보다 reference appearance similarity에 가까워진다.
- 핵심 benchmark의 identity마다 최소 세 역할의 이미지를 확보한다: conditioning reference A, nuisance가 다른 reference B, conditioning에 사용하지 않는 identity-evaluation image.
- 동일 species의 다른 개체들을 hard negative로 포함한 instance retrieval을 함께 평가한다.
- Train/test라는 표현보다는 basis prompt-bank 구성용 development identity와 최종 generation 평가용 held-out identity를 분리한다. 모델 학습은 없지만 prompt-bank overfitting을 방지하기 위해 필요하다.

데이터 규모는 고정할 필요가 없지만, 대략 20개 identity는 pilot 규모로 본다. Pilot variance로 confirmatory benchmark의 identity 수를 정하는 power calculation을 수행한다. 데이터가 더 작다면 seed 수를 늘리는 것보다 identity 단위 paired comparison과 confidence interval을 우선하며, 같은 identity의 여러 seed는 독립 identity sample로 계산하지 않는다.

### 11.3 동작 구성

동물 종에 따라 가능한 action이 다르므로 공통 action과 종별 action을 분리한다.

- 비교적 공통: standing, sitting, lying, running, jumping, eating, sleeping
- 종별/환경 결합: flying, swimming, perching, climbing

불가능하거나 부자연스러운 action prompt는 주 benchmark에서 제외하고 별도 compositional stress test로 둔다.

### 11.4 배경 구성

다음처럼 visual appearance가 명확하고 서로 구분되는 background를 우선한다.

- grass field
- snowy field
- beach
- forest
- indoor room
- rocky area

Action과 background가 본질적으로 묶이는 조합은 독립-factor benchmark와 joint-event benchmark를 분리해야 한다.

### 11.5 Source–target 2×2 설계

각 reference에 대해 다음 네 조건을 균형 있게 구성한다.

| 조건 | Source action vs target | Source background vs target | 측정 목적 |
|---|---|---|---|
| C00 | 같음 | 같음 | 불필요한 editing/identity damage 확인 |
| C10 | 다름 | 같음 | Action leakage와 action controllability |
| C01 | 같음 | 다름 | Background leakage와 background controllability |
| C11 | 다름 | 다름 | 복합 disentanglement 성능 |

C00에서 성능이 떨어지면 projection이 유용한 reference 정보를 과도하게 제거하고 있음을 의미한다. C10/C01은 action과 background 원인을 분리하며, C11은 P3의 필요성을 평가한다.

### 11.6 데이터 분할

최소 다음 분할을 권장한다.

1. **Development split**

   - Basis mode와 $\lambda$ 선택
   - Prompt template 및 neutral replacement 결정
   - 일부 animal identities만 사용

2. **Held-out identity split**

   - Development에 등장하지 않은 개체
   - 동일 species는 허용하되 외형이 다른 개체 사용

3. **Held-out composition split**

   - Development에서 개별적으로 본 action/background이지만 새로운 조합 사용
   - Factorial generalization 평가

4. **Optional held-out action split**

   - Prompt-bank/SVD에 포함되지 않은 action
   - Global basis가 unseen behavior에 일반화되는지 평가

5. **Optional held-out species split**

   - Development에 등장하지 않은 species를 최종 평가에 사용
   - 이 split이 없으면 결론 범위를 관찰한 species 내부의 unseen identity/composition 일반화로 제한

### 11.7 Paired-reference invariance 구성

핵심 benchmark의 각 identity에는 외형은 같지만 source action 또는 background가 다른 reference A/B를 준비한다.

```text
같은 animal identity + 같은 target prompt
    ├─ reference A: sitting on grass
    └─ reference B: standing indoors
```

좋은 disentanglement 방법이라면 reference A/B를 바꾸더라도 생성 결과의 target action/background는 안정적이어야 하고, animal identity는 양쪽 모두에서 유지되어야 한다. 이 paired-reference test는 단순히 reference conditioning 전체를 약화한 방법과 factor-specific projection을 구분하는 데 중요하다.

---

## 12. 생성 전 textual subspace 자체 평가

Generation을 실행하기 전에 basis가 의도한 의미를 담는지 동물 이미지 embedding으로 검사한다. 이렇게 하면 생성 실패가 basis 문제인지 diffusion routing 문제인지 구분할 수 있다.

다만 intervention과 intrinsic score를 모두 SigLIP으로 계산하면 순환 평가가 된다. 따라서 아래 SigLIP projection score는 **진단 지표**로만 사용하며, 방법의 최종 우열이나 탈락 여부는 독립적인 generation metric과 함께 결정한다.

### 12.1 Action selectivity

Action basis $B_A$가 matching action image에는 높은 projection energy를, 다른 action image에는 낮은 energy를 가져야 한다.

$$
S_A(I)=\|B_A^\top \hat v(I)\|_2^2
$$

예를 들어 `running` basis에서 running animal images의 평균 $S_A$가 sitting/standing images보다 높아야 한다.

### 12.2 Background selectivity

Background basis $B_B$가 동일 animal identity/action을 유지하면서 background만 바뀐 이미지에 선택적으로 반응하는지 검사한다.

### 12.3 Cross-factor leakage

- Action basis가 identity 차이를 과도하게 설명하지 않는가?
- Background basis가 action 차이를 과도하게 설명하지 않는가?
- Interaction basis가 단독 action/background보다 joint samples에 선택적인가?

### 12.4 Template stability

다음과 같은 여러 문장 template에서 같은 factor direction이 안정적인지 확인한다.

- `a photo of a dog running`
- `a dog that is running`
- `an image of a running dog`

서로 다른 template로 만든 basis의 principal angle 또는 projection overlap을 측정한다.

### 12.5 Identity contamination

Projection 전후 reference embedding이 animal identity anchor와 얼마나 유사한지 기록한다.

Identity anchor는 generic class text보다 다음이 더 적합하다.

- Subject-only crop의 vision pooled embedding
- Background를 달리한 identity-preserving augmentation들의 평균
- 동일 개체의 여러 reference embedding 평균

---

## 13. Generation 평가 지표

동물 데이터에서는 사람 얼굴용 identity metric만으로 평가할 수 없으므로 subject mask와 동물 특성을 반영해야 한다.

Primary 평가는 intervention과 독립적인 masked DINO 계열 identity score, 별도 검증된 animal-action evaluator, 사람 평가를 중심으로 둔다. SigLIP similarity는 보조 진단 지표로 보고한다. 자동 segmentation과 action/VLM evaluator는 별도의 label이 있는 real-animal subset에서 오류율과 species별 편향을 먼저 측정하고, 신뢰하기 어려운 class는 human audit로 보완한다.

### 13.1 Identity preservation

- Subject-mask 기반 DINO/DINOv2 image similarity
- Masked CLIP-I 또는 SigLIP-I
- 동일 개체 retrieval accuracy가 가능하다면 instance retrieval
- 색·무늬·체형 보존에 대한 VLM 또는 human evaluation

Reference와 generated image의 pose가 다르더라도 identity score가 지나치게 하락하지 않도록, 가능하면 같은 개체의 다양한 pose 이미지를 positive set으로 사용한다.

### 13.2 Target action compliance

- Animal action classifier
- Animal pose/keypoint estimator가 가능한 species에서는 skeleton/pose metric
- VLM/VQA 기반 action 판정
- Human evaluation

### 13.3 Source-action leakage

- Generated action이 target보다 source action에 더 가까운 비율
- Reference–generated pose similarity와 target-action score를 함께 보고
- Source와 target이 다른 C10/C11 조건에서 별도 집계

### 13.4 Target background compliance

- Subject mask 외부 영역의 background classifier/VLM score
- Target background text와의 masked-out similarity

### 13.5 Source-background leakage

- Reference와 generated image의 subject-mask 외부 영역 similarity
- Source background classifier score가 target background score보다 높은 실패 비율

### 13.6 Quality와 diversity

- Image quality/VLM preference
- 동일 조건 여러 seed 간 diversity
- Anatomy failure rate
- Subject duplication 또는 missing-subject rate

최종 결과는 한 개의 평균 점수보다 identity–target compliance–source leakage의 Pareto curve로 보고하는 것이 적절하다.

### 13.7 통계 단위와 신뢰구간

- 주 통계 단위는 생성 이미지나 seed가 아니라 **animal identity**로 둔다.
- 같은 identity에서 여러 reference·prompt·seed를 사용한 결과는 identity 내부 반복 측정으로 처리한다.
- 방법 간 비교는 동일한 reference·prompt·seed에서 얻은 paired difference를 사용한다.
- 가능하면 identity-clustered paired bootstrap으로 95% confidence interval을 보고한다.

Seed 수를 늘리는 것은 생성 변동성을 추정하는 데 유용하지만, 서로 다른 animal identity 수를 늘린 것과 같은 일반화 근거로 해석해서는 안 된다.

Confirmatory 평가 전에는 primary endpoint와 primary contrast를 미리 정한다. 예를 들어 `C10/C11 source-action leakage 감소`를 primary contrast로 두고 `masked identity score 감소 5% 이내`를 제약으로 둘 수 있다. Species·action·background가 불균형하면 identity별 cell-balanced summary 또는 identity/action/background random effect를 포함한 mixed model을 보조 분석으로 사용한다.

---

## 14. 필수 control과 ablation

### 14.1 Geometry control

- 동일 rank의 random orthogonal subspace
- 동일 norm의 random perturbation
- Identity/class phrase projection negative control
- 낮은 IP-Adapter scale만 적용한 baseline
- Projection 결과와 IP residual norm 또는 identity score를 맞춘 scale-matched baseline
- Rank 1/2/4
- Sum vs Mean projection
- Norm restoration On/Off

낮은 IP scale 또는 scale-matched baseline과 차이가 없다면, 개선이 textual factor 제거가 아니라 reference condition 전체를 약화한 결과일 가능성이 있다. 따라서 이 두 control은 최종 disentanglement 주장에 필수적이다.

Scale-matching 기준은 test 결과를 본 뒤 정하지 않는다. Development split에서 `IP residual norm` 또는 `identity score` 중 하나를 matching target으로 사전 고정한다. Rank와 방법을 비교할 때도 동일한 $\lambda$만 맞추지 말고 실제 perturbation 크기 $\|v'-v\|$를 맞춘 결과를 함께 보고한다.

### 14.2 Prompt construction control

- Raw token vs text-head token
- Absolute phrase vs pooled difference
- Deletion vs neutral replacement vs alternative action replacement
- 단일 template vs paraphrase bank
- Target-prompt basis vs reference-caption basis

### 14.3 Projection strength

$$
\lambda\in\{0,0.25,0.5,0.75,1.0\}
$$

`λ=0`은 baseline이며, 모든 방법은 같은 seed·reference·prompt에서 비교한다.

### 14.4 Intermediate logging

각 sample에서 최소 다음 값을 저장한다.

- Original visual pooler norm
- Edited visual pooler norm
- Original–edited cosine similarity
- 각 basis에 대한 projection energy
- Basis rank와 singular values
- InstantX MLP 전후 original–edited cosine
- 가능하면 block별 IP residual norm

---

## 15. 실제 추천 실험 순서

### 단계 0 — 구현 및 shape 검증

1. 동일한 `google/siglip-so400m-patch14-384` checkpoint의 vision/text tower를 freeze한다. 현재 InstantX가 기본적으로 vision model만 로드하므로, 실험용 `SiglipTextModel`과 tokenizer는 추가로 로드하되 학습하지 않는다.
2. Reference vision `pooler_output=[B,1152]` 생성 직후, InstantX MLP 직전에 hook을 둔다.
3. Token mode에서는 actual semantic token mask가 맞는지 사람이 확인할 수 있도록 token과 index를 로그로 남긴다.
4. Projection 전후 shape, dtype, norm을 unit test한다.
5. Model/repository revision과 Transformers 버전을 고정한다.

초기 smoke test에서는 다음 shape를 assertion으로 확인한다.

- Text `last_hidden_state=[K,64,1152]`, `pooler_output=[K,1152]`
- Vision `last_hidden_state=[B,729,1152]`, `pooler_output=[B,1152]`
- Edited visual pool `[B,1152]`
- InstantX IP tokens `[B,128,4096]`

### 단계 1 — 동물 이미지 기반 intrinsic basis screening

Generation 전에 선별된 동물 데이터셋에서 T1~P3 basis의 factor selectivity를 비교한다.

구현·분석 순서:

1. T1 raw token
2. T2 text-head token
3. P1 absolute pooled phrase
4. P2 pooled difference
5. P3 factorial main effect

평가:

- Matching action/background projection energy
- Cross-factor leakage
- Identity contamination
- Template stability

이 단계는 basis의 문제를 조기에 발견하고 generation 결과를 해석하기 위한 진단 단계다. Pooled representation에 유리한 selection bias를 막기 위해 intrinsic score만으로 T1~P3를 탈락시키지 않고, 다섯 방법 모두 최소 generation pilot까지 유지한다. Shape 오류, semantic span 오류, 유효 rank 0처럼 구현 자체가 성립하지 않는 경우만 수정 후 재실행한다.

### 단계 2 — Action-only pilot

C10 조건, 즉 source와 target action이 다르고 background는 같은 sample을 우선 사용한다.

비교 방법:

- Baseline
- T1
- T2
- P1 action phrase
- P2 action span-difference

목표:

- 단일 action에서 token 방식이 실제 장점을 갖는지 확인
- P2가 identity를 더 잘 보존하면서 source-action leakage를 줄이는지 확인

초기 주 가설은 **P2 > T2 ≥ T1**, P1은 단순 pooled control이다.

### 단계 3 — Background-only pilot

C01 조건, 즉 source와 target background가 다르고 action은 같은 sample을 사용한다.

비교 방법:

- Baseline
- T1 background span
- T2 background span
- P1 background phrase
- P2 background span-difference

Background leakage는 subject-mask 밖 영역을 중심으로 평가한다. 이 단계에서는 action score가 유지되는지도 함께 확인하여 background basis가 pose를 훼손하는지 검사한다.

### 단계 4 — Action+background joint pilot

C11 조건에서 다음을 비교한다.

- P2 joint difference
- P2 separate action/background basis
- P1 joint phrase
- P3 main-effect basis $[d_A,d_B]$
- P3 residual-aware basis $[d_A,d_B,d_{AB}]$

목표:

- Joint rank-1이 충분한지 확인
- Action과 background의 분리 조절이 필요한지 확인
- 자연스러운 2×2 조합에서 비가산 residual 방향이 실제 이득을 주는지 확인

P3 factorial variants는 네 cell이 모두 자연스러운 독립 조합 benchmark에만 적용한다. Swimming/flying처럼 환경과 결합된 action은 joint event phrase basis로 평가하고 별도 집계한다.

### 단계 5 — Projection geometry ablation

단계 2~4에서 상위 두 방법을 대상으로 다음을 sweep한다.

- $\lambda$: 0.25/0.5/0.75/1.0
- SVD threshold 이후 최대 retained rank: 1/2/4
- Sum vs Mean
- Norm restoration On/Off
- Single template vs paraphrase bank
- Random rank-matched basis
- 동일 $\lambda$ 비교와 실제 $\|v'-v\|$-matched 비교

이 단계에서 development split의 Pareto frontier를 기준으로 최종 configuration을 선택한다.

### 단계 6 — Target/source/conflict basis 비교

선택된 P2 또는 P3 configuration을 다음 semantic source와 결합한다.

- Target-prompt basis
- Reference-caption basis
- Conflict-aware basis

Reference-caption annotation이 신뢰 가능한 animal subset에서만 source/conflict 방식을 평가한다.

### 단계 7 — Pre-MLP filtering × single-block direct text-row IP update suppression

Embedding projection의 최종 후보를 single-stream IP text-row mask와 결합한다. 이 mask는 single block에서 IP attention residual을 더하기 직전 `[B,T+L,3072]` tensor에 `[1,T+L,1]` token-row mask를 곱해 앞의 text rows는 0, 뒤의 latent rows는 1로 만든다. Feature channel mask가 아니라 token-row mask이다.

| SigLIP projection | Single text-row IP mask | 목적 |
|---|---|---|
| Off | Off | InstantX baseline |
| On | Off | Upstream SigLIP pooled-content filtering의 총효과 |
| Off | On | Single blocks의 direct IP-to-text-row update 제거 효과 |
| On | On | Upstream filtering과 direct text-row update suppression의 상보성 |

이 단계는 textual subspace 방법 자체의 우열을 먼저 확정한 뒤 수행한다. 그렇지 않으면 embedding content와 attention routing의 효과를 구분하기 어렵다.

이 mask 이후에도 base joint/self-attention을 통한 latent↔text 정보 교환, latent rows의 IP update, double block의 IP injection은 유지된다. 따라서 결과를 “모든 reference-to-text routing 차단”으로 해석하지 않는다.

### 단계 8 — Held-out animal benchmark

최종적으로 held-out identity 및 held-out composition split에서 평가한다.

방법 선택, $\lambda$/rank, prompt template, evaluator threshold, scale-matching 규칙은 모두 단계 0~7의 development split에서 확정한다. 단계 8의 held-out test는 configuration을 고정한 뒤 한 번만 열고, test 결과를 보고 재선택하지 않는다.

- 모든 방법에서 동일 reference·prompt·seed 사용
- C00/C10/C01/C11 조건별 성능 별도 보고
- Species별 결과와 전체 평균 함께 보고
- 공통 action과 species-specific action 별도 보고
- 같은 identity의 nuisance-different reference A/B에 대한 target consistency와 identity 유지율 보고
- 자동 지표와 human/VLM evaluation 병행
- Identity 단위 paired difference와 95% confidence interval 보고

---

## 16. 최종 권장안

### 가장 작은 MVP

다음 두 방법부터 구현한다.

1. **T2 — Text-head projected token basis**
2. **P2 — Pooled span-difference basis**

동일한 단일-action prompt에서 두 방법을 비교하면 token granularity의 실제 이득과 pooled alignment의 안정성을 가장 직접적으로 확인할 수 있다.

### 주 방법 후보

첫 주 방법은 다음으로 권장한다.

> **Target-local P2 pooled span-difference + partial projection + norm restoration**

이유:

- SigLIP의 image/text contrastive loss가 사용하는 pooled ambient space를 사용한다.
- 동일 문구를 유지한 counterfactual contrast로 identity/class/template 변화 confound를 줄이도록 설계된다.
- Action-only, background-only, joint prompt에 모두 적용할 수 있다.
- Prompt당 추가 text encoding만 필요하며 timestep과 무관하게 cache할 수 있다.

### 논문 수준의 확장 후보

P2가 유효하면 다음을 주 확장 방법으로 권장한다.

> **P3 factorial action/background average-contrast basis + optional non-additive residual**

특히 동물 데이터에서 자연스럽게 교차 가능한 action–background 조합에는 factorial contrast를, 환경 결합 action에는 joint event basis를 적용해 별도 평가하면, 단순 orthogonalization을 넘어 어떤 semantic factor가 reference leakage를 일으키는지 분석할 수 있다.

### T1/T2의 해석

Token 방식이 단순 animal-action prompt에서 P2보다 우수하다면 중요한 empirical finding이 될 수 있다. 다만 다음을 함께 제시해야 한다.

- Raw token과 text-head token의 차이
- Tokenization/span alignment 정확성
- Template 변화에 대한 안정성
- Identity/background contamination

따라서 token 방법은 배제하지 않되, pooled 방법과 동일한 이론적 확실성을 가진다고 미리 가정하지 않는 것이 적절하다.

---

## 17. 한 문장 요약

단일 동작에서는 **token-derived basis와 pooled counterfactual contrast를 직접 비교**하고, 동작과 배경이 함께 있는 동물 prompt에서는 **factorial average contrast와 비가산 residual을 검증**한 뒤, identity–target compliance–source leakage Pareto frontier를 기준으로 최종 textual subspace 구성을 선택한다.
