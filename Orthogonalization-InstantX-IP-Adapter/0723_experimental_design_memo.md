# 0723 InstantX IP-Adapter Orthogonalization 실험 설계 메모

## 1. 연구 목표

InstantX FLUX IP-Adapter의 파라미터를 학습하지 않고, reference image에서 target 생성으로 새어 나오는 동작·자세·배경 정보를 줄이면서 동물 개체의 identity는 보존한다.

검토할 개입은 다음 세 가지다.

1. **SigLIP pooled-space projection**: IP-Adapter에 들어가기 전 reference visual embedding에서 target 동작·배경 방향을 약화한다.
2. **Single-stream text-row gate**: single-stream block에서 IP attention residual이 text row를 직접 갱신하지 못하게 한다.
3. **결합 방식**: 위 두 개입을 동시에 적용한다.

모든 방법은 training-free/tuning-free inference intervention으로 구현한다.

---

## 2. 실험 1: SigLIP pooled-space projection

### 2.1 공통 공간의 조건

- **차원이 같다는 사실만으로는 같은 embedding space라고 볼 수 없다.**
- 동일한 SigLIP 체크포인트의 paired vision/text tower가 contrastive objective로 정렬한 최종 pooled output을 사용해야 한다.
- InstantX가 사용하는 `google/siglip-so400m-patch14-384`의 `SiglipVisionModel.pooler_output`과 이에 대응하는 `SiglipTextModel.pooler_output`을 사용한다.
- checkpoint, tokenizer, preprocessing 및 `transformers` revision을 고정한다.
- raw token hidden state나 임의로 평균 낸 token embedding은 이번 1차 실험에 사용하지 않는다.
- projection 계산은 FP32와 L2 normalization을 사용하고, 결과는 기존 MLP 입력 dtype으로 되돌린다.

따라서 이 실험은 Harmony의 contextual-token orchestration을 그대로 복제하는 것이 아니라, **SigLIP의 정렬된 pooled ambient space에서 수행하는 nuisance-direction filtering**이다.

### 2.2 Counterfactual prompt 구성

하나의 target 설정에 대해 다음 네 문장을 구성한다.

- \(O\): object
- \(O+A\): object + action/behavior
- \(O+B\): object + background
- \(O+A+B\): object + action/behavior + background

각 prompt의 정규화된 SigLIP text pooled embedding을 다음과 같이 둔다.

$$
e_{00}=e(O),\qquad
e_{10}=e(O+A),\qquad
e_{01}=e(O+B),\qquad
e_{11}=e(O+A+B)
$$

전체 target prompt embedding을 reference embedding에서 바로 빼면 object/identity 관련 성분도 함께 손상될 수 있다. 따라서 동작과 배경의 **평균 counterfactual contrast**를 사용한다.

$$
d_A=
\frac{1}{2}
\left[
(e_{10}-e_{00})+(e_{11}-e_{01})
\right]
$$

$$
d_B=
\frac{1}{2}
\left[
(e_{01}-e_{00})+(e_{11}-e_{10})
\right]
$$

object 방향과 겹치는 성분을 먼저 제거하는 identity-protection ablation도 함께 비교할 수 있다.

$$
u_O=\frac{e_{00}}{\lVert e_{00}\rVert},
\qquad
\widetilde d=(I-u_Ou_O^\top)d
$$

### 2.3 Projection

\(d_A,d_B\)를 열로 쌓은 뒤 QR 또는 SVD로 유효 rank만 남긴 직교정규 basis \(B\)를 만든다. Reference image의 SigLIP vision pooled embedding을 \(v\)라 하면 다음을 적용한다.

$$
\widehat v=\frac{v}{\lVert v\rVert}
$$

$$
\widehat v^{\,\prime}
=
\widehat v-\lambda BB^\top\widehat v
$$

$$
v^\prime
=
\lVert v\rVert
\frac{\widehat v^{\,\prime}}
{\lVert\widehat v^{\,\prime}\rVert+\epsilon}
$$

수정된 \(v^\prime\)는 InstantX의 128 visual token 생성 MLP에 입력한다.

- \(\lambda=1\): 선택한 basis에 대한 엄밀한 orthogonal projection
- \(0<\lambda<1\): 완전한 직교화가 아닌 partial projection 또는 attenuation
- `v - α × text_embedding`과 같은 고정량 subtraction: orthogonalization이 아님
- 초기 권장값: \(\lambda\in\{0.25,0.5,0.75,1.0\}\)

### 2.4 해석상의 한계

Target prompt로 만든 basis는 reference 속의 source 동작·배경을 정확히 식별하는 basis라고 보장할 수 없다. 우선은 **target 의미를 visual branch에서 약화하여 text branch가 담당하게 하는 role allocation**으로 해석한다. 후속 실험에서는 source caption basis와 target basis, 두 basis의 결합을 비교한다.

---

## 3. 실험 2: Single-stream text-row gate

Double-stream에서는 IP attention residual이 image/latent stream에만 직접 더해진다. Single-stream에서는 text row와 latent-image row가 하나의 sequence로 합쳐지므로, IP residual이 text row에도 직접 계산될 수 있다.

Single-stream의 IP residual을 \(R_{\mathrm{IP}}\), text 길이를 \(T\), latent 길이를 \(L\)이라 하면 다음 row gate를 residual addition 직전에 적용한다.

$$
g=
\begin{bmatrix}
0_T\\
1_L
\end{bmatrix},
\qquad
H^\prime=H+s\left(g\odot R_{\mathrm{IP}}\right)
$$

구현 시 유의점은 다음과 같다.

- transformer forward pre-hook에서 실제 text row 수 \(T\)를 읽어 single-stream attention processor에 설정한다.
- text/latent concatenation 순서를 코드에서 확인하고, 고정 길이를 하드코딩하지 않는다.
- double-stream 경로는 변경하지 않는다.
- 이 gate는 **IP residual이 text row를 직접 갱신하는 경로만** 차단한다.
- IP 정보가 latent row에 들어간 뒤 후속 joint self-attention을 통해 text row로 이동하는 간접 경로는 남는다.

따라서 결과는 “모든 visual-to-text 정보 흐름 차단”이 아니라 “single-stream direct IP-to-text update 차단”으로 해석한다.

---

## 4. 실험 3: 결합 및 필수 대조군

세 intervention 설정만 비교하지 말고 원본 baseline을 포함한 2×2 factorial로 구성한다.

| SigLIP projection | Single-stream text gate | 조건 |
|---|---|---|
| Off | Off | 원본 InstantX baseline |
| On | Off | pooled content filtering만 적용 |
| Off | On | direct IP-to-text routing gate만 적용 |
| On | On | 두 개입의 결합 |

동일한 reference, prompt, seed, step 수, guidance 및 IP scale을 사용한다. Projection이 단순히 전체 IP conditioning strength만 낮춘 것인지 구분하기 위해, 가능한 경우 실제 edit magnitude 또는 출력 token norm을 맞춘 IP-scale control도 추가한다.

---

## 5. 동물 중심 데이터셋과 평가

### 데이터 구성

- species와 breed가 한쪽에 치우치지 않게 구성한다.
- 같은 개체의 identity를 확인할 수 있는 reference를 사용하고, source 동작·자세·배경을 함께 기록한다.
- 각 object에 대해 \(O\), \(O+A\), \(O+B\), \(O+A+B\)의 의미가 자연스럽게 유지되는 prompt template을 만든다.
- 동물의 해부학적 특성상 불가능하거나 매우 부자연스러운 동작은 제외한다.
- identity 또는 species가 train/dev/test 사이에 중복되지 않도록 분리하고, 모든 조건에서 동일 seed를 사용한다.

### 핵심 평가축

1. **Identity preservation**: 배경 영향을 줄인 subject crop 또는 mask 기반 유사도
2. **Target behavior adherence**: 목표 동작·자세 반영 여부
3. **Target background adherence**: 목표 배경 반영 여부
4. **Source leakage**: reference의 원래 동작·배경이 남는 정도
5. **Image quality**: 동물의 해부학적 오류와 전체 품질

단일 종합 점수보다 identity–editability Pareto 관계와 behavior/background별 결과를 따로 보고한다.

---

## 6. 코드 수정 방향

기존 `instantx_flux_ip_adapter`는 baseline 재현용으로 보존하고, 현재 폴더에 최소 범위의 실험 코드를 분리한다. Model weight와 Hugging Face cache는 공유한다.
실험용 Modal 환경은 InstantX/SigLIP model revision뿐 아니라 Torch, Diffusers, Transformers 버전도 고정한다.

구현 모듈은 다음과 같다.

- `orthogonalization.py`: counterfactual basis, FP32 SVD, projection strength 및 norm 복원
- `adapter.py`: 동일 checkpoint의 SigLIP text tower와 cache, vision pooled vector 편집, 실제 `text_seq_len`을 읽는 transformer pre-hook
- `attention_processor.py`: single-stream IP residual에 text-row gate 적용
- `model_loader.py`: 고정 revision의 InstantX/SigLIP 로딩과 baseline namespace 분리
- `inference.py`, `modal_inference.py`: 2×2 조건을 독립 flag로 실행하고 진단값 저장
- `config.py`: 별도 Modal app, output 경로 및 model revision 관리

전체 `transformer_flux.py`와 `pipeline_flux_ipa.py`를 복제하지 않고 pre-hook으로 실제 길이를 읽으므로, upstream 연산은 유지하면서 수정 범위를 줄인다.

권장 설정 항목:

```yaml
orthogonalization:
  enabled: false
  basis: factorial_target
  strength: 0.5
  protect_object_direction: false
  restore_visual_norm: true

single_stream:
  mask_ip_text_rows: false
```

Text basis는 prompt/config 조합마다 한 번만 계산해 cache하고, diffusion timestep마다 다시 encoding하지 않는다.

### 필수 검증

- 두 기능을 모두 끈 결과가 원본 InstantX와 수치적으로 동일한지 확인한다.
- 실제 `ip-adapter.bin` 로딩 시 57개 processor의 missing/unexpected key가 없는지 검사한다.
- \(\lambda=1\)일 때 \(B^\top\widehat v^{\,\prime}\)가 0에 가까운지 검사한다.
- projection 전후 shape, dtype 및 복원 norm을 검사한다.
- gate 적용 후 text-row IP residual만 0이고 latent-row residual은 변하지 않는지 검사한다.
- 동일 seed에서 각 조건이 재현되는지 확인한다.

---

## 7. 권장 실험 순서

1. **Baseline parity**: `Off/Off`가 기존 InstantX 결과와 일치하는지 확인
2. **공간 검증**: 같은 SigLIP checkpoint의 vision/text pooled output과 normalization 경로 확인
3. **Projection 단독 pilot**: action-only, background-only에서 \(\lambda\) sweep
4. **Factorial basis 실험**: \(O+A+B\)에서 평균 contrast basis와 object-protected basis 비교
5. **Text-row gate 단독 실험**: leakage 감소와 text adherence 변화를 block별로 확인
6. **2×2 결합 실험**: 두 개입의 독립 효과와 상호작용 평가
7. **후속 basis 비교**: target basis, source-caption basis, source+target basis 비교
8. **동물 전체 데이터셋 확장**: 소규모 pilot에서 안정적인 설정만 고정하여 본 평가 수행

초기 우선순위는 **baseline parity → projection 단독 → gate 단독 → 결합**이다. 이 순서를 지켜야 개선이 embedding 편집 때문인지 attention routing 차단 때문인지 구분할 수 있다.
