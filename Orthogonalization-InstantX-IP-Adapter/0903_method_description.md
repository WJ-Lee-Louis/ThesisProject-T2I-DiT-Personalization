# 0903 method description

## 1. 문서의 목적

이 문서는 현재 코드에 구현된 다음 두 projection 계열을 실제 계산 순서대로
설명한다.

1. Target projection + single-stream gate
2. Target + source-all projection + single-stream gate

각 계열에는 generic object 성분을 다루는 네 가지 변형이 있다.

| Generic 방식 | Target condition | Target + source-all condition |
|---|---|---|
| Absolute direct | `class_target_absolute_direct_ssg` | `class_st_all_absolute_direct_ssg` |
| Absolute residual | `class_target_absolute_residual_ssg` | `class_st_all_absolute_residual_ssg` |
| No-object contextual | `class_target_no_object_ssg` | `class_st_all_no_object_ssg` |
| Superclass contextual | `class_target_superclass_ssg` | `class_st_all_superclass_ssg` |

이하의 설명은 일반화된 아이디어가 아니라 다음 코드의 현재 동작을 기준으로
한다.

- `orthogonalization_instantx_ip_adapter/batch_conditioning.py`
- `orthogonalization_instantx_ip_adapter/adapter.py`
- `orthogonalization_instantx_ip_adapter/orthogonalization.py`
- `orthogonalization_instantx_ip_adapter/attention_processor.py`

현재 0903 실험에서는 nuisance strength를 1.0으로 사용하고, generic class
strength를 0.5 또는 1.0으로 사용한다.

---

## 2. 기호와 실제 embedding 위치

### 2.1 Prompt 구성

다음 기호를 사용한다.

- $O$: `a dog` 또는 `a cat`
- $A_t$: target action 또는 behavior
- $B_t$: target background
- $A_s$: reference caption의 source behavior
- $B_s$: reference caption의 source background
- $S$: superclass prompt이며 현재 기본값은 `an animal`

문자열의 덧셈은 prompt 문자열 결합을 뜻한다. 예를 들어 $O+A_t$는
`a dog flying`과 같은 prompt다.

현재 14개 target prompt의 분류는 다음과 같다.

| 구조 | Prompt suffix |
|---|---|
| $O+A_t$ | `flying`, `sleeping`, `playing the piano`, `holding an umbrella`, `eating watermelon`, `riding a skateboard` |
| $O+B_t$ | `on a stage`, `in the snow` |
| $O+A_t+B_t$ | `flying in the sky`, `sleeping in the snow`, `playing the guitar on a stage`, `swimming in the pool`, `jumping over a wooden fence`, `reading a book on a sofa` |

### 2.2 SigLIP pooled embedding

SigLIP text tower의 pooled output을 $t(p)$라고 하고, 그 단위 벡터를 다음과
같이 정의한다.

$$
e(p)=\frac{t(p)}{\lVert t(p)\rVert_2}
$$

Reference image는 같은 SigLIP checkpoint의 image tower를 통과한다. 그
pooled output을 $v$라고 하고 다음처럼 정규화한다.

$$
z=\frac{v}{\lVert v\rVert_2}
$$

현재 구현은 SigLIP image pooled vector $v$와 SigLIP text pooled vector
$t(p)$를 동일한 차원의 비교 가능한 좌표로 직접 취급한다. 여기서 text
pooled embedding은 방향을 만들기 위해서만 추가로 계산된다. FLUX의 원래
text conditioning을 대체하지 않는다.

Projection은 128개 visual token으로 변환된 이후가 아니라, 변환 전의
global visual pooled vector $v$에 적용된다. 편집된 $v'$가 완성된 후에
InstantX의 `image_proj_model`이 이를 128개 visual token으로 변환한다.

### 2.3 정규화 연산

벡터 정규화를 간단히 다음과 같이 표기한다.

$$
N(x)=\frac{x}{\lVert x\rVert_2}
$$

코드는 다음 순서로 정규화한다.

1. 각 prompt pooled embedding을 먼저 정규화한다.
2. 두 prompt의 차분을 계산한다.
3. 각 차분 방향을 다시 정규화한다.
4. 여러 class contrast를 평균하는 경우, 정규화된 차분들을 평균하고 그
   평균을 다시 정규화한다.

따라서 아래 수식의 차분은 raw pooled vector 차분이 아니라 대부분
$e(p_1)-e(p_2)$ 형태다.

---

## 3. 공통 projection 계산

### 3.1 Nuisance basis

Action과 background 방향을 행으로 쌓은 행렬을 $D_N$이라고 한다. 코드에서는
다음 전치 행렬을 FP32 SVD로 분해한다.

$$
D_N^{\mathsf T}=U_N\Sigma_N V_N^{\mathsf T}
$$

가장 큰 singular value의 $10^{-3}$보다 큰 singular value에 대응하는
$U_N$의 열만 유지한다. 유지된 열을 $Q_N$이라고 하면 nuisance projection
matrix는 다음과 같다.

$$
P_N=Q_NQ_N^{\mathsf T}
$$

$Q_N$의 열은 orthonormal이므로 $P_Nz$는 $z$에서 nuisance subspace에
해당하는 성분이다.

### 3.2 Generic class basis

Generic class 입력 방향들을 행으로 쌓은 행렬을 $C$라고 하자. Class 방향을
nuisance에 대해 residualize하는 설정에서는 각 행에 다음 계산을 적용한다.

$$
R_C=C(I-P_N)
$$

각 residual row를 다시 정규화하고 너무 작은 row는 버린 뒤 SVD를 수행한다.
유지된 left singular vector들을 $Q_C$라고 정의한다.

$$
\widehat{R}_C^{\mathsf T}=U_C\Sigma_CV_C^{\mathsf T}
$$

$$
P_C=Q_CQ_C^{\mathsf T}
$$

Absolute direct만 $R_C$ 대신 원래 $C$를 바로 SVD에 사용한다. Absolute
residual, no-object, superclass는 모두 nuisance residualization을 사용한다.

### 3.3 Visual pooled vector에서의 최종 차감

Nuisance strength를 $\lambda_N$, class strength를 $\lambda_C$라고 하면,
단위 visual vector는 다음처럼 편집된다.

$$
\widetilde{z}
=z-\lambda_NP_Nz-\lambda_CP_Cz
$$

구현은 마지막에 원래 visual pooled vector의 norm을 복원한다.

$$
v'
=\lVert v\rVert_2
\frac{\widetilde{z}}{\lVert\widetilde{z}\rVert_2}
$$

0903 실험 설정은 다음과 같다.

$$
\lambda_N=1.0,\qquad
\lambda_C\in\{0.5,1.0\}
$$

$\lambda_C=0.5$는 class component를 절반만 약화하므로 완전한 직교화를
의미하지 않는다. $\lambda_C=1.0$은 해당 class basis 성분을 전부 차감한다.

---

## 4. Target projection의 nuisance 계산: 모든 target case

이 절의 계산은 다음 다섯 condition에서 nuisance branch로 공통 사용된다.

- `target_projection_ssg`
- `class_target_absolute_direct_ssg`
- `class_target_absolute_residual_ssg`
- `class_target_no_object_ssg`
- `class_target_superclass_ssg`

### 4.1 Target가 $O+A_t$인 경우

예: `a dog flying`

Action direction 하나만 만든다.

$$
d_{A_t}=N\left(e(O+A_t)-e(O)\right)
$$

따라서 nuisance direction matrix는 한 행이다.

$$
D_N=
\begin{bmatrix}
d_{A_t}^{\mathsf T}
\end{bmatrix}
$$

이 경우 nuisance rank는 정상적인 입력이라면 최대 1이다.

### 4.2 Target가 $O+B_t$인 경우

예: `a cat in the snow`

Background direction 하나만 만든다.

$$
d_{B_t}=N\left(e(O+B_t)-e(O)\right)
$$

$$
D_N=
\begin{bmatrix}
d_{B_t}^{\mathsf T}
\end{bmatrix}
$$

이 경우에도 nuisance rank는 최대 1이다.

### 4.3 Target가 $O+A_t+B_t$인 경우

예: `a dog flying in the sky`

단순히 $e(O+A_t+B_t)-e(O)$를 사용하지 않는다. Action과 background의
factorial main effect를 각각 계산한다.

$$
d_{A_t}^{\mathrm{raw}}
=\frac{1}{2}
\left[
e(O+A_t)-e(O)
+e(O+A_t+B_t)-e(O+B_t)
\right]
$$

$$
d_{B_t}^{\mathrm{raw}}
=\frac{1}{2}
\left[
e(O+B_t)-e(O)
+e(O+A_t+B_t)-e(O+A_t)
\right]
$$

$$
d_{A_t}=N\left(d_{A_t}^{\mathrm{raw}}\right),\qquad
d_{B_t}=N\left(d_{B_t}^{\mathrm{raw}}\right)
$$

$$
D_N=
\begin{bmatrix}
d_{A_t}^{\mathsf T}\\
d_{B_t}^{\mathsf T}
\end{bmatrix}
$$

즉 action effect는 background가 없을 때와 있을 때의 두 차분을 평균하고,
background effect도 action이 없을 때와 있을 때의 두 차분을 평균한다.
Interaction direction은 현재 코드에서 포함하지 않는다.

---

## 5. Target projection에 generic object 제거를 결합하는 네 방식

이 절에서는 4절에서 얻은 $P_N$에 어떤 class 방향을 추가하는지 target
prompt 구조별로 모두 정리한다.

### 5.1 Absolute direct

Condition은 `class_target_absolute_direct_ssg`다. 모든 target 구조에서 동일한
절대 anchor를 사용한다.

$$
c_{\mathrm{abs}}=e(O)
$$

Target가 $O+A_t$, $O+B_t$, $O+A_t+B_t$ 중 무엇이더라도 class 입력은
$c_{\mathrm{abs}}$ 하나뿐이다. 이 벡터를 nuisance에 대해 residualize하지
않고 그대로 class basis로 사용한다.

$$
Q_C=c_{\mathrm{abs}}
$$

따라서 $Q_C$와 $Q_N$이 겹칠 수 있다. 최종 식에서 겹친 성분이 nuisance
차감과 class 차감에 의해 두 번 영향을 받을 수 있도록 의도적으로 남긴
ablation이다.

### 5.2 Absolute residual

Condition은 `class_target_absolute_residual_ssg`다. Absolute anchor 자체는
동일하다.

$$
c_{\mathrm{abs}}=e(O)
$$

다만 먼저 nuisance subspace와 겹치는 부분을 제거한다.

$$
r_{\mathrm{abs}}
=(I-P_N)c_{\mathrm{abs}}
$$

$$
q_C=N\left(r_{\mathrm{abs}}\right)
$$

Residual norm이 threshold보다 작으면 class 방향은 drop된다. 그렇지 않으면
$Q_C=q_C$가 된다. 따라서 target 구조가 달라지면 $P_N$이 달라지고, 같은
`a dog` 또는 `a cat` anchor를 사용하더라도 최종 class residual 방향은
달라진다.

### 5.3 No-object contextual

Condition은 `class_target_no_object_ssg`다. Object를 포함한 prompt와 동일
context만 남긴 prompt를 비교한다.

#### Case T-NO-1: $O+A_t$

$$
\delta_{A_t}^{O}
=N\left(e(O+A_t)-e(A_t)\right)
$$

$$
c_t=\delta_{A_t}^{O}
$$

예를 들면 `a dog flying`과 `flying`을 비교한다.

#### Case T-NO-2: $O+B_t$

$$
\delta_{B_t}^{O}
=N\left(e(O+B_t)-e(B_t)\right)
$$

$$
c_t=\delta_{B_t}^{O}
$$

예를 들면 `a cat in the snow`와 `in the snow`를 비교한다.

#### Case T-NO-3: $O+A_t+B_t$

세 context에서 object contribution을 각각 계산한다.

$$
\delta_{A_t}^{O}
=N\left(e(O+A_t)-e(A_t)\right)
$$

$$
\delta_{B_t}^{O}
=N\left(e(O+B_t)-e(B_t)\right)
$$

$$
\delta_{A_tB_t}^{O}
=N\left(e(O+A_t+B_t)-e(A_t+B_t)\right)
$$

정규화된 세 방향을 동일 가중치로 평균하고 다시 정규화한다.

$$
c_t
=N\left(
\frac{
\delta_{A_t}^{O}
+\delta_{B_t}^{O}
+\delta_{A_tB_t}^{O}
}{3}
\right)
$$

그다음 모든 case에서 $c_t$를 nuisance에 대해 residualize한다.

$$
r_t=(I-P_N)c_t
$$

$r_t$를 정규화한 한 방향이 class basis가 된다. 즉 O+A 또는 O+B에서는
한 contrast만 사용하고, O+A+B에서만 세 context의 평균을 사용한다.

### 5.4 Superclass contextual

Condition은 `class_target_superclass_ssg`다. No-object 방식과 계산 순서는
같지만 object를 완전히 지운 prompt 대신 `an animal`로 교체한 prompt를
negative로 사용한다.

#### Case T-SC-1: $O+A_t$

$$
\delta_{A_t}^{O-S}
=N\left(e(O+A_t)-e(S+A_t)\right)
$$

$$
c_t=\delta_{A_t}^{O-S}
$$

#### Case T-SC-2: $O+B_t$

$$
\delta_{B_t}^{O-S}
=N\left(e(O+B_t)-e(S+B_t)\right)
$$

$$
c_t=\delta_{B_t}^{O-S}
$$

#### Case T-SC-3: $O+A_t+B_t$

$$
\delta_{A_t}^{O-S}
=N\left(e(O+A_t)-e(S+A_t)\right)
$$

$$
\delta_{B_t}^{O-S}
=N\left(e(O+B_t)-e(S+B_t)\right)
$$

$$
\delta_{A_tB_t}^{O-S}
=N\left(e(O+A_t+B_t)-e(S+A_t+B_t)\right)
$$

$$
c_t
=N\left(
\frac{
\delta_{A_t}^{O-S}
+\delta_{B_t}^{O-S}
+\delta_{A_tB_t}^{O-S}
}{3}
\right)
$$

모든 case에서 다음 residualization을 거친다.

$$
r_t=(I-P_N)c_t
$$

Superclass contrast는 dog/cat과 animal에 공통인 성분을 차분에서 상쇄하고,
animal보다 더 구체적인 dog/cat class 차이를 추출하려는 설정이다. 반면
no-object는 context-only prompt와 비교하므로 object 존재 자체의 기여를 더
넓게 포착할 수 있다.

---

## 6. ST-all projection의 nuisance 계산: 모든 target case

이 절의 계산은 다음 다섯 condition에서 nuisance branch로 공통 사용된다.

- `st_all_projection_ssg`
- `class_st_all_absolute_direct_ssg`
- `class_st_all_absolute_residual_ssg`
- `class_st_all_no_object_ssg`
- `class_st_all_superclass_ssg`

`all`의 핵심은 target prompt에 action만 있거나 background만 있더라도 source
caption에서는 $A_s$와 $B_s$를 항상 모두 nuisance 후보로 사용한다는 점이다.

### 6.1 Source caption의 두 nuisance 방향

모든 reference caption은 $O+A_s+B_s$로 factorize된다. Source action과
background 방향은 항상 다음 factorial main effect로 계산된다.

$$
d_{A_s}^{\mathrm{raw}}
=\frac{1}{2}
\left[
e(O+A_s)-e(O)
+e(O+A_s+B_s)-e(O+B_s)
\right]
$$

$$
d_{B_s}^{\mathrm{raw}}
=\frac{1}{2}
\left[
e(O+B_s)-e(O)
+e(O+A_s+B_s)-e(O+A_s)
\right]
$$

$$
d_{A_s}=N\left(d_{A_s}^{\mathrm{raw}}\right),\qquad
d_{B_s}=N\left(d_{B_s}^{\mathrm{raw}}\right)
$$

Source interaction은 포함하지 않는다.

### 6.2 Target가 $O+A_t$인 경우

Target에서는 4.1의 $d_{A_t}$ 하나를 만들고, source의 두 방향을 붙인다.

$$
D_N=
\begin{bmatrix}
d_{A_t}^{\mathsf T}\\
d_{A_s}^{\mathsf T}\\
d_{B_s}^{\mathsf T}
\end{bmatrix}
$$

### 6.3 Target가 $O+B_t$인 경우

$$
D_N=
\begin{bmatrix}
d_{B_t}^{\mathsf T}\\
d_{A_s}^{\mathsf T}\\
d_{B_s}^{\mathsf T}
\end{bmatrix}
$$

### 6.4 Target가 $O+A_t+B_t$인 경우

Target factorial main effect 두 개와 source factorial main effect 두 개를
모두 붙인다.

$$
D_N=
\begin{bmatrix}
d_{A_t}^{\mathsf T}\\
d_{B_t}^{\mathsf T}\\
d_{A_s}^{\mathsf T}\\
d_{B_s}^{\mathsf T}
\end{bmatrix}
$$

각 행을 단순히 순차적으로 차감하지 않고, 위 행렬 전체에 한 번의 SVD를
적용하여 공유 nuisance basis $Q_N$을 만든다. 유사하거나 중복되는 target와
source 방향은 singular rank에 따라 하나의 subspace로 정리될 수 있다.

---

## 7. ST-all에 generic object 제거를 결합하는 네 방식

### 7.1 Absolute direct

Condition은 `class_st_all_absolute_direct_ssg`다. Target와 source가 같은 species
object $O$를 사용하므로 absolute class 방향은 별도의 target/source 두
방향이 아니라 다음 한 방향이다.

$$
c_{\mathrm{abs}}=e(O)
$$

Target가 O+A, O+B, O+A+B 중 무엇이든 $c_{\mathrm{abs}}$는 동일하다. 다만
각 case의 $P_N$은 6절처럼 달라진다. Direct 방식에서는 $P_N$과의 겹침을
제거하지 않고 $c_{\mathrm{abs}}$를 그대로 $Q_C$로 사용한다.

### 7.2 Absolute residual

Condition은 `class_st_all_absolute_residual_ssg`다.

$$
c_{\mathrm{abs}}=e(O)
$$

$$
r_{\mathrm{abs}}=(I-P_N)c_{\mathrm{abs}}
$$

$$
q_C=N\left(r_{\mathrm{abs}}\right)
$$

ST-all의 $P_N$에는 target nuisance뿐 아니라 source action과 background도
포함된다. 따라서 Target-only absolute residual보다 더 넓은 nuisance
subspace와 겹치는 class 성분을 제거한 뒤 남은 방향을 사용한다.

### 7.3 Source contextual class 방향의 공통 계산

No-object와 superclass의 ST-all 구현은 target class 방향 $c_t$와 source
class 방향 $c_s$를 각각 하나씩 만든다. 두 방향을 처음부터 한 평균으로
합치지 않는다.

Source 쪽은 target 구조와 무관하게 source caption의 세 context를 항상
사용한다.

#### No-object source 방향

$$
\delta_{A_s}^{O}
=N\left(e(O+A_s)-e(A_s)\right)
$$

$$
\delta_{B_s}^{O}
=N\left(e(O+B_s)-e(B_s)\right)
$$

$$
\delta_{A_sB_s}^{O}
=N\left(e(O+A_s+B_s)-e(A_s+B_s)\right)
$$

$$
c_s^{\mathrm{NO}}
=N\left(
\frac{
\delta_{A_s}^{O}
+\delta_{B_s}^{O}
+\delta_{A_sB_s}^{O}
}{3}
\right)
$$

#### Superclass source 방향

$$
\delta_{A_s}^{O-S}
=N\left(e(O+A_s)-e(S+A_s)\right)
$$

$$
\delta_{B_s}^{O-S}
=N\left(e(O+B_s)-e(S+B_s)\right)
$$

$$
\delta_{A_sB_s}^{O-S}
=N\left(e(O+A_s+B_s)-e(S+A_s+B_s)\right)
$$

$$
c_s^{\mathrm{SC}}
=N\left(
\frac{
\delta_{A_s}^{O-S}
+\delta_{B_s}^{O-S}
+\delta_{A_sB_s}^{O-S}
}{3}
\right)
$$

### 7.4 No-object contextual: 모든 target case

Condition은 `class_st_all_no_object_ssg`다.

#### Case STA-NO-1: $O+A_t$

Target class 방향은 한 contrast다.

$$
c_t=N\left(e(O+A_t)-e(A_t)\right)
$$

Class 입력 행렬은 다음 두 행이다.

$$
C=
\begin{bmatrix}
c_t^{\mathsf T}\\
\left(c_s^{\mathrm{NO}}\right)^{\mathsf T}
\end{bmatrix}
$$

#### Case STA-NO-2: $O+B_t$

$$
c_t=N\left(e(O+B_t)-e(B_t)\right)
$$

$$
C=
\begin{bmatrix}
c_t^{\mathsf T}\\
\left(c_s^{\mathrm{NO}}\right)^{\mathsf T}
\end{bmatrix}
$$

#### Case STA-NO-3: $O+A_t+B_t$

Target 방향은 5.3의 세 normalized contrast 평균이다.

$$
c_t
=N\left(
\frac{
N(e(O+A_t)-e(A_t))
+N(e(O+B_t)-e(B_t))
+N(e(O+A_t+B_t)-e(A_t+B_t))
}{3}
\right)
$$

$$
C=
\begin{bmatrix}
c_t^{\mathsf T}\\
\left(c_s^{\mathrm{NO}}\right)^{\mathsf T}
\end{bmatrix}
$$

세 case 모두 각 class row를 먼저 nuisance에 대해 residualize한다.

$$
R_C=C(I-P_N)
$$

Retained residual rows를 정규화한 뒤 함께 SVD하여 $Q_C$를 만든다. 따라서
target와 source object 방향은 각각 독립적인 한 행을 갖고, 최종 class
subspace rank는 최대 2다. 두 방향이 거의 같다면 SVD rank가 사실상 1로
정리될 수 있다.

### 7.5 Superclass contextual: 모든 target case

Condition은 `class_st_all_superclass_ssg`다.

#### Case STA-SC-1: $O+A_t$

$$
c_t=N\left(e(O+A_t)-e(S+A_t)\right)
$$

$$
C=
\begin{bmatrix}
c_t^{\mathsf T}\\
\left(c_s^{\mathrm{SC}}\right)^{\mathsf T}
\end{bmatrix}
$$

#### Case STA-SC-2: $O+B_t$

$$
c_t=N\left(e(O+B_t)-e(S+B_t)\right)
$$

$$
C=
\begin{bmatrix}
c_t^{\mathsf T}\\
\left(c_s^{\mathrm{SC}}\right)^{\mathsf T}
\end{bmatrix}
$$

#### Case STA-SC-3: $O+A_t+B_t$

$$
c_t
=N\left(
\frac{
N(e(O+A_t)-e(S+A_t))
+N(e(O+B_t)-e(S+B_t))
+N(e(O+A_t+B_t)-e(S+A_t+B_t))
}{3}
\right)
$$

$$
C=
\begin{bmatrix}
c_t^{\mathsf T}\\
\left(c_s^{\mathrm{SC}}\right)^{\mathsf T}
\end{bmatrix}
$$

세 case 모두 다음 순서를 따른다.

1. ST-all nuisance rows로 $Q_N$과 $P_N$을 계산한다.
2. $C(I-P_N)$으로 target/source class 방향 각각의 nuisance overlap을
   제거한다.
3. 남은 class rows를 정규화하고 함께 SVD하여 $Q_C$를 계산한다.
4. $\lambda_NP_Nz$와 $\lambda_CP_Cz$를 visual pooled vector에서 차감한다.

---

## 8. 전체 case 요약표

### 8.1 Nuisance row 구성

| Projection 계열 | Target $O+A_t$ | Target $O+B_t$ | Target $O+A_t+B_t$ |
|---|---|---|---|
| Target | $[d_{A_t}]$ | $[d_{B_t}]$ | $[d_{A_t},d_{B_t}]$ |
| ST-all | $[d_{A_t},d_{A_s},d_{B_s}]$ | $[d_{B_t},d_{A_s},d_{B_s}]$ | $[d_{A_t},d_{B_t},d_{A_s},d_{B_s}]$ |

대괄호 안의 모든 방향을 행으로 쌓아 하나의 nuisance SVD를 수행한다.

### 8.2 Class row 구성

| Generic 방식 | Target $O+A_t$ | Target $O+B_t$ | Target $O+A_t+B_t$ |
|---|---|---|---|
| Absolute | $[e(O)]$ | $[e(O)]$ | $[e(O)]$ |
| Target no-object | $[N(e(O+A_t)-e(A_t))]$ | $[N(e(O+B_t)-e(B_t))]$ | target의 A, B, AB 세 object contrast를 평균한 한 방향 |
| Target superclass | $[N(e(O+A_t)-e(S+A_t))]$ | $[N(e(O+B_t)-e(S+B_t))]$ | target의 A, B, AB 세 O-vs-S contrast를 평균한 한 방향 |
| ST-all no-object | $[c_t,c_s^{\mathrm{NO}}]$ | $[c_t,c_s^{\mathrm{NO}}]$ | $[c_t,c_s^{\mathrm{NO}}]$ |
| ST-all superclass | $[c_t,c_s^{\mathrm{SC}}]$ | $[c_t,c_s^{\mathrm{SC}}]$ | $[c_t,c_s^{\mathrm{SC}}]$ |

Absolute direct만 class rows를 nuisance residualization 없이 사용한다. 나머지
세 방식은 모두 $C(I-P_N)$을 거친다.

---

## 9. SSG가 projection과 결합되는 위치

위의 모든 여덟 generic condition에는 SSG가 항상 켜진다. SSG는 pooled
embedding projection과 별개의 개입이다.

먼저 편집된 visual pooled vector $v'$가 `image_proj_model`을 통과해 128개
IP visual token을 만든다. 각 attention processor는 이 token에서 IP key와
value를 계산하고 현재 hidden-state query와 attention을 수행한다.

Single-stream block의 결합 sequence를 다음처럼 쓰자.

$$
H=
\begin{bmatrix}
H_{\mathrm{text}}\\
H_{\mathrm{latent}}
\end{bmatrix}
$$

IP attention 결과를 $R_{\mathrm{IP}}$라고 하면 SSG mask는 text row에서 0,
latent-image row에서 1이다.

$$
M_i=
\begin{cases}
0, & i\text{가 text row인 경우}\\
1, & i\text{가 latent-image row인 경우}
\end{cases}
$$

Single-stream의 직접 IP residual 업데이트는 다음과 같다.

$$
H' = H + s_{\mathrm{IP}}\left(M\odot R_{\mathrm{IP}}\right)
$$

즉 SSG는 single-stream에서 IP residual이 text row에 직접 더해지는 것만
막는다. Latent-image row의 IP residual은 유지한다. Double-stream에서는
원래 InstantX 구현처럼 IP residual이 image hidden states에만 더해지므로
해당 경로를 변경하지 않는다.

---

## 10. 해석할 때 주의할 점

1. 이 방법이 제거하는 것은 `dog`, `cat`, action 또는 background의 완벽한
   의미 성분이 아니다. SigLIP pooled space에서 prompt contrast로 추정한
   방향에 대한 visual vector의 projection 성분이다.
2. Absolute anchor $e(O)$는 순수 class 차분이 아니라 prompt의 절대 위치다.
   따라서 generic class 외의 공통적 언어 성분을 함께 포함할 수 있다.
3. Absolute direct는 nuisance와 class basis의 overlap을 의도적으로 허용한다.
   $\lambda_N=\lambda_C=1$이어도 두 basis가 서로 직교하지 않기 때문에 결과가
   두 subspace 모두에 완전히 직교한다고 보장할 수 없다.
4. Absolute residual, no-object, superclass는 class 방향을 nuisance에 대해
   residualize하므로 두 차감의 역할이 더 분리된다.
5. No-object와 superclass의 O+A+B case에서 세 contrast는 각각 정규화된 뒤
   평균된다. 길이가 큰 특정 contrast가 평균을 지배하지 않도록 한 구현이다.
6. ST-all의 contextual 방식은 target와 source의 모든 contrast를 하나로
   평균하지 않는다. Target 내부 contrast를 평균해 $c_t$ 하나, source 내부
   contrast를 평균해 $c_s$ 하나를 만든 뒤, 두 방향을 class SVD의 별도 row로
   사용한다.
7. Projection 후 norm을 원래 visual pooled norm으로 복원하므로 실험의 주된
   변화는 vector 크기보다 방향 변화다.

---

## 11. 한 줄 계산 흐름

### Target generic condition

$$
\text{target contrasts}
\longrightarrow Q_N
\longrightarrow \text{target 또는 absolute class direction}
\longrightarrow Q_C
\longrightarrow v'
\longrightarrow 128\text{ visual tokens}
\longrightarrow \mathrm{IP\ attention}+\mathrm{SSG}
$$

### ST-all generic condition

$$
\text{target contrasts}+\text{source action/background contrasts}
\longrightarrow Q_N
\longrightarrow \text{absolute 또는 target/source class directions}
\longrightarrow Q_C
\longrightarrow v'
\longrightarrow 128\text{ visual tokens}
\longrightarrow \mathrm{IP\ attention}+\mathrm{SSG}
$$
