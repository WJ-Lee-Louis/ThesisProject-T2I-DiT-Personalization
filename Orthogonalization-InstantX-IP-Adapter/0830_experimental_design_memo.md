# 0830 Experimental Design Memo

## 1. 연구 목적

Reference image의 SigLIP pooled visual embedding에는 다음 정보가 함께
포함될 수 있다.

- personalization 대상의 고유 identity
- 행동과 자세
- 배경과 촬영 환경
- dog/cat category에서 흔한 generic object 특징

기존 target/source projection은 행동과 배경 leakage를 줄이는 데 초점을
맞췄다. 이번 실험은 이 기능을 유지하면서 generic object 성분도 약하게
제거하되, identity 손상을 분석할 수 있도록 object projection에 별도의
lambda를 적용하는 것이 목적이다. 모든 방법은 training-free이며 baseline을
제외한 intervention condition에는 single-stream text-row gate를 적용한다.

## 2. 단순 union SVD 대신 계층적 직교화를 사용하는 이유

정규화한 SigLIP pooled text embedding을 $\hat{e}(p)$라고 하자. O+A
prompt에서는 다음 두 방향을 생각할 수 있다.

$$
d_A =
\frac{\hat{e}(O+A)-\hat{e}(O)}
{\left\lVert \hat{e}(O+A)-\hat{e}(O) \right\rVert_2}
$$

$$
d_O =
\frac{\hat{e}(O+A)-\hat{e}(A)}
{\left\lVert \hat{e}(O+A)-\hat{e}(A) \right\rVert_2}
$$

$d_A$와 $d_O$를 한 번에 SVD하면 두 의미가 SVD basis 안에서 섞인다.
따라서 어떤 basis 열이 action이고 어떤 열이 object인지 보장할 수 없으며,
object에만 작은 lambda를 지정하기 어렵다. SVD 전에 object row의 크기를
줄이더라도 full-rank subspace 자체는 동일하므로 완전한 정사영 결과는
거의 달라지지 않는다.

이번 설계는 nuisance를 우선하는 계층적 직교화를 사용한다.

1. Action/background 방향들로 nuisance SVD basis $Q_N$을 만든다.
2. Object 방향 $c_O$에서 nuisance와 겹치는 부분을 제거한다.
3. 남은 object residual을 정규화하여 $q_O$를 만든다.
4. 서로 직교하는 두 subspace에 별도의 lambda를 적용한다.

$$
r_O = \left(I-Q_N Q_N^{\mathsf{T}}\right)c_O
$$

$$
q_O = \frac{r_O}{\left\lVert r_O \right\rVert_2}
$$

$$
v^{\prime}
=
v
- \lambda_N Q_N Q_N^{\mathsf{T}}v
- \lambda_O q_O q_O^{\mathsf{T}}v
$$

마지막에만 원래 visual embedding의 norm을 복원한다. $Q_N$과 $q_O$가
직교하므로 중복 차감과 적용 순서 문제가 없다.

## 3. Object 방향을 만드는 두 counterfactual

### 3.1 No-object 방식

같은 문맥에서 object가 있을 때와 없을 때를 비교한다.

$$
\delta_C =
\frac{\hat{e}(O+C)-\hat{e}(C)}
{\left\lVert \hat{e}(O+C)-\hat{e}(C) \right\rVert_2}
$$

예시는 “a dog flying” 대 “flying”이다. 이 방향에는 해당 문맥에서
드러나는 dog/cat 성분과 reference identity에 유효한 object 정보가 함께
들어갈 수 있으므로 $\lambda_O$를 작게 시작해야 한다.

### 3.2 Superclass 또는 animal 방식

Object를 없애는 대신 superclass $S$로 교체한다. 현재 기본 $S$는
“an animal”이다.

$$
\delta_C^{S} =
\frac{\hat{e}(O+C)-\hat{e}(S+C)}
{\left\lVert \hat{e}(O+C)-\hat{e}(S+C) \right\rVert_2}
$$

예시는 “a dog flying” 대 “an animal flying”이다. 이는 animal 공통 성분
자체보다 generic animal과 비교했을 때 남는 dog/cat-specific 성분을
추정한다. 실제 animal 공통 성분 $\hat{e}(S)$를 직접 제거하는 실험과는
구분해야 한다.

## 4. Prompt 구조별 동작

### 4.1 O+A

- Nuisance basis: $\hat{e}(O+A)-\hat{e}(O)$에서 얻은 action 방향 하나
- Object 방향: $(O+A)-A$ 또는 $(O+A)-(S+A)$
- 최종 rank 상한: nuisance 1 + object residual 1

### 4.2 O+B

- Nuisance basis: $\hat{e}(O+B)-\hat{e}(O)$에서 얻은 background 방향 하나
- Object 방향: $(O+B)-B$ 또는 $(O+B)-(S+B)$
- 최종 rank 상한: nuisance 1 + object residual 1

### 4.3 O+A+B

기존 factorial main effect를 유지한다.

$$
d_A =
\frac{1}{2}
\left[
\left(\hat{e}(O+A)-\hat{e}(O)\right)
+
\left(\hat{e}(O+A+B)-\hat{e}(O+B)\right)
\right]
$$

$$
d_B =
\frac{1}{2}
\left[
\left(\hat{e}(O+B)-\hat{e}(O)\right)
+
\left(\hat{e}(O+A+B)-\hat{e}(O+A)\right)
\right]
$$

$d_A$와 $d_B$를 row로 쌓은 행렬을 SVD하여 $Q_N$을 만든다. Object
방향은 다음 세 contextual contrast를 각각 정규화한 뒤 평균하고 다시
정규화한다.

- $(O+A)-A$ 또는 $(O+A)-(S+A)$
- $(O+B)-B$ 또는 $(O+B)-(S+B)$
- $(O+A+B)-(A+B)$ 또는 $(O+A+B)-(S+A+B)$

세 object contrast를 개별 SVD row로 사용하지 않는다. 평균한 object
방향 하나를 $Q_N$에 대해 residualize한다. 최종 rank 상한은
nuisance 2 + object residual 1이다.

## 5. Target와 source 조건의 차이

### Target hierarchical projection

Target prompt에서 nuisance 방향과 object 방향을 모두 계산한다.

- O+A 또는 O+B: 하나의 target nuisance 방향
- O+A+B: 두 factorial target nuisance 방향
- Target prompt가 바뀌면 class direction도 달라진다.

### Source hierarchical projection

Nuisance 방향은 reference caption의 source behavior/background에서
계산한다. Target prompt taxonomy에 맞춰 제거할 source factor를 선택한다.

- Target가 O+A이면 source behavior nuisance만 사용
- Target가 O+B이면 source background nuisance만 사용
- Target가 O+A+B이면 source behavior와 background nuisance를 사용

Object 방향은 source caption의 $(O+A_s)$, $(O+B_s)$,
$(O+A_s+B_s)$ 세 contrast를 평균한다. 같은 reference에서는 평균하기
전의 object 방향 구성이 공통이다. 다만 target prompt 종류에 따라
nuisance basis가 달라지므로 residualized object 방향은 달라질 수 있다.

### Target + source hierarchical projection

Target와 source의 action/background 방향을 `st_matched_projection_ssg`처럼 한
행렬에 쌓아 SVD하고, 하나의 nuisance basis $Q_N$을 만든다. 그다음
generic object 방향은 서로 섞어 평균하지 않고 다음처럼 별도로 계산한다.

- $c_T$: target contextual contrast들의 정규화 평균
- $c_S$: source-caption contextual contrast 3개의 정규화 평균

각 방향에서 joint nuisance 성분을 먼저 제거한다.

$$
r_T = \left(I-Q_N Q_N^{\mathsf{T}}\right)c_T,
\qquad
r_S = \left(I-Q_N Q_N^{\mathsf{T}}\right)c_S
$$

유효한 residual들을 정규화하여 row로 쌓고 두 번째 SVD를 수행한다.

$$
M_O =
\begin{bmatrix}
r_T / \lVert r_T \rVert_2 \\
r_S / \lVert r_S \rVert_2
\end{bmatrix}^{\mathsf{T}},
\qquad
M_O = U_O \Sigma_O V_O^{\mathsf{T}}
$$

$Q_O$는 $U_O$에서 singular-value threshold를 넘는 열들로 구성한다.
$Q_O$의 rank 상한은 2이며, 두 residual이 거의 같은 방향이면 SVD가 rank
1로 줄인다. 최종 편집은 다음과 같다.

$$
v^{\prime}
=v
-\lambda_N Q_N Q_N^{\mathsf{T}}v
-\lambda_O Q_O Q_O^{\mathsf{T}}v
$$

Target contrast가 1개인 O+A/O+B에서도 source의 3개 contrast와 직접 평균하지
않는다. Target와 source를 각각 한 방향으로 먼저 요약하므로 source 쪽이
contrast 개수만으로 세 배 가중되는 것을 방지한다. Source nuisance factor는
target taxonomy에 맞춰 선택하며, O+A이면 양쪽 behavior, O+B이면 양쪽
background, O+A+B이면 양쪽 behavior/background를 joint nuisance SVD에 넣는다.

## 6. 최종 condition 설계

`combined` 대신 single-stream gate를 명확히 표시하는 `ssg` 접미사를
사용한다. 신규 inference condition은 control 2개, 기존 nuisance 4개,
generic 16개로 정확히 22개다.

| Nuisance 출처 | 기존 projection | Absolute direct | Absolute residual | No-object | Superclass |
|---|---|---|---|---|---|
| Target | `target_projection_ssg` | `class_target_absolute_direct_ssg` | `class_target_absolute_residual_ssg` | `class_target_no_object_ssg` | `class_target_superclass_ssg` |
| Source | `source_projection_ssg` | `class_source_absolute_direct_ssg` | `class_source_absolute_residual_ssg` | `class_source_no_object_ssg` | `class_source_superclass_ssg` |
| ST matched | `st_matched_projection_ssg` | `class_st_matched_absolute_direct_ssg` | `class_st_matched_absolute_residual_ssg` | `class_st_matched_no_object_ssg` | `class_st_matched_superclass_ssg` |
| ST all | `st_all_projection_ssg` | `class_st_all_absolute_direct_ssg` | `class_st_all_absolute_residual_ssg` | `class_st_all_no_object_ssg` | `class_st_all_superclass_ssg` |

두 control은 `baseline`과 `ssg`다. 기존 class-only absolute condition과
모호한 `combined` 명칭은 신규 제출 목록에서 제거한다.

Absolute direct도 nuisance projection과 함께 적용하지만 $c_O$를 $Q_N$에
대해 residualize하지 않는다. 따라서 두 projector의 겹침과 이중 차감
가능성을 의도적으로 유지하는 ablation이다. Absolute residual은 기존
계층적 식을 사용하여 겹침을 제거한다. No-object와 superclass는 항상
residual 방식이다.

## 7. Lambda의 의미와 권장 sweep

- orthogonalization-strength: $\lambda_N$, action/background attenuation
- class-projection-strength: $\lambda_O$, generic object attenuation

우선 nuisance 효과를 기존 실험과 맞추기 위해 $\lambda_N=1.0$으로
고정하고 $\lambda_O$만 비교하는 것이 해석하기 쉽다.

| 설정 | $\lambda_N$ | $\lambda_O$ | 목적 |
|---|---:|---:|---|
| Partial | 1.0 | 0.50 | class 억제와 identity 손상의 trade-off |
| Full | 1.0 | 1.00 | full projection 상한 및 실패 양상 확인 |

Generic 16개 condition은 $\lambda_O=0.5$와 $1.0$만 허용한다. 따라서
condition 이름은 22개지만 실제 생성 결과는 control 2개, nuisance 4개,
generic 32개로 총 38개 batch가 된다.

## 8. 반드시 확인할 diagnostics

- nuisance_basis_rank, nuisance_singular_values
- class_overlap_norm: class 방향 중 nuisance와 겹친 크기
- class_residual_norm, class_residual_ratio
- class_direction_count, class_basis_rank, class_dropped_rank
- class_singular_values: target/source object residual의 독립성 및 유효 rank
- class_direction_dropped: 작은 residual을 생략했는지 여부
- class_overlap_policy: absolute direct와 residual 구분
- nuisance_component_norm, class_component_norm
- nuisance_removed_component_norm, class_removed_component_norm
- cosine_to_input, edit_norm
- single-stream gate 활성화 여부

class_residual_ratio가 매우 작으면 object 방향 대부분이 nuisance basis와
겹친다는 의미다. 작은 수치 오차를 확대하지 않도록 이 경우 class
residual을 drop한다. 반대로 residual ratio와
class_removed_component_norm이 모두 크다면 identity 손상 가능성을 정성
결과와 함께 확인해야 한다.

## 9. 권장 비교 순서

1. `baseline`과 `ssg`
2. 기존 nuisance 4개
3. Absolute direct와 residual, $\lambda_O=0.5$
4. No-object와 superclass, $\lambda_O=0.5$
5. Generic 16개를 $\lambda_O=1.0$으로 반복
6. identity 보존, prompt controllability, behavior/background leakage 비교

동물 위주 데이터에서는 털무늬, 귀 모양, 얼굴 비율, 체형과 같은
reference-specific 특징과 generic dog/cat 특징을 분리해 평가해야 한다.
Object attenuation이 generic artifact를 줄이더라도 reference 고유의
털 패턴까지 사라진다면 성공으로 판단하면 안 된다.

Modal 실행 명령은
[CLASS_PROJECTION_MODAL_GUIDE.md](CLASS_PROJECTION_MODAL_GUIDE.md)에 정리되어
있다.
