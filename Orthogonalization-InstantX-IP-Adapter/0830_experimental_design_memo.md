# 0830 Experimental Design Memo

## 1. 연구 목적

Reference image의 SigLIP pooled visual embedding에는 다음 정보가 함께
포함될 수 있다.

- personalization 대상의 고유 identity
- 행동 및 자세
- 배경과 촬영 환경
- dog/cat처럼 category 전체에서 흔한 generic object 특징

기존 target/source projection은 행동과 배경 leakage를 줄이는 데 초점을
맞췄다. 이번 실험은 그 기능을 유지하면서 generic object 성분도 약하게
제거하되, identity 손상을 막기 위해 object projection에 별도의 작은
lambda를 적용하는 것이 목적이다. 모든 실험은 training-free이며
single-stream text-row gate를 함께 사용한다.

## 2. 핵심 설계가 단순 union SVD가 아닌 이유

O+A prompt에서 다음 두 방향을 생각할 수 있다.

$$
d_A=rac{\hat e(O+A)-\hat e(O)}
{\left\lVert\hat e(O+A)-\hat e(O)\right\rVert_2}
$$

$$
d_O=rac{\hat e(O+A)-\hat e(A)}
{\left\lVert\hat e(O+A)-\hat e(A)\right\rVert_2}
$$

여기서 $\hat e(p)$는 정규화한 SigLIP pooled text embedding이다.

$[d_A;d_O]$를 한 번에 SVD하면 두 의미가 SVD basis 안에서 섞인다. 따라서
어떤 basis 열이 action이고 어떤 열이 object인지 보장할 수 없으며,
object에만 작은 lambda를 지정할 수 없다. SVD 전에 object row의 크기를
줄이더라도 full-rank subspace 자체는 같으므로 완전한 정사영 결과는 거의
변하지 않는다.

이번 설계는 nuisance를 우선하는 계층적 직교화를 사용한다.

1. Action/background 방향들로 nuisance SVD basis $Q_N$을 만든다.
2. Object 방향 $c_O$에서 nuisance와 겹치는 부분을 제거한다.
3. 남은 object residual을 정규화하여 $q_O$를 만든다.
4. 두 직교 subspace에 서로 다른 lambda를 적용한다.

$$
r_O=(I-Q_NQ_N^T)c_O
$$

$$
q_O=rac{r_O}{\lVert r_O\rVert_2}
$$

$$
v'=v-\lambda_NQ_NQ_N^Tv-\lambda_Oq_Oq_O^Tv
$$

마지막에만 원래 visual embedding norm을 복원한다. $Q_N$과 $q_O$가
직교하므로 중복 차감과 적용 순서 문제가 없다.

## 3. Object 방향을 만드는 두 counterfactual

### 3.1 No-object 방식

같은 문맥에서 object가 있을 때와 없을 때를 비교한다.

$$
\delta_C=rac{\hat e(O+C)-\hat e(C)}
{\left\lVert\hat e(O+C)-\hat e(C)\right\rVert_2}
$$

예시는 `a dog flying` 대 `flying`이다. 이 방향에는 해당 문맥에서 드러나는
dog/cat 성분과 reference identity에 유효할 수 있는 object 정보가 함께
들어갈 수 있으므로 $\lambda_O$를 작게 시작해야 한다.

### 3.2 Superclass 또는 animal 방식

Object를 없애는 대신 `an animal`로 교체한다.

$$
\delta_C^{S}=rac{\hat e(O+C)-\hat e(S+C)}
{\left\lVert\hat e(O+C)-\hat e(S+C)\right\rVert_2},
\qquad S=\text{an animal}
$$

예시는 `a dog flying` 대 `an animal flying`이다. 이는 animal 공통 성분
자체보다 generic animal과 비교했을 때 남는 dog/cat-specific 성분을
추정한다. 실제 animal 공통 성분 $\hat e(\text{an animal})$을 직접 제거하는
실험과는 구분해야 한다.

## 4. Prompt 구조별 동작

### 4.1 O+A

- Nuisance basis: $d_A=\hat e(O+A)-\hat e(O)$ 한 방향
- Object 방향: $(O+A)-A$ 또는 $(O+A)-(S+A)$
- 최종 basis rank 상한: nuisance 1 + object residual 1

### 4.2 O+B

- Nuisance basis: $d_B=\hat e(O+B)-\hat e(O)$ 한 방향
- Object 방향: $(O+B)-B$ 또는 $(O+B)-(S+B)$
- 최종 basis rank 상한: nuisance 1 + object residual 1

### 4.3 O+A+B

기존 factorial main effect를 유지한다.

$$
d_A=\frac{1}{2}
\left[(\hat e(O+A)-\hat e(O))+
(\hat e(O+A+B)-\hat e(O+B))\right]
$$

$$
d_B=\frac{1}{2}
\left[(\hat e(O+B)-\hat e(O))+
(\hat e(O+A+B)-\hat e(O+A))\right]
$$

$[d_A;d_B]$를 SVD하여 $Q_N$을 만든다. Object 방향은 다음 세 contextual
contrast를 각각 정규화한 뒤 평균하고 다시 정규화한다.

- $(O+A)-A$ 또는 $(O+A)-(S+A)$
- $(O+B)-B$ 또는 $(O+B)-(S+B)$
- $(O+A+B)-(A+B)$ 또는 $(O+A+B)-(S+A+B)$

세 object contrast를 별도 SVD row로 사용하지 않는다. 평균한 object 방향
하나를 $Q_N$에 대해 residualize한다. 최종 rank 상한은 nuisance 2 + object
residual 1이다.

## 5. Target와 source 조건의 차이

### Target hierarchical projection

Target prompt에서 nuisance 방향과 object 방향을 모두 계산한다.

- O+A/O+B: 하나의 target nuisance 방향
- O+A+B: 두 factorial target nuisance 방향
- Target prompt마다 class direction도 달라진다.

### Source hierarchical projection

Nuisance 방향은 reference caption의 source behavior/background에서 계산한다.
Target prompt taxonomy에 맞춰 제거할 source factor를 선택한다.

- Target가 O+A이면 source behavior nuisance만 사용
- Target가 O+B이면 source background nuisance만 사용
- Target가 O+A+B이면 source behavior와 background nuisance를 사용

Object 방향은 source caption의 $(O+A_s)$, $(O+B_s)$,
$(O+A_s+B_s)$ 세 contrast를 평균한다. 같은 reference에서는 이 object
방향이 공통이지만, target prompt 종류에 따라 nuisance basis가 달라지므로
residualized object 방향은 달라질 수 있다.

## 6. 구현된 condition

| Condition | Nuisance 출처 | Object counterfactual | Caption | Gate |
|---|---|---|---|---|
| `class_target_no_object_combined` | target prompt | context only | 불필요 | 적용 |
| `class_target_superclass_combined` | target prompt | `an animal + context` | 불필요 | 적용 |
| `class_source_no_object_combined` | source caption | context only | 필요 | 적용 |
| `class_source_superclass_combined` | source caption | `an animal + context` | 필요 | 적용 |

기존 `class_contextual_combined`는
`class_source_superclass_combined`의 호환 alias다. 두 조건을 동시에 실행할
필요는 없다. `class_absolute_combined`는 nuisance projection 없이
`a dog/cat` 절대 방향만 제거하는 기존 class-only ablation으로 유지한다.

## 7. Lambda의 의미와 권장 sweep

- `--orthogonalization-strength`: $\lambda_N$, action/background attenuation
- `--class-projection-strength`: $\lambda_O$, residual object attenuation

우선 nuisance 효과를 기존 실험과 맞추기 위해 $\lambda_N=1.0$으로 고정하고
$\lambda_O$만 sweep하는 것이 해석하기 쉽다.

권장 1차 sweep:

| 설정 | $\lambda_N$ | $\lambda_O$ | 목적 |
|---|---:|---:|---|
| Control | 1.0 | 0.0 | 기존 nuisance projection과 동등성 확인 |
| Weak | 1.0 | 0.10 | identity 손상을 최소화한 class 약화 |
| Low | 1.0 | 0.25 | 우선 권장값 |
| Medium | 1.0 | 0.50 | class 억제와 identity 손상의 trade-off |
| Full ablation | 1.0 | 1.00 | 상한 및 실패 양상 확인 |

$\lambda_O=0$은 코드 경로와 residualization diagnostics는 유지하면서 실제
object 제거만 끄는 내부 control이다. 먼저 0, 0.25, 0.5를 비교하고 필요할
때 0.1과 1.0을 추가하는 것이 GPU 비용 측면에서 효율적이다.

## 8. 반드시 확인할 diagnostics

- `nuisance_basis_rank`, `nuisance_singular_values`
- `class_overlap_norm`: class 방향 중 nuisance와 겹친 크기
- `class_residual_norm`, `class_residual_ratio`
- `class_direction_dropped`: class residual이 너무 작아 제거를 생략했는지
- `nuisance_component_norm`, `class_component_norm`
- `nuisance_removed_component_norm`, `class_removed_component_norm`
- `cosine_to_input`, `edit_norm`
- single-stream gate 활성화 여부

`class_residual_ratio`가 매우 작으면 object 방향 대부분이 nuisance basis와
겹친다는 의미다. 이때 작은 수치 오차를 확대하지 않도록 class residual을
drop한다. 반대로 residual ratio가 크고 `class_removed_component_norm`도
크다면 identity 손상 가능성을 정성 결과와 함께 확인해야 한다.

## 9. 권장 비교 순서

1. `baseline`
2. `gate`
3. 기존 target 또는 source nuisance projection + gate
4. 대응하는 hierarchical no-object 조건, $\lambda_O=0.25$
5. 대응하는 hierarchical superclass 조건, $\lambda_O=0.25$
6. 더 나은 counterfactual에 대해 $\lambda_O=0,0.5$ 추가
7. identity 보존, prompt controllability, behavior/background leakage를 함께
   비교

동물 위주 데이터에서는 털무늬, 귀 모양, 얼굴 비율, 체형과 같은
reference-specific 특징과 generic dog/cat 특징을 분리해 평가해야 한다.
Object attenuation이 줄무늬 같은 generic artifact를 줄이더라도 reference의
고유 털 패턴까지 사라진다면 성공으로 판단하면 안 된다.

Modal 실행 명령은
[CLASS_PROJECTION_MODAL_GUIDE.md](CLASS_PROJECTION_MODAL_GUIDE.md)에 정리되어
있다.
