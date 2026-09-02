# Final 22-Condition SSG Experiment Guide

## 1. 명칭 규칙

새 inference condition은 정확히 22개다. `ssg`는 single-stream gate를
의미한다. `combined`, `gate`, `sa_*` 같은 과거 명칭은 신규 제출에서
사용하지 않는다.

- `target`: target prompt에서 nuisance 방향 계산
- `source`: reference caption에서 nuisance 방향 계산
- `st_matched`: target 방향과 target taxonomy에 맞는 source 방향을 union SVD
- `st_all`: target 방향과 source behavior/background 모두를 union SVD
- `absolute_direct`: `a dog/cat` 방향과 nuisance의 겹침을 유지
- `absolute_residual`: `a dog/cat` 방향에서 nuisance 겹침을 먼저 제거
- `no_object`: 같은 문맥의 `(O+C)-C`
- `superclass`: 같은 문맥의 `(O+C)-(animal+C)`

## 2. 최종 22개 condition

### Control 2개

| Condition | Projection | SSG |
|---|---|---|
| `baseline` | 없음 | 없음 |
| `ssg` | 없음 | 적용 |

### 기존 nuisance projection 4개

| Condition | Nuisance 출처 |
|---|---|
| `target_projection_ssg` | target |
| `source_projection_ssg` | source |
| `st_matched_projection_ssg` | target + matched source |
| `st_all_projection_ssg` | target + all source |

### Generic projection 16개

| Nuisance 출처 | Absolute direct | Absolute residual | No-object | Superclass |
|---|---|---|---|---|
| Target | `class_target_absolute_direct_ssg` | `class_target_absolute_residual_ssg` | `class_target_no_object_ssg` | `class_target_superclass_ssg` |
| Source | `class_source_absolute_direct_ssg` | `class_source_absolute_residual_ssg` | `class_source_no_object_ssg` | `class_source_superclass_ssg` |
| ST matched | `class_st_matched_absolute_direct_ssg` | `class_st_matched_absolute_residual_ssg` | `class_st_matched_no_object_ssg` | `class_st_matched_superclass_ssg` |
| ST all | `class_st_all_absolute_direct_ssg` | `class_st_all_absolute_residual_ssg` | `class_st_all_no_object_ssg` | `class_st_all_superclass_ssg` |

## 3. Absolute direct와 residual

Nuisance SVD basis를 $Q_N$, 정규화한 absolute class anchor를 $c_O$라고
하자. 두 방식 모두 action/background projection과 class projection을 함께
적용한다.

Absolute direct는 겹침을 그대로 둔다.

$$
v^{\prime}
=v
-\lambda_N Q_N Q_N^{\mathsf{T}}v
-\lambda_O c_O c_O^{\mathsf{T}}v
$$

Absolute residual은 먼저 겹침을 제거한다.

$$
r_O=\left(I-Q_NQ_N^{\mathsf{T}}\right)c_O,
\qquad
q_O=\frac{r_O}{\lVert r_O\rVert_2}
$$

$$
v^{\prime}
=v
-\lambda_N Q_N Q_N^{\mathsf{T}}v
-\lambda_O q_O q_O^{\mathsf{T}}v
$$

No-object와 superclass도 residual 방식으로 동작한다. ST 조건에서는
target/source class 방향을 각각 평균한 후 nuisance에 대해 residualize하고,
두 residual에 두 번째 SVD를 적용한다.

## 4. Lambda 정책

- `--orthogonalization-strength`: action/background의 $\lambda_N$
- `--class-projection-strength`: generic class의 $\lambda_O$
- Generic 16개 condition에서 $\lambda_O$는 `0.5` 또는 `1.0`만 허용한다.
- Control과 기존 nuisance 4개는 class lambda가 결과에 사용되지 않는다.

따라서 방법론은 22개이지만 실제 결과 batch 수는 다음과 같다.

- Control: 2 batch
- 기존 nuisance: 4 batch
- Generic: 16 conditions × 2 lambdas = 32 batch
- 전체 inference 결과 세트: 38 batch

## 5. VS Code CMD 제출 형식

이번 budget-aware 실행은 target과 ST-all만 선택한다. 배포, 16개 batch
제출, 상태 확인 및 다운로드 명령은
[0903_modal_exp_guide.md](0903_modal_exp_guide.md)에 분리했다.

## 6. Diagnostics

각 image sidecar에는 다음 값이 저장된다.

- `basis_origin`
- `class_projection_mode`
- `class_overlap_policy`: `direct` 또는 `residual`
- `nuisance_basis_rank`, `nuisance_singular_values`
- `class_direction_count`, `class_basis_rank`, `class_singular_values`
- `class_overlap_norm`, `class_residual_ratio`
- `nuisance_removed_component_norm`, `class_removed_component_norm`
- `cosine_to_input`, `edit_norm`
- `mask_single_text_rows`

기존 class-only `class_absolute_combined` condition은 제거했다. 과거에 이미
생성한 batch와 이미지 파일은 삭제하거나 변경하지 않는다.
