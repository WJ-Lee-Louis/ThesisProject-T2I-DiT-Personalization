# Class Projection Modal Guide

이 문서는 generic dog/cat 성분을 SigLIP pooled visual embedding에서
training-free 방식으로 약화하는 실험을 정리한다. 모든 `*_combined` 조건은
FLUX single-stream text-row gate를 포함하며, 각 조건은
10 references × 16 prompts = 160 images를 생성한다.

## 조건 이름

| 조건 | 문맥 출처 | counterfactual | source caption |
|---|---|---|---|
| `class_absolute_combined` | 없음 | `a dog/cat` 절대 방향 | 불필요 |
| `class_target_superclass_combined` | target prompt | `O+C` 대 `an animal+C` | 불필요 |
| `class_target_no_object_combined` | target prompt | `O+C` 대 `C` | 불필요 |
| `class_source_superclass_combined` | reference caption | `O+C_s` 대 `an animal+C_s` | 필요 |
| `class_source_no_object_combined` | reference caption | `O+C_s` 대 `C_s` | 필요 |

기존 `class_contextual_combined`는 재현 호환성을 위해 유지하며,
`class_source_superclass_combined`와 같은 계산을 수행한다. 동일 실험에서 두
이름을 모두 실행할 필요는 없다.

## 방향 계산

각 prompt pooled embedding을 먼저 L2 정규화하고, 각 positive/negative 쌍의
차분도 다시 정규화한다.

```text
delta(C) = normalize(normalize(e(O+C)) - normalize(e(counterfactual+C)))
c = normalize(mean(delta(C_1), ..., delta(C_k)))
Q_N = SVD(nuisance directions)
q_C = normalize((I - Q_N Q_N^T)c)
v' = restore_norm(v - lambda_N projection_QN(v) - lambda_C projection_qC(v))
```

- target O+A 또는 O+B: 현재 prompt에 존재하는 문맥 하나만 사용하므로
  `k=1`이다.
- target O+A+B: `(O+A)-(...+A)`, `(O+B)-(...+B)`,
  `(O+A+B)-(...+A+B)`의 세 방향을 평균하므로 `k=3`이다.
- source 조건: caption에 source behavior와 background가 모두 있으므로 항상
  위의 세 문맥을 평균한다. 같은 reference의 16개 target 생성에 같은 source
  방향을 재사용한다.
- superclass 조건의 counterfactual object는 기본값 `an animal`이고,
  no-object 조건은 object 없이 `A`, `B`, `A+B` 자체를 사용한다.

세 class 차분을 SVD에 세 row로 직접 넣는 것이 아니라, 개별 정규화 후
평균한 방향 하나를 사용한다. 반면 기존 action/background 방향들은 먼저
SVD로 nuisance basis를 형성한다. Class 방향은 이 basis와 겹치는 성분을
제거한 뒤 rank-1 basis로 사용하므로 두 projection은 직교하고 lambda를
독립적으로 설정할 수 있다. `class_absolute`는 비교용 class-only ablation으로
`normalize(e("a dog/cat"))` 한 방향을 바로 사용한다.

## VS Code CMD 실행

```cmd
cd /d C:\Users\82107\Downloads\AI_PROJECT
python -m modal deploy Orthogonalization-InstantX-IP-Adapter\batch_modal.py
```

예시 lambda_C=0.5 네 조건을 서로 다른 batch ID로 제출하면 동시에 실행할
수 있다.

```cmd
python Orthogonalization-InstantX-IP-Adapter\batch_client.py submit --condition class_target_superclass_combined --batch-id class-target-superclass-n1-c05-0830 --orthogonalization-strength 1.0 --class-projection-strength 0.5 --class-superclass-prompt "an animal" --yes
python Orthogonalization-InstantX-IP-Adapter\batch_client.py submit --condition class_target_no_object_combined --batch-id class-target-no-object-n1-c05-0830 --orthogonalization-strength 1.0 --class-projection-strength 0.5 --yes
python Orthogonalization-InstantX-IP-Adapter\batch_client.py submit --condition class_source_superclass_combined --batch-id class-source-superclass-n1-c05-0830 --source-captions Orthogonalization-InstantX-IP-Adapter\source_captions.json --orthogonalization-strength 1.0 --class-projection-strength 0.5 --class-superclass-prompt "an animal" --yes
python Orthogonalization-InstantX-IP-Adapter\batch_client.py submit --condition class_source_no_object_combined --batch-id class-source-no-object-n1-c05-0830 --source-captions Orthogonalization-InstantX-IP-Adapter\source_captions.json --orthogonalization-strength 1.0 --class-projection-strength 0.5 --yes
```

Function Call ID가 출력된 뒤에는 VS Code와 PC를 종료해도 Modal 작업은
계속된다. 상태 확인과 다운로드는 각 batch ID에 대해 다음처럼 실행한다.

```cmd
python Orthogonalization-InstantX-IP-Adapter\batch_client.py status --batch-id class-target-superclass-n1-c05-0830
python Orthogonalization-InstantX-IP-Adapter\batch_client.py download --batch-id class-target-superclass-n1-c05-0830
```

`--orthogonalization-strength`는 기존 action/background nuisance basis의
`lambda_N`이고, `--class-projection-strength`는 residual object basis의
`lambda_C`다. 각 image sidecar에는 positive/negative prompt 쌍, nuisance SVD
rank/singular values, class overlap·residual ratio, 두 component의 차감 norm,
최종 edit norm과 gate 상태가 저장된다.
