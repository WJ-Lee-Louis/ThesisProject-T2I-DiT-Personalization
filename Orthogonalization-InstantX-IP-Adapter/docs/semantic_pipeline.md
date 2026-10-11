# Identity 보존을 위한 semantic pipeline

Reference image에서 보존할 concept을 `identity_term`으로 지정하고, source caption과 target prompt의 나머지 의미 성분을 자동으로 분해한다. 이 성분들의 text embedding 방향을 reference visual embedding에서 제거하여 reference의 불필요한 내용과 target의 충돌을 줄이는 것이 목적이다.

연구의 중심은 **projection 기반 disentanglement**다. LLM은 caption과 component 분석을 자동화하며, InstantX의 single-stream gate는 별도의 보조 설정이다.

## 1. 입력과 전체 흐름

| 입력 | 설명 |
|---|---|
| Reference image | 개인화에 사용할 이미지 |
| `identity_term` | 사용자가 보존할 concept. 예: `a dog`, `a red backpack`, `watercolor style` |
| `target_prompt` | 생성할 장면을 설명하는 자유문 |
| `source_caption` (선택) | 직접 제공하면 이미지 captioning을 생략 |

`identity_term`은 target prompt에 **대소문자를 포함해 정확한 문자열과 단어 경계로** 존재해야 한다. 예를 들어 `a dog`는 `A dog`나 `a doghouse`와 일치하지 않는다. 사용자 caption에도 같은 조건을 적용하며, 불일치하면 LLM 호출 전에 중단한다.

기본 설정은 source와 target을 모두 사용하는 `source_target`이다.

```text
reference image + identity_term + target_prompt
 → 입력 검증
 → Module A: reference caption 생성 (사용자 caption이 있으면 생략)
 → Module B: source caption / target prompt 각각 분해
 → JSON 및 원문 span 검증
 → local 또는 factorial로 component 방향 계산
 → source·target 방향을 합쳐 한 번의 SVD / visual projection
 → InstantX IP-Adapter 이미지 생성
```

Component 분석을 위해 만든 부분 문장은 제거 방향 계산에만 사용한다. **실제 이미지 생성의 text conditioning에는 원래 target prompt를 전달한다.**

## 2. LLM 모듈과 분해 규칙

| 모듈 | 입력 | 출력 |
|---|---|---|
| A. Reference Image Captioning | 이미지 + `identity_term` | 지정 identity를 그대로 포함하는 간결한 `source_caption` |
| B. Prompt Component Analysis | text + `identity_term` | 보존 구간, 중립 구간, 제거할 component 목록 |

두 모듈의 system prompt 전문과 입출력 예시는 [LLM 모듈 공유 문서](prompts/llm_modules_share.md)에 정리했다. 실행 시 사용하는 원본은 `semantic_components.py`의 prompt 상수이므로, 공유 문서만 편집해도 실행 prompt가 바뀌지는 않는다.

Module B의 출력 예시:

```json
{
  "identity_term": "a dog",
  "preserved_spans": ["a dog"],
  "neutral_spans": ["a photo of"],
  "components": [
    {"spans": ["playing the guitar"]},
    {"spans": ["on a sofa"]}
  ]
}
```

위 예시의 원문은 `a photo of a dog playing the guitar on a sofa`다.

- **보존 구간**: identity와 필요한 문법적 연결어를 유지한다. 예: identity가 `sunset`이면 `during sunset`을 보존할 수 있다.
- **중립 구간**: `a photo of` 같은 일반적인 도입 표현을 유지하되 제거 방향으로 사용하지 않는다. 구체적인 style/medium이나 물리적인 사진까지 중립 처리하지 않는다.
- **Components**: 제거할 의미 단위다. 개수와 종류를 고정하지 않으며, `playing the guitar`처럼 함께 의미를 이루는 구를 불필요하게 나누지 않는다.

각 span은 원문의 실제 구간이며 서로 겹치지 않는다. 세 종류의 span이 모든 단어를 배정하고, 원래 위치와 구두점·공백을 함께 유지하면 원문을 복원할 수 있어야 한다. Identity가 여러 번 등장하면 모두 보호한다.

Identity가 의미 단위의 중간에 끼면 하나의 component에 여러 span을 묶을 수 있다. 예를 들어 `A tiny dragon perched on a pixel art cloud`에서 identity가 `pixel art`이면 `perched`와 `{"spans":["on a","cloud"]}`를 별도 component로 처리한다. 묶인 span들은 항상 함께 제거하며, component 수는 span 수가 아닌 **그룹 수**다. Style을 보존하는 경우 대상 사물 자체도 제거 component가 될 수 있다.

잘못된 JSON/필드, 빈 span, 원문에 없는 구, identity 침범, 중복·겹침·순서 오류, 미배정 단어를 검증한다. 출력 검증 실패에는 피드백을 포함하여 최대 2회 재시도하고, API·인증·네트워크 오류는 즉시 중단한다.

## 3. Local / factorial과 projection

`P_S`는 선택한 component 집합 `S`만 남긴 부분 문장이다. 보존·중립 구간은 모든 부분 문장에 유지하며, phrase를 identity-first 순서로 재배치하지 않는다. 모든 component를 남기면 원문 그대로다.

`e(P)`는 **정규화한 SigLIP pooled text embedding**이다. Token span의 embedding을 직접 평균하거나 전체 문장에서 단어 벡터를 단순 차감하는 방식은 아니다.

| 방식 | Component i의 제거 방향 |
|---|---|
| `local` (기본) | 전체 문장과 i만 제거한 문장의 embedding 차이 |
| `factorial` | 다른 component의 모든 부분집합 문맥에서, i의 유무에 따른 embedding 차이의 평균 |

```text
local:
d_i = normalize(e(P_all) - e(P_all_without_i))

factorial:
d_i = normalize(mean_{S ⊆ components_without_i}
                (e(P_{S ∪ {i}}) - e(P_S)))
```

Factorial은 **각 차이를 먼저 정규화하지 않고**, 차이를 평균한 다음 방향을 정규화한다. Component가 하나면 두 방식은 같고, 제거할 component가 없으면 제거 방향도 없다.

Source/target에서 얻은 방향을 함께 쌓아 SVD로 직교 기저를 만들고, reference visual pooled embedding에서 해당 부분공간의 projection을 제거한다. 이후 norm restoration과 InstantX image projection을 거친다. 기존 projection 수학은 유지한다.

Ablation은 `source`, `target`, `source_target`으로 선택한다. Source-only는 target 분석을 생략하고, target-only는 caption/source 분석을 생략한다. 기본 경로는 caption 1회와 분석 2회이며, 사용자 caption이 있으면 분석 2회다(재시도 제외).

## 4. 실행과 재사용

프로젝트 루트의 PowerShell에서 `SEMANTIC_LLM_API_KEY`를 환경변수로 제공한다. 키는 코드나 결과 파일에 저장하지 않는다. `SEMANTIC_LLM_MODEL`에는 사용할 실제 API model ID를 설정하고, 다른 Responses-compatible endpoint를 쓰면 `SEMANTIC_LLM_ENDPOINT`도 설정한다. Provider는 교체 가능하며, 사용하는 모델은 이미지 입력과 structured output을 지원해야 한다.

```powershell
$env:SEMANTIC_LLM_MODEL = "<실제 API model ID>"

python -m modal run modal_inference.py `
  --reference-image-path "C:/path/reference.png" `
  --identity-term "a dog" `
  --prompt "a dog playing the guitar on a sofa" `
  --component-origin source_target `
  --component-strategy local `
  --orthogonalization-strength 1.0 `
  --mask-single-text-rows
```

`--source-caption "a dog sleeping on the snow"`를 추가하면 captioning을 생략한다. `--component-strategy factorial`로 계산 방식을 바꾸고, `--component-origin source` 또는 `target`으로 ablation한다. 위 예시는 single-stream gate를 활성화한다.

LLM 분석은 로컬에서 수행하며, Modal GPU에는 검증된 분석 결과를 전달한다. 생성 결과 옆의 `.components.json`을 `--component-analysis-path`로 재사용하면 LLM 호출과 API key가 필요 없다. 재사용 시 target prompt가 정확히 일치해야 하며, origin/strategy는 저장 JSON의 설정을 따른다. Source 분석은 원래 reference image와 함께 관리한다.

## 5. 제한과 구현 위치

- Factorial은 분석 문장별 component N개에 대해 `2^N`개의 부분집합 문맥을 사용한다. 기본 예산 `max_contexts=256`을 초과하면 중단한다.
- SigLIP의 text 길이 제한을 넘으면 오류를 내며 자동으로 잘라내지 않는다.
- Span 검증은 구조를 보장하지만, LLM의 의미 단위 분해 일관성이나 caption의 시각적 정확성까지 보장하지 않는다. 삭제 후 생기는 불완전한 문장과 구두점도 embedding에 영향을 줄 수 있다.
- Identity 문자열을 유지하는 것과 embedding 공간에서 identity를 완전히 보존하는 것은 다르다. 실제 생성 이미지의 의도 반영 및 disentanglement 성능은 후속 평가가 필요하다.
- 이전 분석 JSON도 호환성을 위해 읽는다. 현재 보존·중립 규칙을 적용하려면 과거 cache를 재생성한다.

| 파일 | 역할 |
|---|---|
| [semantic_components.py](../orthogonalization_instantx_ip_adapter/semantic_components.py) | LLM prompt/schema, span 검증, source/target 분석 연결 |
| [llm_provider.py](../orthogonalization_instantx_ip_adapter/llm_provider.py) | API provider 및 환경변수 설정 |
| [component_directions.py](../orthogonalization_instantx_ip_adapter/component_directions.py) | Local/factorial contrast와 방향 계산 |
| [orthogonalization.py](../orthogonalization_instantx_ip_adapter/orthogonalization.py), [adapter.py](../orthogonalization_instantx_ip_adapter/adapter.py) | SVD, visual projection, IP-Adapter 연결 |
| [inference.py](../orthogonalization_instantx_ip_adapter/inference.py), [modal_inference.py](../modal_inference.py) | Python / Modal 실행 진입점 |
| [test_semantic_pipeline.py](../tests/test_semantic_pipeline.py) | Mock LLM, span 검증, component 수·ablation·수학적 계산 검증 |

DreamBench++ style 10개 사례에서 20개 source/target 분석과 local/factorial 계산 계획의 원문 복원·고정 span 유지·차감 쌍·조합 수를 확인했다. 전체 테스트 734개가 통과했으며, 이는 이미지 품질에 대한 평가 결과는 아니다.
