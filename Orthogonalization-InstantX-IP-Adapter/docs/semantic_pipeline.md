# Target-aware source semantic pipeline

Reference의 관찰과 제거 판단을 분리한다. A는 이미지를 caption으로 기록하고, B-source는 target이 요구하는 변경 범위에 따라 source의 보존 영역과 제거 components를 선택한다. B-target은 기존 identity-only 분해를 유지한다. 실제 생성에는 원본 target prompt를 전달한다.

```text
reference image + identity_term → A → source_caption
source_caption + target_prompt + identity_term → B-source → source partition
target_prompt + identity_term → B-target → target partition
origin에 따른 제거 components → local/factorial contrasts → SigLIP pooled directions
→ 기존 SVD + visual projection + norm restoration → InstantX IP-Adapter
```

## 보존 정책

`identity_term`은 사용자가 명시한 보존 개념이다. Benchmark에서는 자연어 class/label을 기본값으로 사용하되 reference ID와 구분한다. 관사는 있어도 없어도 되지만 source와 target 안에 대소문자를 포함하여 정확한 단어 경계로 등장해야 한다. 임의로 소문자화하거나 label을 축약하지 않는다.

- Subject의 색·무늬·형태·재질 등 외형은 기본 보존한다. Target이 **동일 대상의 동일 속성**을 변경할 때 해당 source 표현만 제거 후보로 만든다. 언급 없음과 동일 속성 재확인은 보존이다.
- 빨간 모자는 털 색 변경이 아니며, 수채화 표현은 털 색·무늬 전체 변경을 의미하지 않는다.
- 자세·행동·받침면·배경은 context 전달을 줄이기 위한 기본 제거 후보다. Target에 동일 장소가 있어도 기본 정책을 유지한다. Reference의 자세·배경을 유지하라는 명시적 요청은 예외다.
- 의상·액세서리·표정은 초기의 보수적 정책으로 유지하며, 해당 속성을 교체/제거하거나 명백히 양립할 수 없는 상태를 요구하면 해제한다. 새 액세서리 추가만으로 다른 액세서리를 제거하지 않는다.
- Style/scene concept에서는 해당 개념의 실현 정보를 보존한다. 수채화가 identity이면 고양이의 갈색 털은 subject identity가 아니다. Sunset이면 하늘 색이 개념 보존에 기여할 수 있다.
- Target이 명시적 identity 자체를 모순되게 변경하면 B-source는 원문 evidence와 이유를 담은 `conflicts`를 반환하고, 코드가 `IdentityConflictError`로 중단한다. 의미 판정은 LLM에 의존한다. Target-only는 B-source를 호출하지 않으므로 이 추가 충돌 판정을 하지 않는다.

## Source JSON과 검증

입력:

```json
{"source_caption":"a brown dog with white paws sitting on grass","target_prompt":"a dog wearing a red hat","identity_term":"dog"}
```

출력:

```json
{
  "identity_term": "dog",
  "preserved_spans": ["dog"],
  "retained_spans": ["brown", "with white paws"],
  "neutral_spans": ["a"],
  "components": [{"spans": ["sitting"]}, {"spans": ["on grass"]}],
  "conflicts": []
}
```

`preserved_spans`는 identity와 최소 문법적 carrier다. 각 span은 exact identity를 포함하며 모든 identity 출현을 보호한다. `retained_spans`는 추가 의미 보존 구간이며 identity를 포함할 필요도, 인접할 필요도 없다. `neutral_spans`는 고정 문법 요소만 담는다. `components`는 실제 제거 후보만 담고 보존 속성을 중복 출력하지 않는다.

Partition spans는 source의 verbatim substring이며 상호 비중첩, 단어 경계, 순서, lexical coverage를 검증한다. 한 component의 여러 spans는 함께 삭제한다. Source에 없는 target 표현을 partition에 넣으면 실패다. Conflict evidence는 target 원문에서 검증하며 conflict 응답의 partition arrays는 모두 비운다.

구조 오류는 기존 제한 재시도(기본 2회)를 사용한다. 검증된 conflict는 재시도로 지우지 않고 중단한다. JSON 검증은 의미적 판단의 정답을 보장하지 않는다.

## Contrast의 고정 문맥

`P_S`는 제거 후보 중 S만 남긴 문장이다. 세 종류의 고정 spans는 **모든** P_S에 남는다. 외형 보존 구간은 factorial 조합에서도 꺼지지 않는다. N은 전체 관찰 속성 수가 아닌 제거 후보 그룹 수다. 기존 direction 수식은 그대로 사용한다.

```text
local:     d_i = normalize(e(P_all) - e(P_all_without_i))
factorial: d_i = normalize(mean_S(e(P_(S+{i})) - e(P_S)))
```

`e`는 정규화된 SigLIP pooled text embedding이다. 삭제 후 공백만 정리하고 문장을 다시 쓰지 않는다. Fragment와 남은 구두점은 가능하다. 모든 components를 유지하면 원문을 정확히 복원한다. Empty component list도 유효하다.

Target B는 identity-only 정책과 기존 schema/direction 선택을 유지한다. DreamBench++ 개발 집합 검토에 따라 관사·carrier·관계절·병렬 구문 처리를 보완했다([LLM 분석 문제와 현재 대응](llm_component_issues.md)). Target 방향이 source의 추가 보존 속성과 겹치는 효과는 별도 실험 대상이다. Text span 보존이 embedding에서의 완전한 속성 보존이나 생성 품질을 보장하지 않는다. 삭제 조합의 연결·수식 관계 위험과 대표 snapshot은 [projection 문제 기록](projection_issues.md)에 남긴다.

## API와 저장 형식

- `caption(image_bytes, identity_term)`: 검토한 최신 A prompt 사용.
- `analyze_source(source_caption, target_prompt, identity_term)`: 새 B-source.
- `analyze_target(text, identity_term)`: 기존 B-target. `analyze()`도 target API로 유지.
- `prepare_semantic_conditioning(...)`: origin에 따라 자동 연결.

`source_target`: A + B-source + B-target 각 1회. 수동 caption이면 A 생략.
`source`: A와 B-source만 호출하지만 B-source는 target을 읽는다.
`target`: B-target만 호출하며 caption이 필요 없다. 재시도는 이 횟수에 포함하지 않는다.

저장 envelope는 `schema_version: 2`다. 새 source에는 `retained_spans`, `conditioned_on_target`(정확한 target 원문), `source_policy: target_aware_source_v1`가 함께 저장된다. Target이 다르거나 source metadata 일부가 누락되거나 미지원 policy이면 재사용을 거부한다. 같은 reference라도 target이 바뀌면 B-source를 새로 실행한다. A caption은 재사용할 수 있다.

이전 unversioned JSON은 기존 의미 그대로 읽는다. Retention metadata가 없는 legacy source는 새 정책으로 자동 승격하지 않으며 기존 components를 그대로 제거한다. 새로운 연구 결과에는 새로 분석한 source를 사용한다. 모니터링 스크립트는 legacy cache를 조용히 재사용하지 않는다. 캐시의 reference image 연결은 원본 reference와 함께 관리해야 한다(이번 변경에 image fingerprint 바인딩은 포함되지 않는다).

## 실행

```powershell
python -m modal run modal_inference.py `
  --reference-image-path "C:/path/reference.png" `
  --identity-term "dog" `
  --prompt "a dog wearing a red hat" `
  --source-caption "a brown dog with white paws sitting on grass" `
  --component-origin source_target `
  --component-strategy local
```

기존 CLI/Python 인자는 그대로다. `.components.json` 재사용 시 원본 target과 identity도 확인한다. 기본 모델은 `gpt-6-luna`; 환경변수는 `SEMANTIC_LLM_API_KEY`, 선택적으로 `SEMANTIC_LLM_MODEL`, `SEMANTIC_LLM_ENDPOINT`를 사용한다. 기존 SigLIP 길이 제한과 factorial `max_contexts=256`을 유지한다.

- [전체 runtime prompt와 source schema](prompts/llm_modules_share.md)
- [A prompt와 번역](prompts/module_a_caption_system_prompt.md)
- [B-source prompt·16개 executable few-shot](../orthogonalization_instantx_ip_adapter/source_component_prompt.py)
- [분석·검증·직렬화](../orthogonalization_instantx_ip_adapter/semantic_components.py)
- [Source 정책 테스트](../tests/test_source_semantic_analysis.py)
- [LLM component 분석의 기존 문제·현재 대응·남은 판단](llm_component_issues.md)
- [Projection의 삭제 조합 문제·재현 snapshot](projection_issues.md)
