# LLM component 분석: 구현에 남길 문제와 판단 근거

> **과거 component 경로의 이력 (2026-10-18 snapshot).** 아래의 “현재”, “반영”, partition/span/subset 정책은 당시 legacy 구현을 가리킨다. 새 공동 B의 지시나 정답이 아니다. 후속 승인된 style head 보존·의존 묶음 정책은 [설계 §16](contrast_refactor_design.md)을 따른다. 현재 실행은 [semantic pipeline](semantic_pipeline.md), 사례별 새 수동 기대값과 미해결 scope는 [설계 §7](contrast_refactor_design.md#7-projection-문제-사례의-재분류와-새-기대-출력)을 따른다. 원본 응답과 expected_subset은 비교 이력으로 보존하며 새 API 관측으로 취급하지 않는다.

2026-10-18 정리. 2026-10-10~14의 caption·source/target 분석 기록에서 이후 구현에 필요한 사항만 통합했다. 실행별 전체 응답과 성공률 표는 생략했다. 현재 구현은 [semantic pipeline](semantic_pipeline.md), 정확한 system prompt와 few-shot은 [공유 프롬프트](prompts/llm_modules_share.md)를 기준으로 읽는다.

## 현재 반영된 대응과 회귀를 막아야 할 문제

아래의 ‘반영’은 코드 또는 prompt에 대응이 있다는 뜻이다. 임의 입력과 반복 호출에서 문제가 사라졌다는 뜻은 아니다.

| 문제 | 관찰 사례와 원인 | 현재 대응 / 이후 변경 시 지킬 조건 |
|---|---|---|
| A의 identity 표기 변경 | 입력 `cherry blossoms`를 caption이 `Cherry blossoms`로 시작하여 exact-match 실패. 180개 첫 응답 중 구조 실패 1건이었다. | A에 원문·대소문자 유지 지시가 있다. 검증을 통과시키기 위해 label을 소문자화하거나 caption을 조용히 보정하지 않는다. 시각적 정확성은 별도 확인해야 한다. |
| 원문에 없는 보존 span | `a shiny <dog> dog`에서 `a <dog> dog`를 보존한다고 출력. 연속 substring이 아니다. | identity만 보존하고 떨어진 관사는 neutral에 배정할 수 있다. 원문 순서·단어 경계·비중첩·coverage와 모든 identity 출현을 검증한다. |
| 검증기가 정상 응답을 거절 | `a cube shaped <backpack> backpack`의 neutral `a`를 `shaped` 내부에서도 찾아 ambiguous로 거절했던 초기 구현. | 현재 partition 검증은 lexical boundary를 사용하고 유일한 제거 span이 차지하는 위치를 neutral 후보에서 제외한다. LLM 오류와 validator 오류를 구분한다. 실제 반복으로 모호한 span은 계속 거절한다. |
| 관사·연결사 미배정 | 수식어 제거 시 identity 명사구 관사까지 제거하거나 첫 `a`를 누락. 병렬 술어의 `and` 누락도 관찰. | identity 명사구 관사는 고정하고 `and`는 인접 component에 포함하도록 지시한다. 구조 검증·제한 재시도를 유지한다. **모든 단어 배정과 삭제 후 문법은 별개다.** |
| 같은 관계의 분해 편차 | `with a mountain in the background`가 object/live subject에 따라 한 그룹 또는 두 그룹으로 달랐다. | 도입된 대상과 그 역할·공유 관계는 함께 묶는다. 개별 단어보다 일관된 의미 단위를 우선한다. |
| 너무 큰 장소 그룹 | `on top of a purple rug in a forest`를 통째로 묶으면 rug와 forest를 독립 차감할 수 없다. | 명시적 내부 부착 단서가 없으면 독립 placement를 분리하는 현재 정책. 명시적 관계절은 도입 대상과 묶는다. 원문 자체의 부착 모호성은 남는다. |
| rendering carrier와 물리적 대상 혼동 | `imagined in Ukiyo-e brushwork`는 표현 방식인 반면 `on a pixel art cloud`의 cloud는 가변 내용이다. | 무정보 rendering carrier/head는 style과 고정하고 독립 qualifier는 제거한다. 물리적 명사는 identity 양옆의 비연속 span을 한 component로 묶어 제거할 수 있다. `style` 디렉터리에는 sunset 등 장면 concept도 있다. |
| carrier가 의미 내용을 숨김 | identity가 목적어인 `solving a glowing Magic Cube`에서 solving까지 neutral로 고정하거나, subject의 `a drawing of Nia`에서 `of Nia`를 보존하는 문제. | 동작은 제거 후보, identity 명사 관사는 고정한다. subject의 informative medium wrapper에는 `of`도 포함한다. style carrier와 같은 규칙을 일괄 적용하지 않는다. |
| 필수 관계의 과분할 | 기타 연주와 기타, battle과 상대, comet을 도입하는 riding과 `steering it`, teacup과 `floating in the sky`를 분리하면 의존관계가 끊긴다. | 필수 목적어·상호작용 상대·명시적 관계절·의존 대명사를 관련 구문과 묶도록 prompt/few-shot에 반영했다. 독립 주어의 행동과 장소는 계속 분리할 수 있다. |
| span 복사 오류 | span 양끝 공백, 원문의 `in a STYLE style`에서 `a`를 생략한 substring 등이 중간 응답에 존재. | trimmed verbatim span만 허용하고 임의 trim·관사 삽입·문장 재작성으로 성공 응답을 만들지 않는다. |

## Source와 target 정책을 섞지 말아야 하는 이유

- B-source는 target을 읽고 reference의 외형·의상·표정·style/scene 실현 정보 중 추가로 고정할 내용을 `retained_spans`로 분리한다. B-target은 명시적 identity와 최소 문법적 carrier만 고정하는 정책이다.
- `a brown dog with white paws sitting on grass`는 target이 해변 이동이면 털색을 retained로 남기고, 전신 purple recolor이면 해당 색 표현을 제거한다. 새 모자나 watercolor 요구를 털색 전체 변경으로 취급하지 않는다.
- 의상·표정의 기본 보존은 현재의 보수적인 정책 선택이다. 예컨대 tongue-out 표정 보존은 불변 identity라는 주장이 아니다. 이 정책을 바꿀 때 prompt와 검증·캐시 계약을 함께 검토한다.
- Source API의 backpack shape 사례에는 원래 caption에 명시적인 shape가 없었다. 해당 응답 성공을 ‘관찰된 shape 제거 검증’으로 사용하지 않는다.
- 명시적 identity 충돌은 evidence와 함께 중단한다. target-only는 B-source를 호출하지 않아 이 충돌 검사를 추가 수행하지 않는다.
- 같은 reference라도 target이 바뀌면 B-source 분석을 재사용하지 않는다. Legacy cache를 새 retention 정책의 결과로 간주하지 않는다. reference image fingerprint 바인딩은 현 변경에 포함되지 않았다.

## 남아 있는 판단과 다음 구현의 확인점

1. **분해 granularity:** 독립 placement와 다른 대상 내부 위치·수식 관계를 가르는 기준. object/live나 표현 길이만 달라져도 분해가 바뀌는지 확인한다.
2. **보존 범위와 scope:** sunset을 고정해도 golden horizon을 제거할 때 장면 개념이 약해질 수 있다. style이 문장 전체가 아닌 trees/glow에 걸리는 경우도 있다. 자동으로 보존 범위를 넓히거나 global style suffix로 rewrite하지 않는다.
3. **구조와 의미:** 구조 PASS는 의미 분해 정답이 아니다. 최종 개발 24개와 DreamBench 50개의 첫 응답 통과는 개발 당시 관측이며, 전수·반복 평가 결과가 아니다.
4. **수정의 부작용:** 연결사 coverage, 관사 유지, grouping을 고쳐도 [projection 단계의 삭제 조합 문제](projection_issues.md)는 남을 수 있다. 새 grouping은 독립 방향을 합친다는 효과도 확인해야 한다.
5. **평가의 출처:** 실제 첫 응답, system prompt의 few-shot, 사람이 만든 stress partition을 구분한다. 반복 보완에 사용한 사례를 미사용 평가 집합으로 취급하지 않는다.

## 구현 위치와 기존 검증

- [semantic_components.py](../orthogonalization_instantx_ip_adapter/semantic_components.py): A/B-target prompt, span 해석·coverage, 부분집합 삭제, source/target 연결·저장 계약.
- [source_component_prompt.py](../orthogonalization_instantx_ip_adapter/source_component_prompt.py): B-source 정책과 16개 executable few-shot.
- [test_target_prompt_examples.py](../tests/test_target_prompt_examples.py): 현재 B-target 27개 few-shot의 partition·부분집합과 공유 문서 일치 검증.
- [test_source_semantic_analysis.py](../tests/test_source_semantic_analysis.py): source retention·target binding·직렬화 등의 기존 검증.
- A의 few-shot은 [공유 프롬프트](prompts/llm_modules_share.md)에, A의 한국어 번역은 [별도 prompt 문서](prompts/module_a_caption_system_prompt.md)에 있다. 예시 전문을 이 문제 기록에 중복 저장하지 않는다.

통합 원자료: 기존 DreamBench target monitoring v1~v4, DreamBench++ style/development review, Module A caption monitoring, source policy API audit. 실제 reference·caption 등 기존 데이터셋과 실행 코드는 유지한다. 각 단계의 성공률·중간 prompt hash·전체 응답표는 이후 구현의 요구사항으로 보존하지 않는다.
