# Projection: component 처리에서 남은 문제와 재현 사례

> **과거 component 경로의 이력 (2026-10-18 snapshot).** 아래의 “현재”, “반영”, partition/span/subset 정책은 당시 legacy 구현을 가리킨다. 새 공동 B의 지시나 정답이 아니다. 후속 승인된 style head 보존·의존 묶음 정책은 [설계 §16](contrast_refactor_design.md)을 따른다. 현재 실행은 [semantic pipeline](semantic_pipeline.md), 사례별 새 수동 기대값과 미해결 scope는 [설계 §7](contrast_refactor_design.md#7-projection-문제-사례의-재분류와-새-기대-출력)을 따른다. 원본 응답과 expected_subset은 비교 이력으로 보존하며 새 API 관측으로 취급하지 않는다.

2026-10-18 정리. LLM partition을 받은 뒤 부분집합 문장과 local/factorial 차분을 만드는 과정의 문제를 기록한다. [LLM 분석 문제](llm_component_issues.md)와 연결되지만, 올바른 JSON·단어 coverage만으로 해결되지 않는다. 당시 고정 문맥은 legacy component 구현의 정의이며 새 local 계약과 다르다.

## 관찰과 미확인 사항

- **관찰:** 실제 partition을 현재 삭제 규칙으로 전개하면 연결사 잔존, 선행사 소실, 수식·위치·행동 주체의 재부착이 생길 수 있다. identity 문자열이 남아도 나머지 의미가 같다는 보장은 없다.
- **범위:** 과거 `and` 포함 target 100개 조사에서 수동 기준의 명확한 연결 손상 13개(local 9개, factorial 13개), 추가 관계 위험 5개가 기록됐다. 구조 실패 1개는 projection 진행 전 실패이므로 이 13개에 포함하지 않았다. 이 조사는 최종 B-target 보완 전 snapshot을 포함하므로 현재 prompt의 오류율로 읽지 않는다.
- **최종 partition에도 남은 사례:** 아래 T13은 `and` coverage 보완 후에도 단독 술어를 남기면 연결사가 부적절하게 남는 실제 응답이다. 따라서 LLM 누락 보완과 subset 구성 정책을 분리해서 판단한다.
- **미확인:** 당시 사례집·조사는 문자열과 차분 계획을 검토했다. SigLIP embedding 값, 제거 부분공간, 실제 생성 품질 악화는 측정하지 않았다. 아래 위험을 SVD/projection 수학의 확정 오류나 성능 저하로 표현하지 않는다.

## 문제별로 유지할 판단 근거

| 문제 | 대표 사례 | 이후 구현에서 확인할 내용 |
|---|---|---|
| 연결사의 문맥 의존 | T13, AND011 | `and`를 후행 component에 배정해도 선행 술어를 지우면 연결 손상. local에서 괜찮아도 다른 성분을 함께 지우는 factorial에서 생길 수 있다. |
| 고정 identity가 가변 대상의 수식어 | T01, source의 `A tiny dragon perched on a pixel art cloud` | forest/cloud를 지우면 style이 다른 술어나 대상에 붙는다. 물리적 head까지 고정하면 사용자가 요청한 보존 범위를 넓힌다. |
| 대명사·필수 논항 의존 | T03, T17, T22 | its/it의 선행사 또는 행동 주체가 삭제된다. T17처럼 grouping이 대응할 수 있지만 독립 제거 방향의 수가 줄어든다. |
| 관계의 재부착 | T20, T23, `A dog on a chair beside a table` | web 삭제 후 위치가 Spider-Man에, 변형 삭제 후 leading이 ornament에 붙는지. 자연스러운 문장이어도 제거 대상 외 의미가 바뀔 수 있다. |
| 공유 carrier·수식 범위 | N08, `A dog wearing a red hat and a blue scarf` | with/wearing/red를 공유하는 구문을 분리하면 남은 항의 관계·색이 달라진다. 후자는 의도적 과분할 점검 후보였다. |
| 부정·선택·비교 의존 | N04, N05, `A dog larger than a cat` | either/or, not/but, larger/than을 독립 삭제할 때 원래 논리 관계가 유지되는가. 이들은 실제 LLM 실패가 아닌 수동 stress 후보였다. |
| 관사와 삭제 후 표현 | N01 | 관사 고정은 coverage를 해결하지만 `an excited dog`에서 excited를 지우면 `an dog`. 관사 수정은 현재의 verbatim 삭제 계약을 바꾼다. |
| source retained와 target 방향의 겹침 | S01 / S02 | source의 추가 보존 span은 모든 subset에서 고정돼야 한다. 그래도 target 방향 또는 embedding의 공유 의미가 외형 정보와 겹치는 효과는 별도 확인 대상이다. |

Fragment, 남은 쉼표, 주어가 없는 명사구 자체를 모두 오류로 판정하지 않는다. 제거 대상 이외의 의미·지시·수식 범위가 바뀌는지와 문법적 자연스러움을 구분한다.

## 아래 snapshot을 읽고 재현하는 방법

각 JSON block은 `input`, `response`, `kept_components`, `expected_subset`을 담는다. 번호는 **0부터 시작**한다. 입력/응답은 과거 사례의 해당 내용을 그대로 추출했고, 나열된 subset은 당시 문자열 전개 결과다. 전체 부분집합·차분 쌍은 중복 저장하지 않는다.

- T: 최종 final_v2 실제 target 응답. T17은 의존관계를 묶어 대응한 비교 사례다.
- S: B-source system prompt의 few-shot. 실제 reference를 새로 관찰한 API 응답이 아니다. 동일 source와 다른 target의 고정 문맥을 비교한다.
- N: 사람이 만든 stress partition. LLM이 실제로 이렇게 출력했다는 주장이나 권장 정답이 아니다.
- AND011: 최종 보완 이전 연결사 조사 snapshot. local/factorial 문맥 차이를 보존하기 위한 사례다.

이력 재현은 `semantic_components.SemanticLLM`의 `analyze_target` / `analyze_source`에 응답을 반환하는 mock provider를 연결한 뒤 `prompt_for_subset(kept_components)`를 호출하면 된다. 미래 정책 변경 시 `expected_subset`은 보존해야 할 정답이 아니라 **이전 동작과 비교할 문자열**이다. 실패하거나 통과한 snapshot을 조용히 수정해서 새 정책의 정답으로 만들지 않는다.

### T01 — 보존 style이 수식하던 forest와 장소 carrier를 제거하면 opening pixel art가 남는 문제.

```json
{"input":{"text":"A magical portal opening in a pixel art enchanted forest","identity_term":"pixel art"},"response":{"identity_term":"pixel art","preserved_spans":["pixel art"],"neutral_spans":[],"components":[{"spans":["A magical portal"]},{"spans":["opening"]},{"spans":["in a","enchanted forest"]}]},"kept_components":[1],"expected_subset":"opening pixel art"}
```

### T03 — phoenix 삭제 후 its feathers의 선행사와 of the sunset carrier의 잔존.

```json
{"input":{"text":"A giant phoenix soaring across the sky, its feathers echoing the colors of the sunset","identity_term":"sunset"},"response":{"identity_term":"sunset","preserved_spans":["of the sunset"],"neutral_spans":[],"components":[{"spans":["A giant phoenix"]},{"spans":["soaring"]},{"spans":["across the sky"]},{"spans":["its feathers echoing the colors"]}]},"kept_components":[3],"expected_subset":", its feathers echoing the colors of the sunset"}
```

### T13 — and를 exploring에 귀속해도 wearing 삭제 시 a kitten and exploring이 남음.

```json
{"input":{"text":"A retro-style digital art of a kitten wearing a tiny bow tie and exploring old books","identity_term":"kitten"},"response":{"identity_term":"kitten","preserved_spans":["a kitten"],"neutral_spans":[],"components":[{"spans":["A retro-style digital art of"]},{"spans":["wearing a tiny bow tie"]},{"spans":["and exploring old books"]}]},"kept_components":[2],"expected_subset":"a kitten and exploring old books"}
```

### T17 — riding a comet과 steering it을 묶어 it의 선행사를 함께 제거하는 대응.

```json
{"input":{"text":"A depiction of a space ranger riding a comet, steering it through a field of asteroids","identity_term":"space ranger"},"response":{"identity_term":"space ranger","preserved_spans":["a space ranger"],"neutral_spans":[],"components":[{"spans":["A depiction of"]},{"spans":["riding a comet, steering it through a field of asteroids"]}]},"kept_components":[1],"expected_subset":"a space ranger riding a comet, steering it through a field of asteroids"}
```

### T20 — web 삭제 후 between the moon and Earth가 Spider-Man의 위치로 재부착되는지.

```json
{"input":{"text":"Spider-Man weaving a giant web between the moon and Earth, silhouetted against a cosmic backdrop","identity_term":"Spider-Man"},"response":{"identity_term":"Spider-Man","preserved_spans":["Spider-Man"],"neutral_spans":[],"components":[{"spans":["weaving a giant web"]},{"spans":["between the moon and Earth"]},{"spans":["silhouetted against a cosmic backdrop"]}]},"kept_components":[1],"expected_subset":"Spider-Man between the moon and Earth,"}
```

### T22 — 목적어 identity Magic Cube: character 또는 solving 제거 후 남은 술어/논항.

```json
{"input":{"text":"An anime-style drawing of a character intensely focusing on solving a glowing Magic Cube","identity_term":"Magic Cube"},"response":{"identity_term":"Magic Cube","preserved_spans":["Magic Cube"],"neutral_spans":["a"],"components":[{"spans":["An anime-style drawing of"]},{"spans":["a character"]},{"spans":["intensely focusing on solving"]},{"spans":["glowing"]}]},"kept_components":[2],"expected_subset":"intensely focusing on solving a Magic Cube"}
```

### T23 — spaceship 변형을 제거하면 leading의 주체가 ornament로 바뀌는지.

```json
{"input":{"text":"A Rolls-Royce hood ornament transformed into a sleek spaceship leading a caravan of vintage cars across a galaxy","identity_term":"Rolls-Royce hood ornament"},"response":{"identity_term":"Rolls-Royce hood ornament","preserved_spans":["A Rolls-Royce hood ornament"],"neutral_spans":[],"components":[{"spans":["transformed into a sleek spaceship"]},{"spans":["leading a caravan of vintage cars"]},{"spans":["across a galaxy"]}]},"kept_components":[1],"expected_subset":"A Rolls-Royce hood ornament leading a caravan of vintage cars"}
```

### S01 — 외형 retained는 전 부분집합에서 고정. context만 제거하는 기준 사례.

```json
{"input":{"source_caption":"a brown dog with white paws sitting on grass","target_prompt":"a dog on the beach","identity_term":"dog"},"response":{"identity_term":"dog","preserved_spans":["dog"],"retained_spans":["brown","with white paws"],"neutral_spans":["a"],"components":[{"spans":["sitting"]},{"spans":["on grass"]}],"conflicts":[]},"kept_components":[],"expected_subset":"a brown dog with white paws"}
```

### S02 — S01과 같은 source이지만 target 전신 recolor 때문에 brown/white paws도 제거.

```json
{"input":{"source_caption":"a brown dog with white paws sitting on grass","target_prompt":"a dog with entirely purple fur","identity_term":"dog"},"response":{"identity_term":"dog","preserved_spans":["dog"],"retained_spans":[],"neutral_spans":["a"],"components":[{"spans":["brown"]},{"spans":["with white paws"]},{"spans":["sitting"]},{"spans":["on grass"]}],"conflicts":[]},"kept_components":[],"expected_subset":"a dog"}
```

### N01 — an excited dog에서 excited를 제거하면 an dog. 명사 head identity 반례.

```json
{"input":{"text":"an excited dog","identity_term":"dog"},"response":{"identity_term":"dog","preserved_spans":["dog"],"neutral_spans":["an"],"components":[{"spans":["excited"]}]},"kept_components":[],"expected_subset":"an dog"}
```

### N04 — either/or 상관 의존. 선택지를 독립 제거 성분으로 다뤄도 되는지.

```json
{"input":{"text":"A dog either sitting or standing","identity_term":"dog"},"response":{"identity_term":"dog","preserved_spans":["A dog"],"neutral_spans":[],"components":[{"spans":["either sitting"]},{"spans":["or standing"]}]},"kept_components":[1],"expected_subset":"A dog or standing"}
```

### N05 — not/but 대조와 부정 범위. 남은 성분의 해석을 유지할 단위 검토.

```json
{"input":{"text":"A dog not sitting but standing","identity_term":"dog"},"response":{"identity_term":"dog","preserved_spans":["A dog"],"neutral_spans":[],"components":[{"spans":["not sitting"]},{"spans":["but standing"]}]},"kept_components":[1],"expected_subset":"A dog but standing"}
```

### N08 — 공유 with/red의 범위. ears 제거 후 paws의 red 조건이 사라지는지.

```json
{"input":{"text":"A dog with red ears and paws","identity_term":"dog"},"response":{"identity_term":"dog","preserved_spans":["A dog"],"neutral_spans":[],"components":[{"spans":["with red ears"]},{"spans":["and paws"]}]},"kept_components":[1],"expected_subset":"A dog and paws"}
```

### AND011 — local만 보면 드러나지 않는 factorial 연결 손상

alert만 제거할 때는 crouched가 watching의 연결을 지탱할 수 있다. 아래처럼 crouched와 alert를 함께 제거하면 `and watching`만 남는다. 이는 문법적으로 모호한 local 결과까지 무조건 실패로 세지 않았던 역사적 사례다.

```json
{"input":{"text":"A photograph of a fox crouched in a lush green meadow, alert and watching","identity_term":"fox"},"response":{"identity_term":"fox","components":[{"spans":["crouched"]},{"spans":["in a lush green meadow"]},{"spans":["alert"]},{"spans":["and watching"]}],"preserved_spans":["a fox"],"neutral_spans":["A photograph of"]},"kept_components":[3],"expected_subset":"A photograph of a fox , and watching"}
```

## 향후 정책 변경 시 비교할 기준

1. 연결 의존을 grouping으로 해결할지, subset 제약으로 처리할지, 문장을 다시 구성할지는 아직 채택하지 않았다. 사용자가 지정하는 보완 방향에 따라 결정한다.
2. grouping은 제거 단위·방향 수를 바꾼다. subset을 제한하면 factorial의 문맥 집합·동일 가중 평균 정의가 바뀐다. 문장을 rewrite하면 원문 삭제 기반 contrast가 바뀐다. 문자열만 자연스럽게 만드는 수정으로 취급하지 않는다.
3. 모든 성분 유지 시 원문 정확 복원, identity·source retained 고정, 비연속 span의 동시 토글, source의 정확한 target binding은 현 계약이다. 이를 바꿀 때는 [pipeline 문서](semantic_pipeline.md)와 기존 테스트도 함께 검토한다.
4. [component_directions.py](../orthogonalization_instantx_ip_adapter/component_directions.py)는 정규화된 pooled embedding의 raw 차분을 평균한 뒤 최종 방향을 정규화한다. 각 pair 차분을 먼저 정규화하는 다른 class-direction builder와 혼동하지 않는다.
5. factorial은 성분 수 N에 대해 2^N개 subset을 요구하며 현재 기본 `max_contexts=256`을 초과하면 거절한다. 제한·부분집합 정의를 바꾸는 경우 계산량과 가중치도 명시한다.
6. [orthogonalization.py](../orthogonalization_instantx_ip_adapter/orthogonalization.py)의 SVD·visual projection·norm restoration은 현재 구현을 유지한다. 실제 개선 검증에서는 문자열 위험과 embedding/생성 결과를 따로 관찰한다. 이번 기록만으로 geometry 변경을 정당화하지 않는다.

원자료는 기존 `projection_contrast_casebook.md/json`과 `and_conjunction_projection_audit.md`·`assessment.json`이다. 48개 사례의 반복 전개를 위 문제군과 14개 snapshot으로 압축했다. 실행별 전체 응답·중복 부분집합·검증 수치 파일은 제거했다. 현재 prompt의 few-shot 전문은 [공유 프롬프트](prompts/llm_modules_share.md)와 runtime 코드에 남는다.

