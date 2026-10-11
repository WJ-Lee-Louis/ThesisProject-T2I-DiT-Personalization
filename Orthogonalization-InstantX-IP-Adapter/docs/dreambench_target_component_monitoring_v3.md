# DreamBench target components 전수 모니터링 — 독립 배경 분리 및 few-shot 추가 후 3차 (50개)

- 실행 시작: 2026-10-11T18:04:22+09:00 (Asia/Seoul)
- 실제 API 모델: `gpt-6-luna`; 현재 Module B system prompt와 strict JSON schema 사용
- 대상: object `backpack` 25개 + live subject `dog` 25개, 공식 순서 유지
- Caption/Module A, source 분석, embedding 계산, 이미지 생성은 실행하지 않음
- 기존 annotation/cache는 사용하지 않고 각각 새 API 요청; 재사용용 JSON 파일은 생성하지 않음
- 완료: 50/50; 구조 검증 PASS 50, FAIL 0; API 시도 55회; 재시도 사례 5개
- [공식 prompt 원본](https://raw.githubusercontent.com/google/dreambooth/main/dataset/prompts_and_classes.txt)과 로컬 입력 50개가 모두 일치함
- 공식 원본 SHA-256: `f632a5bdb41c3909ce22a0aefcea14432c0897e8db14ab845e68850af5ada486`
- 실행 system prompt SHA-256: `b9a5316ae56a340fc44850ae40bd6a203cdebae46a90e15eacc6bb6049ac2eec` ([공유 프롬프트](prompts/llm_modules_share.md))

## 읽는 방법

`<backpack>`과 `<dog>`는 benchmark의 unique-token 자리를 나타내는 로컬 문자열이다. Identity는 각각 `<backpack> backpack`, `<dog> dog`로 지정했다. 마지막 5개 prompt는 `a red …`처럼 관사가 identity와 떨어지므로 `a`를 identity에 포함하지 않는다. 원문은 수정하지 않는다.

보존 구간과 중립 구간은 제거하지 않는다. Components의 바깥 배열은 그룹 목록이며, 각 안쪽 배열은 함께 제거되는 span들이다. 예: `[["on a","cloud"]]`는 component 하나다. 빈 배열 `[]`도 실제 LLM 결과 그대로 표시한다.

PASS는 JSON/span 타입·원문 일치·identity 보호·비중첩·순서·단어 배정 검증 통과를 뜻한다. 추가로 전체 component를 유지했을 때의 원문 복원과 모든 부분집합에서 고정 span 유지를 검사했다. 의미적 분해의 적절성까지 자동 판정한 점수는 아니다. 단어 분리·관사의 배정·같은 구문의 object/live 차이를 직접 검토할 수 있도록 결과를 수정하지 않고 기록한다.

## Object: backpack

| 번호 | Target prompt | 보존 구간 | 중립 구간 | 제거 components (그룹별 spans) | 검증 / API 시도 |
|---|---|---|---|---|---|
| 01 | "a &lt;backpack&gt; backpack in the jungle" | ["a &lt;backpack&gt; backpack"] | [] | [["in the jungle"]] | PASS / 1 |
| 02 | "a &lt;backpack&gt; backpack in the snow" | ["a &lt;backpack&gt; backpack"] | [] | [["in the snow"]] | PASS / 2 |
| 03 | "a &lt;backpack&gt; backpack on the beach" | ["a &lt;backpack&gt; backpack"] | [] | [["on the beach"]] | PASS / 1 |
| 04 | "a &lt;backpack&gt; backpack on a cobblestone street" | ["a &lt;backpack&gt; backpack"] | [] | [["on a cobblestone street"]] | PASS / 2 |
| 05 | "a &lt;backpack&gt; backpack on top of pink fabric" | ["a &lt;backpack&gt; backpack"] | [] | [["on top of pink fabric"]] | PASS / 1 |
| 06 | "a &lt;backpack&gt; backpack on top of a wooden floor" | ["a &lt;backpack&gt; backpack"] | [] | [["on top of a wooden floor"]] | PASS / 1 |
| 07 | "a &lt;backpack&gt; backpack with a city in the background" | ["a &lt;backpack&gt; backpack"] | [] | [["with a city in the background"]] | PASS / 2 |
| 08 | "a &lt;backpack&gt; backpack with a mountain in the background" | ["a &lt;backpack&gt; backpack"] | [] | [["with a mountain in the background"]] | PASS / 1 |
| 09 | "a &lt;backpack&gt; backpack with a blue house in the background" | ["a &lt;backpack&gt; backpack"] | [] | [["with a blue house in the background"]] | PASS / 1 |
| 10 | "a &lt;backpack&gt; backpack on top of a purple rug in a forest" | ["a &lt;backpack&gt; backpack"] | [] | [["on top of a purple rug"], ["in a forest"]] | PASS / 1 |
| 11 | "a &lt;backpack&gt; backpack with a wheat field in the background" | ["a &lt;backpack&gt; backpack"] | [] | [["with a wheat field in the background"]] | PASS / 1 |
| 12 | "a &lt;backpack&gt; backpack with a tree and autumn leaves in the background" | ["a &lt;backpack&gt; backpack"] | [] | [["with a tree and autumn leaves in the background"]] | PASS / 1 |
| 13 | "a &lt;backpack&gt; backpack with the Eiffel Tower in the background" | ["a &lt;backpack&gt; backpack"] | [] | [["with the Eiffel Tower in the background"]] | PASS / 1 |
| 14 | "a &lt;backpack&gt; backpack floating on top of water" | ["a &lt;backpack&gt; backpack"] | [] | [["floating"], ["on top of water"]] | PASS / 1 |
| 15 | "a &lt;backpack&gt; backpack floating in an ocean of milk" | ["a &lt;backpack&gt; backpack"] | [] | [["floating"], ["in an ocean of milk"]] | PASS / 1 |
| 16 | "a &lt;backpack&gt; backpack on top of green grass with sunflowers around it" | ["a &lt;backpack&gt; backpack"] | [] | [["on top of green grass"], ["with sunflowers around it"]] | PASS / 1 |
| 17 | "a &lt;backpack&gt; backpack on top of a mirror" | ["a &lt;backpack&gt; backpack"] | [] | [["on top of a mirror"]] | PASS / 1 |
| 18 | "a &lt;backpack&gt; backpack on top of the sidewalk in a crowded street" | ["a &lt;backpack&gt; backpack"] | [] | [["on top of the sidewalk"], ["in a crowded street"]] | PASS / 1 |
| 19 | "a &lt;backpack&gt; backpack on top of a dirt road" | ["a &lt;backpack&gt; backpack"] | [] | [["on top of a dirt road"]] | PASS / 2 |
| 20 | "a &lt;backpack&gt; backpack on top of a white rug" | ["a &lt;backpack&gt; backpack"] | [] | [["on top of a white rug"]] | PASS / 2 |
| 21 | "a red &lt;backpack&gt; backpack" | ["&lt;backpack&gt; backpack"] | ["a"] | [["red"]] | PASS / 1 |
| 22 | "a purple &lt;backpack&gt; backpack" | ["&lt;backpack&gt; backpack"] | ["a"] | [["purple"]] | PASS / 1 |
| 23 | "a shiny &lt;backpack&gt; backpack" | ["&lt;backpack&gt; backpack"] | ["a"] | [["shiny"]] | PASS / 1 |
| 24 | "a wet &lt;backpack&gt; backpack" | ["&lt;backpack&gt; backpack"] | ["a"] | [["wet"]] | PASS / 1 |
| 25 | "a cube shaped &lt;backpack&gt; backpack" | ["&lt;backpack&gt; backpack"] | ["a"] | [["cube shaped"]] | PASS / 1 |

## Live subject: dog

| 번호 | Target prompt | 보존 구간 | 중립 구간 | 제거 components (그룹별 spans) | 검증 / API 시도 |
|---|---|---|---|---|---|
| 01 | "a &lt;dog&gt; dog in the jungle" | ["a &lt;dog&gt; dog"] | [] | [["in the jungle"]] | PASS / 1 |
| 02 | "a &lt;dog&gt; dog in the snow" | ["a &lt;dog&gt; dog"] | [] | [["in the snow"]] | PASS / 1 |
| 03 | "a &lt;dog&gt; dog on the beach" | ["a &lt;dog&gt; dog"] | [] | [["on the beach"]] | PASS / 1 |
| 04 | "a &lt;dog&gt; dog on a cobblestone street" | ["a &lt;dog&gt; dog"] | [] | [["on a cobblestone street"]] | PASS / 1 |
| 05 | "a &lt;dog&gt; dog on top of pink fabric" | ["a &lt;dog&gt; dog"] | [] | [["on top of pink fabric"]] | PASS / 1 |
| 06 | "a &lt;dog&gt; dog on top of a wooden floor" | ["a &lt;dog&gt; dog"] | [] | [["on top of a wooden floor"]] | PASS / 1 |
| 07 | "a &lt;dog&gt; dog with a city in the background" | ["a &lt;dog&gt; dog"] | [] | [["with a city in the background"]] | PASS / 1 |
| 08 | "a &lt;dog&gt; dog with a mountain in the background" | ["a &lt;dog&gt; dog"] | [] | [["with a mountain in the background"]] | PASS / 1 |
| 09 | "a &lt;dog&gt; dog with a blue house in the background" | ["a &lt;dog&gt; dog"] | [] | [["with a blue house in the background"]] | PASS / 1 |
| 10 | "a &lt;dog&gt; dog on top of a purple rug in a forest" | ["a &lt;dog&gt; dog"] | [] | [["on top of a purple rug"], ["in a forest"]] | PASS / 1 |
| 11 | "a &lt;dog&gt; dog wearing a red hat" | ["a &lt;dog&gt; dog"] | [] | [["wearing a red hat"]] | PASS / 1 |
| 12 | "a &lt;dog&gt; dog wearing a santa hat" | ["a &lt;dog&gt; dog"] | [] | [["wearing a santa hat"]] | PASS / 1 |
| 13 | "a &lt;dog&gt; dog wearing a rainbow scarf" | ["a &lt;dog&gt; dog"] | [] | [["wearing a rainbow scarf"]] | PASS / 1 |
| 14 | "a &lt;dog&gt; dog wearing a black top hat and a monocle" | ["a &lt;dog&gt; dog"] | [] | [["wearing a black top hat and a monocle"]] | PASS / 1 |
| 15 | "a &lt;dog&gt; dog in a chef outfit" | ["a &lt;dog&gt; dog"] | [] | [["in a chef outfit"]] | PASS / 1 |
| 16 | "a &lt;dog&gt; dog in a firefighter outfit" | ["a &lt;dog&gt; dog"] | [] | [["in a firefighter outfit"]] | PASS / 1 |
| 17 | "a &lt;dog&gt; dog in a police outfit" | ["a &lt;dog&gt; dog"] | [] | [["in a police outfit"]] | PASS / 1 |
| 18 | "a &lt;dog&gt; dog wearing pink glasses" | ["a &lt;dog&gt; dog"] | [] | [["wearing pink glasses"]] | PASS / 1 |
| 19 | "a &lt;dog&gt; dog wearing a yellow shirt" | ["a &lt;dog&gt; dog"] | [] | [["wearing a yellow shirt"]] | PASS / 1 |
| 20 | "a &lt;dog&gt; dog in a purple wizard outfit" | ["a &lt;dog&gt; dog"] | [] | [["in a purple wizard outfit"]] | PASS / 1 |
| 21 | "a red &lt;dog&gt; dog" | ["&lt;dog&gt; dog"] | ["a"] | [["red"]] | PASS / 1 |
| 22 | "a purple &lt;dog&gt; dog" | ["&lt;dog&gt; dog"] | ["a"] | [["purple"]] | PASS / 1 |
| 23 | "a shiny &lt;dog&gt; dog" | ["&lt;dog&gt; dog"] | ["a"] | [["shiny"]] | PASS / 1 |
| 24 | "a wet &lt;dog&gt; dog" | ["&lt;dog&gt; dog"] | ["a"] | [["wet"]] | PASS / 1 |
| 25 | "a cube shaped &lt;dog&gt; dog" | ["&lt;dog&gt; dog"] | ["a"] | [["cube shaped"]] | PASS / 1 |

## 재시도 및 실패 기록

### backpack 02

시도 1: 피드백 `없음`

```text
{"identity_term":"<backpack> backpack","preserved_spans":["<backpack> backpack"],"neutral_spans":[],"components":[{"spans":["in the snow"]}]}
```

시도 2: 피드백 `unassigned lexical content: a`

```text
{"identity_term":"<backpack> backpack","preserved_spans":["a <backpack> backpack"],"neutral_spans":[],"components":[{"spans":["in the snow"]}]}
```

### backpack 04

시도 1: 피드백 `없음`

```text
{"identity_term":"<backpack> backpack","preserved_spans":["<backpack> backpack"],"neutral_spans":[],"components":[{"spans":["on a cobblestone street"]}]}
```

시도 2: 피드백 `unassigned lexical content: a`

```text
{"identity_term":"<backpack> backpack","preserved_spans":["a <backpack> backpack"],"neutral_spans":[],"components":[{"spans":["on a cobblestone street"]}]}
```

### backpack 07

시도 1: 피드백 `없음`

```text
{"identity_term":"<backpack> backpack","preserved_spans":["<backpack> backpack"],"neutral_spans":[],"components":[{"spans":["with a city in the background"]}]}
```

시도 2: 피드백 `unassigned lexical content: a`

```text
{"identity_term":"<backpack> backpack","preserved_spans":["a <backpack> backpack"],"neutral_spans":[],"components":[{"spans":["with a city in the background"]}]}
```

### backpack 19

시도 1: 피드백 `없음`

```text
{"identity_term":"<backpack> backpack","preserved_spans":["<backpack> backpack"],"neutral_spans":[],"components":[{"spans":["on top of a dirt road"]}]}
```

시도 2: 피드백 `unassigned lexical content: a`

```text
{"identity_term":"<backpack> backpack","preserved_spans":["a <backpack> backpack"],"neutral_spans":[],"components":[{"spans":["on top of a dirt road"]}]}
```

### backpack 20

시도 1: 피드백 `없음`

```text
{"identity_term":"<backpack> backpack","preserved_spans":["<backpack> backpack"],"neutral_spans":[],"components":[{"spans":["on top of a white rug"]}]}
```

시도 2: 피드백 `unassigned lexical content: a`

```text
{"identity_term":"<backpack> backpack","preserved_spans":["a <backpack> backpack"],"neutral_spans":[],"components":[{"spans":["on top of a white rug"]}]}
```

## 검토 메모

### 수정과 비교 조건

[2차 결과](dreambench_target_component_monitoring_v2.md)와 동일한 50개 원문·identity_term·모델 `gpt-6-luna`·schema·최대 재시도 2회를 사용했다. 이전 결과는 그대로 보존하고, 모든 target에 새 API 요청을 수행했다. Caption/source 분석과 이미지 생성은 실행하지 않았고, 재사용용 JSON은 만들지 않았다. 이 실행 중 system prompt와 검증 코드는 변경하지 않았다.

System prompt에서 독립적으로 바꿀 수 있는 배치·환경 관계를 분리하도록 우선순위를 명시했다. 장소 표현이 단순히 연속된 경우에는 독립 조건으로 해석하며, 단순 인접만으로 전체를 묶지 않는다. Background 대상으로 도입된 다른 사물과 그 역할, 공유 관계를 갖는 병렬 대상, 명시적 관계절에 속한 대상 내부 묘사는 함께 유지한다. 관사 유지 및 단어 경계 검증 규칙, local/factorial/projection 수학은 유지했다.

### 추가한 few-shot 3개 (기존 10개 + 신규 3개 = 13개)

정확한 입력·JSON 출력은 [공유 system prompt](prompts/llm_modules_share.md)의 마지막 3개 예시에 있다. Benchmark 문장과 다른 단어를 사용하여 독립 조건과 대상 내부 관계의 차이를 대비한다.

| 입력 / identity | 제거 components | 목적 |
|---|---|---|
| `a fox standing on a stone platform in a garden` / `a fox` | `standing`, `on a stone platform`, `in a garden` | 자세·지지면·환경을 독립 분해 |
| `a fox with a tower and old trees in the background` / `a fox` | `with a tower and old trees in the background` | 도입된 대상과 공유 background 관계를 함께 유지 |
| `a fox on a platform that is floating in a lake` / `a fox` | `on a platform that is floating in a lake` | 관계절이 명시적으로 붙은 대상 내부 묘사를 함께 유지 |

### 결과 비교

| 항목 | 2차 | 3차 |
|---|---|---|
| 최종 구조 검증 | 50/50 | 50/50 |
| 첫 응답 구조 검증 통과 | 49/50 | 45/50 |
| API 요청 수 (재시도 포함) | 51 | 55 |
| 문제였던 복합 장소 3개를 독립 분해 | 0/3 | 3/3 |
| 대상과 background 관계의 결합 유지 | 9/9 | 9/9 |
| 수식어 prompt에서 identity 관사 유지 | 10/10 | 10/10 |
| Object/live 공통 템플릿의 component 분해 일치 | 15/15 | 15/15 |
| Component 수 분포 | 1개 47건 / 2개 3건 | 1개 44건 / 2개 6건 |

### 기존 문제의 해결 확인

| 사례 | 2차 | 3차 |
|---|---|---|
| Object 10 / Live 10 | `on top of a purple rug in a forest` | `on top of a purple rug` + `in a forest` |
| Object 18 | `on top of the sidewalk in a crowded street` | `on top of the sidewalk` + `in a crowded street` |

`with a mountain in the background`, `with a blue house in the background`, `with a tree and autumn leaves in the background`는 여전히 각각 하나의 component다. `wearing a black top hat and a monocle`도 한 predicate로 유지했다. 기존 `floating`과 location의 분리, `on top of green grass`와 `with sunflowers around it`의 분리도 유지했다. 21–25번 수식어는 양 class 모두 관사 없이 `red`, `purple`, `shiny`, `wet`, `cube shaped`만 제거한다.

최종 50개를 원문·고정 span·component 그룹 기준으로 모두 검토했다. 이번 템플릿에서 합의한 독립 장소 분리, background 관계 결합, predicate 유지 및 수식어 분해 기준에 최종 출력 50개가 모두 부합했다. 이는 이번 입력의 최종 결과에 대한 검토이며, 임의의 장문이나 다른 benchmark에서의 일반화 정확도가 100%라는 뜻은 아니다.

### 남은 점: 첫 응답의 관사 누락

Object 02, 04, 07, 19, 20의 첫 응답은 component 분해 자체는 맞았지만, 맨 앞 `a`를 preservation/neutral 어느 쪽에도 배정하지 않았다. 검증기가 `unassigned lexical content: a`로 올바르게 거절했고, 모두 다음 응답에서 실제 원문에 존재하는 `a + identity`를 보존하여 통과했다. 잘못된 word-substring 매칭이나 원문 자동 수정으로 통과시킨 결과는 아니다.

따라서 복합 배경의 granularity 문제는 이번 실행에서 해결됐지만, **첫 응답의 구조 검증 통과율은 2차보다 낮아졌다.** 제한된 재시도로 누락을 복구하는 현재 설계는 동작하지만, 호출 비용과 지연을 줄이려면 초기 관사를 identity에 포함하지 않는 일반 명사구의 coverage 예시를 추가 검토할 수 있다. 단일 실행만으로 prompt 변경이 누락 증가의 원인이라고 단정하거나 첫 응답 정확도까지 개선됐다고 주장하지 않는다.

### 구현 검증

런타임과 공유 문서의 system prompt 일치를 확인했고, few-shot 13개 모두 JSON/span 검증 및 원문 복원을 통과했다. 관련 테스트 **95개가 통과**했다. 50개 실제 최종 응답의 원문 복원, 비중첩·identity 보호 및 모든 부분집합 문맥에서 보존/중립 span 유지도 재확인했다. 의미적 image disentanglement 성능과 이미지 품질은 이번 검토 범위가 아니다.
