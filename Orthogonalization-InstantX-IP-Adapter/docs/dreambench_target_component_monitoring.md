# DreamBench target components 전수 모니터링 (50개)

- 실행 시작: 2026-10-11T17:23:03+09:00 (Asia/Seoul)
- 실제 API 모델: `gpt-6-luna`; 현재 Module B system prompt와 strict JSON schema 사용
- 대상: object `backpack` 25개 + live subject `dog` 25개, 공식 순서 유지
- Caption/Module A, source 분석, embedding 계산, 이미지 생성은 실행하지 않음
- 기존 annotation/cache는 사용하지 않고 각각 새 API 요청; 재사용용 JSON 파일은 생성하지 않음
- 완료: 50/50; 구조 검증 PASS 50, FAIL 0; API 시도 53회; 재시도 사례 2개
- [공식 prompt 원본](https://raw.githubusercontent.com/google/dreambooth/main/dataset/prompts_and_classes.txt)과 로컬 입력 50개가 모두 일치함
- 공식 원본 SHA-256: `f632a5bdb41c3909ce22a0aefcea14432c0897e8db14ab845e68850af5ada486`
- 실행 system prompt SHA-256: `9cac2b66e54cafd63005be18f62608b22c3c7011dacff456e8216eeb5cc921a4` ([공유 프롬프트](prompts/llm_modules_share.md))

## 읽는 방법

`<backpack>`과 `<dog>`는 benchmark의 unique-token 자리를 나타내는 로컬 문자열이다. Identity는 각각 `<backpack> backpack`, `<dog> dog`로 지정했다. 마지막 5개 prompt는 `a red …`처럼 관사가 identity와 떨어지므로 `a`를 identity에 포함하지 않는다. 원문은 수정하지 않는다.

보존 구간과 중립 구간은 제거하지 않는다. Components의 바깥 배열은 그룹 목록이며, 각 안쪽 배열은 함께 제거되는 span들이다. 예: `[["on a","cloud"]]`는 component 하나다. 빈 배열 `[]`도 실제 LLM 결과 그대로 표시한다.

PASS는 JSON/span 타입·원문 일치·identity 보호·비중첩·순서·단어 배정 검증 통과를 뜻한다. 추가로 전체 component를 유지했을 때의 원문 복원과 모든 부분집합에서 고정 span 유지를 검사했다. 의미적 분해의 적절성까지 자동 판정한 점수는 아니다. 단어 분리·관사의 배정·같은 구문의 object/live 차이를 직접 검토할 수 있도록 결과를 수정하지 않고 기록한다.

## Object: backpack

| 번호 | Target prompt | 보존 구간 | 중립 구간 | 제거 components (그룹별 spans) | 검증 / API 시도 |
|---|---|---|---|---|---|
| 01 | "a &lt;backpack&gt; backpack in the jungle" | ["a &lt;backpack&gt; backpack"] | [] | [["in the jungle"]] | PASS / 1 |
| 02 | "a &lt;backpack&gt; backpack in the snow" | ["a &lt;backpack&gt; backpack"] | [] | [["in the snow"]] | PASS / 1 |
| 03 | "a &lt;backpack&gt; backpack on the beach" | ["a &lt;backpack&gt; backpack"] | [] | [["on the beach"]] | PASS / 1 |
| 04 | "a &lt;backpack&gt; backpack on a cobblestone street" | ["a &lt;backpack&gt; backpack"] | [] | [["on a cobblestone street"]] | PASS / 1 |
| 05 | "a &lt;backpack&gt; backpack on top of pink fabric" | ["a &lt;backpack&gt; backpack"] | [] | [["on top of pink fabric"]] | PASS / 1 |
| 06 | "a &lt;backpack&gt; backpack on top of a wooden floor" | ["a &lt;backpack&gt; backpack"] | [] | [["on top of a wooden floor"]] | PASS / 1 |
| 07 | "a &lt;backpack&gt; backpack with a city in the background" | ["a &lt;backpack&gt; backpack"] | [] | [["with a city in the background"]] | PASS / 1 |
| 08 | "a &lt;backpack&gt; backpack with a mountain in the background" | ["a &lt;backpack&gt; backpack"] | [] | [["with a mountain"], ["in the background"]] | PASS / 1 |
| 09 | "a &lt;backpack&gt; backpack with a blue house in the background" | ["a &lt;backpack&gt; backpack"] | [] | [["with a blue house in the background"]] | PASS / 1 |
| 10 | "a &lt;backpack&gt; backpack on top of a purple rug in a forest" | ["a &lt;backpack&gt; backpack"] | [] | [["on top of a purple rug"], ["in a forest"]] | PASS / 1 |
| 11 | "a &lt;backpack&gt; backpack with a wheat field in the background" | ["a &lt;backpack&gt; backpack"] | [] | [["with a wheat field in the background"]] | PASS / 1 |
| 12 | "a &lt;backpack&gt; backpack with a tree and autumn leaves in the background" | ["a &lt;backpack&gt; backpack"] | [] | [["with a tree"], ["and autumn leaves"], ["in the background"]] | PASS / 1 |
| 13 | "a &lt;backpack&gt; backpack with the Eiffel Tower in the background" | ["a &lt;backpack&gt; backpack"] | [] | [["with the Eiffel Tower in the background"]] | PASS / 1 |
| 14 | "a &lt;backpack&gt; backpack floating on top of water" | ["a &lt;backpack&gt; backpack"] | [] | [["floating"], ["on top of water"]] | PASS / 1 |
| 15 | "a &lt;backpack&gt; backpack floating in an ocean of milk" | ["a &lt;backpack&gt; backpack"] | [] | [["floating"], ["in an ocean of milk"]] | PASS / 1 |
| 16 | "a &lt;backpack&gt; backpack on top of green grass with sunflowers around it" | ["a &lt;backpack&gt; backpack"] | [] | [["on top of green grass"], ["with sunflowers around it"]] | PASS / 1 |
| 17 | "a &lt;backpack&gt; backpack on top of a mirror" | ["a &lt;backpack&gt; backpack"] | [] | [["on top of a mirror"]] | PASS / 1 |
| 18 | "a &lt;backpack&gt; backpack on top of the sidewalk in a crowded street" | ["a &lt;backpack&gt; backpack"] | [] | [["on top of the sidewalk"], ["in a crowded street"]] | PASS / 1 |
| 19 | "a &lt;backpack&gt; backpack on top of a dirt road" | ["a &lt;backpack&gt; backpack"] | [] | [["on top of a dirt road"]] | PASS / 1 |
| 20 | "a &lt;backpack&gt; backpack on top of a white rug" | ["a &lt;backpack&gt; backpack"] | [] | [["on top of a white rug"]] | PASS / 1 |
| 21 | "a red &lt;backpack&gt; backpack" | ["&lt;backpack&gt; backpack"] | [] | [["a red"]] | PASS / 1 |
| 22 | "a purple &lt;backpack&gt; backpack" | ["&lt;backpack&gt; backpack"] | [] | [["a purple"]] | PASS / 1 |
| 23 | "a shiny &lt;backpack&gt; backpack" | ["&lt;backpack&gt; backpack"] | [] | [["a shiny"]] | PASS / 1 |
| 24 | "a wet &lt;backpack&gt; backpack" | ["&lt;backpack&gt; backpack"] | [] | [["a wet"]] | PASS / 1 |
| 25 | "a cube shaped &lt;backpack&gt; backpack" | ["&lt;backpack&gt; backpack"] | [] | [["a cube shaped"]] | PASS / 3 |

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
| 09 | "a &lt;dog&gt; dog with a blue house in the background" | ["a &lt;dog&gt; dog"] | [] | [["with a blue house"], ["in the background"]] | PASS / 1 |
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
| 21 | "a red &lt;dog&gt; dog" | ["&lt;dog&gt; dog"] | [] | [["a red"]] | PASS / 1 |
| 22 | "a purple &lt;dog&gt; dog" | ["&lt;dog&gt; dog"] | [] | [["a purple"]] | PASS / 1 |
| 23 | "a shiny &lt;dog&gt; dog" | ["&lt;dog&gt; dog"] | [] | [["a shiny"]] | PASS / 2 |
| 24 | "a wet &lt;dog&gt; dog" | ["&lt;dog&gt; dog"] | [] | [["a wet"]] | PASS / 1 |
| 25 | "a cube shaped &lt;dog&gt; dog" | ["&lt;dog&gt; dog"] | [] | [["a cube shaped"]] | PASS / 1 |

## 재시도 및 실패 기록

### backpack 25

시도 1: 피드백 `없음`

```text
{"identity_term":"<backpack> backpack","preserved_spans":["<backpack> backpack"],"neutral_spans":["a"],"components":[{"spans":["cube shaped"]}]}
```

시도 2: 피드백 `ambiguous span; return a larger unique verbatim span`

```text
{"identity_term":"<backpack> backpack","preserved_spans":["<backpack> backpack"],"neutral_spans":["a"],"components":[{"spans":["cube shaped"]}]}
```

시도 3: 피드백 `ambiguous span; return a larger unique verbatim span`

```text
{"identity_term":"<backpack> backpack","preserved_spans":["<backpack> backpack"],"neutral_spans":[],"components":[{"spans":["a cube shaped"]}]}
```

### dog 23

시도 1: 피드백 `없음`

```text
{"identity_term":"<dog> dog","preserved_spans":["a <dog> dog"],"neutral_spans":[],"components":[{"spans":["shiny"]}]}
```

시도 2: 피드백 `span is absent, unordered, overlapping, or intersects preservation`

```text
{"identity_term":"<dog> dog","preserved_spans":["<dog> dog"],"neutral_spans":[],"components":[{"spans":["a shiny"]}]}
```

## 검토 메모

### 전체 결과

- 50개 모두 최종 구조 검증을 통과했다. 첫 응답 통과는 48/50개이며, 재시도 2개를 포함한 API 요청은 총 53회다.
- 최종 component 수는 1개인 prompt 41개, 2개인 prompt 8개, 3개인 prompt 1개다.
- 단일 location 표현, `floating`과 location의 분리, outfit 표현, `wearing …`과 착용 대상의 결합은 전반적으로 연구 의도에 맞게 나왔다. 최종 출력에는 identity 침범·원문 밖 표현·미배정 단어가 없었다.
- 다음 항목들은 최종 PASS와 별개로 검토할 필요가 있다. 결과를 좋게 보이도록 합치거나 재작성하지 않았으며, 이 실행 중 system prompt와 검증 코드는 변경하지 않았다.

### 의미 단위와 일관성 검토

| 사례 | 관찰 | 검토할 점 |
|---|---|---|
| Object 08 / Live 08 | 같은 `with a mountain in the background`를 object는 2개, live는 1개로 분해 | `in the background`가 앞의 대상과 함께 하나의 관계 표현을 이루는지에 대한 판단이 일관되지 않다. 하나로 묶는 정책이 필요할지 검토해야 한다. |
| Object 09 / Live 09 | 같은 `with a blue house in the background`를 object는 1개, live는 2개로 분해 | 위와 같은 종류의 편차다. 15개의 공통 템플릿 중 component 분해가 달랐던 것은 08, 09 두 개다. |
| Object 12 | `with a tree`, `and autumn leaves`, `in the background`의 3개로 분해 | Tree와 leaves를 독립 내용으로 볼 수는 있지만, 공유되는 background 관계가 따로 제거된다. 전체 background 묘사를 하나로 묶는 편이 의도에 가까운지 검토할 사례다. |
| Object/Live 21–25 | 최종적으로 `a red`, `a shiny`, `a cube shaped`처럼 관사가 수식어 component에 포함됨 | 현재 prompt에서 금지한 결과는 아니다. 다만 수식어 제거 시 관사도 사라져, 속성의 차이 외에 문법적 차이가 contrast에 섞일 수 있다. 관사를 중립으로 유지할지 별도 정책 검토가 필요하다. |
| Live 14 | `wearing a black top hat and a monocle`를 하나로 유지 | Predicate와 착용 대상을 묶는 지시에 부합한다. 두 accessory를 각각 바꾸는 ablation을 원할 때만 더 세분화할지 검토한다. |

### 재시도의 원인: LLM 응답과 검증기를 구분

**Object 25:** 첫 두 응답은 identity를 보존하고 `a`를 중립, `cube shaped`를 제거 component로 배정했다. 이는 원문상 유효한 분해지만 현재 검증기의 span 검색은 단어 경계 없이 문자열을 찾는다. 따라서 중립 `a`를 문장 시작의 관사뿐 아니라 `shaped` 내부에서도 찾아 `ambiguous span`으로 거절했다. 세 번째 응답이 `a cube shaped` 전체를 component로 반환하면서 통과했다. **이 사례는 LLM의 잘못된 분해가 아니라 검증기의 과도한 모호성 판정으로 확인됐다.** 단어 내부의 부분 문자열을 후보에서 제외하도록 span 매칭을 보완할 필요가 있다.

**Live 23:** 첫 응답의 보존 구간 `a <dog> dog`는 원문 `a shiny <dog> dog`에 연속해서 존재하지 않았다. 검증기가 올바르게 거절했고, 재시도에서 보존 구간을 `<dog> dog`로 수정하여 통과했다.

이번 실행은 현 설계의 관찰 기록이다. 구조상 최종 50개가 통과했다는 사실과 의미적 분해가 모두 이상적이라는 판단은 구분해야 한다. 위 항목을 검토한 후 prompt 또는 검증기를 수정한다면, 이 보고서의 기존 결과를 덮어쓰지 않고 별도 실행으로 비교하는 것이 적절하다.
