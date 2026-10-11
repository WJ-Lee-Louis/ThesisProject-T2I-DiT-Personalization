# DreamBench target components 전수 모니터링 — 설계 수정 후 2차 (50개)

- 실행 시작: 2026-10-11T17:48:22+09:00 (Asia/Seoul)
- 실제 API 모델: `gpt-6-luna`; 현재 Module B system prompt와 strict JSON schema 사용
- 대상: object `backpack` 25개 + live subject `dog` 25개, 공식 순서 유지
- Caption/Module A, source 분석, embedding 계산, 이미지 생성은 실행하지 않음
- 기존 annotation/cache는 사용하지 않고 각각 새 API 요청; 재사용용 JSON 파일은 생성하지 않음
- 완료: 50/50; 구조 검증 PASS 50, FAIL 0; API 시도 51회; 재시도 사례 1개
- [공식 prompt 원본](https://raw.githubusercontent.com/google/dreambooth/main/dataset/prompts_and_classes.txt)과 로컬 입력 50개가 모두 일치함
- 공식 원본 SHA-256: `f632a5bdb41c3909ce22a0aefcea14432c0897e8db14ab845e68850af5ada486`
- 실행 system prompt SHA-256: `8d9558f7ed82ff1bf02e0cd03fc8b5f409079791b636547575f3463dc04fd406` ([공유 프롬프트](prompts/llm_modules_share.md))

## 읽는 방법

`<backpack>`과 `<dog>`는 benchmark의 unique-token 자리를 나타내는 로컬 문자열이다. Identity는 각각 `<backpack> backpack`, `<dog> dog`로 지정했다. 마지막 5개 prompt는 `a red …`처럼 관사가 identity와 떨어지므로 `a`를 identity에 포함하지 않는다. 원문은 수정하지 않는다.

보존 구간과 중립 구간은 제거하지 않는다. Components의 바깥 배열은 그룹 목록이며, 각 안쪽 배열은 함께 제거되는 span들이다. 예: `[["on a","cloud"]]`는 component 하나다. 빈 배열 `[]`도 실제 LLM 결과 그대로 표시한다.

PASS는 JSON/span 타입·원문 일치·identity 보호·비중첩·순서·단어 배정 검증 통과를 뜻한다. 추가로 전체 component를 유지했을 때의 원문 복원과 모든 부분집합에서 고정 span 유지를 검사했다. 의미적 분해의 적절성까지 자동 판정한 점수는 아니다. 단어 분리·관사의 배정·같은 구문의 object/live 차이를 직접 검토할 수 있도록 결과를 수정하지 않고 기록한다.

## Object: backpack

| 번호 | Target prompt | 보존 구간 | 중립 구간 | 제거 components (그룹별 spans) | 검증 / API 시도 |
|---|---|---|---|---|---|
| 01 | "a &lt;backpack&gt; backpack in the jungle" | ["&lt;backpack&gt; backpack"] | ["a"] | [["in the jungle"]] | PASS / 1 |
| 02 | "a &lt;backpack&gt; backpack in the snow" | ["a &lt;backpack&gt; backpack"] | [] | [["in the snow"]] | PASS / 1 |
| 03 | "a &lt;backpack&gt; backpack on the beach" | ["a &lt;backpack&gt; backpack"] | [] | [["on the beach"]] | PASS / 1 |
| 04 | "a &lt;backpack&gt; backpack on a cobblestone street" | ["a &lt;backpack&gt; backpack"] | [] | [["on a cobblestone street"]] | PASS / 1 |
| 05 | "a &lt;backpack&gt; backpack on top of pink fabric" | ["a &lt;backpack&gt; backpack"] | [] | [["on top of pink fabric"]] | PASS / 1 |
| 06 | "a &lt;backpack&gt; backpack on top of a wooden floor" | ["&lt;backpack&gt; backpack"] | ["a"] | [["on top of a wooden floor"]] | PASS / 1 |
| 07 | "a &lt;backpack&gt; backpack with a city in the background" | ["a &lt;backpack&gt; backpack"] | [] | [["with a city in the background"]] | PASS / 1 |
| 08 | "a &lt;backpack&gt; backpack with a mountain in the background" | ["a &lt;backpack&gt; backpack"] | [] | [["with a mountain in the background"]] | PASS / 1 |
| 09 | "a &lt;backpack&gt; backpack with a blue house in the background" | ["a &lt;backpack&gt; backpack"] | [] | [["with a blue house in the background"]] | PASS / 1 |
| 10 | "a &lt;backpack&gt; backpack on top of a purple rug in a forest" | ["a &lt;backpack&gt; backpack"] | [] | [["on top of a purple rug in a forest"]] | PASS / 2 |
| 11 | "a &lt;backpack&gt; backpack with a wheat field in the background" | ["a &lt;backpack&gt; backpack"] | [] | [["with a wheat field in the background"]] | PASS / 1 |
| 12 | "a &lt;backpack&gt; backpack with a tree and autumn leaves in the background" | ["a &lt;backpack&gt; backpack"] | [] | [["with a tree and autumn leaves in the background"]] | PASS / 1 |
| 13 | "a &lt;backpack&gt; backpack with the Eiffel Tower in the background" | ["a &lt;backpack&gt; backpack"] | [] | [["with the Eiffel Tower in the background"]] | PASS / 1 |
| 14 | "a &lt;backpack&gt; backpack floating on top of water" | ["a &lt;backpack&gt; backpack"] | [] | [["floating"], ["on top of water"]] | PASS / 1 |
| 15 | "a &lt;backpack&gt; backpack floating in an ocean of milk" | ["a &lt;backpack&gt; backpack"] | [] | [["floating"], ["in an ocean of milk"]] | PASS / 1 |
| 16 | "a &lt;backpack&gt; backpack on top of green grass with sunflowers around it" | ["a &lt;backpack&gt; backpack"] | [] | [["on top of green grass"], ["with sunflowers around it"]] | PASS / 1 |
| 17 | "a &lt;backpack&gt; backpack on top of a mirror" | ["a &lt;backpack&gt; backpack"] | [] | [["on top of a mirror"]] | PASS / 1 |
| 18 | "a &lt;backpack&gt; backpack on top of the sidewalk in a crowded street" | ["a &lt;backpack&gt; backpack"] | [] | [["on top of the sidewalk in a crowded street"]] | PASS / 1 |
| 19 | "a &lt;backpack&gt; backpack on top of a dirt road" | ["a &lt;backpack&gt; backpack"] | [] | [["on top of a dirt road"]] | PASS / 1 |
| 20 | "a &lt;backpack&gt; backpack on top of a white rug" | ["a &lt;backpack&gt; backpack"] | [] | [["on top of a white rug"]] | PASS / 1 |
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
| 10 | "a &lt;dog&gt; dog on top of a purple rug in a forest" | ["a &lt;dog&gt; dog"] | [] | [["on top of a purple rug in a forest"]] | PASS / 1 |
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

### backpack 10

시도 1: 피드백 `없음`

```text
{"identity_term":"<backpack> backpack","preserved_spans":["<backpack> backpack"],"neutral_spans":[],"components":[{"spans":["on top of a purple rug in a forest"]}]}
```

시도 2: 피드백 `unassigned lexical content: a`

```text
{"identity_term":"<backpack> backpack","preserved_spans":["a <backpack> backpack"],"neutral_spans":[],"components":[{"spans":["on top of a purple rug in a forest"]}]}
```

## 검토 메모

### 수정과 비교 조건

[1차 결과](dreambench_target_component_monitoring.md)는 그대로 보존했다. 두 실행의 50개 target prompt, identity_term, 모델 `gpt-6-luna`, schema 및 최대 재시도 2회는 동일하다. 수정한 system prompt와 span 검증기로 새 API 분석을 수행했으며 이전 결과를 재사용하지 않았다. 이 실행 중 코드는 변경하지 않았다.

반영한 일반 규칙:

1. 다른 대상과 그 대상의 위치 관계를 하나로 유지한다. 하나의 predicate/preposition/relation을 공유하는 병렬 대상도 함께 묶는다. Identity 자체의 독립적인 자세와 location을 분리하는 규칙은 유지한다.
2. Identity 명사구의 관사는 제거할 수식어와 분리하여 유지한다. 관사와 identity가 떨어져 있으면 관사를 neutral로 배정하고, 원문에 없는 연속 preservation phrase를 만들지 않는다. 다른 제거 대상의 관사까지 모두 중립화하지 않는다.
3. 현재 span 형식에서 단어 경계를 검사한다. 다른 component가 유일하게 차지하는 원문 위치는 neutral span의 후보에서도 제외한다. 실제 반복 때문에 위치가 불명확한 span은 계속 거절하며 임의로 원문을 고치지 않는다. Legacy 문자열 형식과 local/factorial/projection 수학은 변경하지 않았다.

기존 few-shot 10개는 유지했으며, 추가 few-shot이나 benchmark별 정답을 런타임에 넣지 않았다.

### 결과 비교

| 항목 | 1차 | 2차 |
|---|---|---|
| 최종 구조 검증 | 50/50 | 50/50 |
| 첫 응답 통과 | 48/50 | 49/50 |
| API 요청 수 (재시도 포함) | 53 | 51 |
| Background 묘사 9개 중 관계를 별도 방향으로 분리 | 3개: Object 08, 12 / Live 09 | 0개: 9개 모두 대상과 background 관계를 함께 유지 |
| 수식어 prompt 10개에서 identity 관사 제거 | 10개 | 0개: 10개 모두 `a`를 neutral로 유지 |
| 동일한 object/live 공통 템플릿 15개의 component 분해 차이 | 2개: 08, 09 | 0개 |
| Component 수 분포 | 1개 41건 / 2개 8건 / 3개 1건 | 1개 47건 / 2개 3건 |

**이전 문제의 확인:** Object 08과 Live 08은 모두 `with a mountain in the background`를 한 component로 반환한다. Object 09와 Live 09도 `with a blue house in the background`를 한 component로 반환한다. Object 12의 `with a tree and autumn leaves in the background`는 공유 관계를 포함한 한 component다. Live 14의 `wearing a black top hat and a monocle`도 한 predicate로 유지된다.

**관사와 검증기:** 21–25번은 양 class 모두 `red`, `purple`, `shiny`, `wet`, `cube shaped`만 제거하며 `a`는 남긴다. Object 25의 첫 응답 `neutral_spans: ["a"]`가 추가 재시도 없이 통과해 `shaped` 내부의 `a` 때문에 잘못 거절되던 문제가 재현되지 않았다. 회귀 테스트에서도 해당 정상 응답과 `a red dog wearing a hat`의 관사 위치를 확인했고, 실제 모호한 반복 span 및 단어 조각 분해는 계속 거절된다.

**이번 재시도:** Object 10의 첫 응답에서 맨 앞 `a`가 어느 span에도 배정되지 않아 검증기가 올바르게 거절했다. 두 번째 응답은 관사를 포함하는 실제 연속 보존 구간을 반환하여 통과했다. 이는 잘못된 substring 매칭 문제가 아니라 LLM의 단어 누락이다. 다른 49개는 첫 응답에서 통과했다.

Object 01, 06은 관사를 preservation에 포함하는 대신 neutral로 반환했다. 두 표현 모두 모든 부분 문장에서 동일한 `a + identity`를 유지하므로 component 방향이나 생성할 부분 문장의 차이가 아니다.

### 남은 어려운 유형: 연속된 장소 표현의 부착 관계

| 사례 | 1차 | 2차 | 해석 |
|---|---|---|---|
| Object 10 / Live 10 | `on top of a purple rug` + `in a forest` | `on top of a purple rug in a forest` | Forest를 identity의 독립 장소로 볼지 rug를 수식하는 장소로 볼지 원문만으로 모호하다. |
| Object 18 | `on top of the sidewalk` + `in a crowded street` | `on top of the sidewalk in a crowded street` | Street를 독립 장면 조건으로 볼지 sidewalk의 위치로 볼지 같은 문제가 있다. |

이번 규칙은 이러한 문장을 하나의 복합 location으로 묶는 쪽으로도 영향을 주었다. 전체 장소를 제거하는 목적에는 타당한 해석이지만, rug/forest 또는 sidewalk/street를 독립적으로 차감하려는 목적에는 분해가 너무 큰 결과다. **기존 문제는 개선됐지만 모든 유형에서 이상적인 granularity를 얻었다고 결론 내리지는 않는다.** 일관성 개선도 단일 실행의 관찰이며 다른 subject와 장문으로 일반화된 정확도를 뜻하지 않는다.

### 후속 few-shot 검토 후보 — 런타임 미적용

추가 few-shot을 적용하기 전에 다음 정책을 검토한다: *명시적인 부착 단서가 없으면 identity의 독립적인 placement 조건을 분리하고, 다른 대상의 위치를 직접 설명하거나 공유 관계/관계절로 연결된 부분은 함께 유지한다.* 아래는 그 정책을 택할 때 사용할 후보다. 현재 system prompt에는 추가하지 않았다.

**후보 1: 자세와 두 개의 장면 placement를 독립적으로 제거**

입력:

```json
{"text":"a fox standing on a stone platform in a garden","identity_term":"a fox"}
```

제안 출력:

```json
{"identity_term":"a fox","preserved_spans":["a fox"],"neutral_spans":[],"components":[{"spans":["standing"]},{"spans":["on a stone platform"]},{"spans":["in a garden"]}]}
```

**후보 2: 도입된 병렬 대상과 공유 background 관계는 하나로 유지**

입력:

```json
{"text":"a fox with a tower and old trees in the background","identity_term":"a fox"}
```

제안 출력:

```json
{"identity_term":"a fox","preserved_spans":["a fox"],"neutral_spans":[],"components":[{"spans":["with a tower and old trees in the background"]}]}
```

**후보 3: 관계절이 장소 대상에 명시적으로 붙으면 함께 유지**

입력:

```json
{"text":"a fox on a platform that is floating in a lake","identity_term":"a fox"}
```

제안 출력:

```json
{"identity_term":"a fox","preserved_spans":["a fox"],"neutral_spans":[],"components":[{"spans":["on a platform that is floating in a lake"]}]}
```

이는 benchmark 문장을 그대로 암기시키기보다 독립 placement와 대상 내부 수식의 차이를 가르치려는 후보다. 후보 1의 독립 분해 정책을 채택할지와 style identity에도 같은 원칙을 적용할지는 후속 검토 대상으로 남긴다.

### 구현 검증

새 span 매칭과 관사 유지에 대한 회귀 검증을 포함하여 전체 테스트 **744개가 통과**했다. 50개 최종 출력의 원문 복원과 고정 span 유지도 재확인했다. Captioning, 실제 text/visual embedding 계산과 이미지 품질 평가는 이번 비교 범위에 포함하지 않았다.
