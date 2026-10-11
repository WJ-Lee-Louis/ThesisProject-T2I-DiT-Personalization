# DreamBench target components 전수 모니터링 — 관사 점검 후 첫 응답만 평가한 4차 (50개)

- 실행 시작: 2026-10-11T18:15:36+09:00 (Asia/Seoul)
- 실제 API 모델: `gpt-6-luna`; 현재 Module B system prompt와 strict JSON schema 사용
- 허용 재시도: 0회 (0이면 각 prompt의 첫 응답만 평가)
- 대상: object `backpack` 25개 + live subject `dog` 25개, 공식 순서 유지
- Caption/Module A, source 분석, embedding 계산, 이미지 생성은 실행하지 않음
- 기존 annotation/cache는 사용하지 않고 각각 새 API 요청; 재사용용 JSON 파일은 생성하지 않음
- 완료: 50/50; 구조 검증 PASS 50, FAIL 0; API 시도 50회; 재시도 사례 0개
- [공식 prompt 원본](https://raw.githubusercontent.com/google/dreambooth/main/dataset/prompts_and_classes.txt)과 로컬 입력 50개가 모두 일치함
- 공식 원본 SHA-256: `f632a5bdb41c3909ce22a0aefcea14432c0897e8db14ab845e68850af5ada486`
- 실행 system prompt SHA-256: `02c4b5f1b317f40101e72e66d259073d5028fde9e8e6a11130ee7c283d1e8646` ([공유 프롬프트](prompts/llm_modules_share.md))

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
| 10 | "a &lt;dog&gt; dog on top of a purple rug in a forest" | ["a &lt;dog&gt; dog"] | [] | [["on top of a purple rug"], ["in a forest"]] | PASS / 1 |
| 11 | "a &lt;dog&gt; dog wearing a red hat" | ["a &lt;dog&gt; dog"] | [] | [["wearing a red hat"]] | PASS / 1 |
| 12 | "a &lt;dog&gt; dog wearing a santa hat" | ["&lt;dog&gt; dog"] | ["a"] | [["wearing a santa hat"]] | PASS / 1 |
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

## 검토 메모

### 이번 수정

Module B에 출력 직전 모든 단어의 배정을 조용히 점검하라는 지시를 추가했다. 특히 identity_term에 포함되지 않은 identity 명사구의 첫 관사도 preserved/neutral 중 하나에 명시적으로 배정하고, 제거할 수식어에 붙이지 않도록 했다. 다른 제거 대상의 관사를 전부 보존하는 규칙은 추가하지 않았다.

Few-shot은 13개에서 14개로 늘렸다. 추가한 예시는 다음과 같다.

```json
{"text":"a lamp on a desk","identity_term":"lamp"}
```

```json
{"identity_term":"lamp","preserved_spans":["a lamp"],"neutral_spans":[],"components":[{"spans":["on a desk"]}]}
```

이는 identity 입력에 관사가 없어도 원문의 첫 `a`를 유지하며, 제거 대상인 desk의 관사는 location component에 그대로 포함하는 예시다.

### 첫 응답만 평가한 방법

[3차 실행](dreambench_target_component_monitoring_v3.md)과 동일한 50개 원문·identity_term·모델 `gpt-6-luna`·JSON schema를 사용했다. 입력은 변경하지 않았고, 이전 분석은 재사용하지 않았다.

실행 명령:

```powershell
python scripts/monitor_dreambench_target_components.py --model gpt-6-luna --workers 4 --max-retries 0 --output docs/dreambench_target_component_monitoring_v4_first_pass.md
```

`--max-retries 0`은 이번 모니터링에서 재호출을 금지한다. 첫 응답이 잘못되면 FAIL로 기록하도록 했으며, 오류 피드백을 통한 수정이나 원문/span 자동 보정은 수행하지 않았다. 일반 파이프라인의 기본 최대 재시도 2회 설정은 유지했다. 이 실행 중 prompt와 코드를 변경하지 않았다. Caption/source 분석·이미지 생성·재사용용 JSON 저장은 제외했다.

### 결과

| 항목 | 3차 | 4차: 재시도 금지 |
|---|---|---|
| 첫 응답 구조 검증 통과 | 45/50 | **50/50** |
| 첫 응답의 미배정 관사 누락 | 5개 | **0개** |
| 실제 API 요청 수 | 55회 (재시도 포함) | **50회** |
| 재시도 사례 | 5개 | **0개** |
| 문제였던 복합 장소의 독립 분해 | 3/3 | **3/3** |
| 대상과 background 관계 결합 | 9/9 | **9/9** |
| 수식어 prompt의 관사 유지 | 10/10 | **10/10** |
| Object/live 공통 템플릿의 component 분해 일치 | 15/15 | **15/15** |

**50개 모두 첫 응답만으로 이번 템플릿의 합의된 분해 기준을 충족했다.** Component 그룹은 3차의 검증된 최종 결과와 50개 모두 동일하다. 1 component인 prompt는 44개, 2 components인 prompt는 6개다.

- Object 10 / Live 10: `on top of a purple rug`와 `in a forest`로 분리.
- Object 18: `on top of the sidewalk`와 `in a crowded street`로 분리.
- 대상의 background 관계와 병렬 묘사는 함께 유지. `with a mountain in the background`, `with a blue house in the background`, `with a tree and autumn leaves in the background`를 분절하지 않음.
- 수식어 10개는 `a`를 유지하고 `red`, `purple`, `shiny`, `wet`, `cube shaped`만 제거.
- 이전에 첫 관사를 누락했던 Object 02, 04, 07, 19, 20도 첫 응답에서 실제 연속 `a + identity`를 보존.
- `floating`과 location의 독립 분해, `wearing a black top hat and a monocle`의 predicate 결합도 유지.

Live 12는 3차와 관사의 preserved/neutral 배정 방식만 달라졌으며 두 방식 모두 같은 고정 문구를 유지한다. 모든 component 부분집합에서 생성되는 문장도 3차 최종 결과와 같음을 확인했다.

### 검증과 해석 범위

50개 실제 첫 응답에 대해 원문 복원, identity 보호, 비중첩·순서·모든 단어 배정, 모든 부분집합에서 고정 span 유지와 합의된 component 그룹을 검토했다. Runtime/shared prompt가 일치하며 few-shot 14개도 구조 검증을 통과했다. 모니터링의 재시도 0회 설정을 확인하는 mock 테스트를 포함하여 관련 테스트 **97개가 통과**했다.

이는 선정한 두 subject와 고정 템플릿 50개를 한 번 실행한 결과다. 이번 실행의 첫 응답 성공은 확인했지만, 다른 subject·장문·style identity나 이후 반복 실행에서도 항상 100% 성공한다는 보장은 아니다. 따라서 일반 파이프라인의 검증과 제한된 재시도는 유지한다. 이미지 품질이나 실제 visual disentanglement 성능은 이번 결과로 판단하지 않는다.
