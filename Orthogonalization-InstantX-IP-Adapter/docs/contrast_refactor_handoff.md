# 두 단계 LLM·local contrast 개편: 새 세션 작업 요청서

작성일: 2026-10-18. 이 문서는 아직 구현되지 않은 개편 요구사항과 새 세션에 전달할 프롬프트다. 기존 구현 설명 문서와 구분한다. 사용자가 이 대화에서 확정한 정책을 아래 ‘공통 요구사항’에 정리했으며, 구체적인 schema·API 이름은 다음 설계 단계에서 결정한다.

## 권장 사용 순서

**새 채팅 하나에서 아래 1~4단계를 순서대로 요청한다.** 먼저 전체 입출력 계약과 두 모듈의 프롬프트를 함께 설계하고, 이후 모듈 구현과 projection 연결을 나눈다. 모든 것을 한 번에 구현하도록 요청하면 보존 정책·문법 수정 범위·저장 계약이 구현 도중 서로 다르게 정해질 수 있다. 반대로 각 모듈을 서로 다른 채팅에서 독립적으로 설계하면 전체 의미가 어긋나기 쉽다.

| 단계 | 작업 | 완료 시점 |
|---|---|---|
| 1 | 기존 코드 조사와 통합 설계 | 실제 입력·출력 예시, 두 시스템 프롬프트 초안, 영향 범위가 있는 설계 문서 작성 |
| 2 | 두 LLM 모듈·프롬프트·응답 계약 구현 | 새 응답 생성·검증·직렬화와 모듈 테스트 완료 |
| 3 | local 방향 계산·projection·실행 진입점 연결 | 새 두 단계 경로가 실제 생성 호출과 저장·재사용까지 연결됨 |
| 4 | 통합 검증·few-shot 검토·문서 정리 | 변경된 의미를 검증하고 실제 확인한 범위와 남은 한계 보고 |

이 문서의 공통 요구사항을 매번 다시 붙여 넣을 필요는 없다. 1단계 프롬프트에서 파일 전체를 읽도록 요청한다. 각 단계가 끝나면 결과를 확인하고 다음 프롬프트를 전달한다. 설계 수정이 필요하면 다음 단계 전에 수정 내용을 명시한다.

세션을 다시 바꿔야 한다면 이 파일과 1단계에서 작성한 `docs/contrast_refactor_design.md`를 읽히고, 완료된 단계와 다음 단계 번호를 함께 전달한다. 완료된 작업을 다시 수행하도록 요청하지 않는다.

## 공통 요구사항 — 사용자와 합의한 개편 방향

### 1. 변경 목적과 현재 구현의 위치

기존 경로는 A captioning, B-source component 분석, B-target component 분석의 세 모듈이다. component span을 조합·삭제하여 local/factorial embedding 입력을 만들었다. JSON과 원문 span이 유효해도 연결사 잔존, 수식관계 재부착, 필수 논항 소실 때문에 차분 의미가 흐려질 수 있었다.

새 경로는 **A captioning + B 공동 contrast 생성**의 두 단계다. B는 source-caption·target prompt·identity_term을 함께 읽고 최종 embedding 입력에 쓸 source/target별 contrast 텍스트 목록을 한 번에 제공한다. 런타임이 기존 component span을 기계적으로 삭제해서 문장을 만드는 구조를 새 주 경로로 유지하지 않는다. B-source와 B-target을 내부에서 따로 호출하면서 이름만 합치는 개편도 아니다.

새 방법은 **local만 사용**한다. factorial의 모든 부분집합 생성·평균은 새 경로에서 제외한다. 기존 구현을 참고해 연결부와 검증을 재설계하되, 시스템 프롬프트와 few-shot은 새 목적에 맞춰 전체 구조를 다시 작성한다.

### 2. Identity와 benchmark 입력

- `subject_id`/reference 폴더명과 `identity_term`을 구분한다. 현재 DreamBench identity_term은 원본 class이며 DreamBench++는 원본 label을 사용한다. 폴더명을 자연어 class 대신 무조건 넣지 않는다.
- DreamBench의 `bear_plushie → stuffed animal`, `berry_bowl → bowl`은 원본 class 매핑이다. UNO 기반 target prompt를 임의로 더 구체적인 이름으로 바꾸지 않는다. benchmark의 reference 선택과 target 원문을 유지한다.
- Class 문자열만 유지한다고 개체 identity가 보존되는 것은 아니다. Caption에 관찰된 외형·내용물·구성 관계 등 subject를 특징짓는 정보도 source 보존 판단에 포함한다.
- Source와 target 안에 identity_term을 어떤 표기 규칙으로 유지·검증할지 명시한다. 기존 exact identity 입력 계약을 근거 없이 약화하거나 폴더 ID를 class로 치환하지 않는다.

### 3. A 모듈 — reference image + identity_term → source-caption

- A에는 reference image와 identity_term을 전달한다. Target에 맞춰 관찰을 바꾸지 않도록 target prompt는 전달하지 않는다.
- 기존 A prompt의 적당한 상세도, 시각적으로 확인되는 정보만 쓰기, 불필요한 묘사·추측 금지 원칙을 활용한다.
- Caption은 기본적으로 **identity를 중심으로 특징·행동/상태·배경이 수식하는 자연스러운 영어 명사구**로 생성한다. 예: `A magenta backpack with gold zippers and colorful pins standing on a rocky ledge`.
- 독립 문장의 `stands`를 `standing`과 같은 수식 표현으로 구성하는 취지다. 단순 문자열 치환으로 구현하거나, 모든 caption을 특정 동사 하나로 강제하지 않는다. 정확한 identity 표현을 깨뜨리면서 수식어를 삽입하지 않는다.
- Subject identity이면 medium/style·촬영기법·조명·분위기·화질 묘사를 기본적으로 덧붙이지 않는다. identity_term 자체가 style/medium/scene 개념인 경우 그 개념에 맞는 묘사는 허용하되 추가적인 style 해석을 임의로 생성하지 않는다. 눈에 보이는 subject 색·무늬·재질은 계속 묘사할 수 있다.
- 이 제약은 **생성하는 source-caption**의 기준이다. B가 받는 기존 caption·사용자 caption·benchmark target까지 전부 명사구로 바꾸는 요구가 아니다.

### 4. B 모듈 — source-caption + target prompt + identity_term → source/target contrast 목록

#### Source의 보존·제거 판단

- Subject의 외형·내용물·구성 관계처럼 identity를 특징짓는 표현은 기본 보존한다. Class 단어만 보존하고 나머지를 전부 제거하지 않는다.
- 행동·자세·지지면·배경 등 identity 밖의 context는 기본 제거 후보다. 품사나 caption에서의 위치만으로 특징과 context를 나누지 않고 의미와 관계를 판단한다.
- Target이 **같은 subject의 해당 특징에 반하는 변경을 명시**하면 대응하는 source 특징만 보존에서 해제하고 제거 후보로 삼는다. 다른 보존 특징을 함께 해제하지 않는다.
- Target이 특징을 언급하지 않거나 단순히 새로운 특징을 추가하는 경우, 그것만으로 기존 source 특징을 보존에서 해제하지 않는다. 새 모자 추가를 털색 변경으로 보지 않는 식으로 변경의 대상을 구분한다.
- 예: source가 `A magenta backpack with gold zippers and colorful pins standing on a rocky ledge`일 때 target이 `A backpack in a forest`이면 magenta·gold zippers·colorful pins를 유지한다. Target이 `A purple backpack`이면 magenta만 보존에서 해제하고 zippers·pins는 유지한다.
- 예: identity가 `bowl`이고 caption에 blueberries가 담긴 관계가 관찰되면, 장소 변경만 요구하는 target 때문에 blueberries를 무조건 제거하지 않는다. `berry_bowl`의 내용물도 개체 특징이 될 수 있다는 문제를 반영한다. 모든 그릇·모든 내용물을 무조건 영구 고정하는 benchmark ID별 규칙은 만들지 않는다.
- 보존 영역을 완성된 문구로 설명할 수 있어도 새 schema를 하나의 연속 span으로 한정하지 않는다. 여러 곳의 보존 특징을 담을 수 있어야 하며 구체적인 표현 방식은 설계한다.

#### Target의 보존·제거 판단

- 기본적으로 identity와 그것을 자연스럽게 표현하는 데 필요한 최소 문법 요소를 유지하고, 나머지 의미 조건에서 target 제거 방향을 만든다.
- Target이 source와 충돌하는 특징을 요구하거나 새로운 특징을 추가하면, 그 target 표현은 제거 방향의 후보가 된다. 예컨대 purple은 source의 magenta 보존을 해제하는 근거이면서 동시에 target 쪽 제거 단위다.
- 이때 ‘제거’는 reference visual embedding에서 해당 방향을 projection으로 차감한다는 뜻이다. 실제 생성 모델에 주는 target prompt에서 purple을 삭제하는 뜻이 아니다. 생성에는 원래 target prompt를 전달한다.
- Source와 target에 정확히 같거나 의미가 같은 표현이 나타나는 극단 사례를 위한 별도 일치/불일치 예외 규칙은 추가하지 않는다. 원래 논의했던 ‘source와 같으면 target 제거 목록에서 자동 제외’ 규칙을 별도 기능으로 만들지 않는다. 우선순위는 명시적 충돌에 대한 source 보존 해제와 각 쪽의 일반 보존·제거 정책이다.

#### 의존관계와 문법 수정

- 제거 단위와 의존하는 표현은 필요하면 함께 제거한다. 필수 목적어·대상의 정의 관계절·공유 관계 등을 무조건 단어별로 쪼개지 않는다. 동시에 문장 전체를 항상 하나로 합치지도 않는다.
- 의존 표현을 함께 제거하는 과정에서도 확정된 identity·보존 특징은 유지한다. 제거할 내용과 보존할 내용이 얽힌 사례는 설계 예시에서 실제 출력까지 검토한다.
- 제거 외의 어휘와 표현은 가능한 한 원래대로 유지한다. 문법을 위해 관사·접속사·전치사 등 기능어를 제거·추가·수정할 수 있다. 새로운 실질 내용·색·대상·행동을 만들어 넣거나 광범위한 의역으로 문장을 다시 쓰지 않는다.
- B는 입력이 문장이면 문장 구조, 명사구이면 명사구 구조를 기준으로 자연스러움을 판단한다. 전체 입력을 명사구로 정규화하거나 완전한 문장으로 바꾸도록 강제하지 않는다.
- 대명사만을 위한 별도 복잡한 모듈이나 rewrite 기능은 우선 추가하지 않는다. 일반적인 의존관계 처리에서 다룬다. DreamBench++에서 드물다는 사실은 확인되지 않았으므로 확정된 빈도처럼 서술하지 않는다.
- 자연스러운 텍스트라는 조건과 제거 대상 외 의미를 유지한다는 조건을 별도로 검토한다. T01처럼 head를 지우면 style의 부착 대상이 달라지는 사례를 관사 문제만 고친 것으로 처리하지 않는다.

### 5. Local 방향과 projection 연결

- Source의 공통 positive는 실제 B 입력 source-caption S, target의 공통 positive는 실제 B 입력 target prompt T다. LLM이 만든 각 negative S_minus_i / T_minus_j와 local 차분을 만든다. Source 보존 요약 문구가 positive를 대체하지 않는다.
- `e(X)`를 기존 SigLIP pooled text embedding의 L2 정규화 결과라고 할 때 다음 관계를 사용한다.

```text
source direction_i = normalize(e(S) - e(S_minus_i))
target direction_j = normalize(e(T) - e(T_minus_j))
```

- 각 negative는 **원래 전체 표현에서 해당 제거 단위 하나와 그에 의존하는 표현만 제외한 결과**다. 여러 negative를 순서대로 누적 삭제하지 않는다. Source와 target을 서로 직접 빼지도 않는다.
- 충돌하여 보존에서 해제된 magenta도 하나의 source 제거 후보다. 다른 제거 단위의 negative에서 magenta를 미리 모두 지우는 것이 아니다. 반대로 계속 보존하는 gold zippers·colorful pins는 모든 source negative에서 유지한다.
- 새로 확정한 제거 단위 하나당 negative 하나, 방향 하나를 만든다. 과거 component 개수를 맞출 필요는 없다. 새로 사용하는 방향 수는 선택한 source/target 제거 목록의 길이와 일치한다.
- Source·target 원문이 같은 인코딩을 재사용할 수 있도록 정확히 같은 텍스트의 embedding 중복 호출은 줄일 수 있다. 이것을 source·target 의미 단위를 임의로 합치거나 제거 목록을 의미적으로 필터링하는 규칙과 혼동하지 않는다.
- 빈 제거 목록은 유효하다. Zero/near-zero 차분은 방향 수 기록과 실제 SVD 유효 rank를 구분하고 기존 수치 처리 정책을 확인해 명시한다. 이를 숨기기 위해 새 문장을 임의로 만들지 않는다.
- LLM의 최종 출력에서 핵심은 source/target negative 텍스트다. 파싱은 JSON을 우선 고려하고, 필요하면 제거 단위 식별자·짧은 보존 정보·출처를 포함한다. 장황한 추론문이나 기존 span partition 전체를 필수로 반환시키지 않는다. Schema와 최종 필드는 1단계에서 확정한다.
- 현재 projection 수학·강도·SVD·norm restoration을 바꾸는 것이 이번 개편의 자동 전제는 아니다. 새 contrast를 기존 geometry에 연결하는 코드 수정은 필수이며, 추가적인 수학 변경이 필요하다면 목적과 근거를 별도로 명시한다.

### 6. 기존 구현과의 전환 범위

- 새 두 단계 경로와 충돌하는 기존 span 삭제·검증·factorial 계약을 그대로 억지로 유지하지 않는다. 보존할 의미 정책과 폐기할 구현 제약을 구분한다.
- 새 semantic 경로는 local 전용으로 제공하고 factorial 요청을 조용히 local로 바꾸지 않는다. 기존 별도 class-projection 연구·batch 경로의 factorial까지 단어 검색으로 무차별 삭제하지 않는다. 남길 legacy 범위와 새 경로에서 거절할 인자를 설계에 적는다.
- 기존 `source` / `target` / `source_target`가 있다면 joint 분석의 입력 조건과 제거 방향 선택 옵션을 구분한다. 특히 기존 target-only는 caption 없이 동작했지만 새 B는 source-caption을 요구한다. 이전 호출 수·입력 계약을 그대로 약속하지 말고 새 두 단계 원칙에 맞는 동작과 이전 사용자의 전환 방법을 1단계에서 구체적으로 제안한다.
- 수동 source-caption이나 새 계약과 호환되는 저장 caption이 있으면 A를 생략할 수 있도록 검토한다. 수동 caption까지 명사구로 자동 rewrite하지 않는다.
- 과거 세 모듈 응답·`.components.json`·구 caption cache를 새 결과로 조용히 승격하지 않는다. Schema/policy/prompt 버전과 source-caption·target·identity 연결을 명시하고 재사용 조건을 정의한다. 기존 caption 180개는 이전 prompt의 관찰 자료이므로 새 A가 생성한 결과라고 표시하지 않는다.
- 기존 reference 이미지·benchmark 원문·caption 원본·실험 산출물은 보존한다. 새 중간 결과로 과거 원본을 덮어쓰지 않는다. 과거의 잘못된 subset을 새 정답으로 강제하지 않는다.

## 참고할 로컬 자료

아래 경로는 프로젝트 루트 `Orthogonalization-InstantX-IP-Adapter` 기준이다. 이 파일의 링크는 읽기 편의를 위한 것이며, 새 세션에서는 실제 코드도 확인한다.

| 자료 | 역할 |
|---|---|
| [semantic_pipeline.md](semantic_pipeline.md) | 기존 동작·보존 정책·CLI·저장 형식 |
| [llm_component_issues.md](llm_component_issues.md) | LLM/validator 오류, 보존 범위, 현재 대응과 한계 |
| [projection_issues.md](projection_issues.md) | 삭제 조합 문제, 실제/수동 출처를 구분한 snapshot 14개 |
| [llm_modules_share.md](prompts/llm_modules_share.md) | 기존 세 모듈 prompt·few-shot·schema |
| [module_a_caption_system_prompt.md](prompts/module_a_caption_system_prompt.md) | 기존 A prompt와 한국어 번역 |
| [semantic_components.py](../orthogonalization_instantx_ip_adapter/semantic_components.py) | 호출·파싱·partition 검증·연결·직렬화 |
| [source_component_prompt.py](../orthogonalization_instantx_ip_adapter/source_component_prompt.py) | 기존 source 정책과 executable few-shot |
| [component_directions.py](../orthogonalization_instantx_ip_adapter/component_directions.py) | embedding 입력 계획·방향 계산 |
| [adapter.py](../orthogonalization_instantx_ip_adapter/adapter.py), [orthogonalization.py](../orthogonalization_instantx_ip_adapter/orthogonalization.py) | 방향에서 projection으로의 연결과 geometry |
| [inference.py](../orthogonalization_instantx_ip_adapter/inference.py), [modal_inference.py](../modal_inference.py) | Python/Modal 진입점·캐시·진단 저장 |
| [llm_provider.py](../orthogonalization_instantx_ip_adapter/llm_provider.py) | 기존 structured provider·모델 설정·오류 처리 |
| [tests](../tests) | 영향 받는 기존 검증과 mock provider 사용 방식 |
| [captions.json](../datasets/caption_monitoring/module_a_v1/captions.json) | 기존 180건 caption·raw response·reference 연결·실패 보존 |
| [references.json](../datasets/caption_monitoring/module_a_v1/references.json) | 180개 reference의 원본 label·이미지 위치 |
| [evaluation_750.json](../datasets/dreambench_uno/evaluation_750.json), [manifest.json](../datasets/dreambench_uno/manifest.json) | UNO 기반 DreamBench prompt·class·reference 구분 |
| [DreamBench++ images](../datasets/XVerseBench/dreambench_plus_raw/extracted/data/images) | 실제 reference 확인이 필요할 때 사용 |

구현 단계에서는 `test_semantic_pipeline.py`, `test_source_semantic_analysis.py`, `test_target_prompt_examples.py` 등을 먼저 살핀다. 현재 테스트 이름·개수와 존재 여부를 확인하고 실제 의존 관계에 따라 범위를 정한다.

## 단계 1 — 새 세션의 첫 메시지로 전달할 프롬프트

```text
C:\Users\82107\Downloads\AI_PROJECT\Orthogonalization-InstantX-IP-Adapter 프로젝트의 LLM·contrast·projection 연결을 대대적으로 개편하려 한다.

먼저 docs/contrast_refactor_handoff.md 전체를 읽어라. 이 파일의 ‘공통 요구사항’은 사용자가 확정한 변경 방향이다. 기존 문서의 span 삭제·factorial 정책은 과거 동작을 설명하는 자료이며 새 요구사항보다 우선하지 않는다.

이번 요청은 전체 코드 조사와 설계 단계다. 실행 코드·runtime prompt·데이터셋은 아직 바꾸지 말고 docs/contrast_refactor_design.md에 구체적인 설계를 작성하라.

1. 작업 지침과 Git 상태를 확인하고, 참고 문서·runtime·provider·adapter·Python/Modal 진입점·저장 형식·관련 테스트를 조사하라. 기존 3모듈에서 이미지 생성까지 실제 호출·데이터 흐름을 설명하라.
2. A(image + identity → 명사구 caption), B(caption + target + identity → 공동 source/target contrast)의 새 전체 흐름을 설계하라. 두 stage를 여러 LLM 분석 호출로 숨겨 구현하지 마라. 검증 재시도와 선택적인 A 생략은 구분하라.
3. 두 모듈의 영문 시스템 프롬프트 전체 초안, 대표 few-shot 입력/출력, 실제 JSON schema와 코드 검증 범위를 작성하라. 기존 지시를 덧붙이는 방식이 아니라 새 역할·판단 우선순위·출력 계약에 맞게 다시 구성하라.
4. 기존 few-shot을 목적별로 검토하고 유지·수정·제거할 유형과 근거를 정리하라. 예시 개수를 맞추지 말고 새 정책을 가르치는 최소한의 비중복 사례를 선정하라.
5. Source 보존·명시적 충돌 해제·target 제거·의존 표현 동시 제거·기능어 보정을 실제 예시로 검토하라. Magenta backpack의 forest/purple 두 target과 berry bowl 특징 보존을 반드시 포함하라.
6. projection_issues.md의 대표 사례를 새 local 문제와 과거 factorial 전용 문제로 구분하라. T01/T13/T20/T22와 기존 대응 T16/T17/F02/F04/F08을 검토해 새 출력이 무엇이어야 하는지 제시하라. 근거 없이 모든 사례가 해결됐다고 결론내리지 마라.
7. Local positive는 각 원문이고 negative는 제거 단위 하나를 제외한 문법적으로 자연스러운 표현이라는 계약을 명시하라. 보존 요약만 남기는 누적 삭제 방식과 혼동하지 마라. 방향 개수·부호·정규화·빈 목록·퇴화 차분 처리·projection 연결을 적어라.
8. target-only 입력 계약, 기존 origin 선택, 수동/저장 caption, legacy cache/version, factorial 옵션, CLI·metadata·문서 변경 범위를 구체적으로 결정해 제안하라. 기존의 별도 연구 경로와 새 semantic 경로를 구분하라.
9. 구조 검증으로 확인할 수 있는 조건과 실제 LLM 판단·이미지/embedding 평가가 필요한 조건을 구분하고, 단계 2~4의 구현·검증 계획을 작성하라. 테스트를 보존할 것과 새 의미에 맞춰 교체할 것을 구분하라.

요청서가 이미 정한 정책을 다시 질문하지 마라. 남은 일반 구현 선택은 근거와 함께 제안하고, 방법론을 바꾸는 실질적 모순이 있을 때만 좁혀서 알려라. 검토를 마치면 설계 파일 위치, 핵심 결정, 사용자 확인이 꼭 필요한 미해결 사항만 보고하고 이 단계에서 마쳐라. 프롬프트 효과를 실제 API로 확인한 것처럼 서술하지 마라.
```

## 단계 2 — 설계를 검토한 뒤 두 LLM 모듈 구현 요청

설계에서 수정할 내용이 있다면 이 프롬프트 앞에 구체적으로 덧붙인다. 문제가 없다면 다음을 그대로 전달한다.

```text
앞서 작성한 docs/contrast_refactor_design.md의 설계로 두 LLM 모듈과 새 입출력 계약을 구현하라. docs/contrast_refactor_handoff.md 공통 요구사항을 계속 적용하라. 이번 단계는 LLM·파싱·검증·저장 객체까지이며, 실제 생성 진입점의 전환과 adapter 연결은 단계 3에서 마무리한다.

1. A의 runtime 시스템 프롬프트와 few-shot을 명사구 captioning 목적에 맞춰 전면 재구성하라. 기존 상세도·시각적 근거·style/medium 제한·정확한 identity 표기는 유지하라. 문장 동사 치환에 의존하지 마라.
2. B-source/B-target을 새 공동 contrast 생성 모듈로 구현하라. 한 번의 호출로 source/target 결과를 받고 component span 대신 실제 negative 텍스트를 사용하게 하라. 구체적인 클래스·함수·schema는 설계에 따르되 기존 이름 유지 자체를 목표로 삼지 마라.
3. B 시스템 프롬프트를 역할·보존/충돌 정책·제거 단위·의존관계·문법 보정·출력 순서에 맞춰 새로 작성하라. 기존 component 분석 지시에 문장 생성 지시만 덧붙이지 마라.
4. Source 특징 기본 보존, magenta→purple 해제, target의 신규/충돌 속성 방향, berry bowl 내용물 보존, style/scene identity와 physical head 구분을 few-shot에 균형 있게 반영하라. 동일 표현 중복 등장이나 대명사를 위한 별도 복잡한 예외 체계는 만들지 마라.
5. 기존 component 정답 JSON을 새 few-shot에 그대로 사용하지 마라. 문법 수정과 보존 의미를 설명하는 새 contrast 예시로 수정하거나 삭제하라. 형식적으로 같은 예시만 다수 추가하지 마라.
6. JSON 필드·타입·공백/빈 문자열·identity 연결·목록 길이·입력 바인딩 등의 코드 검증과, 의미적 보존·문법 타당성 같은 LLM 책임을 분리하라. 전체 lexical coverage/verbatim negative 강제를 그대로 이식하지 말고, schema 통과를 의미 정답으로 취급하지 마라.
7. 새 분석 객체와 직렬화·policy/schema version을 구현하라. 기존 결과를 새 schema로 조용히 읽지 마라. 제한된 재시도와 명확한 입력·출력 오류는 설계에 맞게 연결하라.
8. Mock provider로 두 단계 호출 구조, 캡션 생략 조건, 잘못된 응답 거절, source/target 분리, 보존·충돌 예시와 새 few-shot의 구조적 일관성을 검증하라. Mock 성공이 실제 모델의 의미 판단 성공이라는 주장은 하지 마라.
9. 수정된 prompt와 공유 문서를 함께 갱신해 runtime과 문서가 서로 다른 지시를 담지 않게 하라. A의 번역 문서도 최종 prompt와 맞춰라.

이번 단계에 필요한 범위는 끝까지 구현하고 관련 검증을 실행하라. 완료 후 바뀐 모듈/API·few-shot 선정 근거·검증 결과·단계 3에서 연결할 지점을 간단히 보고하라. 기존 원본 caption과 benchmark 자료는 덮어쓰지 마라.
```

## 단계 3 — local 방향·projection·실행 경로 통합 요청

```text
단계 2에서 구현한 두 모듈·새 응답 계약을 실제 local projection 실행 경로에 통합하라. docs/contrast_refactor_handoff.md와 docs/contrast_refactor_design.md를 기준으로 기존 호출자를 조사해 필요한 연결 변경을 완료하라.

1. Source는 e(source-caption)-e(source negative), target은 e(target 원문)-e(target negative)로 각 제거 단위당 하나의 방향을 계산하라. 기존 SigLIP pooled embedding 정규화·방향 정규화·부호를 유지하라. 누적 삭제나 source-target 직접 차분, factorial 평균을 넣지 마라.
2. 계획 생성기가 기존 ComponentAnalysis.prompt_for_subset를 통해 negative를 재생성하지 않게 하라. B가 반환한 텍스트가 인코더에 실제 전달되고 diagnostics에도 남도록 연결하라.
3. 정확히 같은 텍스트의 인코딩은 재사용하되 source/target 방향의 출처와 개수는 보존하라. 빈 목록과 zero/near-zero 차분은 설계대로 다루고 생성 방향 수와 최종 basis rank를 구분하라.
4. Adapter·inference·Modal·관련 script를 새 경로에 연결하고 입력 검증·호출 횟수·origin 선택·수동 caption 처리까지 일관되게 갱신하라. 기존 target-only가 source 없이 B를 호출하는 경로를 실수로 남기지 마라.
5. 새 semantic 경로의 factorial 선택을 제거하거나 명확하게 거절하라. 불필요해진 max_contexts 등의 인자·문서·진단은 정리하되, 별도 legacy class-projection 연구 코드까지 무관하게 삭제하지 마라.
6. 새 schema/version·policy·입력 연결을 저장하고 재사용 검증을 구현하라. 다른 target/identity/source-caption이나 이전 component JSON을 현재 결과로 재사용하지 마라. 실제 reference와 cache의 연결 조건도 설계대로 적용하라.
7. 생성 모델에는 원래 target prompt를 전달하라. contrast에서 purple을 제외한다고 생성 prompt까지 바꾸지 마라. 새 분석 텍스트와 생성 conditioning의 역할을 분리하라.
8. SVD·강도·visual projection·norm restoration은 새 연결 때문에 필요한 수정과 수학 자체 변경을 구분하라. 설계에 없는 수학 변경을 함께 끼워 넣지 마라.
9. 주입 가능한 encoder/mock model을 사용해 negative가 실제 인코딩되는지, 부호·정규화·원문 유지·방향 수·빈/퇴화 입력·저장 왕복·cache 불일치·새 진입점의 실제 데이터 흐름을 검증하라. 과거 geometry를 유지한 부분은 관련 회귀 검증을 실행하라.

새 경로가 실제로 연결될 때까지 필요한 작업을 완료하라. 관련 테스트 결과와 이전 사용법에서 바뀐 옵션·저장 형식을 보고하라. 아직 LLM의 실제 출력이나 이미지 품질을 측정하지 않았다면 그 범위를 명확히 구분하라.
```

## 단계 4 — 최종 검토·검증·문서 정리 요청

```text
두 단계 LLM·local contrast 개편 전체를 처음 요구사항부터 다시 검토하고 누락과 불일치를 수정하라. docs/contrast_refactor_handoff.md와 최종 설계 문서가 기준이다. 단순 코드 리뷰로 끝내지 말고 발견한 문제를 고친 뒤 검증까지 완료하라.

1. 전체 경로가 A→공동 B→source/target local 차분→기존 projection→원래 target으로 생성의 흐름인지 확인하라. 기존 component 분석·span 삭제·factorial 조합을 새 이름 뒤에 숨겨 남기지 않았는지 확인하라.
2. Magenta backpack의 장소 변경/색 변경, berry bowl 내용물 보존, target 신규 속성, style identity, 빈 제거 목록을 확인하라. Source 충돌 단위를 제외한 다른 negative에서도 보존 특징이 유지되는지, 각 negative가 원문 기준인지 확인하라.
3. T01/T13/T20/T22와 T16/T17/F02/F04/F08을 새 출력 관점에서 검토하라. 과거 문제 문서의 잘못된 subset을 새 기대값으로 복사하지 마라. 새 few-shot·수동 기대값·실제 API 관측을 구분하라.
4. 두 시스템 프롬프트를 전체적으로 다시 읽고 기존 역할과 새 역할의 혼재, 중복·상충 지시, caption의 명사구 지시와 B의 문장/명사구 허용 간 모순을 제거하라. Few-shot의 내용과 schema가 최종 지시를 실제로 따르는지 검토하라.
5. 의미를 기계적으로 보장할 수 없는 부분을 명시하라. Offline mock은 배선·검증·수식을 확인하는 것이고 문법·의미 판단 성공률은 실제 LLM 응답 검토가 필요하다. 문자열 개선을 embedding 분리나 생성 품질 개선으로 단정하지 마라.
6. 기본 검증은 offline으로 완료하라. 실제 LLM 호출/GPU 평가가 별도로 요청되지 않았다면 벤치마크 전수 재호출이나 이미지 생성을 시작하지 말고, 다음에 실행할 작은 실제 검토 집합과 관찰 항목을 제안하라. 기존 180개 caption은 검토 입력으로 활용할 수 있지만 새 A 결과라고 간주하지 마라.
7. 현재 설계·공유 prompt·A 번역·CLI 사용법·cache/version·진단 문서를 최종 구현과 일치시켜라. 과거 문제 기록은 이력임을 표시하고 현재 상태와 혼동되는 설명을 정리하라. 새 중간 보고서·반복 JSON을 불필요하게 늘리지 마라.
8. 관련 단위·통합·geometry 회귀 검증을 수행하고, 통과한 확인을 이유 없이 반복하지 마라. 변경된 의미 때문에 대체한 테스트와 계속 유지한 invariant를 설명하라. 테스트를 지워 실패를 숨기지 마라.

최종적으로 변경 결과, 실제 실행한 검증, 남은 불확실성, 현재 실행 방법을 간결하게 보고하라. 이번 요청에 없는 커밋·원격 배포·과거 원본 삭제는 하지 마라.
```

## 다음 세션의 검토에 사용할 대표 예시

다음은 최소한의 설계 확인 기준이다. 모든 예시를 학습용 few-shot으로 넣으라는 요구는 아니다. 동일 입력을 계속 prompt에 넣어 개발한 결과를 독립 평가라고 부르지 않는다.

| 입력/문제 | 기대하는 판단 |
|---|---|
| A: 눈에 보이는 backpack의 색·지퍼·핀·지지면 | 명사구로 관찰을 표현하고 불필요한 medium/style을 추가하지 않음 |
| Source magenta backpack + target forest | 색·지퍼·핀을 보존하고 source context / target forest에서 방향 생성 |
| 같은 source + target purple backpack | Source magenta만 보존 해제; source 다른 특징 유지; target purple 방향 생성 |
| bowl containing blueberries + 장소 변경 | 내용물과 담긴 관계를 identity 특징으로 고려하고 단순 context로 일괄 제거하지 않음 |
| T13: wearing … and exploring … | Wearing 제거 시 불필요한 and를 제거해 exploring의 의미와 남은 표현 유지 |
| T16/F08: teacup/platform + 관계절 | 도입 대상 삭제 시 그 대상을 정의하는 절도 함께 제거 |
| T17: riding a comet, steering it … | 일반 의존관계로 묶는 대응을 검토하되 대명사 전용 하위 시스템은 만들지 않음 |
| T20: web + between the moon and Earth | Web 삭제 후 위치가 주인공에게 재부착되는 결과를 정답으로 승인하지 않음 |
| T22: identity가 목적어인 Magic Cube | 실제 행동·행위자와 보존 identity를 구분하고 문법·논항 관계 검토 |
| T01/F04: style identity + physical head/carrier | 보존 style과 제거할 내용 대상의 scope를 확인; 근거 없는 대상 보존/의역 금지 |
| an excited dog | excited 제거 시 관사 보정 가능; 기존 verbatim 규칙 때문에 an dog를 강제하지 않음 |

추가 확인이 필요한 사례는 설계 단계에서 구체적인 입력과 출력으로 좁힌다. 정의되지 않은 모든 문법 현상을 별도 예외 규칙으로 늘리는 방식은 피한다.

대표 snapshot 문서에 전문이 없는 ID는 다음 입력을 사용한다. 기존 component 출력은 새 정답으로 고정하지 않는다.

| ID | identity_term | 원래 텍스트 |
|---|---|---|
| T16 | `Siamese cat` | `A Siamese cat walking gracefully along the rim of a giant teacup floating in the sky` |
| F02 | `a dog` | `a retro-style digital drawing of a dog playing the guitar on a sofa` |
| F04 | `pastel` | `a cat rendered in soft pastel tones` |
| F08 | `a fox` | `a fox on a platform that is floating in a lake` |
