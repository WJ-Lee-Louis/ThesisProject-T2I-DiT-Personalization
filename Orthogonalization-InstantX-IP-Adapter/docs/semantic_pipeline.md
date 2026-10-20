# Joint local contrast 생성 경로

2026-10-20 현재 경로. GPT-5.6 Sol 전체 A/B 실행 상태는 outputs/semantic_benchmark_gpt56_sol에서 확인한다. GPU 생성·이미지 품질 평가는 별도다.
[설계 §14](contrast_refactor_design.md), [전체 prompt와 schema](prompts/llm_modules_share.md)를 참고한다.

```text
reference + identity → A (target 미전달)
caption + preserve + 원문 target + identity → 공동 B 한 번
origin별 행 선택 → normalize(e(원문) - e(negative))
→ 기존 SVD/visual projection/norm restoration → 원문 target으로 생성
```

모든 origin은 같은 공동 B를 호출한다. 기본 A+B 두 번, 수동/호환 caption 제공 시 B 한 번,
full cache 재사용 시 0번이다. target-only도 source가 필요하며 없으면 A를 실행한다.
구조 검증 재시도는 별도다. 각 negative는 원문에서 단위 하나를 제거한 결과이며 누적 삭제나
보존 요약이 아니다. B가 반환한 text가 직접 인코더에 들어간다.

같은 text만 한 번 인코딩하고 source/target 행 수는 보존한다. 빈 목록에는 identity shape probe
한 번을 사용하되 방향 수는 0이다. raw norm≤1e-12는 zero row로 남긴다. 더 큰 작은 차분은 기존
정규화 정책을 따르며 별도 신뢰도 cutoff를 추가하지 않았다. SVD 수학은 변경하지 않았다.

## Python

```python
from orthogonalization_instantx_ip_adapter.llm_provider import ResponsesJSONProvider
from orthogonalization_instantx_ip_adapter.semantic_contrasts import SemanticLLM
from orthogonalization_instantx_ip_adapter.inference import generate_semantic_png_bytes

# 실제 provider를 사용하면 유료 API/GPU 생성이 실행된다.
png, diagnostics = generate_semantic_png_bytes(
    identity_term="backpack", target_prompt="A purple backpack",
    reference_image_bytes=reference_bytes,
    llm=SemanticLLM(ResponsesJSONProvider.from_env("gpt-5.6-sol")),
    source_caption="A magenta backpack with gold zippers and colorful pins standing on a rocky ledge",
    preserve=["magenta", "with gold zippers and colorful pins"],
    origin="source_target",
)
```

caption_record 또는 contrast_analysis_payload로 새 cache를 전달할 수 있다. full cache이면
llm을 생략할 수 있고 caption 공급과 혼합할 수 없다. origin=None은 저장 선택을 따르며 명시
origin은 행 선택만 바꾼다. strategy는 local만 허용하고 max_contexts는 제거했다.

## Modal

```powershell
python -m modal run modal_inference.py `
  --reference-image-path "C:/path/reference.png" `
  --identity-term "backpack" `
  --prompt "A purple backpack" `
  --source-caption "A magenta backpack with gold zippers and colorful pins standing on a rocky ledge" `
  --preserve-json '["magenta", "with gold zippers and colorful pins"]' `
  --llm-model gpt-5.6-sol `
  --contrast-origin source_target
```

- --contrast-origin source|target|source_target: 방향 선택. 새 실행 기본은 source_target.
- --contrast-analysis-path: schema 4 full cache. identity 생략 시 저장 identity를 사용한다.
  target/reference 및 명시 identity/model의 불일치는 거절한다.
- --source-caption, --caption-cache-path, --import-legacy-caption: 상호 배타적인 caption 공급.
  full cache와 혼합 금지. caption-cache는 CaptionRecord.to_dict() 형식이다.
- legacy import는 aggregate의 임의 첫 행이 아닌 단일 PASS record JSON을 받는다. exact identity,
  원본 image_sha256와 실제 reference bytes가 일치해야 한다. 구 path/file hash/model/prompt를
  provenance에 남기고 fresh B를 수행한다. 새 A 결과로 승격하지 않는다.
- --component-origin/--component-analysis-path는 경고 alias이며 새 schema만 읽는다.
  구/신 alias 동시 지정은 오류. --component-strategy local만 transitional 허용하고 factorial과
  명시 --component-max-contexts는 유료 호출 전 거절한다.
- identity CLI의 --orthogonalization-mode components는 새 semantic_local로 이행하는 alias다.
  legacy behavior/background 선택과 새 semantic 모드를 혼합할 수 없다.

URL은 로컬에서 한 번 읽은 bytes를 A와 원격에 전달한다. 원격 generate_png_bytes는 고정
reference bytes와 새 payload를 검증한 후 GPU 모델을 로드한다. URL 문자열만으로 이미지가
같다고 가정하지 않는다. 생성 prompt는 항상 원문 target이다.

## 저장·검증·진단

PNG, diagnostics .json, 새 .contrasts.json을 같은 stem으로 저장하며 기존 파일과 충돌하면
다음 이름을 선택한다. schema_version=4, policy_version=joint_local_v7, prompt hash/version,
S/T/I, reference RGB pixels+크기 hash, model/attempts를 저장한다. 원본 byte hash도 남긴다.
재인코딩만 다른 동일 RGB는 재사용 가능하다. 구 .components.json/schema 2/unversioned는 새
loader에서 거절한다. target이 바뀌면 B를 다시 실행하고 호환 A caption은 재사용할 수 있다.

이미지 없는 수동 분석 기록은 reference-bound generation cache로 승격할 수 없다. 실제
reference를 제공하여 준비해야 한다. 구조 검증은 시각적 정확성·문법·보존 의미의 증명이 아니다.
T01류는 style의 직접 수식 head를 최소 보존하고 의존 표현을 묶는다. scope/대명사 문제로 분석 불가를 반환하지 않는다. 기본 retry는 2회, 유효 issue/transport 오류는 재시도하지 않는다.

Diagnostics에는 실제 positive/negative/unit/origin, source 보존 구, raw norms, zero flags/count,
unique encoding 수, 입력 행 수와 basis rank, singular values, visual edit/norm/cosine, 분석
prompt/model/attempts를 남긴다. source_preserved는 embedding positive를 대체하지 않는다.

## Monitor와 legacy

monitor_semantic_pipeline.py → semantic_monitor_modal.py → render_semantic_monitor.py는 schema 4,
local만 사용한다. 기본 출력은 outputs/semantic_local_v3이며 구 run 디렉터리 혼용을 거절한다.
GPU 이전에 cache/reference를 검증하고 렌더러는 이미지가 없으면 명시한다. 구 A monitor는 과거 reference ledger 기반의 개발 도구이며 현재 전체 실행에는 사용하지 않는다. 과거 180 caption과 benchmark는 보존한다.

semantic_components.py/source_component_prompt.py/component_directions.py와 components mode,
component_analysis_payload low-level API는 명시적 legacy 재현용이다. 새 identity CLI와 고수준
Python은 이를 호출하지 않는다. 별도 PromptSet factorial/class-contextual/계층 projection 및
batch/eval 연구의 수학·저장 계약은 유지한다. source/target 분리 monitor와 component dataset
도구도 legacy로 표시했다. 기존 batch diagnostics 요약 도구는 별도 경로로 유지한다.

현재 A는 `caption_np_v6` / `identity_np_concept_v4`, B는 `joint_contrast_v8` /
`joint_local_v7`다. Caption schema 2, conditioning/analysis schema 4를 사용한다.

A는 Target 없이 간결한 caption과 추가 보존 표현 목록 preserve를 함께 만든다. Identity는
목록에 반복하지 않는다. 예: `{"source_caption":"A magenta backpack on a ledge","preserve":["magenta"]}`.
B에는 source_caption, target_prompt, identity_term, preserve가 전달된다. B는 전달된 기본
보존 범위에서 명시적 충돌 부분만 해제하고, 각 원문에서 local negative를 만든다. Identity와
최소 필수 수식 관계는 별도로 보호한다. Negative의 관사·전치사 보정은 허용한다.

수동 caption 또는 명시적 legacy import는 preserve 목록이 필수다. Python에서는
`preserve=["magenta"]`, Modal에서는 `--preserve-json '["magenta"]'`를 전달한다.
`[]`는 identity만 보존한다는 명시적 선택이다. 자동 A나 caption/full cache와 외부 preserve는
혼용할 수 없다. Cache는 이미지·identity·source·target·preserve·model·policy/hash에 연결되며
이전 schema를 자동 변환하지 않는다. A가 생성한 목록은 trace의 실제 응답과도 일치해야 한다.

영문 전문 및 번역: [A](prompts/module_a_caption_system_prompt.md),
[B](prompts/module_b_contrast_system_prompt.md). 기본 보존 목록과 B의 source_preserved 감사
필드는 다르며, 후자는 identity를 포함한다. 구조 검증으로 의미·충돌·문법 정확성을 보장하지
않는다. Projection 수학과 원래 Target을 사용하는 생성 경로는 유지했다.

전체 GPT-5.6 Sol 준비/중단 후 재개:
```powershell
python scripts/run_full_semantic_benchmark.py --run
```
실패 항목만 재요청하려면 `--retry-errors`를 추가한다. 성공한 caption/conditioning은 원본 이미지·
입력·모델·prompt 해시를 검증한 뒤 재사용한다. 출력은 `outputs/semantic_benchmark_gpt56_sol`이다.
`generation_manifest.json`의 READY 행은 직접 읽을 수 있는 conditioning JSON 경로와 원래
Target을 포함한다. 실제 image 생성이나 SigLIP embedding 계산은 이 실행에 포함하지 않는다.
원본 180 reference/2100 Target ledger는 `datasets/semantic_benchmark/references.json`에 있다.
중간 pilot 디렉터리·보고서·일회성 스크립트는 활성 프로젝트에서 정리했다.

사용자가 승인한 예외: Target 원문 및 negative 맨 앞에서는 identity 첫 글자의 대문자화를
허용한다. 내부 철자·나머지 대소문자·Source exact identity 계약은 유지한다. 출력 identity_term도
원래 label 그대로이다. Benchmark Target 원문은 수정하지 않는다.

현재 concept-only 규칙: 특정 피사체가 아닌 style·매체·장면 개념은 preserve=[]로 추가 보호를 하지 않는다. Identity 명칭은 별도 보호한다. 묶음/분리 개수 차이는 의미가 유지되는 한 오류가 아니다. 현재 전체 실행의 진행/완료 상태는 출력 progress.json과 summary.json으로 확인한다.

전체 실행 완료: A 180/180, 생성 준비 2100쌍(정상 공동 결과 2098 + 명시적 source-only fallback 2).
[실행 결과 및 재사용 방법](../datasets/semantic_benchmark/README.md)을 참고한다.


## Target 충돌 시 명시적 Source-only fallback

사용자가 승인한 사례만 `prepare_source_only_fallback`으로 처리한다. 기존 caption/reference와
동일 입력·모델의 Target identity_conflict trace가 있어야 한다. 정상 B가 실패했다고 무조건
실행하지 않으며 source-side conflict, transport/구조 오류에는 적용하지 않는다.
원래 Target과 caption/preserve를 읽는 추가 LLM 호출 하나로 source만 구성한다. Target 목록은
빈 배열이며 기존 충돌은 삭제하지 않는다. 별도 stage/prompt hash와 원래 trace를 함께 저장한다.
Kind는 `semantic_source_fallback`, schema 4, fallback policy는 `target_conflict_source_only_v1`이다.
정상 2098개 cache와 기존 A/B hash는 그대로 유효하다. Target 입력이나 reference가 다르면 거절한다.

저장 origin은 source이며 source_target 선택도 동일 Source 방향을 유지한다. Target-only 선택은
Source를 유지하라는 fallback 계약과 충돌하므로 거절한다. Python/Modal은 기존 full-cache 경로로
재사용하며 생성 prompt는 원래 Target이다. Diagnostics의 target_projection_skipped와 fallback_policy,
conditioning의 원래 실패 trace로 일반 성공과 구별한다. Source 제거 목록 자체가 비어 있으면
기존 빈 목록 geometry를 따르며 임의 방향을 만들지 않는다.

명시적 두 건의 준비 명령:
```powershell
python scripts/run_full_semantic_benchmark.py --run --source-only-fallback 092-08 154-07
```
이미 준비된 fallback도 재호출 없이 검증·재사용한다. 전체 생성은 이 명령에 포함하지 않는다.
