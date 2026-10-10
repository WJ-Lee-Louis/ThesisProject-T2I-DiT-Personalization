# 명시적 identity 기반 semantic pipeline 검토 및 구현

## 1. 저장소와 기존 inference 분석

2026-10-10 검토. 기존 `datasets/prompt_component_analysis_v1/`와 `scripts/`는 작업 시작 시 Git untracked 상태였다. benchmark JSONL annotation은 변환하지 않았다. 문서 정리 과정에서 이전 identity-inference prompt/schema와 중복 prompt 문서는 삭제하고, 현재 프롬프트는 통합 공유 문서로 정리했다.

저장소의 주요 영역:

| 영역 | 역할 |
|---|---|
| `orthogonalization_instantx_ip_adapter/` | model loading, adapter overlay, projection, conditioning, manifest/config |
| `modal_inference.py` | 단일 이미지 CLI 및 GPU remote 함수 |
| `batch_client.py`, `batch_modal.py` | 동물 reference batch, 업로드, 재개 및 실행 |
| `eval_batch_client.py`, `eval_batch_modal.py` | benchmark evaluation generation |
| `metric_client.py`, `metric_modal.py`, `datasets/XVerseBench_0907/metrics/` | 정량 평가 |
| `datasets/` | benchmark metadata, source captions, prompt draft, 결과 annotation |
| `tests/` | projection, adapter, gate, batch, metric 회귀 테스트 |
| `docs/`, root 연구 memo/guide | 방법론 및 실행 문서 |
| `outputs/`, `run_history/` | 실험 결과 및 실행 기록 |
| `scripts/build_prompt_component_dataset.py` | 기존 annotation 생성기; runtime LLM 모듈과는 별개 |

기존 처리 위치:

| 처리 | 파일 / 함수 |
|---|---|
| 단일 inference | `modal_inference.main`, `generate_experiment` → `inference.generate_png_bytes` |
| 모델 구성 | `model_loader.load_ip_adapter_model` |
| source 입력 | `source_captions.SourceCaptionRecord` / caption manifest loader; evaluation은 `evaluation_manifest.EvaluationSourceCaption` |
| target 구조 | `prompt_catalog`, `batch_conditioning._validate_job_prompt`, `evaluation_conditioning._target_request`; catalog/manifest에 명시된 action/background/attribute 사용 |
| O/A/B prompt | `orthogonalization.PromptSet`; `SourceCaptionRecord.object_only/object_behavior/object_background/object_behavior_background` |
| 단일 request | `inference.build_orthogonalization_request` |
| batch source/target 합치기 | `batch_conditioning._build_projection_request`, `_build_source_aware_projection_request`, `_build_source_target_projection_request` |
| evaluation source/target 합치기 | `evaluation_conditioning._target_request`, `_source_request`, `_joint_request` |
| pooled text embedding | `adapter._get_siglip_text_encoder`, `_encode_siglip_text_pool` (동일 SigLIP model, cached FP32 `pooler_output`) |
| contrast 생성 | `adapter._build_text_directions`, `_build_direction_set`; `orthogonalization.build_local_directions`, `build_factorial_directions` |
| class 제거 ablation | `adapter._build_class_direction*`, `_build_hierarchical_class_directions`; 새 identity-preserving 경로와는 별개 |
| SVD / projection | `orthogonalization.estimate_orthonormal_basis`, `project_pooled_embedding`, `orthogonalize_pooled_embedding` |
| visual 적용 | `adapter.get_image_embeds`: visual `pooler_output` 편집 후 `image_proj_model` |
| single-stream gate | `attention_processor.build_single_stream_ip_row_gate`, `mask_single_stream_ip_residual`, `IPAFluxAttnProcessor2_0.__call__`; `adapter.set_mask_single_text_rows`, `_capture_text_seq_len` |

```text
reference image → SigLIP visual pool ───────────────────────────┐
manual factorized source caption → source PromptSet ─┐         │
structured target catalog → target PromptSet ────────┤         │
                                                    v         v
                      SigLIP text pool → contrast rows → one SVD
                                                         → visual projection
                                                         → image_proj_model
raw target prompt → upstream FLUX text encoding ──────────→ InstantX generation
                                                         + optional single-stream gate
```

연구 설명과 구현 사이의 중요한 차이:

- 기존 제거 방향은 contextualized token embedding의 span 평균이 아니라 **문맥 prompt의 normalized pooled embedding 간 contrast**다. 새 구현도 이 방식을 유지한다.
- O/A/B 고정은 prompt 구성·contrast 단계에 있다. SVD/projection은 원래부터 `[direction_count, embedding_dim]`을 지원한다.
- source+target은 visual embedding을 두 번 독립 편집하는 방식이 아니라, 방향을 합쳐 **한 번의 SVD**를 사용하는 union subspace다.
- source-only, target-only, ST matched/all 옵션은 이미 존재한다. 새 경로는 predefined category 매칭 없이 선택된 모든 component를 사용한다.
- 읽은 실제 prompt 초안은 hint list 없이 prompt에서 identity를 추론한다. 새 버전은 이 추론을 완전히 제거한다.
- class-removal 실험은 identity/concept 자체를 감쇠할 수 있다. 새 `components` 모드에서는 이 설정과 혼합을 금지한다.

## 2. LLM 설계

정확한 system prompt는 아래 파일에 있으며, 실행 시 동일한 내용의 `semantic_components.py` 상수를 사용한다.

두 모듈의 prompt 전문과 입출력 구조·예시는 하나의 공유 문서로 제공한다:

- [LLM 모듈 프롬프트 및 입출력](prompts/llm_modules_share.md)

공유 문서의 base system prompt가 runtime 상수와 일치함을 확인했다. Markdown만 편집하면 runtime prompt가 바뀌지는 않는다.

- Module A runtime: `semantic_components.py::CAPTION_SYSTEM_PROMPT`, `CAPTION_SCHEMA`
- Module B runtime: `semantic_components.py::COMPONENT_SYSTEM_PROMPT`, `COMPONENT_SCHEMA`

| 항목 | Module A | Module B |
|---|---|---|
| 입력 | reference image + `identity_term` | `text` + `identity_term` (source/target 별도 요청) |
| 출력 | `{"source_caption":"..."}` | `{"identity_term":"...","preserved_spans":["..."],"neutral_spans":[],"components":[{"spans":["..."]}]}` |
| 제한 | identity verbatim 포함, concise visible description, 추론한 identity로 교체 금지 | 원문 spans, source order, no overlap/duplicates, identity 보호, lexical coverage, semantic group 유지, predefined category 없음 |
| style identity | 지정 style phrase를 caption에 그대로 사용 | subject도 non-identity component가 될 수 있음 |
| 빈 결과 | 빈 caption 거절 | identity-only text는 `[]` 허용 |
| 사용자 source caption | 제공하면 A 생략; 먼저 identity 검증 | 제공한 caption도 동일하게 분석 |

`identity_spans`는 case-sensitive verbatim matching과 단어 경계를 사용한다. `a doghouse`는 `a dog`를 충족하지 않으며, `A dog`도 `a dog`를 충족하지 않는다. synonyms나 capitalization을 자동 교정하지 않는다. 모든 identity 등장 위치를 보호한다. 여러 등장 위치가 있는 component는 앞에서부터 가능한 disjoint span에 매핑한다. 동일한 component 문자열의 중복은 거절하므로 반복 내용은 더 큰 unique phrase로 반환해야 한다.

검증과 retry:

1. target identity, 사용자 source caption, origin/strategy를 LLM 호출 전에 검사한다.
2. strict JSON object만 parse한다. Markdown fence, 잘못된 JSON, duplicate JSON key, extra key, 잘못된 필드 타입, identity echo 변경을 거절한다.
3. component 공백/빈 문자열, identity 포함/교차, 원문에 없는 phrase, duplicate, 순서/overlap 오류를 거절한다.
4. 출력 검증 오류에만 최대 2회 retry한다(총 3회). 원래 입력과 validation feedback을 다시 전달하며, 임의 fallback이나 phrase 자동수정을 하지 않는다.
5. 인증/네트워크/API capability 오류, refusal, incomplete 응답은 즉시 중단한다. HTTP request에는 timeout이 있다.

2026-10-10 style 검토 반영: [DreamBench++ style prompt 180개 검토](dreambench_plus_style_prompt_review.md)를 기준으로 Module B의 granularity 규칙을 보완했다. 자세/action과 독립적인 location은 분리하되 `playing the guitar` 같은 predicate는 유지한다. `pixel art cloud`에서 cloud까지 보호하지 않으며, `on a`와 `cloud`를 하나의 component-group으로 묶을 수 있다. LLM 출력 schema와 parser도 이 형식을 지원한다.

새 형식의 `preserved_spans`, `neutral_spans`와 component-group spans는 원문상의 비중첩·identity 포함·순서·모든 identity 보호·lexical coverage를 hard validation한다. `in` 등 미배정 단어는 retry 대상이다. component 하나의 여러 spans는 항상 함께 제거되며, N은 span 수가 아닌 group 수다. 반복 문자열의 위치가 모호하면 더 큰 unique span을 요청한다. 기존 `components: string[]` 및 neutral_spans 도입 전 group 형식의 cache와 mock/provider 결과도 계속 읽으며 당시 정책을 유지한다. 새 규칙을 적용하려면 cached component 분석을 재생성해야 한다.

`neutral_spans`는 identity가 아닌 일반적인 문장 framing이다. `a photo of`, `a photograph of`, `an image of`, `a picture of` 같은 introductory wrapper는 모든 counterfactual에 유지하며 projection 방향을 생성하지 않는다. 구체적 style/medium과 physical photo까지 중립화하지 않는다. 이는 generic framing을 제외하는 연구상 정책이며 photo가 의미를 갖지 않는다는 주장이나 photorealism 보존 보장이 아니다.

`during sunset`, `at sunset`의 연결어는 보존 구간에 포함한다. qualifier가 사이에 끼면 연결어를 neutral로 배정하여 qualifier 제거 시에도 유지한다. 최신 neutral 형식에는 문장 시작의 알려진 generic wrapper와 명백한 직접 temporal connector의 제거를 거절하는 코드 guard도 있다. 모든 문맥의 문법적 attachment를 코드로 추론하는 것은 아니며, 보다 복잡한 관계는 prompt와 검토로 처리한다.

문법/semantic grouping이나 caption의 시각적 정확성은 JSON/span validation만으로 보장할 수 없다. 보존 구간을 최소한으로 잡았는지와 관련 표현의 의미적 연관은 LLM 판단 및 실험으로 확인해야 한다. style을 identity로 지정한다고 encoder 공간에서 identity 방향까지 수학적으로 보존된다는 보장도 없다. 이는 기존 contrast 방법의 한계이며 수학을 임의로 바꿔 해결하지 않았다.

Provider는 `StructuredLLMProvider.complete(...) -> str` Protocol로 교체한다. mock도 같은 interface를 사용한다. 기본 `ResponsesJSONProvider`는 HTTPS Responses endpoint, vision input, strict JSON schema를 사용하며 특정 모델을 기본값으로 고정하지 않는다. Luna를 사용하려면 실제 API model ID와 endpoint의 vision/structured-output 지원을 확인해 설정한다. 모델 가용성을 이번 로컬 작업에서 실호출로 확인하지 않았다. 서로 다른 caption/analysis 모델을 쓰려면 provider 구현에서 schema/역할에 따라 routing할 수 있다.

요청 형식 참고: [official structured output documentation](https://developers.openai.com/api/docs/guides/structured-outputs), [official vision documentation](https://developers.openai.com/api/docs/guides/images-vision).

## 3. 수학과 component 일반화

`P_S`는 원문에서 component subset S만 유지한 prompt다. 새 형식에서는 보존/중립 spans와 선택된 component-group의 모든 spans를 원래 위치에 유지하고 나머지 group을 제거한다. 공백/구두점 separator는 남으며, 삭제된 counterfactual의 whitespace만 정규화한다. 모든 group을 유지하면 원문을 그대로 반환한다. 기존 문자열 형식에서는 identity와 component 외의 원문 내용도 유지한다. phrase 순서를 identity-first로 재배치하지 않는다. `e(P)`는 기존처럼 normalized SigLIP pooled text embedding이다.

Local(default):

```text
d_i = normalize(e(P_all) - e(P_all_without_i))
```

Factorial(option):

```text
d_i = normalize(mean_{S ⊆ components_without_i}(e(P_{S ∪ {i}}) - e(P_S)))
```

평균하기 전에 individual contrast를 normalize하지 않는다. N=2이고 O/A/B 구조가 같으면 각각 기존 local/factorial과 정확히 같으며 이를 테스트했다. N=1도 같은 수식으로 지원한다. N=0은 empty basis가 된다.

Factorial은 unique subset 문맥 수가 `2^N`이다. `max_contexts=256`을 기본 계산량 budget으로 사용하고 초과 시 오류를 낸다. local에는 N 개수 제한을 두지 않는다. factorial 예산은 사용자가 명시적으로 늘릴 수 있다. interaction ablation은 기존 legacy factorial 경로에 남겼고 새 component 경로와 혼합하지 않는다.

Source/target 방향을 선택하여 row로 쌓고 기존 `orthogonalize_pooled_embedding`에 전달한다. SVD tolerance, FP32 계산, rank 진단, projection strength, norm restoration, dtype 변환, checkpoint 및 gate 계산을 변경하지 않았다.

```text
reference image + identity_term + target_prompt
 → target identity validation
 → caption LLM (사용자 source_caption 있으면 생략)
 → source / target analysis LLM
 → validated exact component spans
 → selected local/factorial contextual contrasts
 → SigLIP pooled text embeddings
 → source / target / source_target rows
 → 기존 one-SVD orthogonalization of visual pool
 → InstantX image_proj_model → inference + optional gate
```

source-only는 target identity validation 후 target analysis API를 생략한다. target-only는 caption/source analysis를 생략한다. 기본 source_target에서는 A+B+B 총 세 요청이며 사용자 caption 제공 시 두 요청이다.

## 4. 실행과 재사용

PowerShell에서 key는 환경변수나 기존 secret 관리 도구로 제공한다. 코드/출력에는 credential을 저장하지 않는다.

```powershell
# SEMANTIC_LLM_API_KEY는 현재 shell 환경에 설정
$env:SEMANTIC_LLM_MODEL = "<실제 API model ID>"
# 다른 Responses-compatible endpoint라면 SEMANTIC_LLM_ENDPOINT도 설정

python -m modal run modal_inference.py `
  --reference-image-path "C:/path/reference.png" `
  --identity-term "a dog" `
  --prompt "a retro-style digital drawing of a dog playing the guitar on a sofa" `
  --component-origin source_target `
  --component-strategy local `
  --orthogonalization-strength 1.0 `
  --mask-single-text-rows
```

`--source-caption "a dog sleeping on the snow"`를 추가하면 caption API를 생략한다. `--component-strategy factorial`로 기존 평균 방향의 일반화를 선택한다. `--component-origin source` / `target` / `source_target`으로 ablation한다. CLI의 기존 gate 기본값은 유지했으므로 gate를 원하면 flag를 준다.

LLM은 로컬에서 실행되며 Modal에는 검증된 분석 payload만 전달한다. GPU 모델은 remote payload 재검증 이후 로드한다. 생성 결과 옆 `.components.json`은 `--component-analysis-path`로 재사용할 수 있다(LLM API와 key 불필요). 재사용 시 payload target text와 현재 prompt가 정확히 일치해야 한다. origin/strategy 변경은 저장 JSON에서 명시적으로 수정하며 CLI cached 경로에서는 JSON 설정을 그대로 따른다. 다른 reference에 source 분석을 재사용하지 않도록 사용자가 reference와 함께 관리해야 한다; image hash binding은 이번 범위에 포함하지 않았다.

Python에서 legacy O/A/B 인자를 없이 호출:

```python
from pathlib import Path
from orthogonalization_instantx_ip_adapter.inference import generate_semantic_png_bytes
from orthogonalization_instantx_ip_adapter.llm_provider import ResponsesJSONProvider
from orthogonalization_instantx_ip_adapter.semantic_components import SemanticLLM

png, diagnostics = generate_semantic_png_bytes(
    identity_term="a dog",
    target_prompt="a dog playing the guitar on a sofa",
    reference_image_bytes=Path("reference.png").read_bytes(),
    llm=SemanticLLM(ResponsesJSONProvider.from_env()),
    origin="source_target", strategy="local", strength=1.0,
    mask_single_text_rows=True,
)
```

Python entry의 default strength=1.0, gate=True는 명시적 새 interface의 기본값이다. 기존 CLI/legacy 함수의 strength/gate 기본값은 변경하지 않았다. Provider mock을 주입하면 caption/component 분석을 API 없이 수행할 수 있다. PNG generation까지 mock하려면 `inference.generate_png_bytes`를 대체한다.

긴 자유문 target은 분석 가능하지만 SigLIP text encoder의 positional limit을 넘는 contrast는 `ValueError`로 중단한다. 긴 caption을 silently truncate하거나 독립 phrase embedding으로 자동 대체하면 방법론이 바뀌므로 그렇게 하지 않는다. 초장문을 그대로 inference에 쓰려면 향후 별도 embedding 정책 연구가 필요하다.

기존 animal/evaluation batch manifests와 resumable condition names는 재현성을 위해 유지했다. 새로운 자유문 경로는 단일 Modal CLI와 Python entry에 연결했다. 향후 batch 일반화는 새 caption/component manifest를 이용해 별도로 확장할 수 있다.

## 5. 변경 파일 및 검증

| 파일 | 추가 / 변경 함수 |
|---|---|
| `semantic_components.py` (신규) | prompts/schema, identity_spans, ComponentAnalysis, SemanticLLM, SemanticConditioning, prepare_semantic_conditioning |
| `llm_provider.py` (신규) | StructuredLLMProvider의 HTTP 구현 ResponsesJSONProvider, credential-safe error handling |
| `component_directions.py` (신규) | component_contrast_plan, build_component_directions |
| `adapter.py` | OrthogonalizationRequest의 components mode, _build_text_directions 분기, _encode_component_text_pool 길이 검사, direction count 진단 |
| `inference.py` | generate_png_bytes의 payload 검증/연결, generate_semantic_png_bytes 신규 entry |
| `modal_inference.py` | main의 로컬 LLM/cached 분석, generate_experiment payload 전달, 결과 재사용 파일 |
| `datasets/prompt_component_analysis_v1/` | 기존 benchmark JSONL annotation과 validation summary; 중복 Markdown 및 과거 prompt/schema 삭제 |
| `tests/test_semantic_pipeline.py` | 6가지 case, mock/retry, N=2 수학적 parity, component 수/ablation, CPU visual projection 및 HTTP request mock |

검증 명령:

```powershell
python -m compileall -q orthogonalization_instantx_ip_adapter modal_inference.py
python -m pytest -q
git diff --check -- .
```

최초 구현 검증에서는 실제 caption/component API 호출, 모델 weight 다운로드, GPU 이미지 생성 및 visual/quantitative 평가를 실행하지 않았다.

2026-10-10에는 DreamBench++ style reference 10개에 실제 caption/source/target API 분석을 적용하여 파이프라인을 점검했다. 20개 source/target 분석과 두 방식의 contrast 계획에서 원문 복원, preserved/neutral span 유지, component별 차감 및 조합 수를 확인했다. Local은 전체 문맥에서 해당 component만 제거한 차이이고, factorial은 다른 component의 모든 부분집합 문맥에서 얻은 차이의 평균이다. 정규화한 문장 embedding의 차이를 평균한 뒤 방향을 정규화하며, source/target 방향을 함께 기존 projection에 전달한다. 다만 의미 단위의 분해 일관성은 LLM에 의존하며, span 삭제로 생기는 불완전한 문장과 남는 구두점이 embedding에 영향을 줄 수 있다. 구조 검증 통과가 의미적 disentanglement를 보장하지는 않는다. 실제 이미지의 의도 반영 여부는 후속 평가로 남긴다. 예비 실행 결과 삭제는 사용자 요청으로 시도했으나 도구의 정책 차단으로 수행되지 않았다.
