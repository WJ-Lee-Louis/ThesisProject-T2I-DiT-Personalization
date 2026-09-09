# IP-Sim 문제 진단 및 해결 기록

## 결론

최초의 IP-Sim 전수 0점은 SAM2가 mask 생성에 실패해서 발생한 것이 아니다.
Florence2가 모든 reference 및 생성 이미지에서 bbox를 하나도 반환하지 않아,
파이프라인이 SAM2를 호출하기 전에 종료된 것이 직접적인 원인이었다.

Florence2가 bbox를 만들지 못한 근본 원인은 다음 두 구성 요소 사이의
weight-loading 계약 불일치였다.

1. XVerse가 사용하는 `microsoft/Florence-2-large` checkpoint는 언어모델의
   입력 token embedding과 출력 projection인 `lm_head`가 같은 weight를
   공유하는 tied-weight 구조를 전제로 한다.
2. XVerse에 복사되어 포함된 구버전 Florence2 Python 구현은 설정과
   `_tied_weights_keys`에는 weight 공유 관계를 선언했지만, 실제 모듈을 다시
   연결하는 `_tie_weights()` 구현이 빠져 있었다.
3. 그 결과 checkpoint에 독립 tensor로 저장되지 않은
   `language_model.lm_head.weight`가 shared embedding과 연결되지 않았고,
   loader가 새로 초기화한 별도 weight로 남았다.
4. Florence2 decoder의 hidden state가 이 잘못된 출력 head를 통과하면서
   정상적인 class/location token이 아니라 빈 token 또는 특수 token만
   생성했다.

즉, 단순히 “Microsoft weight가 잘못됐다”거나 “SAM2가 불안정하다”는 문제가
아니다. Microsoft checkpoint 자체와 XVerse가 vendoring한 Florence2 로더
구현이 서로 기대하는 tied-weight 처리가 일치하지 않은 호환성 문제다.

## 관련 구성 요소

| 구성 요소 | 역할 | 문제 발생 당시 상태 |
|---|---|---|
| `language_model.model.shared.weight` | 언어모델의 공통 token embedding | checkpoint에서 정상 로드됨 |
| `encoder.embed_tokens.weight` | encoder 입력 embedding | shared weight와 정상적으로 같은 storage 사용 |
| `decoder.embed_tokens.weight` | decoder 입력 embedding | shared weight와 정상적으로 같은 storage 사용 |
| `language_model.lm_head.weight` | decoder 출력을 vocabulary logit으로 변환 | shared weight와 연결되지 않고 별도 초기화됨 |

문제의 핵심은 마지막 행이다. `lm_head`는 bbox를 직접 회귀하는 layer는
아니지만, Florence2가 `<loc_...>` 형태의 위치 token을 생성하기 위해 반드시
통과하는 vocabulary 출력 projection이다. 따라서 이 weight가 잘못되면
processor가 해석할 location token 자체가 생성되지 않는다.

## 왜 checkpoint에서 missing-weight 경고가 나타났는가

실행 당시 다음과 같은 경고가 출력됐다.

```text
Some weights of Florence2ForConditionalGeneration were not initialized ...
['language_model.lm_head.weight']
```

이 경고는 checkpoint 파일이 손상됐다는 뜻이 아니다. Tied-weight 모델은 같은
값을 여러 이름으로 중복 저장하지 않고 대표 tensor 하나만 저장할 수 있다.
정상 구현에서는 모델을 구성한 뒤 `_tie_weights()`가 `lm_head.weight`를
`shared.weight`와 다시 연결하므로 독립적인 LM-head tensor가 없어도 된다.

Microsoft의 현재 Florence2 구현은
`Florence2LanguageForConditionalGeneration._tie_weights()`에서 다음 세 모듈을
공통 embedding에 연결한다.

```text
encoder.embed_tokens.weight -> model.shared.weight
decoder.embed_tokens.weight -> model.shared.weight
lm_head.weight              -> model.shared.weight
```

반면 이번 XVerse 고정 revision에 포함된 Florence2 클래스에는 이 실제 연결
동작이 빠져 있었다. `_tied_weights_keys`라는 선언만으로는 사용한
`transformers==4.49.0` 로딩 경로에서 LM head가 기대한 storage를 공유하지
못했다. 따라서 missing-weight 경고와 비어 있는 grounding 결과가 함께
나타났다.

이 조합은 XVerse GitHub issue #26에서 보고된 증상과 같다. 해당 issue도 모든
이미지에서 grounding 결과가 비었으며 동시에
`language_model.lm_head.weight` missing 경고를 기록하고 있다.

## 수정 전 실측

A10G에서 reference polar bear 한 장과 같은 subject의 생성 이미지 한 장을
XVerse의 실제 Florence2 호출 설정으로 검사했다.

```text
tie_word_embeddings = true
encoder_tied_to_shared = true
decoder_tied_to_shared = true
lm_head_tied_to_shared = false
```

두 이미지 모두 raw generation이 다음과 같았다.

```text
</s><s><pad><pad><pad></s>
```

즉, 정상적인 subject label과 `<loc_...>` token이 전혀 생성되지 않았다.
`post_process_generation()` 결과의 bbox 수도 0개였다.

이때 실제 평가 경로는 다음과 같다.

```text
생성 이미지
  -> Florence2 OPEN_VOCABULARY_DETECTION
  -> bbox 0개
  -> XVerse ObjectDetector가 빈 instance 목록 반환
  -> SAM2 호출 안 됨
  -> DINO 호출 안 됨
  -> 기존 wrapper가 IP-Sim 0.0 기록
```

따라서 당시 저장된 0.0은 낮은 identity similarity가 아니라 측정 실패값이다.

## Reference만 진행된 것처럼 보였던 이유

최초 대응에서는 고정된 29개 reference에 사람이 확인한 bbox를 제공해
Florence2 단계를 우회하고 그 bbox를 SAM2 prompt로 사용했다. 이 때문에
reference에서는 non-empty mask와 foreground crop을 만들 수 있었다.

그러나 생성 이미지는 1,236장으로 수동 bbox가 없었다. 생성 이미지에서는
Florence2가 bbox 0개를 반환한 즉시 종료됐으므로 SAM2 mask가 만들어지지
않았다. 이것이 reference와 생성 이미지의 결과가 비대칭이었던 이유다.

## 적용한 수정

현재 evaluator는 다음 두 조치를 수행한다.

1. `microsoft/Florence-2-large`를 revision
   `21a599d414c4d928c9032694c424fb94458e3594`로 고정해 모델 파일과 processor의
   조합이 실행마다 바뀌지 않게 한다.
2. XVerse `ObjectDetector`가 모델을 로드한 직후, 첫 inference 전에 아래
   weight들이 실제로 같은 storage를 사용하도록 명시적으로 연결한다.

```python
encoder.weight = shared.weight
decoder.weight = shared.weight
lm_head.weight = shared.weight
```

이 수정은 새로운 weight를 만들거나 Florence2를 재학습하는 작업이 아니다.
Checkpoint에서 정상 로드된 shared embedding을 Microsoft 구현이 의도한
tied-weight 구조대로 다시 연결하는 최소 호환성 수정이다. XVerse의
`OPEN_VOCABULARY_DETECTION -> SAM2 -> DINO` 평가 알고리즘, query, SAM2,
DINO model 및 점수 계산 방식은 변경하지 않았다.

현재 수정은 모델 객체가 생성된 직후 적용되므로, underlying loader가
`lm_head.weight`를 처음 확인하는 시점에는 missing-weight 경고가 먼저 보일 수
있다. 중요한 검증 기준은 경고 문자열의 유무가 아니라 inference 전에 기록한
compatibility report에서 `lm_head_tied_to_shared=true`인지 여부다. Evaluator는
수정 전후의 tensor storage pointer 관계를 점수 JSON에 함께 기록한다.

## 수정 후 실측

같은 두 이미지로 다시 검사한 결과는 다음과 같다.

```text
lm_head_tied_to_shared = true
reference Florence2 bbox 수 = 1
reference SAM2 score = 0.98798
generated Florence2 bbox 수 = 1
generated SAM2 score = 0.99299
```

이후 전체 실행 전에 다음 gate를 통과했다.

- Reference 29장: Florence2 검출 및 SAM2 mask 생성 29/29 성공
- 여섯 조건의 생성 이미지 smoke test: Florence2 검출 6/6 성공
- Reference SAM2 mask score 범위: 약 0.950~0.994

최종 전수 재평가 결과는 다음과 같다.

| 벤치마크 | 이미지 수 | Florence2 검출 성공 | 검출률 | 평균 IP-Sim |
|---|---:|---:|---:|---:|
| subset | 696 | 696 | 100% | 75.8187 |
| custom | 540 | 540 | 100% | 74.7478 |

모든 생성 이미지가 Florence2 bbox 검출 후 SAM2와 DINO 단계까지 진행됐으므로,
현재 IP-Sim은 유효한 identity metric으로 사용할 수 있다.

## 수동 bbox 제거 상태

Florence2 자체 문제가 해결됐으므로 이전의 reference-only 수동 bbox 우회는
평가에서 제거했다.

- 수동 bbox override JSON 삭제
- Reference 준비 코드의 수동 bbox 입력 및 fallback 함수 삭제
- `segmentation_manifest.json` schema 4 사용
- Manifest 최상위에 `fallback_method: null` 기록
- 29개 record 모두
  `segmentation_method: florence2_open_vocabulary+sam2` 사용
- 과거 수동 bbox로 만든 로컬 artifact 디렉터리 삭제
- Validator가 수동 bbox fallback이 표시된 manifest를 거부하도록 변경

Reference에서 exact query가 실패할 때 `polar bear -> bear`처럼 같은 개체를
가리키는 head-noun query를 시도할 수는 있다. 이것은 사람이 bbox를 지정하는
fallback이 아니라 Florence2 open-vocabulary detection 내부의 query 전략이며,
시도한 query 목록은 manifest에 모두 기록된다.

## MOSAIC 및 관련 연구와의 관계

MOSAIC은 XVerseBench에서 DPG, ID-Sim, IP-Sim, AES를 보고하지만, 현재 공개된
공식 repository에는 별도의 XVerseBench metric 구현이 포함되어 있지 않다.
관련 ByteDance 연구인 UMO의 공개 평가 script도 별도 IP-Sim을 만들지 않고
XVerse의 `eval.tools.idip_sam-dino_score`를 직접 호출한다.

따라서 이번 평가는 다른 metric으로 대체하지 않고 XVerse의 공식
Florence2 -> SAM2 -> DINO 경로를 유지했다. 변경 범위는 위에서 설명한
Florence2 checkpoint-loading 호환성 수정으로 제한했다.

## 재발 방지 장치

- Florence2 model repository와 snapshot revision 고정
- 수정 전후 encoder/decoder/LM-head storage 공유 상태 기록
- 점수 JSON에 이미지 수, 검출 성공 수, 검출률 저장
- 전 이미지 bbox 0개인 IP-Sim 실행은 완료로 인정하지 않고 오류 처리
- Reference artifact validator가 schema 4, Florence-only segmentation,
  repaired LM-head 상태 및 `fallback_method: null`을 확인
- 로컬 집계기가 과거의 invalid all-zero IP 파일을 유효 점수로 포함하지 않음

## 참고 자료

- XVerse issue #26: <https://github.com/bytedance/XVerse/issues/26>
- 사용한 XVerse 고정 revision의 평가 pipeline: <https://github.com/bytedance/XVerse/blob/65e2581aac0faffe4607881a10cc486186f6499d/eval/grounded_sam/grounded_sam2_florence2_autolabel_pipeline.py>
- Microsoft Florence2 현재 구현: <https://huggingface.co/microsoft/Florence-2-large/blob/main/modeling_florence2.py>
- MOSAIC 논문: <https://arxiv.org/abs/2509.01977>
- MOSAIC 공식 repository: <https://github.com/bytedance-fanqie-ai/MOSAIC>
- UMO XVerseBench 평가 script: <https://github.com/bytedance/UMO/blob/main/scripts/eval_xversebench.sh>
