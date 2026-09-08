# XVerseBench 0907 평가 설정

## 1. 목적

본 디렉터리는 두 종류의 평가를 분리하여 관리한다.

1. `subset_bench`: XVerseBench 공식 single-animal 29개 레코드의 재현 평가
2. `custom_bench`: 같은 reference pool의 일부를 이용해 action과 background controllability를 통제 비교하는 자체 평가

두 결과는 합산하지 않는다. `subset_bench`는 공식 XVerseBench single-animal subset 결과로 보고하고, `custom_bench`는 `XVerse-reference Controlled Animal Benchmark`처럼 별도 명칭으로 보고한다.

기준 upstream은 XVerse 공식 저장소 commit `65e2581aac0faffe4607881a10cc486186f6499d`이다.

- 공식 저장소: <https://github.com/bytedance/XVerse>
- 공식 benchmark 설명: <https://bytedance.github.io/XVerse/>

## 2. Reference 검증 결과

### subset_bench

- 이미지 수: 29장
- 구성: XVerseBench single-subject animal 레코드 index 0--28에 대응하는 reference 전체
- 앞서 DreamBench++에서 공식 filename mapping에 따라 재구성한 prepared source와 SHA-256이 모두 일치함
- 누락, 추가, 이름 충돌 없음

### custom_bench

- 이미지 수: 15장
- 모든 이미지가 공식 29장 안에 포함됨
- 재구성한 prepared source와 SHA-256이 모두 일치함
- 선택된 파일:

```text
00_kitten.jpg
02_kitten.jpg
07_dog.jpg
08_raccoon.jpg
10_fox.jpg
12_dog.jpg
13_polar bear.jpg
20_hamster.jpg
25_Sphynx cat.jpg
27_wolf.jpg
33_panda.jpg
40_white tiger.jpg
42_puppy.jpg
43_French bulldog.jpg
44_cat.jpg
```

해시와 파일 크기는 `dataset_integrity.json`에 고정되어 있다. 다음 명령은 이미지 구성과 생성된 metadata를 다시 검증한다.

`reference_index_mapping.csv`에는 공식 record index와 실제 filename의 대응, 공식 prompt, custom 포함 여부가 정리되어 있다. 파일명 앞 숫자는 DreamBench++ asset 번호이고 공식 record index와는 다르므로 이 표를 기준으로 caption을 작성한다.

```cmd
cd /d C:\Users\82107\Downloads\AI_PROJECT\Orthogonalization-InstantX-IP-Adapter
python datasets\XVerseBench_0907\build_evaluation_assets.py --check-only
```

## 3. 공식 subset_bench

### 평가 레코드

각 reference에는 공식 target prompt 하나가 1:1로 연결된다. 공식 target prompt와 공식 DSG 질문은 수정하지 않았다.

- `subset_bench/benchmark_manifest.json`: 현재 프로젝트에서 사용할 manifest
- `subset_bench/XVerseBench_single_animal_29.json`: 공식 single JSON의 animal 29개 레코드
- `subset_bench/XVerseBench_single_animal_29_DSG.json`: 대응하는 공식 DSG 29개

방법당 4 seeds를 사용한다.

```text
42, 43, 44, 45
```

이는 upstream 생성 함수의 기본 seed 42와 `seed + i` 방식에 대응한다. 방법 간에는 동일한 reference, prompt, seed를 사용해야 한다.

### 공식 subset metric

1. DPG: 공식 mPLUG VQA와 공식 DSG dependency를 사용한다.
2. IP-Sim: 생성 이미지에서 Florence2+SAM2로 subject를 검출·분할한 뒤 DINO embedding cosine similarity를 계산한다.
3. AES: XVerse가 사용한 SigLIP 기반 aesthetic predictor v2.5를 사용한다.

공식 DSG의 DPG 전체 점수를 주 점수로 사용한다. O+A/O+B/O+A+B 분류는 본 연구의 사후 분석 label이며 공식 annotation이 아니다.

### O+attribute projection 처리

공식 prompt와 DSG는 그대로 유지하고, projection 계산에서만 5개의 `O+attribute` 레코드를 독립 factor로 분해한다. Core object prompt는 각 레코드의 `subject_caption`으로 고정하며 attribute direction은 다음과 같다.

$$
d_{\mathrm{attr}}
=
\frac{e(p_{O+\mathrm{attr}})-e(p_O)}
{\left\lVert e(p_{O+\mathrm{attr}})-e(p_O)\right\rVert_2}
$$

- `target_projection_ssg`: target attribute direction을 제거한다.
- `source_projection_ssg`: source caption에 대응 attribute가 없으므로 attribute-only 레코드에서는 projection을 적용하지 않는다.
- `st_matched_projection_ssg`: source 쪽 matched factor가 없으므로 attribute-only 레코드에서는 target projection과 정확히 동일하다.
- `st_all_projection_ssg`: target attribute와 source behavior/background direction을 하나의 union-SVD basis로 처리한다.

`target_factor_annotations.json`의 `attribute`는 projection 선택과 metadata 기록에 사용된다. 실제 target image 생성에는 항상 공식 prompt 원문을 사용한다.

## 4. Custom benchmark

### Prompt 구성

15 references 각각에 아래 여섯 prompt를 모두 적용하여 총 90개 reference--prompt pair를 만든다.

| Prompt ID | 구성 | Suffix |
|---|---|---|
| `oa_sleeping` | O+A | `sleeping` |
| `oa_playing_guitar` | O+A | `playing the guitar` |
| `ob_snow` | O+B | `in the snow` |
| `ob_sofa` | O+B | `on a sofa` |
| `oab_sleeping_snow` | O+A+B | `sleeping in the snow` |
| `oab_playing_guitar_sofa` | O+A+B | `playing the guitar on a sofa` |

O+A, O+B, O+A+B가 각각 두 개이므로 composition별 macro-average를 계산할 수 있다.

### Seed

Custom은 pair당 한 장을 생성한다. 각 pair에는 다음 deterministic seed를 부여했다.

$$
s_r = 42 + r
$$

여기서 $r$은 custom record index 0--89이다. 따라서 seed 범위는 42--131이다. 같은 pair를 생성하는 모든 condition은 반드시 같은 seed를 재사용한다. 이는 여러 seed 반복을 의미하지 않으며, 90개 pair가 동일한 초기 latent 하나만 공유하지 않도록 하는 비용 없는 통제 장치다.

### Custom DPG 질문 설계

질문은 공식 DSG의 `qid2tuple`, `qid2dependency`, `qid2question` 형식을 유지한다. 추가 분석을 위해 `qid2component`와 `qid2factor`를 덧붙였다. 공식 DPG scorer는 추가 field를 무시하므로 질문 응답과 dependency 계산 방식은 동일하다.

#### O+A: sleeping

```text
Q1 Is there a {subject}?                      dependency: 0
Q2 Is the {subject} sleeping?                dependency: Q1
```

#### O+A: playing the guitar

```text
Q1 Is there a {subject}?                     dependency: 0
Q2 Is there a guitar?                        dependency: 0
Q3 Is the {subject} playing the guitar?      dependency: Q1, Q2
```

#### O+B: in the snow

```text
Q1 Is there a {subject}?                     dependency: 0
Q2 Is there snow?                            dependency: 0
Q3 Is the {subject} in the snow?             dependency: Q1, Q2
```

#### O+B: on a sofa

```text
Q1 Is there a {subject}?                     dependency: 0
Q2 Is there a sofa?                          dependency: 0
Q3 Is the {subject} on the sofa?             dependency: Q1, Q2
```

#### O+A+B: sleeping in the snow

```text
Q1 subject 존재                              dependency: 0
Q2 snow 존재                                 dependency: 0
Q3 sleeping                                  dependency: Q1
Q4 subject가 snow 환경에 있음                dependency: Q1, Q2
```

#### O+A+B: playing the guitar on a sofa

```text
Q1 subject 존재                              dependency: 0
Q2 guitar 존재                               dependency: 0
Q3 sofa 존재                                 dependency: 0
Q4 subject가 guitar를 연주함                  dependency: Q1, Q2
Q5 subject가 sofa 위에 있음                   dependency: Q1, Q3
```

질문 원본은 `custom_bench/XVerseBench_custom_animal_15x6_DSG.json`에 있다.

### DPG 집계

Custom DPG는 다음 수준으로 각각 집계한다.

- 전체 DPG: 한 이미지의 모든 dependency-filtered proposition 평균
- Object: personalized subject 존재 여부
- Action: sleeping state 또는 guitar 존재와 playing relation
- Background: snow/sofa 존재와 spatial relation
- Joint: O+A+B에서 action 및 background proposition을 모두 만족한 비율

질문 수가 서로 다르므로 단순히 모든 질문을 하나로 합쳐 action/background 효과를 비교하지 않는다. 먼저 pair별 factor 점수를 계산하고, prompt별 평균을 낸 후 두 prompt를 동일 가중치로 평균하는 macro 방식으로 해석한다.

방법 $m$의 action 및 background 개선량은 같은 reference--prompt--seed에 대한 baseline 차이로 계산한다.

$$
\Delta_A(m)=S_A(m)-S_A(\mathrm{baseline})
$$

$$
\Delta_B(m)=S_B(m)-S_B(\mathrm{baseline})
$$

`aggregate_scores.py`는 condition, composition, prompt별 평균과 baseline 대비 paired delta를 생성한다.

## 5. IP-Sim 주의사항

현재 `subset_bench`와 `custom_bench`의 JPEG는 생성 입력용 원본 복제본이다. 공식 XVerse IP-Sim 코드는 reference가 이미 foreground segmentation된 상태라고 가정한다. 따라서 현재 JPEG를 곧바로 strict official IP-Sim reference로 사용하면 background가 similarity에 섞일 수 있다.

평가 전 다음 원칙을 적용한다.

1. 원본 reference는 덮어쓰지 않는다.
2. 공식 Florence2+SAM2로 29개 reference를 분할한다.
3. foreground crop은 `evaluation_references/segmented/crops`, 대응하는 binary mask는 `evaluation_references/segmented/masks`에 PNG로 저장한다.
4. subset과 custom 모두 같은 segmented reference를 공유한다.

`metrics/prepare_segmented_references.py`가 이를 수행한다. GPU와 공식 XVerse checkpoint 환경이 준비된 뒤 실행한다.

Mask는 공식 코드와 같은 방식으로 SAM2가 흰 배경에 합성한 instance에서 흰색이 아닌 픽셀을 foreground로 복원한다. 값은 foreground 255, background 0이다. `segmentation_manifest.json`에는 각 crop/mask 경로, bounding box와 SHA-256이 기록된다.

또한 XVerse 논문과 README는 DINOv2 similarity라고 표현하지만, 공개된 `eval/tools/dino.py` 기본 checkpoint는 `facebook/dino-vits16`이다. 공식 공개 코드와의 수치 재현을 우선할 경우 해당 checkpoint를 고정하고 논문에 정확한 구현을 밝혀야 한다.

## 6. AES 단위

공식 XVerse DPG와 IP-Sim은 2x2 grid를 네 이미지로 나누지만, 공개 AES script는 grid 전체를 한 이미지로 평가한다. 본 manifest 기반 `score_aes.py`는 생성 이미지 한 장씩 평가한다.

- 공식 공개 script와 byte-level 동작까지 맞춘 수치가 필요하면 `make_official_grids.py`로 grid를 만들고 upstream AES를 실행한다.
- 방법 간 공정한 per-image 분석에는 현재 `score_aes.py` 결과를 사용한다.

두 AES 방식을 같은 표에서 혼용하지 않는다.

## 7. Source caption

`source_captions.template.json`에 29개 official index, 파일명, SHA-256, subject caption을 기록했다. Caption을 다른 reference에 잘못 연결하는 일을 막기 위해 hash도 함께 검증한다. 사용자가 제공한 29개 caption은 `source_captions.json`에 반영되어 있다.

```json
{
  "official_index": 0,
  "reference_filename": "13_polar bear.jpg",
  "content_sha256": "...",
  "subject_caption": "a polar bear",
  "behavior": "standing",
  "background": "in the snow",
  "full_caption": "a polar bear standing in the snow"
}
```

이 caption은 source-aware projection 계산에만 사용한다. DPG 질문이나 공식 target prompt를 바꾸지 않는다. 다음 명령으로 index/파일/hash 누락을 검사하고 projection용 O, O+A, O+B, O+A+B prompt manifest를 다시 생성할 수 있다.

```cmd
python datasets\XVerseBench_0907\validate_source_captions.py datasets\XVerseBench_0907\source_captions.json --runtime-output datasets\XVerseBench_0907\source_captions.runtime.json
```

수동 caption을 사용하는 사실은 최종 논문에서 추가 annotation으로 명시해야 한다.

## 8. Generated-image manifest

평가 스크립트는 파일명을 추측하지 않고 generated-image manifest를 입력받는다. JSON schema는 `generated_images_manifest.schema.json`에 있다.

```json
{
  "schema_version": 1,
  "benchmark_id": "xverse_ref_custom_animal_15x6",
  "images": [
    {
      "record_id": "custom_000",
      "condition": "baseline",
      "seed": 42,
      "image_path": "outputs/custom/baseline/custom_000.png"
    }
  ]
}
```

동일한 `(record_id, condition, seed)` 조합은 하나만 존재해야 한다. 경로는 manifest 파일 기준 상대 경로 또는 절대 경로를 사용할 수 있다.

평가 전에 condition별 누락과 seed 불일치를 검사한다.

```cmd
python datasets\XVerseBench_0907\metrics\validate_generated_manifest.py --benchmark-root datasets\XVerseBench_0907 --benchmark custom --generated-manifest generated_custom.json
```

## 9. Metric 실행 구조

GPU 환경은 현재 InstantX inference 환경과 분리하는 것을 권장한다. XVerse의 Florence2, SAM2, ModelScope mPLUG 및 aesthetic predictor 의존성이 현재 diffusers 환경과 충돌할 수 있기 때문이다.

준비된 runner는 다음과 같다.

```text
metrics/score_dpg.py
metrics/score_ip_similarity.py
metrics/score_aes.py
metrics/aggregate_scores.py
metrics/make_official_grids.py
metrics/prepare_segmented_references.py
metrics/validate_generated_manifest.py
```

GPU checkpoint 설치와 Modal 배포는 이미지 생성 결과가 준비된 후 별도 작업으로 진행한다. 현재 runner는 official XVerse checkout 경로를 `--xverse-root`로 받아 upstream model wrapper를 그대로 재사용한다.

예시 구조는 다음과 같다.

```cmd
python datasets\XVerseBench_0907\metrics\score_dpg.py --benchmark-root datasets\XVerseBench_0907 --benchmark custom --generated-manifest generated_custom.json --xverse-root C:\path\to\XVerse --output metric_outputs\custom_dpg.json
```

IP-Sim에는 segmented reference를 명시한다.

```cmd
python datasets\XVerseBench_0907\metrics\score_ip_similarity.py --benchmark-root datasets\XVerseBench_0907 --benchmark custom --generated-manifest generated_custom.json --xverse-root C:\path\to\XVerse --segmented-reference-root datasets\XVerseBench_0907\evaluation_references\segmented --output metric_outputs\custom_ip.json
```

위 명령은 생성 이미지에서 검출한 최대 3개 candidate 각각의 foreground crop과 binary mask도 기본적으로 `metric_outputs/custom_ip_artifacts/generated/<condition>/<record_id>/seed_<seed>`에 저장한다. 별도 위치가 필요하면 `--artifact-root <path>`를 지정한다. 점수 JSON에는 모든 candidate의 경로·점수와 실제 IP-Sim에 선택된 최고점 candidate index가 함께 기록된다.

집계 예시는 다음과 같다.

```cmd
python datasets\XVerseBench_0907\metrics\aggregate_scores.py --benchmark-root datasets\XVerseBench_0907 --benchmark custom --dpg metric_outputs\custom_dpg.json --ip metric_outputs\custom_ip.json --aes metric_outputs\custom_aes.json --baseline-condition baseline --output-dir metric_outputs\custom_summary
```

## 10. 예상 평가 산출물

`aggregate_scores.py`는 다음 파일을 만든다.

- `per_image_scores.csv`: 모든 pair와 metric 원자료
- `summary_by_condition.csv`: 방법별 전체 평균
- `summary_by_composition.csv`: O+A/O+B/O+A+B별 평균
- `summary_by_prompt.csv`: 여섯 prompt별 평균
- `paired_delta_vs_baseline.csv`: 동일 pair 기준 baseline 대비 변화량
- `summary.json`: 위 결과의 기계 판독용 통합본

주 결과에서는 DPG 향상만 제시하지 않고 IP-Sim과 AES를 함께 확인한다. Text controllability가 상승했더라도 IP-Sim이 크게 하락하면 identity를 희생한 결과일 수 있기 때문이다.

## 11. 이미지 수와 예산

6개 condition을 모두 평가한다고 가정하면 공식 subset은 다음과 같다.

$$
29 \times 4 \times 6 = 696
$$

Custom은 15 references, 6 prompts, pair별 1 seed이므로 다음과 같다.

$$
15 \times 6 \times 6 = 540
$$

합계는 1,236장이다. 기존 실측치인 약 33달러/2,000장을 단순 적용하면 이미지 생성비는 약 20.4달러다. DPG, IP-Sim, AES는 별도 GPU 평가비가 발생하므로 총액은 모델 다운로드 cache 상태와 처리 속도에 따라 달라진다. 동일 모델을 매 condition마다 다시 로드하지 말고, metric별로 모델을 한 번 로드한 뒤 모든 condition 이미지를 연속 평가해야 한다.

## 12. 현재 완료 범위와 다음 입력

현재 완료된 항목:

- 29/15 reference 구성 및 SHA-256 검증
- 공식 29개 prompt/DSG 동결
- custom 90개 prompt record 생성
- custom dependency-aware DPG 질문 300개 생성
- composition, factor, component 분석 label 생성
- pair별 deterministic seed 동결
- generated-image manifest schema와 metric runner 구성
- source caption 29개 입력 및 filename/index/hash 검증
- source-aware projection용 O, O+A, O+B, O+A+B runtime prompt 구성

다음 단계에서는 `source_captions.runtime.json`을 새로운 benchmark generation manifest 및 source-aware projection loader에 연결한다.
