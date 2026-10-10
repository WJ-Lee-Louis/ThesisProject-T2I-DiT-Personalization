# DreamBench++ style reference 180 prompt 실례 검토

검토일: 2026-10-10. API 호출 및 inference 없이 로컬 원문 20개 caption 파일과 대응 reference 이미지 20개를 확인했다. 아래 분류와 보존 후보는 설계 검토용 수작업 annotation이며 공식 정답이나 실제 LLM 분석 결과가 아니다. 검토 시에는 runtime 코드를 변경하지 않았으며, 후속 사용자 요청에 따라 아래 설계를 Module B prompt/schema와 span-group parsing에 반영했다. Projection 계산식은 유지했다. 현재 prompt 전문은 [공유 문서](prompts/llm_modules_share.md)에 있다.

원문 위치: `datasets/XVerseBench/dreambench_plus_raw/extracted/data/captions/style/00.txt`–`19.txt`. 각 파일은 첫 줄 label + target prompt 9개로 구성된다. 이미지 위치: 같은 data 디렉터리의 `images/style/00.jpg`–`19.jpg`.

## 확인 결과

- 20개 reference × 9개 = 180개 prompt. 서로 다른 label은 18종이다. `pixel art`(01, 12), `cosmic nebula`(13, 17)가 중복된다.
- 첫 줄 label을 명시적 identity_term으로 입력한다고 가정하면 180개 모두 현재의 case-sensitive lexical-boundary exact-match 검사를 통과한다. 모델이 identity를 추론하도록 바꿀 근거는 없다.
- 문장 구조를 D/M/C 세 가지로 수작업 분류했다. D는 명시적인 표현 방식 또는 concept 제시 구간, M은 명사구 내부의 modifier, C는 장면/사건 내용으로 포함된 concept이다. 이 분류는 이번 검토용이며 LLM에 predefined category로 요구할 계획이 아니다.
- style directory에는 회화 기법뿐 아니라 `sunset`, `volcanic landscape`, `desert moonscape`, `cosmic nebula`, `serene desert`처럼 장면·환경 concept도 있다. 이들 reference의 실제 이미지도 대응하는 풍경/천체를 담고 있다. 이 차이는 이미지 확인에 따른 해석이며 공식 category를 재정의하지 않는다.
- 기존 `dreambench_plus_non_style_components.jsonl` 생성기는 style 디렉터리를 제외하므로 이 180개 검토를 대신할 수 없다.
- 후속 수정: 문장 시작의 generic `A photo of`, `A photograph of`, `An image of`, `A picture of` 후보는 38개다. 전체 표에 중립 후보 열을 추가했으며 `neutral_spans`로 배정하여 removal direction에서는 제외한다. 의미가 있는 다른 medium/style 표현까지 일괄 제외한 것은 아니다.

## 실례가 드러낸 설계 차이

1. `in pencil sketch style`, `rendered in pixel art`, `depicted as a monochromatic ink sketch`, `with vibrant cosmic nebula colors`: 전치사뿐 아니라 rendering verb, article, style/medium head, qualifier도 등장한다. `in + identity`만 특수 처리하는 규칙으로는 충분하지 않다.
2. 같은 label의 surface carrier가 달라진다. watercolor sketch의 9개 prompt에는 `depicted in watercolor sketch`, `in watercolor sketch style`, `in a watercolor sketch style` 등이 있다. 사용자의 identity_term을 항상 전체 carrier로 바꾸면 benchmark target 원문을 일괄 처리하기 어렵다.
3. `A tiny dragon perched on a pixel art cloud`: `pixel art cloud`를 모두 보존하면 cloud까지 고정한다. 반면 identity만 남겨 주변을 연속 span으로만 나누면 `perched on a`와 `cloud`가 갈라진다. 같은 의미 component가 identity를 사이에 두고 여러 span에 걸칠 수 있다.
4. `A cup of tea on a pastel tablecloth, inviting a moment of relaxation`: `pastel tablecloth` 전체 보존은 tablecloth 보존까지 포함한다. `pastel`만 보호하면서 `on a ... tablecloth`를 하나의 component로 묶는 편이 제거 대상의 의미를 더 잘 유지한다.
5. `A photo of a serene beach during sunset, capturing the golden horizon`: sunset은 rendering suffix가 아니라 scene condition이다. `during sunset`이라는 최소 carrier를 선택할 수 있지만 `golden horizon` 역시 보존 concept와 의미적으로 연결되어 있다. exact span partition만으로 이 의미적 연관을 해결했다고 볼 수 없다.
6. `A mythical forest with trees that glow in the Impressionist style`: style clause가 문장 전체보다 tree/glow에 연결되어 읽힐 수 있다. 문장을 항상 global style suffix로 정규화하면 원문의 scope가 바뀔 수 있다.
7. `An astronaut floating in space surrounded by nebula-like colors, in a cosmic nebula style`: exact identity 바깥에도 concept와 관련된 표현이 있다. exact identity 보호와 의미적으로 유사한 표현 전체 보호는 다른 정책이다. 후자를 자동 적용하면 사용자가 요청한 보존 범위를 넓힌다.

## 권장 설계

identity를 명사 위치로 강제하거나 target 원문을 rewrite하지 않는다. identity_term은 사용자 지정 concept로 유지한다. 별도로 원문상의 보존 span과 제거하지 않는 generic framing인 neutral_spans를 명시하고, component들과의 비중첩 및 lexical coverage를 검증한다. 공백/구두점은 원문 separator로 보관한다.

사용자가 지적한 02-01의 `sunset`만 표시한 후보는 당시 core concept 표시였으나, 실제 removal 설계로 읽으면 `during`이 다른 component에 흡수될 수 있어 불충분했다. 현재 표에서는 `during sunset`으로 수정했다. `at sunset`, `of sunset` 등 명백한 인접 carrier도 수정했다. runtime prompt에 이미 있던 `during sunset` 예시를 일반 규칙과 최신 형식의 validation guard로 보강했다. 단, modifier로 사용된 경우 앞의 전치사가 다른 명사에 연결될 수 있으므로 무조건 인접 단어를 보존하지 않는다.

02-01의 출력 예시:

```json
{
  "identity_term": "sunset",
  "preserved_spans": ["during sunset"],
  "neutral_spans": ["A photo of"],
  "components": [
    {"spans": ["a serene beach"]},
    {"spans": ["capturing the golden horizon"]}
  ]
}
```

모든 component를 제거해도 `A photo of during sunset`이 남는다. 이 fragment가 완전한 문법 문장이라는 보장은 없지만, generic wrapper와 identity 관계 표현은 contrast 양쪽에 유지된다. `golden horizon`을 제거하는 방향이 sunset 보존에 미치는 의미적 영향은 별도 실험 대상이다.

**단순한 경우:** 의미가 없는 연결어와 일반적인 rendering carrier를 포함한 보존 span + 연속 component string을 사용한다. 예: `A cat napping by a sunny window, in watercolor sketch style` → identity `watercolor sketch`, 보존 `in watercolor sketch style`, component 후보 `A cat`, `napping`, `by a sunny window`.

**명사구 내부 삽입 사례:** 보존 span 확장만으로 해결하지 않는다. 다른 내용 concept(cloud, tablecloth, planet, forest)을 보존 영역에 넣지 않는다. 하나의 component가 여러 원문 span을 가질 수 있도록 component-group 표현을 고려한다. 예:

```json
{
  "identity_term": "pixel art",
  "preserved_spans": ["pixel art"],
  "neutral_spans": [],
  "components": [
    {"spans": ["A tiny dragon"]},
    {"spans": ["perched"]},
    {"spans": ["on a", "cloud"]}
  ]
}
```

각 spans 값은 원문 verbatim substring이다. 위치 component 제거 시 두 구간을 함께 제거하며 자세는 별도로 제거한다. 재구성은 array 순서대로 concatenate하는 방식이 아니라 원문상의 위치를 사용한다. 모든 component를 유지하면 원문을 정확히 복원하고, 모두 제거하면 보존/중립 span과 separator만 남는다. component의 의미 단위와 연속 문자열을 구분하는 설계다. 내부적으로 start/end 위치를 보관하고, 같은 문자열이 여러 위치에 매칭되어 모호하면 더 큰 unique span을 요청하거나 validation error를 반환한다. API schema에 offsets는 요구하지 않는다.

수학적 direction 정의는 유지한다. local은 component-group 하나를 제거한 prompt와 원문의 embedding 차이, factorial은 component-group의 포함/제외 조합으로 계산한다. N은 span 조각 수가 아닌 semantic component-group 수다. 단, group을 바꾸면 실제 embedding/direction 값은 달라지므로 기존 decomposition과 실험적으로 동등하다고 주장하지 않는다.

**coverage는 hard validation, 자연스러운 문장은 별도 문제다.** 보호/중립 span과 component spans가 원문의 lexical content를 빠짐없이 배정하도록 검사한다. `in` 등 미배정 단어를 조용히 보존하지 않는다. full prompt는 원문 그대로 복원하되 counterfactual fragment까지 항상 완전한 문법 문장이라고 보장하지 않는다. LLM로 counterfactual을 rewrite하면 원문 삭제 기반 direction이 달라지므로 이번 변경에 포함하지 않는다.

`dynamic`, `vibrant`, `soft`, `traditional`과 같은 qualifier를 concept realization에 포함할지 별도 component로 제거할지는 분해 granularity 정책이다. 아래 D 후보는 문맥상의 표현 방식 구간을 넓게 표시한 것이며, 모든 qualifier를 반드시 보존하자는 정답 annotation이 아니다. C 유형의 environmental concept와 관련된 다른 표현을 얼마나 제거할지는 projection 품질 평가 대상으로 남는다.

## 구현 전 확인할 사례

00-01(후치 style), 16-08(관사 포함 carrier), 01-08 및 01-09(명사구 내부 pixel art), 18-06(tablecloth), 18-07/08(hyphenated pastel-colored), 02-01(sunset condition), 13-07(주어 위치 cosmic nebula), 17-07(의미 관련 표현), 11-08(좁은 style scope)를 mock output + span coverage/counterfactual 테스트에 포함한다. source caption에도 같은 규칙을 적용하되, 이 검토는 target 원문에 대한 것이므로 실제 captioning 결과의 안정성을 검증한 것은 아니다.

## 전체 180개 원문과 보존 후보

ID의 두 번째 숫자는 파일 내 prompt 순서(1–9)다. D 후보는 명시적 carrier 포함 표현이다. M/C 후보에서도 명백한 인접 carrier는 포함했으며, qualifier가 끼거나 다른 명사를 수식하는 경우 core concept를 표시했다. 이때 실제 출력에서는 주변 연결어를 neutral span으로 배정하거나 해당 의미 component에 결합하는 추가 판단이 필요하며 core 표기만으로 주변 표현의 제거를 허용한 것은 아니다. 보존 후보는 원문에 한 번 등장하는 실제 substring이며 label을 verbatim 포함한다. 중립 후보 열은 introductory generic wrapper만 표시하며 필요한 다른 carrier는 LLM 분석에서 추가될 수 있다.

유형별 개수: D 151개, M 14개, C 15개.

검증: 문서의 180개 행과 20개 원문 링크를 확인했다. 별도 API 없는 설계 fixture 4개(pixel art cloud, pastel tablecloth, pastel-colored macaron, watercolor sketch suffix)에서 span 비중첩, lexical coverage, 모든 group 유지 시 원문 그대로 복원, 모든 group 제거 시 보존 표현만 남는 조건을 확인했다. 이는 제안한 span-group 처리의 동작 확인이며 실제 LLM 분해 품질이나 projection 성능 검증은 아니다.

### 00: pencil sketch

원문: [00.txt](../datasets/XVerseBench/dreambench_plus_raw/extracted/data/captions/style/00.txt), SHA-256 `beba7456ee23c0beac92e7e72b5deb71f056d4788d61975b4e741de56b2de70e`

| ID | 유형 | Target prompt 원문 | 보존 후보 | 중립 후보 |
|---|---|---|---|---|
| 00-01 | D | A photo of a quiet forest path, in pencil sketch style | `in pencil sketch style` | `A photo of` |
| 00-02 | D | A photograph of an old wooden bridge over a stream, in pencil sketch style | `in pencil sketch style` | `A photograph of` |
| 00-03 | D | A painting of a bustling street market, captured in pencil sketch style | `captured in pencil sketch style` | - |
| 00-04 | D | A detailed drawing of an antique clock, in pencil sketch style | `in pencil sketch style` | - |
| 00-05 | D | A pair of worn leather boots, rendered in pencil sketch style | `rendered in pencil sketch style` | - |
| 00-06 | D | A serene landscape with rolling hills and a lone tree, in pencil sketch style | `in pencil sketch style` | - |
| 00-07 | D | An expressive portrait of an elderly man with deep wrinkles, in pencil sketch style | `in pencil sketch style` | - |
| 00-08 | D | A mythical unicorn resting under an ancient oak tree, in pencil sketch style | `in pencil sketch style` | - |
| 00-09 | D | A scene of tiny humans living inside a giant's watch, in pencil sketch style | `in pencil sketch style` | - |

### 01: pixel art

원문: [01.txt](../datasets/XVerseBench/dreambench_plus_raw/extracted/data/captions/style/01.txt), SHA-256 `ec28bc0775f4bd92fa8fa53d01a0b16ea5626ba493b44d15092fb8fabae58e82`

| ID | 유형 | Target prompt 원문 | 보존 후보 | 중립 후보 |
|---|---|---|---|---|
| 01-01 | D | A photo of a bustling cityscape, rendered in pixel art | `rendered in pixel art` | `A photo of` |
| 01-02 | D | A photograph of a quaint cottage surrounded by flowers, in pixel art style | `in pixel art style` | `A photograph of` |
| 01-03 | D | A painting of a serene lake at dusk, depicted in pixel art | `depicted in pixel art` | - |
| 01-04 | D | A simple street scene with pedestrians, captured in pixel art | `captured in pixel art` | - |
| 01-05 | D | A vintage car parked on a cobblestone street, in pixel art style | `in pixel art style` | - |
| 01-06 | D | A collection of arcade video game machines, in pixel art style | `in pixel art style` | - |
| 01-07 | D | A playful puppy chasing a ball, illustrated in pixel art | `illustrated in pixel art` | - |
| 01-08 | M | A tiny dragon perched on a pixel art cloud | `pixel art` | - |
| 01-09 | M | A magical portal opening in a pixel art enchanted forest | `pixel art` | - |

### 02: sunset

원문: [02.txt](../datasets/XVerseBench/dreambench_plus_raw/extracted/data/captions/style/02.txt), SHA-256 `9f669a8ac82ec93e2784f05b3303ff709af4af96a936504c31a386e726d0da46`

| ID | 유형 | Target prompt 원문 | 보존 후보 | 중립 후보 |
|---|---|---|---|---|
| 02-01 | C | A photo of a serene beach during sunset, capturing the golden horizon | `during sunset` | `A photo of` |
| 02-02 | C | A photograph of a bustling city skyline illuminated by the fading sunset glow | `sunset` | `A photograph of` |
| 02-03 | C | A painting of a tranquil lake reflecting the vibrant colors of sunset | `of sunset` | - |
| 02-04 | C | A quiet countryside road basking in the soft light of sunset | `of sunset` | - |
| 02-05 | C | A couple walking hand-in-hand along a shoreline at sunset | `at sunset` | - |
| 02-06 | C | A lone tree silhouetted against a fiery sunset sky | `sunset` | - |
| 02-07 | M | A sunset view from a mountain peak, bathing the landscape in orange hues | `sunset` | - |
| 02-08 | C | A fantastical city floating amongst clouds, bathed in eternal sunset | `sunset` | - |
| 02-09 | C | A giant phoenix soaring across the sky, its feathers echoing the colors of the sunset | `of the sunset` | - |

### 03: cyberpunk neon

원문: [03.txt](../datasets/XVerseBench/dreambench_plus_raw/extracted/data/captions/style/03.txt), SHA-256 `8108ccfbdc702e454e7b3e58036e8893d08f75552c19aae227e2c45e95ccd751`

| ID | 유형 | Target prompt 원문 | 보존 후보 | 중립 후보 |
|---|---|---|---|---|
| 03-01 | D | A photo of a bustling city street at night illuminated by neon lights, in a cyberpunk neon style | `in a cyberpunk neon style` | `A photo of` |
| 03-02 | D | A photograph of a rainy urban alley with glowing neon signs, in a cyberpunk neon style | `in a cyberpunk neon style` | `A photograph of` |
| 03-03 | D | A painting of a futuristic motorcycle with bright neon accents, in a cyberpunk neon style | `in a cyberpunk neon style` | - |
| 03-04 | D | A vivid neon-lit skyline of a high-tech city, in a cyberpunk neon style | `in a cyberpunk neon style` | - |
| 03-05 | D | An underground subway station bathed in neon light, cyberpunk neon style | `cyberpunk neon style` | - |
| 03-06 | D | A hacker's den with multiple screens and neon blue lighting, in a cyberpunk neon style | `in a cyberpunk neon style` | - |
| 03-07 | D | A neon-lit bar with eclectic patrons, in a cyberpunk neon style | `in a cyberpunk neon style` | - |
| 03-08 | D | A cybernetic owl with glowing neon eyes perched in a neon-lit tree, in a cyberpunk neon style | `in a cyberpunk neon style` | - |
| 03-09 | D | A floating market with neon signs hovering above boats, in a cyberpunk neon style | `in a cyberpunk neon style` | - |

### 04: Impressionist swirls

원문: [04.txt](../datasets/XVerseBench/dreambench_plus_raw/extracted/data/captions/style/04.txt), SHA-256 `37b501857d8496cf8dce138aa118a2c578f5e758efc15469c88789f96a05bc7f`

| ID | 유형 | Target prompt 원문 | 보존 후보 | 중립 후보 |
|---|---|---|---|---|
| 04-01 | D | A photo of a quiet garden with blooming flowers, brushed in Impressionist swirls | `brushed in Impressionist swirls` | `A photo of` |
| 04-02 | D | A painting of a bustling city street, captured in dynamic Impressionist swirls | `captured in dynamic Impressionist swirls` | - |
| 04-03 | D | A photograph of an old wooden bridge over a tranquil pond, rendered in Impressionist swirls | `rendered in Impressionist swirls` | `A photograph of` |
| 04-04 | D | A serene lakeside at sunset, depicted with vibrant Impressionist swirls | `depicted with vibrant Impressionist swirls` | - |
| 04-05 | D | A tree-lined pathway, shaded and mysterious, in the style of Impressionist swirls | `in the style of Impressionist swirls` | - |
| 04-06 | D | A cozy cafe corner with patrons chatting, illustrated in Impressionist swirls | `illustrated in Impressionist swirls` | - |
| 04-07 | D | An old man reading a newspaper on a park bench, enveloped in Impressionist swirls | `enveloped in Impressionist swirls` | - |
| 04-08 | D | A mystical forest where trees have leaves of shimmering gold, envisioned in Impressionist swirls | `envisioned in Impressionist swirls` | - |
| 04-09 | D | A night sky where stars dance in patterns of bright, swirling colors, depicted in Impressionist swirls | `depicted in Impressionist swirls` | - |

### 05: minimalist line art

원문: [05.txt](../datasets/XVerseBench/dreambench_plus_raw/extracted/data/captions/style/05.txt), SHA-256 `7e50025b384bdb29126c5b6ec9f8891c34036fff9aae8eb88c9d53c283030afc`

| ID | 유형 | Target prompt 원문 | 보존 후보 | 중립 후보 |
|---|---|---|---|---|
| 05-01 | D | A sketch of a single tree with sparse leaves in minimalist line art | `in minimalist line art` | - |
| 05-02 | D | A photo of a slender vase containing a single flower, rendered in minimalist line art | `rendered in minimalist line art` | `A photo of` |
| 05-03 | D | A simple drawing of a cat sitting still, in minimalist line art | `in minimalist line art` | - |
| 05-04 | D | A portrait of a woman with her hair in a bun, illustrated in minimalist line art | `illustrated in minimalist line art` | - |
| 05-05 | M | A minimalist line art representation of a bicycle leaning against a wall | `minimalist line art` | - |
| 05-06 | D | A set of geometric shapes, like circles and triangles, composed in minimalist line art | `composed in minimalist line art` | - |
| 05-07 | M | A minimalist line art drawing of a coffee cup on a saucer | `minimalist line art` | - |
| 05-08 | M | A minimalist line art depiction of a city floating among clouds | `minimalist line art` | - |
| 05-09 | D | A creature from a fantasy world where plants and animals merge, illustrated in minimalist line art | `illustrated in minimalist line art` | - |

### 06: volcanic landscape

원문: [06.txt](../datasets/XVerseBench/dreambench_plus_raw/extracted/data/captions/style/06.txt), SHA-256 `c667d04f256f30d083c7b8df77b2e2c49974da413272309e90a43e938a94b070`

| ID | 유형 | Target prompt 원문 | 보존 후보 | 중립 후보 |
|---|---|---|---|---|
| 06-01 | D | A photograph of a smoldering volcano with streams of molten lava, in a volcanic landscape | `in a volcanic landscape` | `A photograph of` |
| 06-02 | D | A painting of a rugged terrain with ash-covered rocks, in the style of a volcanic landscape | `in the style of a volcanic landscape` | - |
| 06-03 | D | A photo of a sunset casting an eerie glow over a volcanic crater, in a volcanic landscape | `in a volcanic landscape` | `A photo of` |
| 06-04 | D | A close-up photograph of igneous rock formations, characteristic of a volcanic landscape | `characteristic of a volcanic landscape` | - |
| 06-05 | D | A serene lake formed in an ancient volcanic caldera, in the style of a volcanic landscape | `in the style of a volcanic landscape` | - |
| 06-06 | D | A forest recovering after a volcanic eruption, with new growth sprouting through ash, in the style of a volcanic landscape | `in the style of a volcanic landscape` | - |
| 06-07 | D | A night-time scene of a starlit sky above a silhouette of a volcanic mountain, in a volcanic landscape | `in a volcanic landscape` | - |
| 06-08 | D | A fantastical city built around the rim of an active volcano, glowing with lava, in the style of a volcanic landscape | `in the style of a volcanic landscape` | - |
| 06-09 | D | A giant phoenix rising from volcanic ashes under a molten sky, in a volcanic landscape | `in a volcanic landscape` | - |

### 07: Impressionist pointillism

원문: [07.txt](../datasets/XVerseBench/dreambench_plus_raw/extracted/data/captions/style/07.txt), SHA-256 `cfc425a0adecd9d106f95d4b52ce27bc217a63a6919b1c42d005f6213bd0786d`

| ID | 유형 | Target prompt 원문 | 보존 후보 | 중립 후보 |
|---|---|---|---|---|
| 07-01 | D | A painting of a garden with blooming flowers, in Impressionist pointillism | `in Impressionist pointillism` | - |
| 07-02 | D | A photo of a bustling street market, captured in Impressionist pointillism | `captured in Impressionist pointillism` | `A photo of` |
| 07-03 | D | A photograph of a serene lakeside at dusk, rendered in Impressionist pointillism | `rendered in Impressionist pointillism` | `A photograph of` |
| 07-04 | D | A painting of a busy café terrace at night, in the style of Impressionist pointillism | `in the style of Impressionist pointillism` | - |
| 07-05 | D | A child playing with a colorful kite, Impressionist pointillism style | `Impressionist pointillism style` | - |
| 07-06 | D | A woman reading under a tree, depicted in Impressionist pointillism | `depicted in Impressionist pointillism` | - |
| 07-07 | D | A streetscape with old-fashioned lampposts, in Impressionist pointillism style | `in Impressionist pointillism style` | - |
| 07-08 | D | A mythical phoenix soaring above a misty forest, in Impressionist pointillism | `in Impressionist pointillism` | - |
| 07-09 | D | A scene of tiny, illuminated fairies dancing around a moonlit pond, in Impressionist pointillism | `in Impressionist pointillism` | - |

### 08: desert moonscape

원문: [08.txt](../datasets/XVerseBench/dreambench_plus_raw/extracted/data/captions/style/08.txt), SHA-256 `cc4689c67673aebdee2dae42e0150715baebe547e7721b87a9b72cba3e9cf51c`

| ID | 유형 | Target prompt 원문 | 보존 후보 | 중립 후보 |
|---|---|---|---|---|
| 08-01 | D | A photograph of a barren desert under the soft glow of the moon, showcasing a desert moonscape | `showcasing a desert moonscape` | `A photograph of` |
| 08-02 | D | A picture of sand dunes illuminated by moonlight, captured in a desert moonscape style | `captured in a desert moonscape style` | `A picture of` |
| 08-03 | D | An image of a lone cactus silhouetted against a nocturnal desert backdrop, in the style of a desert moonscape | `in the style of a desert moonscape` | `An image of` |
| 08-04 | D | A painting of a weathered rock formation under a starry sky, depicted in a desert moonscape | `depicted in a desert moonscape` | - |
| 08-05 | D | A desert fox prowling silently, bathed in moonlight, in desert moonscape style | `in desert moonscape style` | - |
| 08-06 | D | An oasis reflecting the moon's image, surrounded by the quiet desert, in a desert moonscape style | `in a desert moonscape style` | - |
| 08-07 | D | A caravan of camels trekking through a moonlit desert landscape, captured in desert moonscape style | `captured in desert moonscape style` | - |
| 08-08 | D | An ancient temple rising from the desert sands, bathed in otherworldly moonlight, in a desert moonscape style | `in a desert moonscape style` | - |
| 08-09 | D | A surreal scene of floating desert islands under a gigantic moon, depicted in desert moonscape style | `depicted in desert moonscape style` | - |

### 09: low-poly geometric

원문: [09.txt](../datasets/XVerseBench/dreambench_plus_raw/extracted/data/captions/style/09.txt), SHA-256 `a0967a79cc9bc9a5ca5be36cef91a6d9e55cfeb1e26c9e7382e3b3d0c65da934`

| ID | 유형 | Target prompt 원문 | 보존 후보 | 중립 후보 |
|---|---|---|---|---|
| 09-01 | D | A photograph of a mountain landscape in a low-poly geometric style | `in a low-poly geometric style` | `A photograph of` |
| 09-02 | D | A painting of a bustling city skyline, rendered in low-poly geometric shapes | `rendered in low-poly geometric shapes` | - |
| 09-03 | D | A photo of a serene lake, depicted with crisp low-poly geometric contours | `depicted with crisp low-poly geometric contours` | `A photo of` |
| 09-04 | D | A digital illustration of a cat, composed of low-poly geometric patterns | `composed of low-poly geometric patterns` | - |
| 09-05 | D | A pine tree depicted with angular, low-poly geometric shapes | `depicted with angular, low-poly geometric shapes` | - |
| 09-06 | D | A fox crafted from sharp, low-poly geometric lines | `crafted from sharp, low-poly geometric lines` | - |
| 09-07 | D | A portrait of a woman, stylized with low-poly geometric facets | `stylized with low-poly geometric facets` | - |
| 09-08 | D | A mythical phoenix reborn in flames, designed in vibrant low-poly geometric style | `designed in vibrant low-poly geometric style` | - |
| 09-09 | D | An alien landscape with floating islands, visualized in low-poly geometric form | `visualized in low-poly geometric form` | - |

### 10: Ukiyo-e

원문: [10.txt](../datasets/XVerseBench/dreambench_plus_raw/extracted/data/captions/style/10.txt), SHA-256 `168bf1928d436d3bc799fa5926b781cdd246cf94f7e7744ae5f342b6d08f404b`

| ID | 유형 | Target prompt 원문 | 보존 후보 | 중립 후보 |
|---|---|---|---|---|
| 10-01 | D | A painting of Mount Fuji with delicate Ukiyo-e brush strokes | `with delicate Ukiyo-e brush strokes` | - |
| 10-02 | D | A photograph of a cherry blossom tree, rendered in the Ukiyo-e style | `rendered in the Ukiyo-e style` | `A photograph of` |
| 10-03 | D | A serene river landscape, captured in the traditional Ukiyo-e art form | `captured in the traditional Ukiyo-e art form` | - |
| 10-04 | D | A woodblock print of a bustling Edo period marketplace, in Ukiyo-e style | `in Ukiyo-e style` | - |
| 10-05 | D | A geisha in a colorful kimono, depicted in Ukiyo-e style | `depicted in Ukiyo-e style` | - |
| 10-06 | D | A samurai in battle garb, illustrated in the Ukiyo-e technique | `illustrated in the Ukiyo-e technique` | - |
| 10-07 | D | A traditional tea ceremony, portrayed with Ukiyo-e aesthetics | `portrayed with Ukiyo-e aesthetics` | - |
| 10-08 | D | A mythical dragon coiling around Mount Fuji, depicted in Ukiyo-e style | `depicted in Ukiyo-e style` | - |
| 10-09 | D | A scene of moon jellyfish floating in the sky, imagined in Ukiyo-e brushwork | `imagined in Ukiyo-e brushwork` | - |

### 11: Impressionist

원문: [11.txt](../datasets/XVerseBench/dreambench_plus_raw/extracted/data/captions/style/11.txt), SHA-256 `abe7a714d56fcf62e7285510579bf11d5096f47f9e03cde31ce2a248151547df`

| ID | 유형 | Target prompt 원문 | 보존 후보 | 중립 후보 |
|---|---|---|---|---|
| 11-01 | D | A photo of a blossoming garden in the Impressionist style | `in the Impressionist style` | `A photo of` |
| 11-02 | D | A painting of a bustling street cafe in the Impressionist style | `in the Impressionist style` | - |
| 11-03 | D | A photograph of a serene lakeside at dusk, captured in the Impressionist style | `captured in the Impressionist style` | `A photograph of` |
| 11-04 | D | A vibrant market scene depicted in the Impressionist style | `depicted in the Impressionist style` | - |
| 11-05 | D | A woman holding a parasol, painted in the Impressionist style | `painted in the Impressionist style` | - |
| 11-06 | D | A small boat floating on a gentle river, in the Impressionist style | `in the Impressionist style` | - |
| 11-07 | D | A child playing in a sunlit field, captured in the Impressionist style | `captured in the Impressionist style` | - |
| 11-08 | D | A mythical forest with trees that glow in the Impressionist style | `in the Impressionist style` | - |
| 11-09 | D | A dreamlike cityscape where the buildings softly blur into the sky, in the Impressionist style | `in the Impressionist style` | - |

### 12: pixel art

원문: [12.txt](../datasets/XVerseBench/dreambench_plus_raw/extracted/data/captions/style/12.txt), SHA-256 `9a9961ac7ae1c893d75f18f21b4c46549ec4bf4225e8d144c7055d9f2c998d13`

| ID | 유형 | Target prompt 원문 | 보존 후보 | 중립 후보 |
|---|---|---|---|---|
| 12-01 | D | A photo of a bustling cityscape at night, rendered in pixel art | `rendered in pixel art` | `A photo of` |
| 12-02 | D | A painting of a serene mountain landscape, depicted in pixel art | `depicted in pixel art` | - |
| 12-03 | D | A photograph of a colorful coral reef, stylized in pixel art | `stylized in pixel art` | `A photograph of` |
| 12-04 | D | An image of a quaint village street, visualized in pixel art | `visualized in pixel art` | `An image of` |
| 12-05 | D | A vintage car parked on an urban street, in pixel art style | `in pixel art style` | - |
| 12-06 | D | A cup of steaming coffee on a saucer, pixel art representation | `pixel art representation` | - |
| 12-07 | D | A houseplant with lush green leaves, illustrated in pixel art | `illustrated in pixel art` | - |
| 12-08 | D | A scene of tiny astronauts exploring a cheese moon, in pixel art | `in pixel art` | - |
| 12-09 | D | An enchanted forest with trees casting spells, visualized in pixel art | `visualized in pixel art` | - |

### 13: cosmic nebula

원문: [13.txt](../datasets/XVerseBench/dreambench_plus_raw/extracted/data/captions/style/13.txt), SHA-256 `9ae194d35c67bd28c842ae79d21605f90dd661e0d4b5772dbe3e9957cd2d942a`

| ID | 유형 | Target prompt 원문 | 보존 후보 | 중립 후보 |
|---|---|---|---|---|
| 13-01 | M | A photo of a serene galaxy with swirling cosmic nebula colors | `cosmic nebula` | `A photo of` |
| 13-02 | C | A painting of a star being born within a vibrant cosmic nebula | `cosmic nebula` | - |
| 13-03 | D | A photograph of an interstellar cloud in the style of cosmic nebula | `in the style of cosmic nebula` | `A photograph of` |
| 13-04 | M | An illustration of a distant universe filled with cosmic nebula patterns | `cosmic nebula` | - |
| 13-05 | C | A telescope view capturing the intricate movement of a cosmic nebula | `of a cosmic nebula` | - |
| 13-06 | C | A star cluster surrounded by a luminous cosmic nebula | `cosmic nebula` | - |
| 13-07 | C | A cosmic nebula enveloping a newly forming solar system | `A cosmic nebula` | - |
| 13-08 | C | A mythical phoenix rising from a cosmic nebula | `from a cosmic nebula` | - |
| 13-09 | C | A grand castle floating amidst a colorful cosmic nebula | `cosmic nebula` | - |

### 14: serene desert

원문: [14.txt](../datasets/XVerseBench/dreambench_plus_raw/extracted/data/captions/style/14.txt), SHA-256 `5a53a18bf5248b67e2757c271227738dd4562d4871c823f9bdd03fc98952258f`

| ID | 유형 | Target prompt 원문 | 보존 후보 | 중립 후보 |
|---|---|---|---|---|
| 14-01 | D | A photograph of a tranquil desert landscape at dawn, serene desert style | `serene desert style` | `A photograph of` |
| 14-02 | D | A painting of a lone cactus under a vast, clear sky, serene desert style | `serene desert style` | - |
| 14-03 | D | A photo of sand dunes softly lit by the setting sun, in a serene desert style | `in a serene desert style` | `A photo of` |
| 14-04 | D | An image of a desert fox quietly navigating through sparse vegetation, serene desert style | `serene desert style` | `An image of` |
| 14-05 | M | A serene desert scene featuring an old wooden cart partially buried in sand | `serene desert` | - |
| 14-06 | D | A moonlit desert with gentle winds stirring the sand, serene desert style | `serene desert style` | - |
| 14-07 | D | A quiet desert oasis surrounded by palm trees, in a serene desert style | `in a serene desert style` | - |
| 14-08 | C | A mirage of an ancient city appearing over the dunes in a serene desert | `in a serene desert` | - |
| 14-09 | D | A photo of a floating desert island in the sky, serene desert style | `serene desert style` | `A photo of` |

### 15: monochromatic ink sketch

원문: [15.txt](../datasets/XVerseBench/dreambench_plus_raw/extracted/data/captions/style/15.txt), SHA-256 `f2b5560b2ef1d02b1e9072554d5f35cf538f21ee20119f745846a1847fe6bddc`

| ID | 유형 | Target prompt 원문 | 보존 후보 | 중립 후보 |
|---|---|---|---|---|
| 15-01 | D | A photo of a serene lake bordered by willow trees, depicted in a monochromatic ink sketch style | `depicted in a monochromatic ink sketch style` | `A photo of` |
| 15-02 | D | A drawing of a bustling city street, with pedestrians and cars, rendered in monochromatic ink sketch | `rendered in monochromatic ink sketch` | - |
| 15-03 | D | A painting of an old stone bridge over a quiet stream, in a monochromatic ink sketch style | `in a monochromatic ink sketch style` | - |
| 15-04 | D | An illustration of a quaint cottage surrounded by a flower garden, in monochromatic ink sketch | `in monochromatic ink sketch` | - |
| 15-05 | D | A cat perched on a windowsill, monochromatic ink sketch style | `monochromatic ink sketch style` | - |
| 15-06 | D | A teapot steaming gently on an old wooden table, in a monochromatic ink sketch style | `in a monochromatic ink sketch style` | - |
| 15-07 | D | A pair of worn leather boots, illustrated in a monochromatic ink sketch style | `illustrated in a monochromatic ink sketch style` | - |
| 15-08 | D | A mythical forest with trees that whisper secrets, visualized in a monochromatic ink sketch style | `visualized in a monochromatic ink sketch style` | - |
| 15-09 | D | A city floating in the clouds, depicted as a monochromatic ink sketch | `depicted as a monochromatic ink sketch` | - |

### 16: watercolor sketch

원문: [16.txt](../datasets/XVerseBench/dreambench_plus_raw/extracted/data/captions/style/16.txt), SHA-256 `82fff1e3f3ea1672099c57a5ba51fb7de47c1a8c694eefe8460d9304c77fcafd`

| ID | 유형 | Target prompt 원문 | 보존 후보 | 중립 후보 |
|---|---|---|---|---|
| 16-01 | D | A cottage surrounded by blooming flowers, depicted in watercolor sketch | `depicted in watercolor sketch` | - |
| 16-02 | D | A serene riverside scene with fishermen in boats, painted in watercolor sketch | `painted in watercolor sketch` | - |
| 16-03 | D | A bustling city street with pedestrians and vintage cars, illustrated in watercolor sketch | `illustrated in watercolor sketch` | - |
| 16-04 | D | A peaceful meadow under a clear blue sky, rendered in watercolor sketch | `rendered in watercolor sketch` | - |
| 16-05 | D | A cat napping by a sunny window, in watercolor sketch style | `in watercolor sketch style` | - |
| 16-06 | D | A row of old books on a wooden shelf, depicted in watercolor sketch style | `depicted in watercolor sketch style` | - |
| 16-07 | D | A woman reading under a tree, illustrated in watercolor sketch style | `illustrated in watercolor sketch style` | - |
| 16-08 | D | An ancient tree with glowing leaves in a watercolor sketch style | `in a watercolor sketch style` | - |
| 16-09 | D | A city floating on clouds, illustrated in watercolor sketch style | `illustrated in watercolor sketch style` | - |

### 17: cosmic nebula

원문: [17.txt](../datasets/XVerseBench/dreambench_plus_raw/extracted/data/captions/style/17.txt), SHA-256 `3a919e3bb9195afeeb0a251de8ba5f01e88b7ed58d263ed09c684f9e88563b2f`

| ID | 유형 | Target prompt 원문 | 보존 후보 | 중립 후보 |
|---|---|---|---|---|
| 17-01 | D | A photograph of a distant galaxy swirling in hues of blue and pink, in a cosmic nebula style | `in a cosmic nebula style` | `A photograph of` |
| 17-02 | D | A photo of a star cluster emitting radiant light, rendered in cosmic nebula aesthetics | `rendered in cosmic nebula aesthetics` | `A photo of` |
| 17-03 | D | A painting of a newly forming star within a cloud of cosmic dust, in cosmic nebula style | `in cosmic nebula style` | - |
| 17-04 | D | An image of the aurora borealis with vibrant cosmic nebula colors | `with vibrant cosmic nebula colors` | `An image of` |
| 17-05 | D | A telescope pointed towards a starry sky, painted in a cosmic nebula style | `painted in a cosmic nebula style` | - |
| 17-06 | D | A night sky filled with constellations, depicted in cosmic nebula aesthetics | `depicted in cosmic nebula aesthetics` | - |
| 17-07 | D | An astronaut floating in space surrounded by nebula-like colors, in a cosmic nebula style | `in a cosmic nebula style` | - |
| 17-08 | M | A mythical phoenix reborn from cosmic nebula dust | `cosmic nebula` | - |
| 17-09 | D | A ancient sailboat navigating through a sea of stars and nebulae in cosmic nebula style | `in cosmic nebula style` | - |

### 18: pastel

원문: [18.txt](../datasets/XVerseBench/dreambench_plus_raw/extracted/data/captions/style/18.txt), SHA-256 `d3ea564c6afb404a59beb4f2f68d2e1e7c2022688a84bf2087863d2397eba0ec`

| ID | 유형 | Target prompt 원문 | 보존 후보 | 중립 후보 |
|---|---|---|---|---|
| 18-01 | D | A photograph of a serene beach at dusk, rendered in soft pastel tones | `rendered in soft pastel tones` | `A photograph of` |
| 18-02 | D | A painting of a tranquil garden with blooming flowers in a pastel palette | `in a pastel palette` | - |
| 18-03 | D | A photo of a gentle sunrise over a calm lake, captured in pastel colors | `captured in pastel colors` | `A photo of` |
| 18-04 | D | A pastoral landscape with rolling hills and a clear sky, depicted in pastel hues | `depicted in pastel hues` | - |
| 18-05 | D | A kitten sleeping in a sunny window, bathed in light pastel colors | `bathed in light pastel colors` | - |
| 18-06 | M | A cup of tea on a pastel tablecloth, inviting a moment of relaxation | `pastel` | - |
| 18-07 | M | A close-up of a pastel-colored macaron, delicate and sweet | `pastel-colored` | - |
| 18-08 | M | A whimsical forest where the trees have pastel-colored leaves | `pastel-colored` | - |
| 18-09 | M | A dreamy pastel planet with rings made of stardust and light | `pastel` | - |

### 19: traditional Chinese painting

원문: [19.txt](../datasets/XVerseBench/dreambench_plus_raw/extracted/data/captions/style/19.txt), SHA-256 `b04d5adf63530a711233201f4787cfeecb4fff86b20fd5617ed734d6852ba550`

| ID | 유형 | Target prompt 원문 | 보존 후보 | 중립 후보 |
|---|---|---|---|---|
| 19-01 | D | A mountain landscape shrouded in mist, in traditional Chinese painting style | `in traditional Chinese painting style` | - |
| 19-02 | D | A delicate bamboo grove with slender leaves, depicted in traditional Chinese painting | `depicted in traditional Chinese painting` | - |
| 19-03 | D | A serene koi pond reflecting the sky, captured in the style of traditional Chinese painting | `captured in the style of traditional Chinese painting` | - |
| 19-04 | D | A scholar sitting under a willow tree, in traditional Chinese painting style | `in traditional Chinese painting style` | - |
| 19-05 | D | An ancient pagoda amidst rolling hills, in the style of traditional Chinese painting | `in the style of traditional Chinese painting` | - |
| 19-06 | D | A pair of cranes standing in a marshy reed bed, rendered in traditional Chinese painting style | `rendered in traditional Chinese painting style` | - |
| 19-07 | D | A bustling ancient market scene with vendors and shoppers, traditional Chinese painting style | `traditional Chinese painting style` | - |
| 19-08 | D | A dragon coiling around a moonlit pagoda, envisioned in traditional Chinese painting style | `envisioned in traditional Chinese painting style` | - |
| 19-09 | D | A mythical phoenix soaring above a blooming cherry orchard, in traditional Chinese painting style | `in traditional Chinese painting style` | - |

