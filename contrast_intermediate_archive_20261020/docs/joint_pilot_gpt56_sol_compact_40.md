# 40개 공동 contrast — 전체 원문과 차분 문장

> 후속 정책 안내: 이 실험은 concept-only 보존 규칙 수정 전 결과다. 현재는 의미가 같은 묶음/분리의 개수 차이 자체를 오류로 보지 않는다. 아래 실제 응답과 당시 관측은 변경하지 않았다.


각 방향은 normalize(e(아래 positive) − e(각 negative))이다. 각 negative는 독립된 원문 기준이며 누적 삭제가 아니다.
PASS는 구조 검증 통과만 뜻한다. 아래 응답과 오류는 수정하지 않은 실제 API 결과다.

<!-- REVIEW_START -->
## 2026-10-20 현재 프롬프트 실제 실행

이 보고서는 기존 내용 대신 **A의 preserve 전달 계약을 사용한 새 실제 응답**으로 갱신했다.

- 모델: **GPT-5.6 Sol**. A `caption_np_v4` / `identity_np_preserve_v3`, B `joint_contrast_v5` / `joint_local_v5`.
- 입력은 기존과 동일한 reference **19개**, DreamBench/DreamBench++ 고정 Target **40쌍**이다. 180개 전체나 2,100쌍 전체를 실행한 것은 아니다.
- A 19개를 새로 생성한 뒤 각 caption과 preserve를 해당 reference의 여러 Target에 재사용했다. 기존 실험의 caption이나 B 응답을 재사용하지 않았다.
- 실제 호출은 **A 19회 + B 40회 = 59회**. 검증 재시도 0회, 최종 구조 PASS **40/40**, negative **165개**. PASS는 의미·문법 정답률이 아니다.
- A caption 공백 기준 평균 길이는 **15.5단어**(9~24)다. 직전 compact Sol 실행은 15.4단어였다. 보존 목록은 caption과 별도로 전달한다.
- 실행 도중 시스템 프롬프트를 변경하지 않았으며, 의미가 마음에 들지 않는 응답을 골라 재호출하거나 고쳐 쓰지 않았다.
- 40쌍의 원문·A preserve·B 보존 결과·Source/Target negative를 모두 텍스트로 대조했다. 원본 이미지 19장과 caption의 정밀 전수 대조는 하지 않았다.
- 개발 few-shot과 유사하거나 겹치는 사례가 있으므로 독립적인 일반화 성능 측정이 아니다. 같은 정확한 입력을 여러 번 반복한 실험도 아니므로 재현성/일관성의 성공률을 계산하지 않는다.
- 실제 SigLIP/GPU 생성·benchmark metric 평가는 하지 않았다. 청구 usage가 저장되지 않아 실제 달러 비용은 확정하지 않는다.

## 주요 관측

**기본 보존 범위 전달은 실제로 연결됐다.** 저장된 40개 B 입력의 preserve는 해당 A 응답과 모두 일치했다.
A 목록에는 identity를 중복하지 않았고, B 결과 source_preserved에는 감사 목적으로 identity가 포함된다.
두 필드는 서로 다른 역할이며 B 감사 필드의 관사·전치사 표현 차이를 곧바로 의미 변화로 계산하지 않았다.

- **Backpack:** 000-09의 purple rug는 magenta 해제 근거가 되지 않았다. 000-21 purple backpack은 magenta만 해제했으며 위치 제거 negative에는 magenta와 다른 외형이 남았다.
- **Berry bowl:** 네 Target 모두 blueberries를 유지했다. `white patterned`라는 하나의 A 보존 표현에서 purple 요청은 white만 해제하고 patterned는 남겼다.
- **Magic Cube:** A가 `partially twisted`를 preserve에서 제외했고, 115-04/06/07 모두 이 상태를 제거했다. 이전의 Target별 회전 상태 보존 편차는 이번 세 응답에서 재발하지 않았다.
- **T23:** winged female figure는 spaceship 변형과 충돌하는 외형으로 해제했고 polished silver는 유지했다. 변형과 후속 leading 동작을 묶은 negative도 생성됐으며 identity conflict로 중단되지 않았다.
- **T01/cloud:** pixel art가 직접 수식하는 forest/cloud는 유지했다. T13의 bow tie 제거 뒤 exploring, T17의 comet/steering 묶음, T20의 web 관계, T22의 character/solving 묶음, T03의 phoenix/feathers 의존 제거를 아래 실제 응답에서 확인했다.
- **기능어:** 176-07은 `watercolor sketch style`, 178-00은 `A photograph rendered in soft pastel tones`를 생성했다. `in ...`만 남던 형태는 이 사례에서 재발하지 않았다.

## 남은 문제: A의 범위와 B의 단위 선택을 구분

| 사례 | 실제 관측 | 판단 |
|---|---|---|
| **169-03/05 low-poly** | A가 `forested cliff island`와 `with stylized trees, a stream, and a waterfall`을 preserve로 지정했다. B는 두 Target에서 모두 Source 제거 목록을 비웠다. | **가장 명확한 A 과보존 사례.** 전역 low-poly style이 부수적인 섬·나무·물까지 보존해야 한다는 뜻은 아니다. B는 새 계약을 따랐지만, 잘못된 A 범위가 두 Target에 그대로 전파됐다. |
| 162-01/08 sunset | A가 `over ocean waves and a sandy beach`까지 보호했고 B Source가 모두 비었다. | Sunset과 주변 장소의 경계가 넓게 잡힌 과보존 후보다. 목록 전달의 일관성과 목록의 적절성을 구분해야 한다. |
| 161-00/07/08 pixel art | Source preserve는 모두 []이지만 Source negative 수는 1/1/4개다. | 보존 의미는 일치하지만 **제거 단위 크기와 수는 여전히 변동**한다. 동일 Source에 같은 방향 집합이 보장된 것은 아니다. |
| 178-00/05/07/08 pastel | 같은 Source와 preserve=[]에서 Source negative 수는 6/1/4/5개다. 178-05는 전체 장면을 묶고, 178-07은 dunes 자체의 제거가 없다. | 단위 묶음·후보 포함 범위의 변동이 남았다. 단순 개수 차이만 오류로 집계하지 않지만, 제거 후보 누락 가능성은 검토 대상이다. |
| 161-00 및 162-01 Target | depicted cityscape/skyline을 제거한 negative에서 독립적으로 제거할 수 있는 photo/photograph framing까지 사라졌다. | **제거 범위를 넓힌 후보.** `A photo rendered in pixel art`처럼 framing을 남길 여지가 있다. 178-00에서 framing이 유지됐다고 다른 사례도 해결됐다고 볼 수 없다. |
| 010-21 purple dog | orange-and-white 기본 털색은 해제하고 broad white facial stripe는 유지했다. | 전체 색 변경과 식별 무늬의 경계 문제다. 이번 응답이 의도한 recolor 범위를 만족한다고 확정하지 않는다. |
| 115-07 geometric morphing | A가 보호한 multicolored square tiles와 black borders는 그대로 유지했다. | 전체 형태 변형이 개별 타일 형태/색까지 바꾸는지 불명확하다. 부분 속성의 충돌 해제 성공으로 집계하지 않는다. |
| 075-07 Source | 자세 제거 후 `...one shoulder on an alien landscape`가 이어진다. | 원래 공간 부착 관계가 문법적으로 덜 명확해질 여지는 남아 있다. 주체가 실제로 바뀌었다고 단정하지는 않는다. |

이번 결과는 **기본 보존 범위의 변동을 줄이는 방향은 유효해 보이지만, A의 과보존과 B의 제거 단위 선택 문제는 별도로 남는다**는 관측이다.
Caption도 새로 생성됐으므로 이전 실행과의 차이를 preserve 전달 구조 하나의 효과로 분리해 단정할 수 없다.
다음 보완 우선순위는 사례별 재호출보다 A의 전역 style/주변 장면 경계(169, 162)를 먼저 좁히고,
B의 단위 크기·누락을 별도의 소규모 반복 실험으로 확인하는 것이다. 이번 요청에서는 프롬프트를 추가 변경하지 않았다.

## 이번 실행 결과의 offline 검증

- 새 A 19개와 B 40개 모두 직렬화 왕복과 reference/identity/target/caption/preserve/model 바인딩을 확인했다. A/B prompt hash도 현재 runtime과 일치한다.
- 실제 저장 응답 40개를 source/target/source_target로 선택한 **120개 계획**을 주입형 CPU encoder로 검증했다. 원문 positive, 실제 B negative, 차분 부호·정규화·같은 텍스트 인코딩 재사용·빈 목록·diagnostics 연결을 확인했다.
- 이 encoder는 해시 기반 테스트 벡터이며 실제 SigLIP 의미 평가가 아니다. 165개는 생성된 제거 방향 수이며 최종 projection basis rank가 아니다.
- 원래 Target은 저장 입력과 계획의 Target positive에 유지됐다. GPU 생성은 하지 않았으므로 이번 실험을 이미지 품질 검증으로 표현하지 않는다.

원본 응답·trace·prompt snapshot: [`outputs/joint_pilot_gpt56_sol_preserve_v5`](../outputs/joint_pilot_gpt56_sol_preserve_v5).
기존 raw JSON과 benchmark 입력은 유지했으며, 본 Markdown의 이전 분석 내용만 새 관측으로 교체했다.

## 새 A caption과 기본 preserve — 19개

Preserve에는 identity를 넣지 않는다. []도 유효하다. 아래 목록은 수정하지 않은 실제 A 출력이다.

| Reference index | Caption | preserve |
|---|---|---|
| 000 | A magenta backpack with a large zippered front pocket, gold-toned zippers, and two character pins on a rocky ledge | `["magenta", "with a large zippered front pocket", "gold-toned zippers", "two character pins"]` |
| 003 | A white patterned bowl filled with blueberries on white fabric | `["white patterned", "filled with blueberries"]` |
| 010 | A fluffy orange-and-white dog with upright ears and a broad white facial stripe sitting outdoors | `["fluffy orange-and-white", "with upright ears and a broad white facial stripe"]` |
| 030 | A gray-and-white kitten with pale green eyes sitting upright on a bed | `["gray-and-white", "with pale green eyes"]` |
| 051 | A cream-colored Siamese cat with blue eyes and dark markings on its face, ears, and tail, sitting on a wooden tabletop | `["cream-colored", "with blue eyes and dark markings on its face, ears, and tail"]` |
| 075 | A space ranger in a white and green armored suit with a small orange cat perched on one shoulder, standing on an alien landscape | `["in a white and green armored suit", "with a small orange cat perched on one shoulder"]` |
| 089 | Spider-Man in a red and dark blue web-patterned suit with large white eye lenses, swinging above a colorful city street | `["in a red and dark blue web-patterned suit", "with large white eye lenses"]` |
| 098 | pale pink cherry blossoms clustered on a reddish-brown branch | `["pale pink", "clustered on a reddish-brown branch"]` |
| 115 | A partially twisted Magic Cube with multicolored square tiles and black borders | `["with multicolored square tiles and black borders"]` |
| 131 | A cream ceramic bowl with a green botanical pattern and short rounded feet | `["cream", "with a green botanical pattern and short rounded feet"]` |
| 135 | A polished silver Rolls-Royce hood ornament depicting a winged female figure mounted above the car grille | `["polished silver", "depicting a winged female figure"]` |
| 161 | pixel art of a beige desktop computer with a blue screen and wired mouse | `[]` |
| 162 | sunset with an orange-pink cloudy sky over ocean waves and a sandy beach | `["with an orange-pink cloudy sky", "over ocean waves and a sandy beach"]` |
| 166 | volcanic landscape with a glowing orange lava flow winding through black rock beneath a gray cloudy sky | `["with a glowing orange lava flow", "black rock"]` |
| 169 | a forested cliff island with stylized trees, a stream, and a waterfall in low-poly geometric style | `["forested cliff island", "with stylized trees, a stream, and a waterfall"]` |
| 173 | A cosmic nebula with bright turquoise and blue gas clouds, dark dust lanes, and glowing golden edges amid a dense star field | `["with bright turquoise and blue gas clouds", "dark dust lanes", "glowing golden edges"]` |
| 176 | a rocky mountain landscape with sparse shrubs in watercolor sketch | `[]` |
| 178 | rolling blue and pink dunes beneath a pale moon in pastel | `[]` |
| 179 | traditional Chinese painting of a pavilion garden among tall pine trees and misty mountains, rendered with fine lines and muted colors | `["with fine lines and muted colors"]` |

<!-- REVIEW_END -->

| 번호 | ID | identity | 결과 |
|---|---|---|---|
| 1 | [000-00](#case-000-00) | backpack | PASS |
| 2 | [000-09](#case-000-09) | backpack | PASS |
| 3 | [000-21](#case-000-21) | backpack | PASS |
| 4 | [000-24](#case-000-24) | backpack | PASS |
| 5 | [003-02](#case-003-02) | bowl | PASS |
| 6 | [003-14](#case-003-14) | bowl | PASS |
| 7 | [003-21](#case-003-21) | bowl | PASS |
| 8 | [003-24](#case-003-24) | bowl | PASS |
| 9 | [010-10](#case-010-10) | dog | PASS |
| 10 | [010-13](#case-010-13) | dog | PASS |
| 11 | [010-21](#case-010-21) | dog | PASS |
| 12 | [030-06](#case-030-06) | kitten | PASS |
| 13 | [030-07](#case-030-07) | kitten | PASS |
| 14 | [051-07](#case-051-07) | Siamese cat | PASS |
| 15 | [075-06](#case-075-06) | space ranger | PASS |
| 16 | [075-07](#case-075-07) | space ranger | PASS |
| 17 | [075-08](#case-075-08) | space ranger | PASS |
| 18 | [089-07](#case-089-07) | Spider-Man | PASS |
| 19 | [115-04](#case-115-04) | Magic Cube | PASS |
| 20 | [115-06](#case-115-06) | Magic Cube | PASS |
| 21 | [115-07](#case-115-07) | Magic Cube | PASS |
| 22 | [135-01](#case-135-01) | Rolls-Royce hood ornament | PASS |
| 23 | [135-07](#case-135-07) | Rolls-Royce hood ornament | PASS |
| 24 | [161-00](#case-161-00) | pixel art | PASS |
| 25 | [161-07](#case-161-07) | pixel art | PASS |
| 26 | [161-08](#case-161-08) | pixel art | PASS |
| 27 | [162-01](#case-162-01) | sunset | PASS |
| 28 | [162-08](#case-162-08) | sunset | PASS |
| 29 | [169-03](#case-169-03) | low-poly geometric | PASS |
| 30 | [169-05](#case-169-05) | low-poly geometric | PASS |
| 31 | [178-00](#case-178-00) | pastel | PASS |
| 32 | [178-05](#case-178-05) | pastel | PASS |
| 33 | [178-07](#case-178-07) | pastel | PASS |
| 34 | [178-08](#case-178-08) | pastel | PASS |
| 35 | [098-00](#case-098-00) | cherry blossoms | PASS |
| 36 | [131-00](#case-131-00) | ceramic bowl | PASS |
| 37 | [166-06](#case-166-06) | volcanic landscape | PASS |
| 38 | [173-07](#case-173-07) | cosmic nebula | PASS |
| 39 | [176-07](#case-176-07) | watercolor sketch | PASS |
| 40 | [179-08](#case-179-08) | traditional Chinese painting | PASS |

<a id="case-000-00"></a>
## 01. 000-00 — backpack

- Reference: `dreambench/backpack`
- 결과: **PASS**
- 알려진 사례: 추가 benchmark 사례

**Source positive — 원문 caption**

> A magenta backpack with a large zippered front pocket, gold-toned zippers, and two character pins on a rocky ledge

**Target positive — 생성에도 그대로 전달하는 원문**

> a backpack in the jungle

**A가 전달한 기본 preserve (identity 별도 보호)**

["magenta", "with a large zippered front pocket", "gold-toned zippers", "two character pins"]

**Source 보존 정보**

`backpack`, `magenta`, `large zippered front pocket`, `gold-toned zippers`, `two character pins`

### Source 차분 (1개)

**source[0] — rocky ledge placement**

A magenta backpack with a large zippered front pocket, gold-toned zippers, and two character pins

### Target 차분 (1개)

**target[0] — jungle setting**

a backpack

모델: `gpt-5.6-sol` · prompt: `joint_contrast_v5` · API attempts: 1


<a id="case-000-09"></a>
## 02. 000-09 — backpack

- Reference: `dreambench/backpack`
- 결과: **PASS**
- 알려진 사례: 추가 benchmark 사례

**Source positive — 원문 caption**

> A magenta backpack with a large zippered front pocket, gold-toned zippers, and two character pins on a rocky ledge

**Target positive — 생성에도 그대로 전달하는 원문**

> a backpack on top of a purple rug in a forest

**A가 전달한 기본 preserve (identity 별도 보호)**

["magenta", "with a large zippered front pocket", "gold-toned zippers", "two character pins"]

**Source 보존 정보**

`backpack`, `magenta`, `large zippered front pocket`, `gold-toned zippers`, `two character pins`

### Source 차분 (1개)

**source[0] — rocky ledge placement**

A magenta backpack with a large zippered front pocket, gold-toned zippers, and two character pins

### Target 차분 (3개)

**target[0] — placement on top of a rug**

a backpack in a forest

**target[1] — purple rug color**

a backpack on top of a rug in a forest

**target[2] — forest setting**

a backpack on top of a purple rug

모델: `gpt-5.6-sol` · prompt: `joint_contrast_v5` · API attempts: 1


<a id="case-000-21"></a>
## 03. 000-21 — backpack

- Reference: `dreambench/backpack`
- 결과: **PASS**
- 알려진 사례: 추가 benchmark 사례

**Source positive — 원문 caption**

> A magenta backpack with a large zippered front pocket, gold-toned zippers, and two character pins on a rocky ledge

**Target positive — 생성에도 그대로 전달하는 원문**

> a purple backpack

**A가 전달한 기본 preserve (identity 별도 보호)**

["magenta", "with a large zippered front pocket", "gold-toned zippers", "two character pins"]

**Source 보존 정보**

`backpack`, `large zippered front pocket`, `gold-toned zippers`, `two character pins`

### Source 차분 (2개)

**source[0] — magenta body color**

A backpack with a large zippered front pocket, gold-toned zippers, and two character pins on a rocky ledge

**source[1] — rocky ledge placement**

A magenta backpack with a large zippered front pocket, gold-toned zippers, and two character pins

### Target 차분 (1개)

**target[0] — purple body color**

a backpack

모델: `gpt-5.6-sol` · prompt: `joint_contrast_v5` · API attempts: 1


<a id="case-000-24"></a>
## 04. 000-24 — backpack

- Reference: `dreambench/backpack`
- 결과: **PASS**
- 알려진 사례: 추가 benchmark 사례

**Source positive — 원문 caption**

> A magenta backpack with a large zippered front pocket, gold-toned zippers, and two character pins on a rocky ledge

**Target positive — 생성에도 그대로 전달하는 원문**

> a cube shaped backpack

**A가 전달한 기본 preserve (identity 별도 보호)**

["magenta", "with a large zippered front pocket", "gold-toned zippers", "two character pins"]

**Source 보존 정보**

`backpack`, `magenta`, `large zippered front pocket`, `gold-toned zippers`, `two character pins`

### Source 차분 (1개)

**source[0] — rocky ledge placement**

A magenta backpack with a large zippered front pocket, gold-toned zippers, and two character pins

### Target 차분 (1개)

**target[0] — cube shape**

a backpack

모델: `gpt-5.6-sol` · prompt: `joint_contrast_v5` · API attempts: 1


<a id="case-003-02"></a>
## 05. 003-02 — bowl

- Reference: `dreambench/berry_bowl`
- 결과: **PASS**
- 알려진 사례: 추가 benchmark 사례

**Source positive — 원문 caption**

> A white patterned bowl filled with blueberries on white fabric

**Target positive — 생성에도 그대로 전달하는 원문**

> a bowl on the beach

**A가 전달한 기본 preserve (identity 별도 보호)**

["white patterned", "filled with blueberries"]

**Source 보존 정보**

`bowl`, `white patterned`, `filled with blueberries`

### Source 차분 (1개)

**source[0] — white fabric placement**

A white patterned bowl filled with blueberries

### Target 차분 (1개)

**target[0] — beach placement**

a bowl

모델: `gpt-5.6-sol` · prompt: `joint_contrast_v5` · API attempts: 1


<a id="case-003-14"></a>
## 06. 003-14 — bowl

- Reference: `dreambench/berry_bowl`
- 결과: **PASS**
- 알려진 사례: 추가 benchmark 사례

**Source positive — 원문 caption**

> A white patterned bowl filled with blueberries on white fabric

**Target positive — 생성에도 그대로 전달하는 원문**

> a bowl floating in an ocean of milk

**A가 전달한 기본 preserve (identity 별도 보호)**

["white patterned", "filled with blueberries"]

**Source 보존 정보**

`bowl`, `white patterned`, `filled with blueberries`

### Source 차분 (1개)

**source[0] — white fabric placement**

A white patterned bowl filled with blueberries

### Target 차분 (2개)

**target[0] — floating state**

a bowl in an ocean of milk

**target[1] — ocean of milk setting**

a bowl floating

모델: `gpt-5.6-sol` · prompt: `joint_contrast_v5` · API attempts: 1


<a id="case-003-21"></a>
## 07. 003-21 — bowl

- Reference: `dreambench/berry_bowl`
- 결과: **PASS**
- 알려진 사례: 추가 benchmark 사례

**Source positive — 원문 caption**

> A white patterned bowl filled with blueberries on white fabric

**Target positive — 생성에도 그대로 전달하는 원문**

> a purple bowl

**A가 전달한 기본 preserve (identity 별도 보호)**

["white patterned", "filled with blueberries"]

**Source 보존 정보**

`bowl`, `patterned`, `filled with blueberries`

### Source 차분 (2개)

**source[0] — white bowl color**

A patterned bowl filled with blueberries on white fabric

**source[1] — white fabric placement**

A white patterned bowl filled with blueberries

### Target 차분 (1개)

**target[0] — purple bowl color**

a bowl

모델: `gpt-5.6-sol` · prompt: `joint_contrast_v5` · API attempts: 1


<a id="case-003-24"></a>
## 08. 003-24 — bowl

- Reference: `dreambench/berry_bowl`
- 결과: **PASS**
- 알려진 사례: 추가 benchmark 사례

**Source positive — 원문 caption**

> A white patterned bowl filled with blueberries on white fabric

**Target positive — 생성에도 그대로 전달하는 원문**

> a cube shaped bowl

**A가 전달한 기본 preserve (identity 별도 보호)**

["white patterned", "filled with blueberries"]

**Source 보존 정보**

`bowl`, `white patterned`, `filled with blueberries`

### Source 차분 (1개)

**source[0] — white fabric placement**

A white patterned bowl filled with blueberries

### Target 차분 (1개)

**target[0] — cube shape**

a bowl

모델: `gpt-5.6-sol` · prompt: `joint_contrast_v5` · API attempts: 1


<a id="case-010-10"></a>
## 09. 010-10 — dog

- Reference: `dreambench/dog`
- 결과: **PASS**
- 알려진 사례: 추가 benchmark 사례

**Source positive — 원문 caption**

> A fluffy orange-and-white dog with upright ears and a broad white facial stripe sitting outdoors

**Target positive — 생성에도 그대로 전달하는 원문**

> a dog wearing a red hat

**A가 전달한 기본 preserve (identity 별도 보호)**

["fluffy orange-and-white", "with upright ears and a broad white facial stripe"]

**Source 보존 정보**

`dog`, `fluffy orange-and-white`, `upright ears`, `broad white facial stripe`

### Source 차분 (2개)

**source[0] — sitting pose**

A fluffy orange-and-white dog with upright ears and a broad white facial stripe outdoors

**source[1] — outdoor setting**

A fluffy orange-and-white dog with upright ears and a broad white facial stripe sitting

### Target 차분 (1개)

**target[0] — wearing a red hat**

a dog

모델: `gpt-5.6-sol` · prompt: `joint_contrast_v5` · API attempts: 1


<a id="case-010-13"></a>
## 10. 010-13 — dog

- Reference: `dreambench/dog`
- 결과: **PASS**
- 알려진 사례: 추가 benchmark 사례

**Source positive — 원문 caption**

> A fluffy orange-and-white dog with upright ears and a broad white facial stripe sitting outdoors

**Target positive — 생성에도 그대로 전달하는 원문**

> a dog wearing a black top hat and a monocle

**A가 전달한 기본 preserve (identity 별도 보호)**

["fluffy orange-and-white", "with upright ears and a broad white facial stripe"]

**Source 보존 정보**

`dog`, `fluffy orange-and-white`, `upright ears`, `broad white facial stripe`

### Source 차분 (2개)

**source[0] — sitting posture**

A fluffy orange-and-white dog with upright ears and a broad white facial stripe outdoors

**source[1] — outdoor setting**

A fluffy orange-and-white dog with upright ears and a broad white facial stripe sitting

### Target 차분 (2개)

**target[0] — black top hat**

a dog wearing a monocle

**target[1] — monocle**

a dog wearing a black top hat

모델: `gpt-5.6-sol` · prompt: `joint_contrast_v5` · API attempts: 1


<a id="case-010-21"></a>
## 11. 010-21 — dog

- Reference: `dreambench/dog`
- 결과: **PASS**
- 알려진 사례: 추가 benchmark 사례

**Source positive — 원문 caption**

> A fluffy orange-and-white dog with upright ears and a broad white facial stripe sitting outdoors

**Target positive — 생성에도 그대로 전달하는 원문**

> a purple dog

**A가 전달한 기본 preserve (identity 별도 보호)**

["fluffy orange-and-white", "with upright ears and a broad white facial stripe"]

**Source 보존 정보**

`dog`, `fluffy`, `upright ears`, `broad white facial stripe`

### Source 차분 (3개)

**source[0] — orange-and-white coat color**

A fluffy dog with upright ears and a broad white facial stripe sitting outdoors

**source[1] — sitting pose**

A fluffy orange-and-white dog with upright ears and a broad white facial stripe outdoors

**source[2] — outdoor setting**

A fluffy orange-and-white dog with upright ears and a broad white facial stripe sitting

### Target 차분 (1개)

**target[0] — purple coat color**

a dog

모델: `gpt-5.6-sol` · prompt: `joint_contrast_v5` · API attempts: 1


<a id="case-030-06"></a>
## 12. 030-06 — kitten

- Reference: `dreambench_plus/live_subject/animal/00`
- 결과: **PASS**
- 알려진 사례: T13

**Source positive — 원문 caption**

> A gray-and-white kitten with pale green eyes sitting upright on a bed

**Target positive — 생성에도 그대로 전달하는 원문**

> A retro-style digital art of a kitten wearing a tiny bow tie and exploring old books

**A가 전달한 기본 preserve (identity 별도 보호)**

["gray-and-white", "with pale green eyes"]

**Source 보존 정보**

`kitten`, `gray-and-white`, `pale green eyes`

### Source 차분 (2개)

**source[0] — sitting upright posture**

A gray-and-white kitten with pale green eyes on a bed

**source[1] — bed placement**

A gray-and-white kitten with pale green eyes sitting upright

### Target 차분 (4개)

**target[0] — retro style**

Digital art of a kitten wearing a tiny bow tie and exploring old books

**target[1] — retro-style digital art representation**

A kitten wearing a tiny bow tie and exploring old books

**target[2] — wearing a tiny bow tie**

A retro-style digital art of a kitten exploring old books

**target[3] — exploring old books**

A retro-style digital art of a kitten wearing a tiny bow tie

모델: `gpt-5.6-sol` · prompt: `joint_contrast_v5` · API attempts: 1


<a id="case-030-07"></a>
## 13. 030-07 — kitten

- Reference: `dreambench_plus/live_subject/animal/00`
- 결과: **PASS**
- 알려진 사례: 추가 benchmark 사례

**Source positive — 원문 caption**

> A gray-and-white kitten with pale green eyes sitting upright on a bed

**Target positive — 생성에도 그대로 전달하는 원문**

> A kitten floating on a leaf down a gentle stream, surrounded by magical glowing fireflies

**A가 전달한 기본 preserve (identity 별도 보호)**

["gray-and-white", "with pale green eyes"]

**Source 보존 정보**

`kitten`, `gray-and-white`, `with pale green eyes`

### Source 차분 (2개)

**source[0] — sitting upright posture**

A gray-and-white kitten with pale green eyes on a bed

**source[1] — bed placement**

A gray-and-white kitten with pale green eyes sitting upright

### Target 차분 (2개)

**target[0] — floating on a leaf down a gentle stream**

A kitten surrounded by magical glowing fireflies

**target[1] — magical glowing firefly surroundings**

A kitten floating on a leaf down a gentle stream

모델: `gpt-5.6-sol` · prompt: `joint_contrast_v5` · API attempts: 1


<a id="case-051-07"></a>
## 14. 051-07 — Siamese cat

- Reference: `dreambench_plus/live_subject/animal/21`
- 결과: **PASS**
- 알려진 사례: T16

**Source positive — 원문 caption**

> A cream-colored Siamese cat with blue eyes and dark markings on its face, ears, and tail, sitting on a wooden tabletop

**Target positive — 생성에도 그대로 전달하는 원문**

> A Siamese cat walking gracefully along the rim of a giant teacup floating in the sky

**A가 전달한 기본 preserve (identity 별도 보호)**

["cream-colored", "with blue eyes and dark markings on its face, ears, and tail"]

**Source 보존 정보**

`Siamese cat`, `cream-colored`, `blue eyes`, `dark markings on its face, ears, and tail`

### Source 차분 (2개)

**source[0] — sitting pose**

A cream-colored Siamese cat with blue eyes and dark markings on its face, ears, and tail, on a wooden tabletop

**source[1] — wooden tabletop placement**

A cream-colored Siamese cat with blue eyes and dark markings on its face, ears, and tail, sitting

### Target 차분 (4개)

**target[0] — walking gracefully**

A Siamese cat along the rim of a giant teacup floating in the sky

**target[1] — teacup-rim placement and dependent floating setting**

A Siamese cat walking gracefully

**target[2] — giant teacup scale**

A Siamese cat walking gracefully along the rim of a teacup floating in the sky

**target[3] — floating in the sky**

A Siamese cat walking gracefully along the rim of a giant teacup

모델: `gpt-5.6-sol` · prompt: `joint_contrast_v5` · API attempts: 1


<a id="case-075-06"></a>
## 15. 075-06 — space ranger

- Reference: `dreambench_plus/live_subject/human/00`
- 결과: **PASS**
- 알려진 사례: 추가 benchmark 사례

**Source positive — 원문 caption**

> A space ranger in a white and green armored suit with a small orange cat perched on one shoulder, standing on an alien landscape

**Target positive — 생성에도 그대로 전달하는 원문**

> A retro-style drawing of a space ranger with a classic bubble helmet, exploring a vintage-looking space station

**A가 전달한 기본 preserve (identity 별도 보호)**

["in a white and green armored suit", "with a small orange cat perched on one shoulder"]

**Source 보존 정보**

`space ranger`, `in a white and green armored suit`, `with a small orange cat perched on one shoulder`

### Source 차분 (2개)

**source[0] — standing pose**

On an alien landscape, a space ranger in a white and green armored suit with a small orange cat perched on one shoulder

**source[1] — alien landscape placement**

A space ranger in a white and green armored suit with a small orange cat perched on one shoulder, standing

### Target 차분 (4개)

**target[0] — retro-style drawing representation**

A space ranger with a classic bubble helmet, exploring a vintage-looking space station

**target[1] — classic bubble helmet**

A retro-style drawing of a space ranger exploring a vintage-looking space station

**target[2] — exploring the space station**

A retro-style drawing of a space ranger with a classic bubble helmet

**target[3] — vintage-looking station appearance**

A retro-style drawing of a space ranger with a classic bubble helmet, exploring a space station

모델: `gpt-5.6-sol` · prompt: `joint_contrast_v5` · API attempts: 1


<a id="case-075-07"></a>
## 16. 075-07 — space ranger

- Reference: `dreambench_plus/live_subject/human/00`
- 결과: **PASS**
- 알려진 사례: 추가 benchmark 사례

**Source positive — 원문 caption**

> A space ranger in a white and green armored suit with a small orange cat perched on one shoulder, standing on an alien landscape

**Target positive — 생성에도 그대로 전달하는 원문**

> A space ranger floating through a nebula, with stars reflecting in the visor of his helmet

**A가 전달한 기본 preserve (identity 별도 보호)**

["in a white and green armored suit", "with a small orange cat perched on one shoulder"]

**Source 보존 정보**

`space ranger`, `white and green armored suit`, `small orange cat perched on one shoulder`

### Source 차분 (2개)

**source[0] — standing pose**

A space ranger in a white and green armored suit with a small orange cat perched on one shoulder on an alien landscape

**source[1] — alien landscape placement**

A space ranger in a white and green armored suit with a small orange cat perched on one shoulder, standing

### Target 차분 (2개)

**target[0] — floating through a nebula**

A space ranger with stars reflecting in the visor of his helmet

**target[1] — stars reflected in the helmet visor**

A space ranger floating through a nebula

모델: `gpt-5.6-sol` · prompt: `joint_contrast_v5` · API attempts: 1


<a id="case-075-08"></a>
## 17. 075-08 — space ranger

- Reference: `dreambench_plus/live_subject/human/00`
- 결과: **PASS**
- 알려진 사례: T17

**Source positive — 원문 caption**

> A space ranger in a white and green armored suit with a small orange cat perched on one shoulder, standing on an alien landscape

**Target positive — 생성에도 그대로 전달하는 원문**

> A depiction of a space ranger riding a comet, steering it through a field of asteroids

**A가 전달한 기본 preserve (identity 별도 보호)**

["in a white and green armored suit", "with a small orange cat perched on one shoulder"]

**Source 보존 정보**

`space ranger`, `white and green armored suit`, `small orange cat perched on one shoulder`

### Source 차분 (2개)

**source[0] — standing pose**

A space ranger in a white and green armored suit with a small orange cat perched on one shoulder is on an alien landscape

**source[1] — alien landscape setting**

A space ranger in a white and green armored suit with a small orange cat perched on one shoulder, standing

### Target 차분 (2개)

**target[0] — depiction representation**

A space ranger riding a comet, steering it through a field of asteroids

**target[1] — riding and steering the comet**

A depiction of a space ranger

모델: `gpt-5.6-sol` · prompt: `joint_contrast_v5` · API attempts: 1


<a id="case-089-07"></a>
## 18. 089-07 — Spider-Man

- Reference: `dreambench_plus/live_subject/human/14`
- 결과: **PASS**
- 알려진 사례: T20

**Source positive — 원문 caption**

> Spider-Man in a red and dark blue web-patterned suit with large white eye lenses, swinging above a colorful city street

**Target positive — 생성에도 그대로 전달하는 원문**

> Spider-Man weaving a giant web between the moon and Earth, silhouetted against a cosmic backdrop

**A가 전달한 기본 preserve (identity 별도 보호)**

["in a red and dark blue web-patterned suit", "with large white eye lenses"]

**Source 보존 정보**

`Spider-Man`, `red and dark blue web-patterned suit`, `large white eye lenses`

### Source 차분 (2개)

**source[0] — swinging action**

Spider-Man in a red and dark blue web-patterned suit with large white eye lenses above a colorful city street

**source[1] — colorful city street placement**

Spider-Man in a red and dark blue web-patterned suit with large white eye lenses, swinging

### Target 차분 (2개)

**target[0] — weaving a giant web between the moon and Earth**

Spider-Man silhouetted against a cosmic backdrop

**target[1] — silhouetted cosmic backdrop**

Spider-Man weaving a giant web between the moon and Earth

모델: `gpt-5.6-sol` · prompt: `joint_contrast_v5` · API attempts: 1


<a id="case-115-04"></a>
## 19. 115-04 — Magic Cube

- Reference: `dreambench_plus/object/20`
- 결과: **PASS**
- 알려진 사례: 추가 benchmark 사례

**Source positive — 원문 caption**

> A partially twisted Magic Cube with multicolored square tiles and black borders

**Target positive — 생성에도 그대로 전달하는 원문**

> A cartoon of a Magic Cube with animated, expressive eyes trying to solve itself

**A가 전달한 기본 preserve (identity 별도 보호)**

["with multicolored square tiles and black borders"]

**Source 보존 정보**

`Magic Cube`, `multicolored square tiles`, `black borders`

### Source 차분 (1개)

**source[0] — partially twisted configuration**

A Magic Cube with multicolored square tiles and black borders

### Target 차분 (3개)

**target[0] — cartoon representation**

A Magic Cube with animated, expressive eyes trying to solve itself

**target[1] — animated, expressive eyes**

A cartoon of a Magic Cube trying to solve itself

**target[2] — trying to solve itself**

A cartoon of a Magic Cube with animated, expressive eyes

모델: `gpt-5.6-sol` · prompt: `joint_contrast_v5` · API attempts: 1


<a id="case-115-06"></a>
## 20. 115-06 — Magic Cube

- Reference: `dreambench_plus/object/20`
- 결과: **PASS**
- 알려진 사례: T22

**Source positive — 원문 caption**

> A partially twisted Magic Cube with multicolored square tiles and black borders

**Target positive — 생성에도 그대로 전달하는 원문**

> An anime-style drawing of a character intensely focusing on solving a glowing Magic Cube

**A가 전달한 기본 preserve (identity 별도 보호)**

["with multicolored square tiles and black borders"]

**Source 보존 정보**

`Magic Cube`, `multicolored square tiles`, `black borders`

### Source 차분 (1개)

**source[0] — partially twisted state**

A Magic Cube with multicolored square tiles and black borders

### Target 차분 (3개)

**target[0] — anime-style drawing representation**

A character intensely focusing on solving a glowing Magic Cube

**target[1] — character focusing on solving**

An anime-style drawing of a glowing Magic Cube

**target[2] — glowing appearance**

An anime-style drawing of a character intensely focusing on solving a Magic Cube

모델: `gpt-5.6-sol` · prompt: `joint_contrast_v5` · API attempts: 1


<a id="case-115-07"></a>
## 21. 115-07 — Magic Cube

- Reference: `dreambench_plus/object/20`
- 결과: **PASS**
- 알려진 사례: 추가 benchmark 사례

**Source positive — 원문 caption**

> A partially twisted Magic Cube with multicolored square tiles and black borders

**Target positive — 생성에도 그대로 전달하는 원문**

> A Magic Cube morphing into different geometric shapes, each block displaying a different scene from nature

**A가 전달한 기본 preserve (identity 별도 보호)**

["with multicolored square tiles and black borders"]

**Source 보존 정보**

`Magic Cube`, `multicolored square tiles`, `black borders`

### Source 차분 (1개)

**source[0] — partially twisted configuration**

A Magic Cube with multicolored square tiles and black borders

### Target 차분 (2개)

**target[0] — morphing into geometric shapes**

A Magic Cube, each block displaying a different scene from nature

**target[1] — nature scenes displayed on each block**

A Magic Cube morphing into different geometric shapes

모델: `gpt-5.6-sol` · prompt: `joint_contrast_v5` · API attempts: 1


<a id="case-135-01"></a>
## 22. 135-01 — Rolls-Royce hood ornament

- Reference: `dreambench_plus/object/40`
- 결과: **PASS**
- 알려진 사례: 추가 benchmark 사례

**Source positive — 원문 caption**

> A polished silver Rolls-Royce hood ornament depicting a winged female figure mounted above the car grille

**Target positive — 생성에도 그대로 전달하는 원문**

> A photograph of a Rolls-Royce hood ornament with raindrops beading on its polished surface

**A가 전달한 기본 preserve (identity 별도 보호)**

["polished silver", "depicting a winged female figure"]

**Source 보존 정보**

`Rolls-Royce hood ornament`, `polished silver`, `depicting a winged female figure`

### Source 차분 (1개)

**source[0] — mounting above the car grille**

A polished silver Rolls-Royce hood ornament depicting a winged female figure

### Target 차분 (3개)

**target[0] — photograph representation**

A Rolls-Royce hood ornament with raindrops beading on its polished surface

**target[1] — raindrops on the surface**

A photograph of a Rolls-Royce hood ornament with a polished surface

**target[2] — polished surface quality**

A photograph of a Rolls-Royce hood ornament with raindrops beading on its surface

모델: `gpt-5.6-sol` · prompt: `joint_contrast_v5` · API attempts: 1


<a id="case-135-07"></a>
## 23. 135-07 — Rolls-Royce hood ornament

- Reference: `dreambench_plus/object/40`
- 결과: **PASS**
- 알려진 사례: T23

**Source positive — 원문 caption**

> A polished silver Rolls-Royce hood ornament depicting a winged female figure mounted above the car grille

**Target positive — 생성에도 그대로 전달하는 원문**

> A Rolls-Royce hood ornament transformed into a sleek spaceship leading a caravan of vintage cars across a galaxy

**A가 전달한 기본 preserve (identity 별도 보호)**

["polished silver", "depicting a winged female figure"]

**Source 보존 정보**

`Rolls-Royce hood ornament`, `polished silver`

### Source 차분 (2개)

**source[0] — winged female figure depiction**

A polished silver Rolls-Royce hood ornament mounted above the car grille

**source[1] — mounting above the car grille**

A polished silver Rolls-Royce hood ornament depicting a winged female figure

### Target 차분 (5개)

**target[0] — spaceship transformation and subsequent caravan-leading action**

A Rolls-Royce hood ornament

**target[1] — sleek appearance**

A Rolls-Royce hood ornament transformed into a spaceship leading a caravan of vintage cars across a galaxy

**target[2] — leading a caravan across a galaxy**

A Rolls-Royce hood ornament transformed into a sleek spaceship

**target[3] — vintage car age**

A Rolls-Royce hood ornament transformed into a sleek spaceship leading a caravan of cars across a galaxy

**target[4] — galaxy setting**

A Rolls-Royce hood ornament transformed into a sleek spaceship leading a caravan of vintage cars

모델: `gpt-5.6-sol` · prompt: `joint_contrast_v5` · API attempts: 1


<a id="case-161-00"></a>
## 24. 161-00 — pixel art

- Reference: `dreambench_plus/style/01`
- 결과: **PASS**
- 알려진 사례: 추가 benchmark 사례

**Source positive — 원문 caption**

> pixel art of a beige desktop computer with a blue screen and wired mouse

**Target positive — 생성에도 그대로 전달하는 원문**

> A photo of a bustling cityscape, rendered in pixel art

**A가 전달한 기본 preserve (identity 별도 보호)**

[]

**Source 보존 정보**

`pixel art`

### Source 차분 (1개)

**source[0] — depicted desktop computer and accessories**

pixel art

### Target 차분 (2개)

**target[0] — photo representation**

A bustling cityscape, rendered in pixel art

**target[1] — depicted bustling cityscape**

pixel art

모델: `gpt-5.6-sol` · prompt: `joint_contrast_v5` · API attempts: 1


<a id="case-161-07"></a>
## 25. 161-07 — pixel art

- Reference: `dreambench_plus/style/01`
- 결과: **PASS**
- 알려진 사례: cloud development

**Source positive — 원문 caption**

> pixel art of a beige desktop computer with a blue screen and wired mouse

**Target positive — 생성에도 그대로 전달하는 원문**

> A tiny dragon perched on a pixel art cloud

**A가 전달한 기본 preserve (identity 별도 보호)**

[]

**Source 보존 정보**

`pixel art`

### Source 차분 (1개)

**source[0] — depicted desktop computer and accessories**

pixel art

### Target 차분 (1개)

**target[0] — tiny dragon and perching relation**

A pixel art cloud

모델: `gpt-5.6-sol` · prompt: `joint_contrast_v5` · API attempts: 1


<a id="case-161-08"></a>
## 26. 161-08 — pixel art

- Reference: `dreambench_plus/style/01`
- 결과: **PASS**
- 알려진 사례: T01

**Source positive — 원문 caption**

> pixel art of a beige desktop computer with a blue screen and wired mouse

**Target positive — 생성에도 그대로 전달하는 원문**

> A magical portal opening in a pixel art enchanted forest

**A가 전달한 기본 preserve (identity 별도 보호)**

[]

**Source 보존 정보**

`pixel art`

### Source 차분 (4개)

**source[0] — depicted desktop computer and accessories**

pixel art

**source[1] — beige computer color**

pixel art of a desktop computer with a blue screen and wired mouse

**source[2] — blue screen**

pixel art of a beige desktop computer with a wired mouse

**source[3] — wired mouse**

pixel art of a beige desktop computer with a blue screen

### Target 차분 (4개)

**target[0] — magical portal quality**

A portal opening in a pixel art enchanted forest

**target[1] — portal and opening event**

A pixel art enchanted forest

**target[2] — portal opening state**

A magical portal in a pixel art enchanted forest

**target[3] — enchanted forest quality**

A magical portal opening in a pixel art forest

모델: `gpt-5.6-sol` · prompt: `joint_contrast_v5` · API attempts: 1


<a id="case-162-01"></a>
## 27. 162-01 — sunset

- Reference: `dreambench_plus/style/02`
- 결과: **PASS**
- 알려진 사례: 추가 benchmark 사례

**Source positive — 원문 caption**

> sunset with an orange-pink cloudy sky over ocean waves and a sandy beach

**Target positive — 생성에도 그대로 전달하는 원문**

> A photograph of a bustling city skyline illuminated by the fading sunset glow

**A가 전달한 기본 preserve (identity 별도 보호)**

["with an orange-pink cloudy sky", "over ocean waves and a sandy beach"]

**Source 보존 정보**

`sunset`, `with an orange-pink cloudy sky`, `over ocean waves and a sandy beach`

### Source 차분 (0개)

빈 목록: 이쪽 제거 방향 없음.

### Target 차분 (4개)

**target[0] — photograph representation**

A bustling city skyline illuminated by the fading sunset glow

**target[1] — bustling city quality**

A photograph of a city skyline illuminated by the fading sunset glow

**target[2] — city skyline and illumination relation**

the fading sunset glow

**target[3] — fading quality**

A photograph of a bustling city skyline illuminated by the sunset glow

모델: `gpt-5.6-sol` · prompt: `joint_contrast_v5` · API attempts: 1


<a id="case-162-08"></a>
## 28. 162-08 — sunset

- Reference: `dreambench_plus/style/02`
- 결과: **PASS**
- 알려진 사례: T03

**Source positive — 원문 caption**

> sunset with an orange-pink cloudy sky over ocean waves and a sandy beach

**Target positive — 생성에도 그대로 전달하는 원문**

> A giant phoenix soaring across the sky, its feathers echoing the colors of the sunset

**A가 전달한 기본 preserve (identity 별도 보호)**

["with an orange-pink cloudy sky", "over ocean waves and a sandy beach"]

**Source 보존 정보**

`sunset`, `with an orange-pink cloudy sky`, `over ocean waves and a sandy beach`

### Source 차분 (0개)

빈 목록: 이쪽 제거 방향 없음.

### Target 차분 (3개)

**target[0] — giant size**

A phoenix soaring across the sky, its feathers echoing the colors of the sunset

**target[1] — depicted phoenix and its dependent features**

the sunset

**target[2] — soaring across the sky**

A giant phoenix, its feathers echoing the colors of the sunset

모델: `gpt-5.6-sol` · prompt: `joint_contrast_v5` · API attempts: 1


<a id="case-169-03"></a>
## 29. 169-03 — low-poly geometric

- Reference: `dreambench_plus/style/09`
- 결과: **PASS**
- 알려진 사례: 추가 benchmark 사례

**Source positive — 원문 caption**

> a forested cliff island with stylized trees, a stream, and a waterfall in low-poly geometric style

**Target positive — 생성에도 그대로 전달하는 원문**

> A digital illustration of a cat, composed of low-poly geometric patterns

**A가 전달한 기본 preserve (identity 별도 보호)**

["forested cliff island", "with stylized trees, a stream, and a waterfall"]

**Source 보존 정보**

`low-poly geometric`, `style`, `forested cliff island`, `stylized trees`, `stream`, `waterfall`

### Source 차분 (0개)

빈 목록: 이쪽 제거 방향 없음.

### Target 차분 (2개)

**target[0] — digital illustration representation**

A cat composed of low-poly geometric patterns

**target[1] — depicted cat**

A digital illustration composed of low-poly geometric patterns

모델: `gpt-5.6-sol` · prompt: `joint_contrast_v5` · API attempts: 1


<a id="case-169-05"></a>
## 30. 169-05 — low-poly geometric

- Reference: `dreambench_plus/style/09`
- 결과: **PASS**
- 알려진 사례: 추가 benchmark 사례

**Source positive — 원문 caption**

> a forested cliff island with stylized trees, a stream, and a waterfall in low-poly geometric style

**Target positive — 생성에도 그대로 전달하는 원문**

> A fox crafted from sharp, low-poly geometric lines

**A가 전달한 기본 preserve (identity 별도 보호)**

["forested cliff island", "with stylized trees, a stream, and a waterfall"]

**Source 보존 정보**

`low-poly geometric`, `forested cliff island`, `stylized trees`, `a stream`, `a waterfall`

### Source 차분 (0개)

빈 목록: 이쪽 제거 방향 없음.

### Target 차분 (2개)

**target[0] — depicted fox and crafted-from relation**

sharp, low-poly geometric lines

**target[1] — sharp line quality**

A fox crafted from low-poly geometric lines

모델: `gpt-5.6-sol` · prompt: `joint_contrast_v5` · API attempts: 1


<a id="case-178-00"></a>
## 31. 178-00 — pastel

- Reference: `dreambench_plus/style/18`
- 결과: **PASS**
- 알려진 사례: 추가 benchmark 사례

**Source positive — 원문 caption**

> rolling blue and pink dunes beneath a pale moon in pastel

**Target positive — 생성에도 그대로 전달하는 원문**

> A photograph of a serene beach at dusk, rendered in soft pastel tones

**A가 전달한 기본 preserve (identity 별도 보호)**

[]

**Source 보존 정보**

`pastel`

### Source 차분 (6개)

**source[0] — rolling dune form**

blue and pink dunes beneath a pale moon in pastel

**source[1] — blue and pink dune colors**

rolling dunes beneath a pale moon in pastel

**source[2] — depicted dunes**

a pale moon in pastel

**source[3] — beneath relation**

rolling blue and pink dunes and a pale moon in pastel

**source[4] — pale moon color**

rolling blue and pink dunes beneath a moon in pastel

**source[5] — depicted moon**

rolling blue and pink dunes in pastel

### Target 차분 (5개)

**target[0] — photograph representation**

A serene beach at dusk, rendered in soft pastel tones

**target[1] — serene quality**

A photograph of a beach at dusk, rendered in soft pastel tones

**target[2] — depicted beach**

A photograph rendered in soft pastel tones

**target[3] — dusk setting**

A photograph of a serene beach, rendered in soft pastel tones

**target[4] — soft tone quality**

A photograph of a serene beach at dusk, rendered in pastel tones

모델: `gpt-5.6-sol` · prompt: `joint_contrast_v5` · API attempts: 1


<a id="case-178-05"></a>
## 32. 178-05 — pastel

- Reference: `dreambench_plus/style/18`
- 결과: **PASS**
- 알려진 사례: 추가 benchmark 사례

**Source positive — 원문 caption**

> rolling blue and pink dunes beneath a pale moon in pastel

**Target positive — 생성에도 그대로 전달하는 원문**

> A cup of tea on a pastel tablecloth, inviting a moment of relaxation

**A가 전달한 기본 preserve (identity 별도 보호)**

[]

**Source 보존 정보**

`pastel`

### Source 차분 (1개)

**source[0] — depicted dunes and moon scene**

pastel

### Target 차분 (3개)

**target[0] — cup of tea setup and relaxation invitation**

A pastel tablecloth

**target[1] — tea contents**

A cup on a pastel tablecloth, inviting a moment of relaxation

**target[2] — relaxation invitation**

A cup of tea on a pastel tablecloth

모델: `gpt-5.6-sol` · prompt: `joint_contrast_v5` · API attempts: 1


<a id="case-178-07"></a>
## 33. 178-07 — pastel

- Reference: `dreambench_plus/style/18`
- 결과: **PASS**
- 알려진 사례: 추가 benchmark 사례

**Source positive — 원문 caption**

> rolling blue and pink dunes beneath a pale moon in pastel

**Target positive — 생성에도 그대로 전달하는 원문**

> A whimsical forest where the trees have pastel-colored leaves

**A가 전달한 기본 preserve (identity 별도 보호)**

[]

**Source 보존 정보**

`pastel`

### Source 차분 (4개)

**source[0] — rolling dune form**

blue and pink dunes beneath a pale moon in pastel

**source[1] — blue and pink dune colors**

rolling dunes beneath a pale moon in pastel

**source[2] — pale moon appearance**

rolling blue and pink dunes beneath a moon in pastel

**source[3] — moon and beneath relation**

rolling blue and pink dunes in pastel

### Target 차분 (2개)

**target[0] — whimsical quality**

A forest where the trees have pastel-colored leaves

**target[1] — forest and tree context**

pastel-colored leaves

모델: `gpt-5.6-sol` · prompt: `joint_contrast_v5` · API attempts: 1


<a id="case-178-08"></a>
## 34. 178-08 — pastel

- Reference: `dreambench_plus/style/18`
- 결과: **PASS**
- 알려진 사례: 추가 benchmark 사례

**Source positive — 원문 caption**

> rolling blue and pink dunes beneath a pale moon in pastel

**Target positive — 생성에도 그대로 전달하는 원문**

> A dreamy pastel planet with rings made of stardust and light

**A가 전달한 기본 preserve (identity 별도 보호)**

[]

**Source 보존 정보**

`pastel`

### Source 차분 (5개)

**source[0] — rolling dune form**

blue and pink dunes beneath a pale moon in pastel

**source[1] — blue and pink dune colors**

rolling dunes beneath a pale moon in pastel

**source[2] — depicted dunes**

a pale moon in pastel

**source[3] — pale moon appearance**

rolling blue and pink dunes beneath a moon in pastel

**source[4] — moon and overhead placement**

rolling blue and pink dunes in pastel

### Target 차분 (3개)

**target[0] — dreamy atmosphere**

A pastel planet with rings made of stardust and light

**target[1] — planetary rings**

A dreamy pastel planet

**target[2] — stardust and light ring composition**

A dreamy pastel planet with rings

모델: `gpt-5.6-sol` · prompt: `joint_contrast_v5` · API attempts: 1


<a id="case-098-00"></a>
## 35. 098-00 — cherry blossoms

- Reference: `dreambench_plus/object/03`
- 결과: **PASS**
- 알려진 사례: 추가 benchmark 사례

**Source positive — 원문 caption**

> pale pink cherry blossoms clustered on a reddish-brown branch

**Target positive — 생성에도 그대로 전달하는 원문**

> A photo of cherry blossoms gently falling on a serene lake surface

**A가 전달한 기본 preserve (identity 별도 보호)**

["pale pink", "clustered on a reddish-brown branch"]

**Source 보존 정보**

`cherry blossoms`, `pale pink`

### Source 차분 (1개)

**source[0] — clustered on a reddish-brown branch**

pale pink cherry blossoms

### Target 차분 (4개)

**target[0] — photo representation**

cherry blossoms gently falling on a serene lake surface

**target[1] — gentle manner**

A photo of cherry blossoms falling on a serene lake surface

**target[2] — falling action**

A photo of cherry blossoms on a serene lake surface

**target[3] — serene lake surface placement**

A photo of cherry blossoms gently falling

모델: `gpt-5.6-sol` · prompt: `joint_contrast_v5` · API attempts: 1


<a id="case-131-00"></a>
## 36. 131-00 — ceramic bowl

- Reference: `dreambench_plus/object/36`
- 결과: **PASS**
- 알려진 사례: 추가 benchmark 사례

**Source positive — 원문 caption**

> A cream ceramic bowl with a green botanical pattern and short rounded feet

**Target positive — 생성에도 그대로 전달하는 원문**

> A photograph of a ceramic bowl filled with vibrant citrus fruits on a rustic wooden table

**A가 전달한 기본 preserve (identity 별도 보호)**

["cream", "with a green botanical pattern and short rounded feet"]

**Source 보존 정보**

`ceramic bowl`, `cream`, `green botanical pattern`, `short rounded feet`

### Source 차분 (0개)

빈 목록: 이쪽 제거 방향 없음.

### Target 차분 (3개)

**target[0] — photograph representation**

A ceramic bowl filled with vibrant citrus fruits on a rustic wooden table

**target[1] — vibrant citrus fruit contents**

A photograph of a ceramic bowl on a rustic wooden table

**target[2] — rustic wooden table placement**

A photograph of a ceramic bowl filled with vibrant citrus fruits

모델: `gpt-5.6-sol` · prompt: `joint_contrast_v5` · API attempts: 1


<a id="case-166-06"></a>
## 37. 166-06 — volcanic landscape

- Reference: `dreambench_plus/style/06`
- 결과: **PASS**
- 알려진 사례: 추가 benchmark 사례

**Source positive — 원문 caption**

> volcanic landscape with a glowing orange lava flow winding through black rock beneath a gray cloudy sky

**Target positive — 생성에도 그대로 전달하는 원문**

> A night-time scene of a starlit sky above a silhouette of a volcanic mountain, in a volcanic landscape

**A가 전달한 기본 preserve (identity 별도 보호)**

["with a glowing orange lava flow", "black rock"]

**Source 보존 정보**

`volcanic landscape`, `with a glowing orange lava flow`, `black rock`

### Source 차분 (2개)

**source[0] — lava flow winding through the black rock**

volcanic landscape, with a glowing orange lava flow and black rock, beneath a gray cloudy sky

**source[1] — gray cloudy sky**

volcanic landscape with a glowing orange lava flow winding through black rock

### Target 차분 (4개)

**target[0] — night-time setting**

A scene of a starlit sky above a silhouette of a volcanic mountain, in a volcanic landscape

**target[1] — starlit sky**

A night-time scene of a silhouette of a volcanic mountain in a volcanic landscape

**target[2] — sky-above-mountain arrangement**

A night-time scene of a starlit sky and a silhouette of a volcanic mountain, in a volcanic landscape

**target[3] — silhouetted volcanic mountain**

A night-time scene of a starlit sky in a volcanic landscape

모델: `gpt-5.6-sol` · prompt: `joint_contrast_v5` · API attempts: 1


<a id="case-173-07"></a>
## 38. 173-07 — cosmic nebula

- Reference: `dreambench_plus/style/13`
- 결과: **PASS**
- 알려진 사례: 추가 benchmark 사례

**Source positive — 원문 caption**

> A cosmic nebula with bright turquoise and blue gas clouds, dark dust lanes, and glowing golden edges amid a dense star field

**Target positive — 생성에도 그대로 전달하는 원문**

> A mythical phoenix rising from a cosmic nebula

**A가 전달한 기본 preserve (identity 별도 보호)**

["with bright turquoise and blue gas clouds", "dark dust lanes", "glowing golden edges"]

**Source 보존 정보**

`cosmic nebula`, `with bright turquoise and blue gas clouds`, `dark dust lanes`, `glowing golden edges`

### Source 차분 (1개)

**source[0] — dense star field setting**

A cosmic nebula with bright turquoise and blue gas clouds, dark dust lanes, and glowing golden edges

### Target 차분 (2개)

**target[0] — mythical nature**

A phoenix rising from a cosmic nebula

**target[1] — phoenix and rising relation**

A cosmic nebula

모델: `gpt-5.6-sol` · prompt: `joint_contrast_v5` · API attempts: 1


<a id="case-176-07"></a>
## 39. 176-07 — watercolor sketch

- Reference: `dreambench_plus/style/16`
- 결과: **PASS**
- 알려진 사례: 추가 benchmark 사례

**Source positive — 원문 caption**

> a rocky mountain landscape with sparse shrubs in watercolor sketch

**Target positive — 생성에도 그대로 전달하는 원문**

> An ancient tree with glowing leaves in a watercolor sketch style

**A가 전달한 기본 preserve (identity 별도 보호)**

[]

**Source 보존 정보**

`watercolor sketch`

### Source 차분 (1개)

**source[0] — rocky mountain landscape and sparse shrubs**

watercolor sketch

### Target 차분 (1개)

**target[0] — ancient tree and glowing leaves**

watercolor sketch style

모델: `gpt-5.6-sol` · prompt: `joint_contrast_v5` · API attempts: 1


<a id="case-179-08"></a>
## 40. 179-08 — traditional Chinese painting

- Reference: `dreambench_plus/style/19`
- 결과: **PASS**
- 알려진 사례: 추가 benchmark 사례

**Source positive — 원문 caption**

> traditional Chinese painting of a pavilion garden among tall pine trees and misty mountains, rendered with fine lines and muted colors

**Target positive — 생성에도 그대로 전달하는 원문**

> A mythical phoenix soaring above a blooming cherry orchard, in traditional Chinese painting style

**A가 전달한 기본 preserve (identity 별도 보호)**

["with fine lines and muted colors"]

**Source 보존 정보**

`traditional Chinese painting`, `fine lines`, `muted colors`

### Source 차분 (3개)

**source[0] — pavilion garden**

traditional Chinese painting of tall pine trees and misty mountains, rendered with fine lines and muted colors

**source[1] — tall pine trees**

traditional Chinese painting of a pavilion garden among misty mountains, rendered with fine lines and muted colors

**source[2] — misty mountains**

traditional Chinese painting of a pavilion garden among tall pine trees, rendered with fine lines and muted colors

### Target 차분 (2개)

**target[0] — mythical phoenix and its soaring relation**

A blooming cherry orchard, in traditional Chinese painting style

**target[1] — blooming cherry orchard placement**

A mythical phoenix soaring, in traditional Chinese painting style

모델: `gpt-5.6-sol` · prompt: `joint_contrast_v5` · API attempts: 1

