# 지표 분석 가이드

## 핵심 해석 원칙

프롬프트 제어력의 주지표로는 dependency-aware DPG를 사용하고, IP-Sim과
AES는 각각 주체 정체성 보존과 일반 화질의 안전 지표로 해석한다. 세 지표는
척도와 분산, 의미가 서로 다르므로 원점수를 단순 평균한 하나의 종합점수로
순위를 정하지 않는다.

이 문서의 두 조건별 표에서 `DPG`로 표시한 값은 subset과 custom 모두
`dpg_with_dependency`이다. Custom DPG도 dependency-free가 아니다. 평가기는
비교·진단을 위해 `dpg_without_dependency`도 함께 계산해 원점수 JSON과 집계
CSV에 보존하지만, 이 값은 주 결과표의 DPG로 사용하지 않는다.

### 두 DPG 출력의 의미

- `dpg_with_dependency`: 선행 질문이 실패하면 그 질문에 의존하는 후속
  질문을 유효한 성공으로 인정하지 않는다. 이번 분석의 주지표다.
- `dpg_without_dependency`: 각 VQA 질문의 yes/no 결과를 선행 질문의 성공
  여부와 관계없이 평균한 보조 진단값이다. 여기서 `without_dependency`는
  Custom 벤치마크의 종류를 뜻하지 않는다.

Custom DSG에도 다음과 같이 dependency가 명시되어 있다.

- O+A: action/state 질문은 subject 존재 질문에 의존한다.
- O+B: subject-background 관계 질문은 subject와 background의 존재 질문에
  의존한다.
- O+A+B: action/state는 subject 존재에 의존하고, spatial relation은 subject와
  background 존재 모두에 의존한다.

따라서 Custom의 주 분석값도 dependency-aware DPG로 해석해야 한다. 특정
이미지에서 두 값이 같을 수는 있지만, 이는 그 이미지에서 parent 질문이 모두
통과했거나 dependency 적용이 최종 평균을 바꾸지 않았기 때문이지 Custom
평가가 dependency-free이기 때문은 아니다.

## 조건별 결과

### 공식 벤치마크 기반 subset

| 조건 | DPG | ΔDPG | IP-Sim | ΔIP | AES | ΔAES |
|---|---:|---:|---:|---:|---:|---:|
| 기준선 (`baseline`) | 85.47 | — | 76.62 | — | 64.37 | — |
| SSG | 91.85 | +6.38 | 75.81 | -0.81 | 65.09 | +0.72 |
| source projection + SSG | 91.85 | +6.38 | 75.81 | -0.81 | 64.95 | +0.58 |
| target projection + SSG | 94.44 | +8.97 | 75.63 | -0.99 | 65.19 | +0.82 |
| ST-matched projection + SSG | **95.34** | **+9.87** | 75.67 | -0.95 | 64.84 | +0.47 |
| ST-all projection + SSG | 95.00 | +9.53 | 75.38 | -1.24 | 64.85 | +0.48 |

임시 선택 기준을 `ΔIP >= -1.0` 및 `ΔAES >= 0`으로 두면, subset에서는
ST-matched가 가장 강한 선택이다. target projection은 DPG가 약간 낮지만 AES가
가장 높다. ST-all은 거의 같은 DPG 향상을 보이지만 IP-Sim 하락폭이 1점을
넘으므로 정체성 보존 기준을 더 엄격하게 적용할 때는 제외된다.

### Custom 벤치마크

| 조건 | DPG | ΔDPG | IP-Sim | ΔIP | AES | ΔAES |
|---|---:|---:|---:|---:|---:|---:|
| 기준선 (`baseline`) | 71.94 | — | 74.41 | — | 65.78 | — |
| SSG | 84.11 | +12.17 | **75.37** | **+0.96** | 66.16 | +0.37 |
| source projection + SSG | 85.13 | +13.19 | 74.49 | +0.08 | 66.19 | +0.40 |
| target projection + SSG | 91.24 | +19.30 | 74.84 | +0.43 | 66.22 | +0.43 |
| ST-matched projection + SSG | 91.69 | +19.74 | 74.73 | +0.32 | 66.33 | +0.54 |
| ST-all projection + SSG | **92.98** | **+21.04** | 74.65 | +0.25 | **66.56** | **+0.77** |

Custom 벤치마크에서는 ST-all이 IP-Sim 하락 없이 제어력과 심미성 모두에서
가장 강하다. SSG의 IP-Sim은 가장 높지만, 세 projection 조건보다 DPG가 크게
낮다.

## 권장 분석 순서

1. `summary_by_condition.csv`에서 조건별 전체 trade-off를 확인한다.
2. `paired_delta_vs_baseline.csv`를 주 비교표로 사용한다. 주체와 프롬프트의
   난이도가 섞이지 않도록 같은 record와 seed끼리 기준선과 실험 조건을
   대응시킨 차이를 사용한다.
3. DPG를 composition과 atomic component별로 분해한다. Custom 벤치마크는
   composition마다 30장으로 균형이 맞아 이 비교에 적합하다.
4. `summary_by_prompt.csv`와 이미지별 paired delta를 사용해 평균만으로는
   드러나지 않는 최대 개선·최대 악화 사례를 찾는다.
5. 최종 보고서에는 reference 또는 record를 군집 단위로 재표집한
   cluster-bootstrap 95% 신뢰구간을 추가한다. 같은 reference와 prompt를
   공유하는 이미지들은 완전히 독립적이지 않으므로 이미지 단위 신뢰구간은
   실제보다 지나치게 좁아질 수 있다.
6. 대표적인 향상·악화 사례는 생성 이미지와 `evaluation/ip/gen/...`의
   Florence2/SAM2 mask 및 foreground crop을 함께 확인한다.

## 해석 시 주의사항

- 주 결과에는 subset과 custom 모두 dependency-aware DPG를 사용한다.
  `dpg_without_dependency`는 compositional dependency를 적용하지 않았을 때의
  보조 진단값이지 별도의 Custom DPG나 대체 주지표가 아니다.
- subset은 composition별 수가 불균형하다. 조건당 O+A 4장, O+B 8장,
  O+attribute 20장, O+A+B 84장이므로 작은 subgroup의 평균을 안정적인 결론으로
  해석하지 않는다.
- Custom 벤치마크의 큰 향상은 공식 기반 subset 및 신뢰구간으로 재확인해야
  한다. Custom은 균형 잡힌 세부 분석에는 적합하지만 외부 비교의 주
  벤치마크는 아니다.
- IP-Sim은 이번 수정 후 Florence2 검출률이 100%이기 때문에 유효하다. 향후
  표에도 검출률을 함께 기록해 segmentation 실패가 정체성 하락으로 오인되지
  않게 한다.
- 하나의 조건을 선택해야 한다면 보정되지 않은 평균점수를 만들기보다
  `ΔIP >= -1.0`, `ΔAES >= 0`을 만족하는 조건 중 DPG를 최대화하는 것처럼
  명시적인 제약식을 먼저 정한다.

## 권장 표와 시각화

- 주 결과표: subset과 custom을 나누어 조건별 DPG, IP-Sim, AES 및 기준선 대비
  paired delta를 표시한다.
- Trade-off 산점도: x축은 ΔIP, y축은 ΔDPG, 점 색상은 ΔAES로 두고 Pareto
  frontier를 표시한다.
- Composition heatmap: 조건과 composition별 paired ΔDPG를 표시한다.
- 정성 비교 grid: DPG 최대 향상 사례, IP 최대 하락 사례, 중앙값 사례에서
  baseline과 ST-matched/ST-all을 나란히 놓고 mask도 함께 표시한다.

현재 계산된 DPG/IP-Sim 최대 향상·하락 Top-5와 각 생성 이미지, baseline,
reference, mask, foreground crop의 경로는 `QUALITATIVE_CASES.md`에서 확인한다.

CSV가 권위 있는 원자료이며, 렌더링한 표 이미지는 CSV에서 생성한 발표용
산출물로 관리한다.
