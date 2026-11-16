# VIP-Adapter algorithm and preliminaries

논문 삽입용 두 PNG와 편집 가능한 LaTeX 원문을 관리합니다.

- `vip_adapter_algorithm.png`: LLM 기반 source/target contrast 구성, joint SVD
  projection, image-token 생성, single-stream gate, frozen FLUX sampling.
- `vip_adapter_preliminaries.png`: 같은 알고리즘을 정의하는 notation과 핵심 수식.
- `vip_adapter_algorithm.tex`, `vip_adapter_preliminaries.tex`: 단독 컴파일 문서.
- `*_body.tex`: 논문 본문에 삽입할 내용. 공유 패키지/매크로는
  `paper_preamble.tex`에 있습니다. 기존 논문에 통합할 때 geometry 설정까지
  복사할 필요는 없습니다.
- `build_algorithm_figures.py`: 실제 LaTeX를 컴파일하여 여백을 정리한
  400-DPI PNG를 생성합니다. PDF와 컴파일 부산물은 임시 폴더에만 생성합니다.
- `figure_audit.json`: 원문·관련 코드 해시, 조판 폰트, PNG 크기와 검증 결과.

## 폰트와 Overleaf

본문은 Times New Roman입니다. 수식은 Times 계열의 OpenType 수학 폰트인
TeX Gyre Termes Math를 사용합니다. 수식 전용 글리프가 필요하므로 본문용
Times New Roman과 수학 폰트를 구분했습니다.

Overleaf에서는 XeLaTeX 또는 LuaLaTeX를 선택합니다. Times New Roman이 없는
환경에서는 공유 preamble이 본문을 Times 호환 TeX Gyre Termes로 대체합니다.
정확히 Times New Roman을 유지하려면 사용 권한이 있는 해당 폰트를 환경에
설정해야 합니다. 생성된 PNG는 이 Mac의 실제 Times New Roman을 사용합니다.

```sh
python -m pip install -r requirements.txt
python build_algorithm_figures.py --tectonic /path/to/tectonic
```

XeLaTeX가 PATH에 있으면 `--tectonic` 없이 실행할 수도 있습니다. 논문 본문에서는
`algorithm`/`algpseudocode`와 수학 패키지를 로드하고, 필요한 매크로와 body를
포함합니다. 알고리즘은 학습 루프가 아닌 추론 알고리즘입니다.

## 표기와 구현을 맞춘 결정

1. **Identity term은 iota**, caption은 s, 원문 target prompt는 y로 구별했습니다.
   Identity는 벤치마크의 subject 또는 보존할 style/concept 명칭이 될 수 있습니다.
2. **SigLIP 텍스트 encoder와 FLUX 텍스트 encoder는 다릅니다.** 전자는 제거
   방향 구성에, 후자는 원문 target의 생성 conditioning에 사용합니다.
3. **두 projection은 하나의 joint subspace로 구현됩니다.** Source와 target의
   각 local removal direction을 모두 열로 쌓고 SVD 기저를 한 번 만듭니다.
   Source projection 후 target projection을 순차 적용하는 알고리즘이 아닙니다.
4. **Projection은 image-token projector 앞**의 SigLIP visual pooled vector에서
   수행합니다. 각 text feature를 먼저 정규화하고 positive-minus-negative 차이를
   다시 정규화합니다. 합친 방향들의 평균을 취하지 않습니다.
5. **Norm restoration**을 명시했습니다. lambda는 projection strength이며,
   alpha는 IP-conditioning scale입니다. 선택된 Ours 설정은 alpha 0.8,
   lambda 1, rho 0.001, 24 steps, guidance 3.5입니다.
6. **Single-stream gate는 IP residual의 text 행만 0으로 만듭니다.** Base joint
   attention이나 text encoder를 끄지 않으며, image token을 통한 간접적인 상호
   작용도 제거하지 않습니다. Double-stream의 IP branch는 기존처럼 image 행만
   갱신합니다. 원래 output projection, FFN, modulation, normalization을 유지합니다.
7. **IP query와 base-attention query의 위치를 구분했습니다.** 현재 processor는
   IP-attention을 rotary embedding 적용 전에 계산합니다. 수식의 per-head query/key
   normalization도 원본 IP-Adapter에서 사용하는 연산을 뜻합니다.
8. **원문 target은 재작성하지 않습니다.** 제거 문장은 방향을 계산하는 text
   counterfactual이며 생성 prompt를 대체하지 않습니다.

## 알고리즘에 생략한 실행 세부 사항

본문 알고리즘은 검증된 semantic-local 정상 경로를 나타냅니다. 실제 코드는 LLM
JSON 검증, reference/model/prompt에 묶인 cache와 호출 trace도 관리합니다. Target이
identity 자체와 충돌하는 경우에는 해당 오류 trace를 보존하고, 명시적인
source-only fallback 경로를 사용할 수 있습니다. 이때 target direction은
사용하지 않습니다. 이는 별도 실행 정책이며 일반적인 local-removal 수식과
혼동하지 않도록 메인 알고리즘에는 넣지 않았습니다.

Zero-safe normalization은 norm이 epsilon 이하이면 zero를 반환합니다. 방향이
없으면 basis가 비어 projection은 아무 성분도 제거하지 않습니다. 투영 후
잔차 norm이 epsilon 이하인 경우 norm restoration도 zero를 반환합니다.

Identity 보호는 **LLM의 contrast 작성 정책**입니다. 제거 방향이 identity의
수학적으로 분리된 축이라고 보장하거나, projection이 identity를 완전히 보존한다는
정리를 가정하지 않습니다. LLM 분해 오류와 embedding의 의미 중첩은 실험으로
검토할 문제입니다. 전체 파이프라인은 학습된 IP-Adapter에 의존하지만 제안한
projection과 gate를 위해 새 parameter learning을 수행하지 않습니다.

Sampling은 시간 감소 순서의 flow Euler step으로 명시했습니다. 알고리즘의
`v_theta`는 원래 FLUX의 timestep/text/guidance conditioning을 포함한 frozen
velocity prediction을 나타냅니다. Packed latent 처리, VAE scale, prompt encoding
구현과 dtype/cache 관리는 원래 pipeline의 내부 연산으로 묶었습니다.

## 코드 대응

- `../Orthogonalization-InstantX-IP-Adapter/orthogonalization_instantx_ip_adapter/semantic_contrasts.py`:
  `SemanticLLM.caption`, `SemanticLLM.contrast`, `prepare_semantic_conditioning`.
- `../Orthogonalization-InstantX-IP-Adapter/orthogonalization_instantx_ip_adapter/contrast_directions.py`:
  `contrast_plan`, `build_contrast_directions`.
- `../Orthogonalization-InstantX-IP-Adapter/orthogonalization_instantx_ip_adapter/orthogonalization.py`:
  `normalize_fp32`, `estimate_orthonormal_basis`, `project_pooled_embedding`.
- `../Orthogonalization-InstantX-IP-Adapter/orthogonalization_instantx_ip_adapter/adapter.py`:
  `_encode_siglip_text_pool`, `_build_text_directions`, `get_image_embeds`.
- `../Orthogonalization-InstantX-IP-Adapter/orthogonalization_instantx_ip_adapter/attention_processor.py`:
  `build_single_stream_ip_row_gate`, `IPAFluxAttnProcessor2_0.__call__`.

알고리즘에 새로운 최적화, 학습 loss 또는 gradient update를 추가하지 않았습니다.
