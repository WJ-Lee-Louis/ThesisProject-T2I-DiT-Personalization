# MOSAIC benchmark on server GPU 0

논문: [MOSAIC: Multi-Subject Personalized Generation via Correspondence-Aware Alignment and Disentanglement](https://arxiv.org/abs/2509.01977). 공식 [코드](https://github.com/bytedance-fanqie-ai/MOSAIC)와 [가중치](https://huggingface.co/ByteDance-FanQie/MOSAIC)를 사용한다. 원본 추론 관련 파일 7개는 수정하지 않고 `upstream/`에 보존했으며 commit과 SHA256은 `UPSTREAM.json`에 기록했다.

## 실행 조건

공개 체크포인트 `subject_512.safetensors`에 맞춰 **512×512**, reference 512, 28 steps, guidance 3.5, adapter `subject` weight 1을 사용한다. 공식 `inference.py`가 실제 호출하는 `src/flux_omini.py`의 `Condition`과 `generate`, `utils.process_image(scale=0.9, white padding)`, reference position delta `[0,-32]`를 그대로 사용한다. 미공개 1024 모델을 가정하거나 양자화·KV cache·추가 segmentation을 적용하지 않는다. 생성 결과는 평가용 lossless PNG로 저장한다.

DreamBench **750장**, DreamBench++ **1,350장(style 180장 포함)**, 합계 **2,100장**이다. 기존 Diptych/FreeCus/DreamO/UNO/XVerse와 같은 manifest bytes, conditioning reference 한 장, 원본 target prompt, case당 seed42 한 장을 유지한다. 공식 single-reference 호출을 사용하며 multi-subject 입력을 새로 만들지 않는다. 원본 코드는 style 전용 adapter를 제공하지 않으므로 style도 공식 subject adapter와 같은 전처리를 적용해 평가한다. 이 정책은 config/provenance에 남기며 style case를 누락하지 않는다.

추론은 BF16, batch1이다. 공식 diffusers 0.32.0, transformers 4.46.3, peft 0.15.0, accelerate 1.1.1을 유지하고, RTX PRO 6000 Blackwell 지원을 위해 PyTorch 2.7.1+cu128을 별도 Python 3.11 환경에 설치한다. 이미지별 global RNG와 CUDA Generator를 seed42로 초기화해 reference VAE sampling도 재현 가능하게 만든다. GPU 1·2는 사용하지 않는다.

## 실행 및 상태

프로젝트 루트에서:

```bash
~/.local/bin/uv venv --python 3.11 mosaic/.venv-local
~/.local/bin/uv pip install --python mosaic/.venv-local/bin/python --index-strategy unsafe-best-match -r mosaic/requirements-local.txt
# FLUX.1-dev 접근 승인을 받은 Hugging Face 로그인은 기존 서버 계정 설정을 재사용
bash mosaic/run_local.sh prepare
mkdir -p mosaic/outputs/local
nohup bash mosaic/run_local.sh run > mosaic/outputs/local/workflow.log 2>&1 &
bash mosaic/run_local.sh status
tail -f mosaic/outputs/local/workflow.log
```

`run_local.sh`는 PCI bus 순서의 GPU 0번만 노출한다. workflow는 입력/가중치 준비 → 두 벤치마크의 범주별 첫 case 시범 → DreamBench 전체 생성·압축·평가 → DreamBench++ 전체 생성·압축·평가 → 통합 보고서 순서로 진행한다. 시범 생성도 동일 run의 최종 case로 재사용한다. 단일 workflow lock을 사용하고 완성된 SHA256 검증 이미지부터 건너뛰므로 같은 명령으로 재개할 수 있다. 코드·모델·설정·의존성이 달라지면 기존 결과와 섞지 않고 중단한다.

서버에서 독립된 background session으로 실행하면 접속 PC 종료나 SSH 종료와 무관하게 유지된다. 서버 자체가 종료되면 다시 같은 명령으로 재개해야 한다. 실패하거나 일부 case가 누락되면 완료로 표시하지 않는다.

## 기존 비교군과 동일한 보관 구조

bundle ID: `local-gpu0-seed42-bf16-v1`.

```text
mosaic/
  config.json, models.lock.json, UPSTREAM.json, requirements-local.txt
  upstream/                             # 고정한 원본 추론 소스
  manifests/                            # canonical seed42 JSONL
  outputs/local/                        # 추적 제외: 로그, 모델 경로, 작업 상태, source snapshot
    <benchmark>/<bundle_id>/             # 재개용 PNG, metadata, text, run.json
  artifacts/<benchmark>/<bundle_id>/
    images-NNN.zip                      # 각 파일 1 GiB 이하, Git LFS
  results/<benchmark>/<bundle_id>/
    bundle_index.json, manifest.jsonl, metrics.json
    provenance.json, verification.json, generation_source.zip
    environment.freeze.txt, evaluation_plan.json
  results/reports/                       # 이미지별 CSV, scores JSON, summary CSV, audit
  results/transfer_inventory.json
```

기존 `freecus.operations.artifact_bundle`을 재사용해 원본 참조, 생성 PNG, prompt, metadata, run identity를 ZIP에 보관한다. 동일 참조 PNG는 SHA256으로 중복 제거하고 경로 복원 index를 기록한다. ZIP 및 모든 멤버를 다시 읽어 SHA256을 검증하며, 이미지 해시와 metric score의 연결도 확인한다. 모델 가중치·venv·로그는 Git에 추가하지 않는다.

metric은 변경하지 않은 기존 evaluator의 **CLIP ViT-B/32, DINOv1 vits16, 원본 target prompt, selected conditioning reference, subject macro**를 사용한다. Mosaic 전용 validator가 512×512 출력과 seed/case/reference/fingerprint/완전성을 확인한다. 다른 방법의 생성 코드·결과·평가 소스는 변경하지 않는다. 출력 해상도와 라이브러리 버전 차이는 별도로 기록한다. 기존 논문 표를 자동으로 덮어쓰지 않으며 Git commit/push도 이 실행에 포함하지 않는다.
