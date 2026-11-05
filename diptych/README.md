# Diptych Prompting benchmark

## 로컬 GPU 0 실행 (2026-11-04)

로컬 실행은 `run_local.sh`로 물리 GPU 0번만 노출한다. RTX PRO 6000 Blackwell 지원을 위해 별도 Python 3.11 환경과 PyTorch 2.7.1+cu128을 사용한다. Modal용 설정·예산은 로컬 실행에 적용하지 않는다. 원본 `config.json`은 유지하고 로컬 설정은 `outputs/local/config.json`에 기록한다. upstream 파일과 모델 revision lock은 그대로 사용한다.

```bash
# 최초 환경 설치 (uv가 설치되어 있어야 함)
~/.local/bin/uv venv --python 3.11 diptych/.venv-local
~/.local/bin/uv pip install --python diptych/.venv-local/bin/python --index-strategy unsafe-best-match -r diptych/requirements-local.txt
# FLUX.1-dev 접근 승인 계정 로그인
diptych/.venv-local/bin/huggingface-cli login
# 입력 확인 및 모델 다운로드; 실제 생성 없이 준비만 수행
bash diptych/run_local.sh prepare
# 750 + 1350장 생성 → 각 벤치마크 ZIP 검증 → 동일 evaluator로 평가 → 통합 보고서
mkdir -p diptych/outputs/local
nohup bash diptych/run_local.sh run > diptych/outputs/local/workflow.log 2>&1 &
# 상태 확인
bash diptych/run_local.sh status
tail -f diptych/outputs/local/workflow.log
```

한 개 workflow만 실행한다. 중단 후 같은 `run` 명령으로 재개하면 SHA256으로 검증된 완료 이미지는 건너뛴다. 코드·설정·의존성이 바뀌면 기존 결과와 섞지 않고 오류로 중단한다. 범주별 첫 case를 먼저 시범 생성하고 동일 run의 최종 이미지로 재사용한다. 실패를 무시하거나 참조·prompt를 임의 변경하지 않는다. SSH 종료 후에도 `nohup` 작업은 유지되지만 서버 종료 시에는 같은 명령으로 재개해야 한다.

DreamBench++ 누락 입력은 공개 `yuangpeng/dreambench_plus` revision `459b06499fad3dac842c4b1f7a053fb3328d91c7`의 ZIP에서 필요한 원본만 복원하고 기존 manifest SHA256과 대조한다. DreamBench 750장과 DreamBench++ 1,350장(style 180장 포함), seed42, selected_one 참조, 768×768 출력 및 기존 Diptych prompt/segmentation/attention 정책을 유지한다.

저장 구조는 기존 비교군과 같다. bundle ID는 `local-gpu0-seed42-bf16-v1`이다.

```text
outputs/local/<benchmark>/<bundle_id>/  # 재개용 PNG·metadata·prompt·run.json
outputs/local/status.json              # preparing/generating/evaluating/complete/failed
artifacts/<benchmark>/<bundle_id>/     # 각각 1 GiB 이하 images-NNN.zip (LFS)
results/<benchmark>/<bundle_id>/       # manifest, bundle_index, metrics, provenance,
                                      # verification, generation_source.zip
results/reports/                       # benchmark별 scores JSON·per_image CSV·요약 CSV·audit
results/transfer_inventory.json        # 검증된 ZIP과 평가 범위 목록
```

전체 이미지의 완전성·SHA256 검증 후에만 평가한다. metric은 기존 고정 evaluator의 CLIP ViT-B/32, DINOv1 vits16, 원본 target prompt 및 subject macro 집계를 그대로 사용한다. 환경 버전 차이는 provenance에 남긴다. ZIP 안의 참조 PNG 중복 제거 및 원본 참조 보관도 기존 `artifact_bundle`을 재사용한다. 최종 완료 표시는 두 벤치마크 생성·압축·평가·보고서 작성이 모두 끝난 경우에만 기록한다. Git commit/push는 실행하지 않는다.

아래 내용은 이전 Modal 시범 실행 기록이다.

768×768 출력의 현재 BF16 구현은 peak allocated 약 42.4 GiB, peak reserved 약 44.7 GiB를 측정했다. CUDA context 등을 포함한 장치 전체 메모리는 별도이므로 48GB 카드에 그대로 적합하다고 보장하지 않는다. 메모리 효율을 우선하면 SAM/검출 모델을 배경 제거 후 CPU로 이동하고, prompt embedding 생성 후 T5/CLIP text encoder를 offload하는 48GB 구성부터 검증하는 것이 합리적이다. 이때 peak 약 35~40 GiB는 상주 모델 용량으로 계산한 추정이며 아직 검증하지 않았다. 24GB에서는 추가 layer/sequential offload가 필요할 수 있어 현재 구현의 검증된 최소 VRAM으로 제시하지 않는다. 속도·비용을 배제한 무수정 안정 기준은 80GB다. offload 실험이나 추가 GPU 실행은 아직 수행하지 않았다.

공식 [DiptychPrompting](https://github.com/chaehunshin/DiptychPrompting)의 추론 필수 파일 6개를 원본 그대로 `upstream/`에 보존했다. commit과 파일별 SHA256은 `UPSTREAM.json`에 기록한다. 모델 가중치와 전체 Git 이력은 복제하지 않는다.

논문 권장 출력 패널 **768×768**, 공식 border 8px, 내부 캔버스 1568×784를 사용한다. 참조 선정·원본 target prompt·seed42·case당 1장은 기존 FreeCus/DreamO/UNO 비교와 동일하다. DreamBench 750장, DreamBench++는 style을 포함해 1350장을 생성한다. 원본 prompt와 실제 diptych generation prompt는 각각 보존한다. subject에는 공식 segmentation과 attention 1.3, style에는 논문 부록 G의 full-reference/attention 1.0 정책을 적용한다.

Modal workspace `louislee-85468` (profile `method-comparison`), app `diptych-method-comparison`, volume `diptych-runs`를 사용한다. H100 80GB 최대 4개 worker 설정이며 신규 작업 총예산은 $40이다. A100 시범은 80.86초/장, H100 시범은 42.41초/장이었다. H100 DreamBench의 여유분 포함 추정은 약 $49로 현재 예산을 넘어 전체 생성은 `budget_review_required` 상태다. 이전 시범을 위한 $0.5를 제외해 H100 admission은 $39.5이며 추가 예산 결정 전에는 전체 생성을 제출하지 않는다. PC 종료 후에도 제출한 원격 작업은 유지된다. 기본 모델 캐시는 기존 workspace의 FLUX snapshot을 재사용한다.

로컬 상태 확인:

```powershell
& freecus/.venv-modal311/Scripts/python.exe -X utf8 -m comparison_candidates.client status --method diptych
& freecus/.venv-modal311/Scripts/python.exe -X utf8 -m comparison_candidates.client logs --method diptych
```

실행 receipt와 source snapshot은 `outputs/`, 최종 metric을 상태 명령으로 수신하면 `results/<benchmark>/<bundle_id>/metrics.json`에 저장한다. 생성 이미지는 원격 `diptych-runs`의 `results/<bundle_id>/H100/lane-*/`에 보관한다. 전체/누락 없는 평가만 수행하며 예산 또는 오류로 중단하면 완료로 표시하지 않는다. 이미지 다운로드와 LFS 추적은 별도 후속 작업이다. 모델·metric revision과 검증 설명은 [공통 안내](../comparison_candidates/README.md)를 참고한다.


