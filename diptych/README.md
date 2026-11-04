# Diptych Prompting benchmark

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


