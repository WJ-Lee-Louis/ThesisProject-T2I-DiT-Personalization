# XVerse benchmark

2026-11-03 최종 보관 결과: DreamBench **750/750장**, DreamBench++ **1,350/1,350장**의 이미지와 CLIP-T/CLIP-I/DINO-I 평가를 저장했다. DreamBench의 누락 17장은 `louislee-85468`에서 원래 source snapshot으로 복구했다. 최종 DreamBench bundle은 `2e73e10891f643f354a846f4-complete-v1`이다. 이미지 ZIP, 이미지별 점수, 재현·검증 기록은 [RESULTS.md](RESULTS.md)와 [요약 CSV](results/reports/metrics_summary.csv)를 참조한다. 아래 예산·진행 상태 설명은 최초 실행 기록이다.

DreamBench++의 최신 실행은 [DBPP_SPLIT.md](DBPP_SPLIT.md)를 따른다. 사용자 추가 지시에 따라 wj-lee-louis 459장/$21, woojin716 216장/$13, louislee-85468 675장/$28로 분리해 제출했다. 아래의 기존 louislee DreamBench 전용 예산 감시는 그대로 유지되며, 그 workspace에 DreamBench++를 추가하지 않는다.

현재 실행 제약은 사용자 정정에 따라 **잔여 예산 약 $30, DreamBench만 진행**이다. 기존 $37.5 parent workflow는 중단하고 정상 GPU worker는 유지한 채 `xverse-dreambench-30usd-guard`가 관리한다. generation 추정 상한 $26.5와 metric·감시용 여유 $3.5를 배정했으며, DreamBench++는 제출하지 않는다. 실패한 lane 2는 동일한 고정 source/settings로 한 번 재시도했다. 새 guard receipt는 `outputs/dreambench-30usd-guard.json`, 상태는 `outputs/budget-guard-status.json`에 기록한다. 아래 status 명령은 guard를 우선 조회한다. 이전 예산·workflow 설명은 당시 기록이며 이 제약이 우선한다.

전체 생성 초반 28개 신규 샘플의 평균 37.32초/장으로 단순 생성비는 약 $24.14/750장이다. 초기화·재시도·metric을 포함하면 약 $26~29를 예상하지만, 시범 표본과 실제 전체 속도가 달라질 수 있어 guard가 비용을 감시한다. 이 추정은 Modal 청구 API 잔액 조회가 아니다.

공식 [XVerse](https://github.com/bytedance/XVerse)의 추론 필수 소스·설정·라이선스 18개를 원본 그대로 `upstream/`에 보존했다. commit과 파일별 SHA256은 `UPSTREAM.json`에 기록한다. 모델 가중치·학습 데이터·전체 Git 이력은 복제하지 않는다.

공식 CLI 기본 출력인 **768×768**을 사용한다. 이것을 논문이 명시한 DreamBench 평가 해상도라고 단정하지 않는다. 참조 선정·원본 target prompt·seed42·case당 1장은 기존 FreeCus/DreamO/UNO 비교와 동일하다. DreamBench 750장, DreamBench++는 style을 포함해 1350장을 생성한다.

공식 pretrained modulation adapter와 LoRA를 로드한다. 공식 single-sample `generate_image` 함수를 그대로 추출하여 사용하며, 추론에 사용하지 않는 Gradio/face 평가 도구 import는 제외한다. 28 steps, conditioning 256, IP weight 5/human ID weight 3, latent scale 0.85, VAE scale 1.3을 사용한다. CLI 기본 int8-quanto 대신 공식 unquantized BF16 branch를 선택하고 config에 기록한다. conditioning caption은 원본 target prompt에 존재하는 subject_word의 정확한 표기를 사용하며, 평가 prompt는 변경하지 않는다.

Modal workspace `louislee` (profile `method-comparison-dbpp`), app `xverse-method-comparison`, volume `xverse-runs`를 사용한다. A100 80GB 최대 4개 worker、신규 작업 총예산 $40이다. A100 시범 31.81초/장과 H100 시범 31.16초/장의 차이가 작아 A100을 선택했다. 이전 시범 몫 $2.5를 제외한 현재 admission은 $37.5다. DreamBench 750장 후 T4 metric을 진행하며 DreamBench++는 잔여 예산 검사 후 진행한다. PC 종료 후에도 원격으로 유지된다.

로컬 상태 확인:

```powershell
& freecus/.venv-modal311/Scripts/python.exe -X utf8 -m comparison_candidates.client status --method xverse
& freecus/.venv-modal311/Scripts/python.exe -X utf8 -m comparison_candidates.client logs --method xverse
```

실행 receipt와 source snapshot은 `outputs/`, 최종 metric을 상태 명령으로 수신하면 `results/<benchmark>/<bundle_id>/metrics.json`에 저장한다. 생성 이미지는 원격 `xverse-runs`의 `results/<bundle_id>/A100-80GB/lane-*/`에 보관한다. 전체/누락 없는 평가만 수행하며 예산 또는 오류로 중단하면 완료로 표시하지 않는다. 이미지 다운로드와 LFS 추적은 별도 후속 작업이다. 모델·metric revision과 검증 설명은 [공통 안내](../comparison_candidates/README.md)를 참고한다.

