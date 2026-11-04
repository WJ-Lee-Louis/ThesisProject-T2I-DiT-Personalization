# XVerse 결과 보관 및 평가

생성 결과는 `artifacts/<benchmark>/<bundle_id>/images-*.zip`에, metric과 검증 기록은 `results/<benchmark>/<bundle_id>/`에 보관한다. ZIP은 각각 1 GiB 이하이며 Git LFS 대상으로 설정한다. 원본 manifest, 생성 시점 source snapshot, run 정보, 이미지별 metadata와 prompt, 참조 이미지를 함께 보관한다. 반복되는 참조 PNG는 SHA256으로 중복 제거하며 기존 `freecus.operations.artifact_bundle` 도구로 원래 경로를 복원할 수 있다.

2026-11-03 최초 수집에서 DreamBench는 예산 중단으로 **733/750 case**가 완료되었다. 누락 17개는 `026-11`부터 `026-24`, `029-22`부터 `029-24`이다. 사용자 추가 지시에 따라 이 17개를 `louislee-85468`에서 생성 당시 원본 source snapshot과 모델 revision으로 복구하여 **750/750 case**를 평가한다. 기존 733개 압축본은 보존하고, 최종 전체 결과는 bundle ID 뒤에 `-complete-v1`을 붙여 구분한다. 완성된 metadata가 없는 기존 PNG 1개는 최초 bundle의 `incomplete_artifacts.zip`에 별도로 보존한다.

DreamBench++는 `wj-lee-louis` 459개, `woojin716` 216개, `louislee-85468` 675개로 총 **1,350/1,350 case**이며 style 180개를 포함한다. 각 workspace에서 이미 완료된 T4 평가를 수집하고, case 점수를 합친 뒤 subject macro를 재계산한다. 분할 평균을 단순 평균하지 않는다. 각 partition의 원본 metric도 보존한다.

`louislee`에는 새 원격 계산 작업을 제출하지 않는다. 해당 workspace에서는 Volume 조회와 다운로드만 수행한다. 복구 생성은 `louislee-85468`의 A100 80GB 1개 worker에서 최대 1,800초로 제한한다. 기존 733개 저장본을 이 workspace로 옮긴 뒤 복구한 17개와 합쳐 T4에서 평가한다. 평가 함수는 1개 container, 재시도 없음, 최대 1,800초로 제한한다. 이 제한은 실제 과금 잔액 감시가 아니다. 최초 `woojin716` 업로드 시도는 평가 제출 전에 중단했다.

평가는 FreeCus/DreamO/UNO와 동일한 고정된 CLIP/DINO evaluator 및 selected-one reference protocol을 사용한다. XVerse 생성 해상도는 공식 CLI 기본값인 **768×768**이며, 기존 세 방법의 1024×1024와 차이가 있다. 기존 733개를 확대하거나 재생성하지 않는다. 전체 750개 최종 보고서와 최초 733개 보관본의 범위를 구분한다.

`results/reports/`에는 benchmark별 전체 score JSON, 이미지별 CSV, 요약 CSV, audit JSON을 저장한다. `results/transfer_inventory.json`에는 ZIP 목록과 검증 결과가 있다. 검증은 ZIP 및 모든 member SHA256, canonical prompt/reference, 누락·중복, metric의 generated SHA256과 보관 이미지 SHA256 일치 여부를 확인한다.

수집과 검증 재실행:

```powershell
& freecus/.venv-modal311/Scripts/python.exe -X utf8 -m xverse.operations.collect download louislee
& freecus/.venv-modal311/Scripts/python.exe -X utf8 -m xverse.operations.collect download wj-lee-louis
& freecus/.venv-modal311/Scripts/python.exe -X utf8 -m xverse.operations.collect download woojin716
& freecus/.venv-modal311/Scripts/python.exe -X utf8 -m xverse.operations.collect download louislee-85468
& freecus/.venv-modal311/Scripts/python.exe -X utf8 -m xverse.operations.collect bundle dreambench
& freecus/.venv-modal311/Scripts/python.exe -X utf8 -m xverse.operations.collect bundle dreambench_plus
& freecus/.venv-modal311/Scripts/python.exe -X utf8 -m xverse.operations.recover_dreambench
& freecus/.venv-modal311/Scripts/python.exe -X utf8 -m xverse.operations.collect download louislee-85468-dreambench
& freecus/.venv-modal311/Scripts/python.exe -X utf8 -m xverse.operations.collect bundle dreambench_complete
& freecus/.venv-modal311/Scripts/python.exe -X utf8 -m xverse.operations.evaluate_saved --complete --status
& freecus/.venv-modal311/Scripts/python.exe -X utf8 -m xverse.operations.report
```

기존 metric이 있는 DreamBench++는 재평가하지 않는다. 전체 평가 준비는 `evaluate_saved --complete --prepare-only`, 복구 완료 후 제출은 `evaluate_saved --complete --submit-prepared`로 수행한다. receipt가 있으면 중복 제출을 거부한다. commit/pull/push는 수행하지 않는다.
