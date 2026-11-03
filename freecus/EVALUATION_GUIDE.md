# FreeCus 생성 결과 수집 및 공통 metric 평가

**2026-11-01 완료:** 두 원본 ZIP 다운로드/SHA-256 검증/압축 해제, 로컬 lane 병합 및 750/1,170장 전체 검증, 두 계정 T4 평가, 점수 JSON 수집과 기존 방법의 동일 subset 비교를 완료했다. 아래 명령은 수행한 작업의 재현 절차다. 현재 결과를 확인하려면 `outputs/metrics/comparison_summary.csv`, `comparison.json`, 각 벤치마크 `*_scores.json`을 열면 된다. 평가를 새로 중복 제출할 필요는 없다.

## 평가 범위

| 벤치마크 | 참조 수 | 평가 이미지 | 선택 규칙 |
|---|---:|---:|---|
| DreamBench | 30 | 750 | 전체 |
| DreamBench++ non-style | 130 | 1,170 | style 참조 20개 × prompt 9개 = 180개 case 전체 제외 |

DreamBench++는 animal 405, human 180, object 585개 case다. 이미 생성된 style 18장도 평가에서 제외한다. 점수·실패 여부에 따라 개별 이미지를 골라 제외하지 않는다. 기존 baseline과 source_target_ssg에도 같은 case 목록을 적용한다. 이 결과는 DreamBench++ 전체 1,350장 점수가 아닌 **non-style subset**으로 표기한다.

모든 명령은 프로젝트 루트 PowerShell에서 실행한다. Modal 명령은 별도 Python 3.11 client, 로컬 무결성 검증은 기존 Python 3.10 환경을 사용한다. profile은 명령에서 벤치마크별로 자동 지정되며 기본 Modal profile은 바꾸지 않는다.

## 1. 원본 결과 다운로드

```powershell
freecus/.venv-modal311/Scripts/python.exe -X utf8 -m freecus.operations.collect_results --benchmark dreambench
freecus/.venv-modal311/Scripts/python.exe -X utf8 -m freecus.operations.collect_results --benchmark dreambench_plus
```

CPU 원격 함수가 결과를 ZIP으로 묶고 로컬에서 SHA-256과 파일 크기를 검증해 해제한다. 이미지 재생성은 하지 않는다. PNG·원문 prompt·caption·metadata·run.json·manifest·모델 revision·workflow 기록을 보관하며, 모델 weight는 다운로드하지 않는다. 생성된 style 18장도 원본 보관본에는 남는다.

이번 원본 묶음:

- DreamBench: 750 generated PNG, ZIP 2,901,909,391 bytes, 약 2.70 GiB.
- DreamBench++: 1,188 generated PNG, ZIP 2,073,909,444 bytes, 약 1.93 GiB.

다운로드/해제 위치:

```text
freecus/outputs/modal/collected/results/477fc2c07d3451850b37b400/lane-{0,1}/
freecus/outputs/modal-method-comparison-dbpp/collected/results/0fa5460f44b4b8672480a187/lane-{0,1}/
```

`tgt_image`가 생성 이미지, `src_image`는 참조 이미지 사본이다. ZIP과 `export-verified.json`은 각 `collected/`에 남는다. 같은 명령을 다시 실행하면 이미 해시가 맞는 ZIP은 전송하지 않는다. 다운로드 중 로컬 PC를 종료하면 로컬 전송이 중단된다. 원격 생성 결과는 Volume에 그대로 남는다.

## 2. 두 GPU의 결과를 검증·병합

```powershell
freecus/.venv/Scripts/python.exe -X utf8 -m freecus.operations.assemble_run --benchmark dreambench
freecus/.venv/Scripts/python.exe -X utf8 -m freecus.operations.assemble_run --benchmark dreambench_plus
```

lane별 동일 run fingerprint/config/manifest/library/GPU 환경, case 중복·누락, metadata와 artifact hash, reference·원문 prompt·seed 42·1024×1024를 검사한다. 원본 보관본은 유지하고 아래에 평가용 사본을 만든다. DreamBench++ 평가용 사본에는 non-style 1,170장만 포함된다.

```text
freecus/outputs/evaluation_inputs/dreambench/
freecus/outputs/evaluation_inputs/dreambench_plus_no_style/
```

각 폴더의 `assembly.json`에 검증 결과와 제외 case/reference 목록을 보존한다. 750/1,170장 중 하나라도 누락·손상되면 평가를 진행하지 않는다. 원래 1,350개 manifest를 그대로 검증하고 subset 선택을 따로 기록하므로 생성 당시 run identity를 바꾸지 않는다.

## 3. 권장: 기존 실험과 같은 T4 평가 환경에서 실행

기존 `scores_A.json`/`scores_B.json`은 T4, torch 2.7.1, torchvision 0.22.1, numpy 1.26.4, Pillow 11.3.0, OpenAI CLIP 1.0 환경이었다. 아래 평가 함수는 같은 GPU 종류·기록된 패키지 버전을 사용하고 CLIP Git revision과 DINO revision을 기존 evaluator에 맞춰 고정한다. CUDA driver 등 환경 정보는 실제 실행 결과에도 기록한다. 서로 다른 설치 환경의 bitwise 동일성을 보장하는 것은 아니다.

평가 GPU는 **벤치마크별 T4 1개**다. 생성용 FLUX/Qwen/BiRefNet은 로드하지 않는다. 요청한 metric 계산을 위해 두 평가 작업의 배포·접수를 진행했다. 아래 명령은 새 작업을 접수하므로 현재 receipt가 있으면 중복 실행하지 않고 `--status`를 사용한다.

```powershell
freecus/.venv-modal311/Scripts/python.exe -X utf8 -m freecus.operations.evaluate_modal --benchmark dreambench
freecus/.venv-modal311/Scripts/python.exe -X utf8 -m freecus.operations.evaluate_modal --benchmark dreambench_plus
```

각 벤치마크는 기존 생성 계정과 Volume을 사용한다. 원격 저장 결과를 직접 읽어 검증하므로 다운로드한 이미지를 다시 업로드할 필요가 없다. 접수 call ID 출력 이후에는 로컬 PC를 종료해도 평가가 계속된다. 동일 작업 중복 제출은 receipt로 차단한다. 실패 시 자동 유료 재시도는 하지 않는다.

완료 확인 및 점수 JSON 다운로드:

```powershell
freecus/.venv-modal311/Scripts/python.exe -X utf8 -m freecus.operations.evaluate_modal --benchmark dreambench --status
freecus/.venv-modal311/Scripts/python.exe -X utf8 -m freecus.operations.evaluate_modal --benchmark dreambench_plus --status
```

완료 시 아래 JSON으로 저장한다.

```text
freecus/outputs/metrics/dreambench_scores.json
freecus/outputs/metrics/dreambench_plus_no_style_scores.json
```

원격 평가 상태와 최종 성공 여부는 `evaluation.json`의 call ID 및 `--status`로 확인한다. 배포·접수와 metric 전체 완료는 구분한다.

## 4. 대안: 로컬 CPU 평가

로컬 FreeCus 환경은 CPU torch이므로 `--device cpu`를 명시한다. 위 T4 경로와 이 경로 중 하나를 선택한다. CPU는 시간이 더 걸리며 GPU의 FP16 CLIP 계산과 작은 수치 차이가 가능하므로 기존 T4 점수와의 엄격한 비교에는 위 경로를 권장한다. 모델 weight가 없다면 최초 실행 시 CLIP/DINO weight를 다운로드한다.

```powershell
freecus/.venv/Scripts/python.exe -X utf8 -m freecus.evaluate --manifest freecus/manifests/dreambench_project_seed42.jsonl --data-root . --run freecus/outputs/evaluation_inputs/dreambench --output freecus/outputs/metrics/dreambench_scores.json --device cpu --batch-size 16

freecus/.venv/Scripts/python.exe -X utf8 -m freecus.evaluate --manifest freecus/manifests/dreambench_plus_project_seed42.jsonl --data-root . --run freecus/outputs/evaluation_inputs/dreambench_plus_no_style --output freecus/outputs/metrics/dreambench_plus_no_style_scores.json --selection dbpp-no-style --device cpu --batch-size 16
```

`--validate-only`를 붙이면 모델 없이 전체 artifact를 검증하며, `--dry-run`은 파일 존재 수만 확인한다. 로컬 평가에는 batch별 `.checkpoint.json`, `.plan.json`, 최종 점수 JSON이 생성된다. 같은 최종 점수 경로의 덮어쓰기는 거부한다. 중단 후 최종 JSON이 아직 없으면 같은 명령으로 checkpoint부터 재개할 수 있다.

## 5. 점수와 비교표

평가 지표는 기존 해시 보존 evaluator의 `uno-selected-reference-v1`이다.

- CLIP-T: 생성 이미지와 **원문 target prompt**. Qwen이 확장한 prompt로 평가하지 않는다.
- CLIP-I: 생성 이미지 전체와 실제 conditioning에 사용한 reference 한 장.
- DINO-I: 동일 이미지 쌍, 기존 **DINOv1 vits16**.
- 원본 PNG를 사용하고 기존 CLIP/DINO 전처리와 scale 1을 유지한다.
- case별 raw score, reference별 macro, case별 micro, category별 집계, class macro를 저장한다. 주 비교값은 기존과 같은 `subject_macro`다.

기존 baseline·우리 방법은 재생성하거나 embedding을 다시 계산할 필요 없이 저장된 case별 점수를 같은 750/1,170개 case로 재집계할 수 있다. prompt/reference hash/metric protocol을 확인하며 저장한다.

```powershell
# 이미 생성한 비교 대상 점수 집합. 같은 출력이 있으면 다시 실행하지 않는다.
freecus/.venv/Scripts/python.exe -X utf8 -m freecus.operations.compare_metrics --prepare-only

# FreeCus 두 점수 파일을 받은 뒤 실행한다.
freecus/.venv/Scripts/python.exe -X utf8 -m freecus.operations.compare_metrics
```

`freecus/outputs/metrics/existing_matched_cohorts.json`은 이미 준비한 기존 방법의 동일 subset 점수다. 최종 `comparison.json`에는 세 조건의 점수와 `ours_minus_freecus_subject_macro`를 저장한다. CLIP-T 개선과 identity 지표 CLIP-I/DINO-I의 변화를 함께 확인한다. subset·GPU·라이브러리 환경·실패 이력은 결과와 함께 보고한다.

최종 비교는 완료되어 `comparison_summary.csv`(6개 benchmark/condition 조합)와 `comparison_case_scores.csv`(세 방법의 5,760개 case 점수)도 저장했다. CSV는 Excel에서 열 수 있는 UTF-8 BOM 형식이다. source_target_ssg가 우리 방법, baseline이 기존 InstantX baseline, freecus가 이번 FreeCus다.

이번 결과는 1024×1024·seed 42·단일 reference·1장/prompt의 공통 방법 비교다. FreeCus 논문의 512×512·4장/pair·SAM 평가 점수 직접 재현과는 구분한다.
