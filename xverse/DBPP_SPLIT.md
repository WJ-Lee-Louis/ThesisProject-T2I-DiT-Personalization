# DreamBench++: 세 workspace 분할 실행

2026-11-03 사용자 지시: 기존 DreamBench는 `louislee`에서 계속 진행한다. DreamBench++는 `wj-lee-louis`, `woojin716`, `louislee-85468`에서 독립 실행한다. 초기 잔액 $16/$8/$23에 각각 $5까지 추가 지출을 허용했다. 작업 설정의 한도는 $21/$13/$28이며 실제 계정의 결제 제한을 변경하지 않는다.

| Workspace | Profile | 참조 | 이미지 | A100 80GB worker | 실행 상한 |
|---|---|---:|---:|---:|---:|
| wj-lee-louis | wj-lee-louis | 51 | 459 | 2 | $21 |
| woojin716 | woojin716 | 24 | 216 | 1 | $13 |
| louislee-85468 | method-comparison | 75 | 675 | 3 | $28 |
| 합계 | | 150 | 1,350 | 6 | $62 |

배정은 처음의 2:1:2를 고정하지 않고 초기 예산의 고정비 여유를 제외한 비율로 정했다. 추가 $5는 재시도·속도 변동의 여유로 유지한다. 모든 참조의 9개 프롬프트를 같은 workspace와 같은 worker에 묶는다. animal/human/object/style을 분산하며 style 180개 case를 포함한다. 전역 manifest는 1350개를 그대로 보존하고 별도의 assigned_case_ids로 작업 범위만 지정한다. 세 집합의 합집합·교집합과 참조 단위 소유권은 테스트로 확인했다.

공통 출력 768×768, seed42, 원본 target prompt, 참조 1장, 동일 upstream/model revision 및 BF16 설정을 사용한다. 기존 DreamBench의 배포·source snapshot·결과는 수정하지 않고 별도 앱 `xverse-dbpp-split` 및 `xverse-dbpp-split-supervisor`를 세 계정에 배포한다. 결과 volume은 각 workspace의 `xverse-runs`이며 서로 다른 계정의 volume이므로 동일 bundle ID여도 물리적으로 분리되어 있다.

각 workflow는 CPU 모델 준비 → GPU 생성 → 해당 partition T4 metric 순으로 원격 진행한다. PC를 종료해도 유지된다. 호스트 RAM은 32 GiB 요청/64 GiB 상한으로 설정하고, 실행시간 한도는 64 GiB 요율을 적용해 보수적으로 계산한다. 각 예산에서 $1.2는 CPU 준비·감시·metric 여유로 남긴다. 시간 한도로 중단되거나 오류가 나면 이미지를 보존하고 budget_paused/generation_failed로 기록한다. 불완전한 partition을 최종 점수로 표시하지 않는다. 설정 한도는 추정 compute이며 청구 API 기반의 정확한 잔액 감시는 아니다.

DreamBench 본 생성의 평균 약 40초/장을 적용하면 각 partition의 생성 compute는 대략 $15~16/$7~8/$22~23 수준에 초기화와 metric이 더해진다. 이는 category별 속도 및 실제 RAM 사용에 따라 바뀐다. human/style은 전체 실행으로 아직 검증되지 않았으므로 정확한 완료 비용을 보장하지 않는다. 병렬 작업은 초반 설정을 제외하고 대략 2.5~3시간 수준을 예상하되, GPU 할당·모델 준비·재시도에 따라 늘어날 수 있다.

계획/원본 source hash/입력 checksum은 `verification/dbpp_split_plan.json`에, 실제 receipt/status는 `outputs/dbpp_split/<workspace>/`에 보관한다. 입력 묶음과 모델은 CPU에서 준비하며 GPU가 다운로드를 기다리지 않게 한다. 기존 두 UNO 계정은 자신의 `uno-models`/`uno-huggingface`를 재사용하며, 새 계정으로 인증정보를 복사하지 않는다.

상태와 해당 workspace의 최종 metric 수신:

```powershell
& freecus/.venv-modal311/Scripts/python.exe -X utf8 -m comparison_candidates.split_dbpp status --workspace wj-lee-louis
& freecus/.venv-modal311/Scripts/python.exe -X utf8 -m comparison_candidates.split_dbpp status --workspace woojin716
& freecus/.venv-modal311/Scripts/python.exe -X utf8 -m comparison_candidates.split_dbpp status --workspace louislee-85468
```

모든 partition 완료 및 metric 수신 후:

```powershell
& freecus/.venv-modal311/Scripts/python.exe -X utf8 -m comparison_candidates.merge_dbpp
```

병합은 workspace 평균을 다시 평균내지 않고 1350개 case score를 합친 후 reference macro를 재계산한다. 모델/전처리 protocol, metric environment, 참조·원문 prompt, 정확한 partition 소유권과 누락·중복을 검증한다. 생성 이미지 다운로드/압축/LFS 추적은 후속 작업으로 수행한다. 이 작업에서는 commit/pull/push를 수행하지 않는다.
