# FreeCus / DreamO / UNO: 로컬 결과 통합 및 Git 추적

> 현재 이미지 저장 정책은 [GENERATED_IMAGE_STORAGE.md](GENERATED_IMAGE_STORAGE.md)를 따른다.
> 아래 Git/LFS 이미지 ZIP 업로드 설명은 과거 작업 기록이다. 현재 생성 이미지 ZIP은
> Google Drive에 보관하며, GitHub에는 코드·설정·평가·검증 기록만 저장한다.

목적은 **로컬의 FreeCus·DreamO·UNO 결과를 먼저 GitHub에 반영해 공통 기준점을 만든 뒤, RunPod의 기존 InstantX 실험 결과를 통합하는 것**이다. 별도의 RunPod 복원 과정은 필요하지 않다. 이번 작업은 이미지 다운로드·검증과 Git/LFS 추적 설정 및 staging까지 수행했으며, **commit, fetch, pull, push는 사용자가 직접 수행한다.**

## 보관 구조

```text
freecus/  dreamo/  uno/
  <source, requirements, model revision locks, original upstream, tests>
  manifests/                             # 현재 seed42 입력 목록
  results/reports/                        # metric JSON/CSV, 비교표, 감사 기록
  results/<benchmark>/<bundle_id>/
    bundle_index.json                    # ZIP·멤버 SHA256, 크기, 생성 및 누락 case
    manifest.jsonl                       # 실행 당시 원본 bytes
    metrics.json                         # 실제 평가 범위·이미지별 점수
    provenance.json
    generation_source.zip                # 생성 당시 source snapshot 원본 bytes
    verification.json
  artifacts/<benchmark>/<bundle_id>/
    images-000.zip                       # Git LFS, 각 파일 1 GiB 이하
    images-001.zip
  outputs/                               # 캐시·중복 결과: 추적 제외
```

생성 이미지는 로컬의 ZIP에 저장되어 있으며, Git LFS로 추적한다. Git 이력에는 LFS 포인터가 들어간다. `results/`와 현재 manifest는 일반 Git으로 추적하고 줄바꿈 변환을 비활성화해 checksum을 보존한다. ZIP에는 생성 PNG, metadata, prompt, run.json 및 선택된 원본 참조 입력을 보관한다. 반복되는 `src_image` PNG는 SHA256별로 한 번씩 보관하고 원래 경로와의 대응을 기록한다. 모델 가중치, venv, 모델 캐시, 폐기된 다중 seed 입력은 추적하지 않는다.

| 방법 | DreamBench 생성·평가 | DreamBench++ 생성 | DreamBench++ 평가 |
|---|---:|---:|---:|
| FreeCus | 750 | 1,188 | style 제외 1,170 |
| DreamO | 750 | 1,350 | style 포함 1,350 |
| UNO | 750 | 1,350 | style 포함 1,350 |

전체 생성 PNG는 **6,138장**, 평가 대상은 **6,120장**이다. FreeCus DB++는 원래 1,350-case manifest와 생성되지 않은 162개 case ID를 함께 보존한다. 평가에 포함되지 않은 생성 이미지 18장도 보관한다. 이미지 ZIP은 총 **10개, 약 7.17 GiB**이며, 전체 목록은 [transfer_inventory.json](freecus/results/transfer_inventory.json)에 기록되어 있다.

## 사용자가 진행할 통합 순서

1. 로컬에 stage된 세 방법의 코드·manifest·metric·이미지 ZIP을 검토하고 commit/push한다.
2. 이후 RunPod에서 이 기준점을 반영한 상태로 기존 InstantX 실험 결과를 통합하고, 필요한 commit/push를 진행한다.
3. RunPod에서 반영한 결과를 로컬에 가져와 최종 통합 상태를 맞춘다.

각 방법의 결과를 별도 경로로 관리하므로 결과 파일 간 충돌 가능성을 줄일 수 있다. 다만 공통 문서나 설정 파일을 양쪽에서 수정하면 충돌이 생길 수 있으므로, 첫 로컬 반영 이후의 공통 변경을 기준으로 통합한다. 현재 세 방법과 무관한 다른 작업의 변경·삭제는 staging에 포함하지 않았다.

아래는 로컬 검토 및 사용자가 직접 실행할 명령의 예시다. 자동 실행하지 않는다.

```bash
git status
git diff --cached --stat
git lfs ls-files
# stage된 내용을 검토한 뒤 직접 실행
git commit -m "Add FreeCus DreamO UNO benchmark results and LFS image bundles"
git push
```

LFS pre-push hook이 `git push` 시 ZIP 객체를 업로드한다. GitHub 반영은 Git commit과 LFS 업로드가 모두 완료되어야 한다. 약 7.17 GiB는 저장소 소유자의 LFS 저장·전송 용량에 반영될 수 있다. `.git/lfs` 캐시 자체는 Git에 추가하지 않는다.

현재 Windows 로컬에서는 저장 공간을 절약하기 위해 일부 ZIP과 LFS 캐시가 검증된 동일 파일의 NTFS hardlink를 공유한다. **이 ZIP은 불변 산출물로 취급하고 파일 내용을 제자리 수정하지 않는다.** 수정이 필요하면 새 run/bundle 폴더에 새 ZIP을 생성한다. ZIP 조각은 각각 독립된 ZIP 파일이며 이어 붙이는 형식이 아니다.

별도의 RunPod 이미지 복원이나 재평가 없이, 보관된 `metrics.json`과 보고서로 결과를 통합할 수 있다. 추후 이미지 열람·분석이 필요할 때 해당 ZIP을 필요한 위치에 압축 해제하면 된다.
