# RunPod 실험 개요

**실행은 [RUNPOD_EXECUTION_KO.md](RUNPOD_EXECUTION_KO.md)의 1번부터 진행한다.**
팀 계정에 이미 있는 100GB Network Volume을 사용한다. 여유 공간은 충분하다고 가정하며 새 볼륨을 만들지 않는다.

## 확정한 실험

| 조건 | IP-Adapter scale | Projection strength | SSG | 이미지 |
|---|---:|---:|---|---:|
| source_target_ssg | 0.8 | 1.0 | 켬 | 2,100 |
| source_target | 0.7 | 1.0 | 끔 | 2,100 |

각 조건은 DreamBench 750 + DreamBench++ 1,350장이다. seed 42, 24 steps, 1024×1024, guidance 3.5 및 기존 모델 revision/cache를 유지한다. 평가는 기존 CLIP-T·CLIP-I·DINO-I이며 공식 GPT 평가와 별개다.

## 파일 관리

- 코드·설치 스크립트·가이드: GitHub `WJ-Lee-Louis/AI_PROJECT`.
- 입력: `outputs/runpod_delivery/runpod_inputs.tar.gz`를 팀 볼륨에 한 번 업로드하고 SHA256 검증 후 복원한다.
- 팀 볼륨의 실험 경로: `/workspace/instantx-0807`. 코드·가상환경·모델·결과를 이 경로 안에 보관한다.
- 기존 `runpod_bundle.tar.gz`는 코드까지 포함한 이전 전달 방식이다. GitHub 방식에서는 새 **입력 전용 번들**을 사용한다.
- 토큰·개인키·모델·생성 결과는 Git에 커밋하지 않는다.

## GPU 운영

1. 1 GPU Pod에서 설치하고 20장 생성·평가를 검증한다.
2. 같은 볼륨에 8 GPU Pod를 배포한다. `8 max`는 실제 재고를 보장하지 않는다.
3. 재고가 부족하면 4+4장 등 여러 Pod에 분할한다. 8 worker면 각각 525장이다.
4. 생성 완료 후 1 GPU로 평가하고 결과를 회수한다. 본인 Pod는 Stop하되 공동 볼륨은 삭제하지 않는다.

동일 GPU당 단가·처리속도에서는 순수 생성비가 비슷하다. 초기화·I/O·대기 시간 때문에 정확히 같지는 않다. 시간·비용은 소량 검증 기록과 배포 화면 단가로 계산한다.

## 구현 및 검증 범위

- 줄바꿈이 달라도 같은 RunPod 소스 해시를 사용한다. 실제 코드 변경은 이전 결과 재개를 차단한다.
- 완료 PNG/JSON을 검증해 재실행 시 건너뛴다. 연속 3회 실패하면 worker를 중지한다.
- 여러 Pod의 작업 범위를 분리하고, 평가 전에 전체 누락을 검사한다.
- `--stop-pod`는 API 권한이 필요한 선택 기능이다. 평가용 Pod 배포는 사이트에서 수행한다.
- 원격 Linux 설치·GPU 생성·평가는 최초 1 GPU Pod에서 확인해야 한다.
