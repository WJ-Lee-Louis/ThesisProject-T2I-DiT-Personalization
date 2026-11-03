# 로컬 검증 기록 — 기존 프로젝트 프로토콜 반영

검증일: 2026-10-30 (Asia/Seoul)

- 독립 Python 3.10.9 venv 설치 완료. PyTorch 2.6.0+cpu / torchvision 0.21.0+cpu.
- `python -m pip check`: No broken requirements found.
- `python -m freecus.verify`: FLUX, Qwen2-VL, BiRefNet 의존성 및 runtime import 통과. 원본 파일 SHA-256 검증 통과.
- `python -m unittest discover -s freecus/tests -v`: **10 tests passed**.
- DreamBench 주 manifest: 기존 지정 참조 30장, **750 samples, seed 42**.
- DreamBench++ 주 manifest: 기존 지정 참조 150장, **1,350 samples, seed 42**. style 180개 포함.
- 기존 프로젝트의 실제 A/B 원격 plan에서도 모든 job의 seed 42 / 1024×1024 / 24 steps / guidance 3.5를 확인했다. FreeCus는 권장 30 steps를 유지한다.
- 원문 target 대조: DreamBench 750/750, DreamBench++ 1,350/1,350 공식 자료와 일치. 회귀 테스트에 포함했다.
- 기존 selected-reference evaluator의 소스 해시와 CLIP ViT-B/32 / DINOv1 vits16 정책을 그대로 재사용한다. CLIP 패키지의 고정 Git revision 설치와 import를 확인했다.
- CPU KV cache 변경: 테스트 입력에서 원본 attention 출력과 정확히 일치.
- 실제 diffusers FLUX transformer 축소 모델: 19 dual + 38 single blocks, 512 text tokens, rotary embedding, KV save/use 및 유한한 출력 확인. 가중치를 다운로드하지 않은 API 호환성 검사다.
- ANS: 1024에서 packed token 4096 → μ=1.15 → 참조 negative shift -1.15 확인. 공개 코드의 중복 축소 계산도 별도 모드로 검사.
- 결과 파일 손상 감지, 해상도 강제, 입력 이미지 변경 감지, 빈 DreamBench++ dataset 거부 검사 통과.

미수행: CUDA 환경 설치/실행, Docker build, 실제 pretrained model 추론, SAM 평가용 foreground 생성, CLIP/DINO 모델 추론, GPT judge, 전체 벤치마크, 논문 점수 재현 판단, Modal 계정 및 배포.

환경 상세는 `environment-local-cpu.json`, 설치 버전은 `requirements-local-cpu.txt`, 현재 비교 설정은 `project-protocol.json`을 참조한다. `datasets.json`의 초기 4-seed 개수는 superseded 상태로 보존한다. 논문은 512 해상도·pair당 4장이므로 이번 1024·1장 설정은 동일 조건의 수치 재현이 아니다.
