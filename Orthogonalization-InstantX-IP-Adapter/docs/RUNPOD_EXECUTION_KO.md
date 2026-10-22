# RunPod 실행: 팀 볼륨 + GitHub + SSH

기존 팀 계정의 100GB Network Volume을 사용하며 공간은 충분하다고 가정한다.
코드는 GitHub, 입력 데이터는 별도 번들, 모델·결과는 팀 볼륨에 보관한다. 최초 1 GPU 검증 후 확대한다.

## 1. RunPod 사이트에서 1 GPU 배포

1. 공동작업자가 사용하는 **팀 계정/조직으로 전환**한다.
2. **Storage**에서 기존 100GB 볼륨의 **이름·지역**을 확인한다. 새 볼륨을 만들지 않는다.
3. **Pods → Deploy → Network Volume**에서 그 볼륨을 선택한다.
4. **A100 SXM 80GB / GPU Count 1 / On-Demand**, Python **3.11**과 SSH를 제공하는 템플릿을 선택한다.
5. Full SSH / exposed TCP 22를 사용하도록 설정하고 배포한다. 기존 SSH 공개키를 사용한다.
6. 이후 Pod도 같은 템플릿과 같은 볼륨을 사용한다. 공동작업자의 파일·볼륨은 삭제하지 않는다.

이번 실험은 `/workspace/instantx-0807` 아래를 사용한다. 볼륨 지역에 따라 GPU 재고가 제한된다.
공식: [볼륨 연결](https://docs.runpod.io/storage/network-volumes), [Full SSH](https://docs.runpod.io/pods/configuration/use-ssh).

## 2. 내 PC에서 VS Code Remote - SSH 연결

새 Pod의 **Connect → SSH over exposed TCP**에서 IP와 포트를 확인한다.
VS Code에 Microsoft **Remote - SSH** 확장을 설치한다.
`Ctrl+Shift+P → Remote-SSH: Open SSH Configuration File...`에서 `C:\Users\82107\.ssh\config`를 연다.
아래 `실제_IP`, `실제_포트`는 화면의 값으로 바꾼다.

```sshconfig
Host runpod-ai
    HostName 실제_IP
    Port 실제_포트
    User root
    IdentityFile C:/Users/82107/.ssh/runpod_ai_project
    IdentitiesOnly yes
    ServerAliveInterval 30
```

`Remote-SSH: Connect to Host... → runpod-ai`로 연결한다. 운영체제는 Linux.
`Enter passphrase for key`에는 **SSH 키 암호**를 입력한다. RunPod 로그인 비밀번호가 아니다.
`root@... password`가 나오면 공개키 인증 문제이므로 취소하고 주소·키 등록을 확인한다.

## 3. RunPod 원격 터미널에서 GitHub 코드 받기

VS Code 좌측 아래가 **SSH: runpod-ai**인지 확인하고 **터미널 → 새 터미널**을 연다.

```bash
nvidia-smi
python3.11 --version
git --version
df -h /workspace
mkdir -p /workspace/instantx-0807
cd /workspace/instantx-0807
git clone --depth 1 https://github.com/WJ-Lee-Louis/AI_PROJECT.git
cd AI_PROJECT/Orthogonalization-InstantX-IP-Adapter
git rev-parse HEAD
```

- Python 3.11 또는 git이 없으면 여기서 멈추고 템플릿/오류를 확인한다.
- 이미 clone했다면 같은 저장소에서 `git pull --ff-only`한다. **생성·평가 중에는 pull하지 않는다.**
- 비공개 저장소는 GitHub 저장소 읽기 권한으로 인증한다. RunPod SSH 키와는 별개이며 토큰을 URL에 넣지 않는다.
- **파일 → 폴더 열기**에서 `/workspace/instantx-0807/AI_PROJECT/Orthogonalization-InstantX-IP-Adapter`를 열면 원격 코드를 볼 수 있다.

## 4. 내 PC PowerShell에서 입력 번들 한 번 업로드

**아래는 RunPod 터미널이 아니라 내 PC PowerShell에서 실행한다.** Windows 시작 메뉴에서 PowerShell을 열면 구분하기 쉽다.

```powershell
Set-Location 'C:\Users\82107\Downloads\AI_PROJECT\Orthogonalization-InstantX-IP-Adapter'
scp .\outputs\runpod_delivery\runpod_inputs.tar.gz runpod-ai:/workspace/instantx-0807/
```

새 파일은 **runpod_inputs.tar.gz**이며 이전 runpod_bundle.tar.gz와 다르다.
manifest·참조 이미지 180개·conditioning cache 2,100개가 들어 있다. 코드는 없다.
팀 볼륨에 이미 이 번들이 있고 다음 검증이 성공하면 재업로드하지 않는다.

## 5. RunPod 원격 터미널에서 입력 복원·설치·다운로드

```bash
cd /workspace/instantx-0807/AI_PROJECT/Orthogonalization-InstantX-IP-Adapter
python3.11 runpod/input_bundle.py restore /workspace/instantx-0807/runpod_inputs.tar.gz
bash runpod/setup.sh
source runpod/env.sh
```

Git의 `runpod/input_bundle.json`과 번들 SHA256을 비교하며 기존 입력이 다르면 덮어쓰지 않는다.
설치 완료 문구는 **Setup complete**다. 가상환경·캐시는 실험 전용 경로에 저장된다.

아래에는 **Hugging Face의 hf_... 읽기 토큰**을 입력한다. SSH 키 암호가 아니며 입력 문자는 표시되지 않는다.

```bash
read -rsp 'Hugging Face read token: ' HF_TOKEN
echo
export HF_TOKEN
python runpod/prefetch.py
python scripts/benchmark_runpod.py plan --gpus 1
```

토큰은 현재 터미널 환경에만 둔다. 새 접속에서는 다시 입력한다.
**Generation weights cached** 이후 plan의 **expected: 4200**을 확인한다.
이전 `/workspace/venvs`와 이전 결과는 자동 이동/덮어쓰기하지 않는다.

## 6. 1 GPU에서 20장 생성·평가 검증

토큰을 입력한 터미널에서 새 tmux 서버를 시작한다. 별도 소켓 이름으로 공동작업자의 tmux와 분리한다.

```bash
tmux -L instantx-pilot new -s pilot
```

열린 tmux 안에서:

```bash
cd /workspace/instantx-0807/AI_PROJECT/Orthogonalization-InstantX-IP-Adapter
source runpod/env.sh
python scripts/benchmark_runpod.py all --gpus 1 --limit 5 --output /workspace/instantx-0807/results/pilot
```

각 데이터셋 5개 × 두 조건 = **20장**, 이어서 CLIP-T·CLIP-I·DINO-I 평가.
**Ctrl+B → D**로 빠져나오면 PC를 꺼도 계속된다. 재접속은 `tmux -L instantx-pilot attach -t pilot`.

```bash
tail -n 10 /workspace/instantx-0807/results/pilot/worker-0.log
cat /workspace/instantx-0807/results/pilot/status-all-0.json
cat /workspace/instantx-0807/results/pilot/metrics/summary.csv
```

**complete / images=20**, 평가표 4행 및 이미지 품질을 확인한다.
오류 수정 후 동일 명령으로 재개하면 완료 이미지를 건너뛴다.
연속 3회 실패하면 worker를 중지하고 같은 Pod의 다른 worker도 정리한다.

## 7. 같은 볼륨에 최대 8 GPU로 본 실험

검증 후 본인 1 GPU Pod를 Stop하고 **같은 볼륨·템플릿**으로 본 작업 Pod를 생성한다.
재고가 있으면 **Pod 하나, GPU Count 8**이다. `8 max`는 재고 보장이 아니다.
SSH config의 IP/포트를 갱신한다. 같은 볼륨이면 clone·입력 업로드·모델 다운로드는 재사용한다.

새 Pod 원격 터미널에서:

```bash
cd /workspace/instantx-0807/AI_PROJECT/Orthogonalization-InstantX-IP-Adapter
source runpod/env.sh
read -rsp 'Hugging Face read token: ' HF_TOKEN
echo
export HF_TOKEN
command -v tmux || { apt-get update && apt-get install -y tmux; }
tmux -L instantx-full new -s full
```

tmux 안에서 환경을 활성화하고 **구성에 맞는 명령만 선택**한다.

```bash
source runpod/env.sh
# 8 GPU Pod 하나
python scripts/benchmark_runpod.py generate --gpus 8 --output /workspace/instantx-0807/results/full

# 4 GPU Pod 두 개라면 각 줄을 해당 Pod에서만 실행
# Pod A
python scripts/benchmark_runpod.py generate --gpus 4 --total-workers 8 --worker-offset 0 --output /workspace/instantx-0807/results/full
# Pod B
python scripts/benchmark_runpod.py generate --gpus 4 --total-workers 8 --worker-offset 4 --output /workspace/instantx-0807/results/full
```

총 4,200장, 8 worker면 각각 525장. 4장만 확보하면 `--gpus 4`로 전체 처리 가능하다.
여러 Pod는 **같은 볼륨·코드 버전**을 사용한다. 두 Pod 구성은 status-generate-0.json과 status-generate-4.json을 모두 확인한다.
분할을 바꿀 때는 **모든 기존 작업 중지 후에만** 결과 폴더의 topology.json을 제거한다. 완료 PNG/JSON은 유지한다.

## 8. 생성 완료 후 1 GPU에서 평가

생성 Pod들을 Stop하고 같은 볼륨을 연결한 1 GPU Pod에서 환경을 활성화한다.
tmux 안에서 아래 명령을 실행한다. 전체 4,200개 파일을 검증하고 누락이 없을 때만 평가한다.

```bash
cd /workspace/instantx-0807/AI_PROJECT/Orthogonalization-InstantX-IP-Adapter
source runpod/env.sh
python scripts/benchmark_runpod.py evaluate --output /workspace/instantx-0807/results/full
```

최종 파일: **metrics/scores.json, summary.csv, summary.json**.
기존 Modal 평가는 T4였으므로 GPU 정보를 함께 기록한다. DreamBench++ 공식 GPT 평가는 포함하지 않는다.
1 GPU에서 생성·평가를 연속 실행하려면 7–8단계 대신 `all --gpus 1 --output /workspace/instantx-0807/results/full`을 사용한다.
8 GPU에서도 all은 가능하지만 평가 중 7장도 과금된다.

## 9. 종료·결과 회수

완료 후 본인 Pod를 **Stop**한다. **공동 볼륨은 삭제하지 않는다.**
자동 Stop은 실험 전용 Pod에서 Pod 제어 권한 API key를 입력한 뒤 실행 명령 끝에 **--stop-pod**를 추가한다.

```bash
read -rsp 'RunPod API key: ' RUNPOD_API_KEY
echo
export RUNPOD_API_KEY
```

현재 Pod만 중지한다. API 오류·강제 종료·호스트 장애에서는 자동 Stop을 보장하지 않으므로 콘솔도 확인한다.
평가 Pod 자동 생성 기능은 없다.

Pod가 실행 중일 때 결과를 압축한 뒤 내 PC로 내려받는다. 결과와 압축파일을 함께 보관할 공간이 필요하다.

```bash
# RunPod 터미널
tar -czf /workspace/instantx-0807/runpod_results.tar.gz -C /workspace/instantx-0807/results full
sha256sum /workspace/instantx-0807/runpod_results.tar.gz
```

```powershell
# 내 PC PowerShell: 서버의 해시와 비교
scp runpod-ai:/workspace/instantx-0807/runpod_results.tar.gz .
Get-FileHash .\runpod_results.tar.gz -Algorithm SHA256
```

## 개발자 참고

- 입력 재생성: 로컬 `python runpod/input_bundle.py build`. JSON 체크섬 파일도 커밋한다. 데이터 자체는 커밋하지 않는다.
- RunPod 소스 해시는 CRLF/LF만 정규화한다. 실제 코드 변경은 여전히 재개를 차단한다. 이전 결과의 해시를 새 값으로 덮어쓰지 않는다.
- 로컬 테스트가 원격 CUDA·네트워크 볼륨 잠금을 검증하는 것은 아니다. 1 GPU 검증 후 확대한다.
