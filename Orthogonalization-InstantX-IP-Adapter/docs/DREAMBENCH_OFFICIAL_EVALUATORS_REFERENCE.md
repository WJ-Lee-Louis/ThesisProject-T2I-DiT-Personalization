# ?? ?? evaluator ?? ?? ? ?? ?? ??? ??

?? CLIP/DINO ??? ??? [?? ???](DREAMBENCH_METRIC_EVALUATION_GUIDE.md)? ???. ??? ?? all-reference?DreamBench++ ?? evaluator?CP/PF ?? ????. ?? source_target ??? ?? ??? ?? ? ?? ??? ???? ???.

# DreamBench / DreamBench++ 이미지 생성 완료 후 metric 평가 참고

이미지 생성이 모두 끝난 뒤 읽는다. 지금 진행할 인퍼런스 명령은 [Modal 이미지 생성 가이드](../DREAMBENCH_EVALUATION_MODAL_GUIDE.md)를 따른다. 평가 환경 설치·export·CLIP/DINO·CP/PF 명령은 생성 실행에 필요하지 않다.

UNO 공개 evaluator의 all-reference 동작과 논문 점수 산출 프로토콜을 동일시하지 않는다. 비교군 9개 연구의 차이와 공통 평가 권고는 [비교 평가 프로토콜 검토](BENCHMARK_COMPARISON_PROTOCOL_REVIEW.md)를 따른다. 아래 UNO all-reference 명령은 공개 코드 동작을 재현하는 보조 평가다. 외부 evaluator와 GPU/GPT 평가는 아직 실행 검증되지 않았다.

## 1. 먼저 결정할 평가 프로토콜

### DreamBench와 UNO

DreamBench에서 사용하는 CLIP-I, CLIP-T, DINO-I를 UNO도 single-subject 평가에 사용한다. 다만 이름만 맞추면 충분하지 않다.

| 항목 | UNO 공개 single-subject evaluator에서 확인한 정의 |
|---|---|
| CLIP-I | OpenAI CLIP ViT-B/32, 생성 이미지와 subject reference들의 정규화 feature cosine 평균 |
| CLIP-T | 같은 CLIP 모델, 원래 target prompt와 생성 이미지의 cosine. 음수는 0으로 clipping, 가중치 1 |
| DINO-I | DINOv1 `dino_vits16`, 생성 이미지와 subject reference들의 cosine 평균 |
| 평가 reference | 입력으로 고른 한 장의 **상위 subject 폴더 안에 있는 모든 이미지** |
| 집계 | subject별 점수 후 subject 평균. 전체가 30×25로 완전하면 이미지별 균등 평균과 같음 |
| 생성 파일명 | manifest 순서에 따라 `0_0.png` … `749_0.png`, 마지막 0은 evaluator의 rank |

근거: [UNO 공식 평가 코드](https://github.com/bytedance/UNO/blob/a563432dcfb4caf98df4368facb8102bf97d376c/eval/evaluate_clip_dino_score_single_subject.py), [UNO 공식 실행 안내](https://github.com/bytedance/UNO#-evaluation).

현재 로컬 `datasets/dreambench_uno/images`에는 subject당 선택된 reference 한 장만 있다. 이것으로 CLIP-I/DINO-I를 계산하면 **selected-reference 평가**다. 사용자가 별도로 확인한 UNO 논문 점수 산출 방식도 한 장 비교이며, 공개 코드의 전체 폴더 평균과 불일치한다. 따라서 공개 코드만 보고 논문 표 비교에 전체 reference가 필수라고 했던 이전 설명은 철회한다. 주 비교는 모든 방법에 동일한 selected reference를 적용하도록 권고한다. 전체 DreamBooth subject 폴더는 all-reference 민감도 분석이나 공개 코드 동작 재현을 할 때 별도로 확보한다. 한 장 생성 입력과 여러 장 평가 reference는 서로 다른 개념이다.

추가로 확인한 공개 코드 불일치: 현재 pinned UNO JSON은 `prompt`/`image_paths` 키인데, evaluator는 `text`/`image_path`를 읽는다. 마지막 평가 실행 예제는 이 키를 메모리에서 변환한다. [입력 JSON](https://github.com/bytedance/UNO/blob/a563432dcfb4caf98df4368facb8102bf97d376c/datasets/dreambench_singleip.json), [데이터 submodule](https://github.com/bytedance/UNO/blob/a563432dcfb4caf98df4368facb8102bf97d376c/.gitmodules).

### DreamBench++

| 지표 | 공식 구현 | 보고 시 주의 |
|---|---|---|
| CLIP-I / CLIP-T | `openai/clip-vit-base-patch32`, 대응하는 이미지/텍스트 cosine ×100 | UNO의 0~1 관례와 스케일이 다름. CLIP-T에 UNO와 같은 0 clipping을 추가하지 말 것 |
| DINOv1-I | `dino_vits8`, cosine ×100 | UNO의 `vits16`과 다른 백본 |
| DINOv2-I | `facebook/dinov2-small`, CLS feature cosine ×100 | DINOv1과 별도 열로 보고 |
| CP | Concept Preservation: reference와 생성 이미지의 concept 보존 | subject와 style에 서로 다른 rubric |
| PF | Prompt Following: **원래 target prompt**와 생성 이미지의 일치 | reference/source-caption/차분 문장을 target 대신 넣지 않음 |

근거: [공식 CLIP 구현](https://github.com/yuangpeng/dreambench_plus/blob/66ca27e52b5e7daa350bdb92ac8c60d4a9339c5e/dreambench_plus/metrics/clip_score.py), [공식 DINO 구현](https://github.com/yuangpeng/dreambench_plus/blob/66ca27e52b5e7daa350bdb92ac8c60d4a9339c5e/dreambench_plus/metrics/dino_score.py), [전체 지표 entrypoint](https://github.com/yuangpeng/dreambench_plus/blob/66ca27e52b5e7daa350bdb92ac8c60d4a9339c5e/dreambench_plus/eval_clip_and_dino.py).

**공식 CP/PF 원점수는 각각 정수 0~4점이다. 0~5점이 아니다.** 5개 등급이라는 것과 최대 점수 5는 다르다. 집계 코드는 각 평균을 4로 나눠 0~1로 만들고, CP×PF도 출력한다. `CP×PF`는 집계된 CP와 PF의 곱이며 이미지별 곱의 평균으로 바꾸지 않는다. 0~5 rubric을 새로 사용하면 별도 사용자 정의 지표로 명명하고, 공식 점수와 같은 열에 합치지 않는다. 단순히 5로 나누어도 공식 rubric과 같아지지 않는다.

근거: [subject CP rubric](https://github.com/yuangpeng/dreambench_plus/blob/66ca27e52b5e7daa350bdb92ac8c60d4a9339c5e/dreambench_plus/prompts/user_prompt_subject_full.txt), [style CP rubric](https://github.com/yuangpeng/dreambench_plus/blob/66ca27e52b5e7daa350bdb92ac8c60d4a9339c5e/dreambench_plus/prompts/user_prompt_style_full.txt), [PF rubric](https://github.com/yuangpeng/dreambench_plus/blob/66ca27e52b5e7daa350bdb92ac8c60d4a9339c5e/dreambench_plus/prompts/user_prompt_text_full.txt), [정규화·반복 집계](https://github.com/yuangpeng/dreambench_plus/blob/66ca27e52b5e7daa350bdb92ac8c60d4a9339c5e/dreambench_plus/benchmarking.py).

공개 GPT evaluator 기본값은 `gpt-4o-2024-05-13`, temperature=1이며 무작위 seed를 사용한다. CP는 reference+생성 이미지, PF는 target+생성 이미지를 512×512로 resize해 전달한다. 집계 스크립트는 `full/full2/full3`의 3회 평가를 사용한다. 한 번만 평가했다면 1회라고 명시한다. 원래 snapshot을 현재 계정에서 호출할 수 있는지는 미확인이며, 접근 불가 시 다른 모델로 조용히 대체하지 않는다. 대체 judge는 모델명과 rubric을 따로 기록한다. 근거: [CP evaluator](https://github.com/yuangpeng/dreambench_plus/blob/66ca27e52b5e7daa350bdb92ac8c60d4a9339c5e/dreambench_plus/eval_concept_preservation.py), [PF evaluator](https://github.com/yuangpeng/dreambench_plus/blob/66ca27e52b5e7daa350bdb92ac8c60d4a9339c5e/dreambench_plus/eval_prompt_following.py).

추천 보고 방식: DreamBench는 selected-reference 기반 공통 CLIP-I/T·DINO-I를 주 지표로 보고하고, UNO 공개 코드의 all-reference 결과가 필요하면 별도 열로 분리한다. DreamBench++는 고정한 공통 CLIP-I/T·DINOv1/v2와 CP/PF를 함께 보고한다. DreamBench++의 animal/human/object/style 및 prompt 0~3/4~6/7~8 그룹도 분리한다. 현재 ++ 전체에는 style 20개×9=180개가 포함된다. 과거 `dreambench_plus_non_style_components.jsonl`의 1,170개를 전체 1,350개 대신 사용하지 않는다.

## 2. 전체 완료 검증 및 공식 evaluator용 export

프로젝트 폴더에서 다음을 실행한다. 아래는 `source_target` 한 조건에 대한 검증과 export다. 다른 조건은 `--condition`과 export 목적지의 조건 이름을 함께 바꿔 실행한다.

```powershell
# selected=2100, complete=2100, pending=0 확인
& 'C:\Users\82107\.venvs\orthogonalization-modal311\Scripts\python.exe' scripts/benchmark_modal.py --condition source_target --num-shards 1 --shard-index 0

# 목적지는 새 폴더여야 함. 일부 이미지가 없거나 설정이 섞이면 export 실패
& 'C:\Users\82107\.venvs\orthogonalization-modal311\Scripts\python.exe' scripts/benchmark_modal.py --condition source_target --benchmark dreambench --export-dir outputs/eval_export_v1/source_target/dreambench
& 'C:\Users\82107\.venvs\orthogonalization-modal311\Scripts\python.exe' scripts/benchmark_modal.py --condition source_target --benchmark dreambench_plus --export-dir outputs/eval_export_v1/source_target/dreambench_plus
```

DreamBench export는 manifest 순서로 UNO의 `0_0.png` … `749_0.png`를 만든다. ++ export는 공식 `src_image/<collection_id>/<prompt_index>_0.jpg`, `tgt_image/...`, `text/...txt` 구조다. ++ evaluator가 JPG 확장자를 하드코딩하므로 reference와 결과를 RGB JPEG quality=95로 변환한다. 원본 PNG는 보존된다. 모든 조건을 같은 변환 규칙으로 export하고 이 규칙을 보고한다. JPEG export로 평가한 점수와 원본 PNG로 별도 평가한 점수를 섞지 않는다. [공식 파일 구조와 실행 순서](https://github.com/yuangpeng/dreambench_plus#step-2-generate-images).

## 3. 평가 실행 예제와 재현 범위

이 절은 생성 완료 후 사용할 **공식 evaluator 연결 예제**다. 현재 실행 검증 범위는 입력·분할·재개·export까지이며, 아래 외부 패키지 설치와 실제 GPU/GPT 점수 계산은 수행하지 않았다. generation Python 환경을 보존하도록 평가용 환경을 분리한다.

### DreamBench++ — 별도 PowerShell 평가 환경

```powershell
Set-Location 'C:\Users\82107\Downloads\AI_PROJECT\Orthogonalization-InstantX-IP-Adapter'
$env:PYTHONUTF8 = '1'
$env:BENCH_ROOT = (Get-Location).Path
New-Item -ItemType Directory -Force external | Out-Null
git clone https://github.com/yuangpeng/dreambench_plus.git external/dreambench_plus
if ($LASTEXITCODE -ne 0) { throw 'Clone failed; inspect existing checkout before proceeding' }
git -C external/dreambench_plus checkout 66ca27e52b5e7daa350bdb92ac8c60d4a9339c5e
py -3.11 -m venv 'C:\Users\82107\.venvs\orthogonalization-benchmark-eval'
$env:BENCH_EVAL_PY = 'C:\Users\82107\.venvs\orthogonalization-benchmark-eval\Scripts\python.exe'
& $env:BENCH_EVAL_PY -m pip install --upgrade pip
# 평가 경로에 필요한 패키지만 설치. 생성 모델용 diffusers를 구버전으로 내리지 않음
& $env:BENCH_EVAL_PY -m pip install torch==2.7.1 torchvision==0.22.1 transformers==4.35.2 accelerate==1.10.0 'numpy<2' pillow==11.3.0 fire megfile loguru omegaconf openai requests
& $env:BENCH_EVAL_PY -m pip install --no-deps -e external/dreambench_plus
$env:CPP_SAMPLES = Join-Path $env:BENCH_ROOT 'outputs/eval_export_v1/source_target/dreambench_plus'
$env:CPP_DATA = Join-Path $env:BENCH_ROOT 'datasets/XVerseBench/dreambench_plus_raw/extracted/data'
& $env:BENCH_EVAL_PY -m dreambench_plus.eval_clip_and_dino --dir $env:CPP_SAMPLES
& $env:BENCH_EVAL_PY -m pip freeze > outputs/benchmark_eval_requirements.txt
```

로컬에 CUDA가 없으면 ++ 고전 지표는 CPU에서 실행되어 오래 걸린다. GPU로 평가하려면 평가 전용 CUDA 환경에서 같은 명령을 실행한다. 생성에 사용한 Modal GPU 할당과는 별개다. 위 환경은 평가 경로에 필요한 의존성만 설치하며, 상류 저장소의 모든 생성 모델 설치를 재현하지는 않는다.

**CP/PF는 별도의 유료 API 호출**이며 Modal credit으로 결제되지 않는다. 아래는 해당 snapshot의 사용 가능 여부를 확인한 뒤 실행한다. 평가 입력에 A/B caption이나 negative를 추가하지 않는다. 공식 README의 GPT 명령에는 스크립트명이 빠져 있어, 아래는 Python 함수를 명시적으로 호출한다.

```powershell
$env:OPENAI_API_KEY = [System.Net.NetworkCredential]::new('', (Read-Host 'OpenAI API key' -AsSecureString)).Password
$env:CPP_JUDGE = 'gpt-4o-2024-05-13'
@'
import glob, json, os
from pathlib import Path
import dreambench_plus.constants as constants
# Override before importing evaluator modules; leave upstream source files unchanged.
constants.OPENAI_API_KEYS = [os.environ['OPENAI_API_KEY']]
constants.METHODS = {'instantx_st': Path(os.environ['CPP_SAMPLES']).as_posix()}
# Upstream dataset parsing splits paths on '/'; normalize Windows glob results.
original_glob = glob.glob
glob.glob = lambda *a, **k: [p.replace('\\', '/') for p in original_glob(*a, **k)]
from dreambench_plus.eval_concept_preservation import main as cp
from dreambench_plus.eval_prompt_following import main as pf
base = Path(os.environ['BENCH_ROOT']) / 'outputs/gpt_rating_v1'
data = Path(os.environ['CPP_DATA']).as_posix()
for repeat in ('full', 'full2', 'full3'):
    cp_dir = base / ('concept_preservation_' + repeat) / 'instantx_st'
    pf_dir = base / ('prompt_following_' + repeat) / 'instantx_st'
    for category in ('subject', 'style'):
        cp(data_dir=data, method='instantx_st', out_dir=str(cp_dir), category=category,
           ablation_settings='full', few_shot=0, model_name=os.environ['CPP_JUDGE'], num_workers=4)
    pf(data_dir=data, method='instantx_st', out_dir=str(pf_dir),
       ablation_settings='full', model_name=os.environ['CPP_JUDGE'], num_workers=4)
    means = []
    for folder in (cp_dir, pf_dir):
        records = [json.loads(p.read_text(encoding='utf-8')) for p in folder.glob('*.json')]
        keys = {(r['collection_id'], r['prompt_index']) for r in records}
        assert len(records) == len(keys) == 1350, (folder, len(records), len(keys))
        assert all(type(r['score']) is int and 0 <= r['score'] <= 4 for r in records)
        means.append(sum(r['score'] for r in records) / 1350 / 4)
    summary = dict(judge=os.environ['CPP_JUDGE'], repeat=repeat, CP=means[0], PF=means[1],
                   CP_times_PF=means[0]*means[1], n=1350)
    (base / (repeat + '_summary.json')).write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(summary)
'@ | & $env:BENCH_EVAL_PY -
Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue
```

CP subject/style 합계 1,350건, PF 1,350건이며, 3회 반복하면 조건당 총 8,100 judge task다. parse/transport 재시도로 실제 요청 수는 늘어난다. 공식 코드는 raw response와 score를 저장하지만 완전한 API usage ledger를 남기지는 않으므로 API 측 usage도 기록한다. 예제 summary는 각 회차의 정규화 평균이다. 최종 표는 3회 평균±표준편차와 category별 값도 집계하고, 최종 CP×PF는 **3회 평균 CP × 3회 평균 PF**로 계산한다. baseline 등은 입력 폴더·method명·rating 출력 경로를 바꿔 별도로 평가한다. 모델명·prompt·이미지 변환·반복 횟수를 혼합하지 않는다.

### DreamBench — UNO 공개 코드 all-reference 보조 평가

UNO evaluator는 CUDA를 고정하고 top-level에서 DataLoader를 실행한다. 아래는 **Linux/WSL2의 CUDA 지원 평가 환경**에서 공개 코드의 all-reference 동작을 확인하는 보조 명령이다. selected-reference 주 평가 구현은 이 명령에 포함되어 있지 않다. 현재 CPU-only Windows Python에서 그대로 실행할 수는 없다. Modal에서 생성했다고 로컬 CUDA가 제공되는 것은 아니다.

```bash
cd /mnt/c/Users/82107/Downloads/AI_PROJECT/Orthogonalization-InstantX-IP-Adapter
export BENCH_ROOT="$PWD"
git clone https://github.com/bytedance/UNO.git external/UNO
git -C external/UNO checkout a563432dcfb4caf98df4368facb8102bf97d376c
git -C external/UNO submodule update --init datasets/dreambooth
python3 -m venv "$HOME/.venvs/orthogonalization-uno-eval-linux"
source "$HOME/.venvs/orthogonalization-uno-eval-linux/bin/activate"
# 이 평가 환경에는 CUDA를 지원하는 torch/torchvision이 필요함
python -m pip install torch==2.7.1 torchvision==0.22.1 'numpy<2' pandas scikit-learn packaging pillow tqdm ftfy regex 'git+https://github.com/openai/CLIP.git'
python -c 'import torch; assert torch.cuda.is_available(), "CUDA evaluation environment required"'
cd external/UNO
python - <<'PY'
import json, os, runpy, sys
original_load = json.load
def load_with_manifest_compat(file, *args, **kwargs):
    value = original_load(file, *args, **kwargs)
    if str(getattr(file, 'name', '')).replace('\\', '/').endswith('datasets/dreambench_singleip.json'):
        value = [dict(r, text=r['prompt'], image_path=r['image_paths'][0]) for r in value]
    return value
json.load = load_with_manifest_compat
root = os.environ['BENCH_ROOT']
sys.argv = ['evaluate', '--result_root', root+'/outputs/eval_export_v1/source_target/dreambench',
            '--save_dir', root+'/outputs/uno_metrics_v1/source_target']
runpy.run_path('eval/evaluate_clip_dino_score_single_subject.py', run_name='__main__')
PY
python -m pip freeze > "$BENCH_ROOT/outputs/uno_eval_requirements.txt"
```

이 환경 구축은 아직 실행하지 않았다. 외부 CLIP·torch hub DINO revision도 최종 실험 시 기록한다. UNO evaluator는 누락 이미지를 조용히 skip하므로 반드시 먼저 전체 750장 export를 성공시킨다. 로컬 CPU-only 상태로 모든 평가를 완료하려면 UNO 지표용 Modal evaluator를 별도로 구현해야 한다. 새 runner는 생성용이며 기존 `metric_client.py`는 XVerse용이므로 대체 사용하지 않는다.

## 4. 분석·보고 시 남길 항목

모든 조건에서 같은 reference/target/seed를 대응시키고 이미지별 지표 차이와 subject 평균을 저장한다. 계정별 평균을 단순히 평균내지 말고 case ID로 전체 결과를 합쳐 집계한다. 누락을 제외할 경우 건수와 이유를 명시하고 방법 간 공통 집합을 사용한다. 통계 불확실성은 같은 reference의 prompt들이 독립적이지 않으므로 reference 단위 paired bootstrap 등으로 추정하고, 1개 seed 결과는 그 범위로 해석한다.

projection 전후 visual embedding cosine, 편집 norm, basis rank, singular values, source/target 방향 수를 CP/PF·CLIP/DINO 변화와 대응시킨다. 배경 제거나 style 변경으로 CLIP-I가 낮아져도 PF는 개선될 수 있으므로 하나의 지표만으로 판단하지 않는다. 이는 실험 해석 제안이며 효과를 확인했다는 뜻은 아니다. 2건의 fallback은 플래그를 붙인 전체 결과와 해당 2건을 공통 제외한 민감도 분석을 함께 보고하면 판단하기 쉽다.

이번 변경의 검증 범위: shard 중복 없는 전체 커버리지, PNG/hash/설정 기반 resume, ++ 공식 collection명 export, 전체 2,100건의 입력 캐시 검증, Modal 클래스의 로컬 정의 및 mock 모델 재사용·입력 전달. 실제 GPU 생성·생성 품질·외부 지표 계산·GPT judge API 연결은 미검증이다.
