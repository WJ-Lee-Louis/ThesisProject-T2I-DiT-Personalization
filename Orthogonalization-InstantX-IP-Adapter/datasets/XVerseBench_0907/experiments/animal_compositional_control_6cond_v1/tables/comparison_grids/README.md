# Reference–prompt–seed별 6-condition 비교 grid

`index.html`을 브라우저로 열면 subset/custom 전체 사례를 검색하고, 선택한
condition의 DPG·IP-Sim·AES 점수 변화량으로 정렬할 수 있다. 각 행의 `Open grid`를
누르면 해당 reference, prompt, seed에 대한 6-condition 비교 이미지를 연다.

- `subset/`: 공식 기반 subset, 29 record × 4 seed = 116장
- `custom/`: custom 평가, 15 reference × 6 prompt × 1 seed = 90장
- `grid_manifest.csv`: 각 grid에 대응하는 입력 정보, 이미지 경로, 조건별 원점수와 Δ

DPG는 `dpg_with_dependency`를 사용한다. 모든 점수는 이미지별 0–100 척도이며,
Δ는 동일 record와 seed의 baseline 이미지 점수를 뺀 값이다. 색은 점수의 증가·감소만
표시하며 통계적 유의성이나 정성적 우열을 뜻하지 않는다. 원본 생성 이미지와 reference
이미지는 복제하지 않고 각 grid 이미지에 축소 표시한다.

재생성 명령은 `datasets/XVerseBench_0907`의 상위 프로젝트 디렉터리에서:

```powershell
python datasets/XVerseBench_0907/metrics/make_comparison_grids.py
```

비교 grid에는 생성 이미지 원본을 그대로 포함한다. SAM2 mask/foreground crop은
이 grid에 포함하지 않으며, 필요한 사례는 `evaluation/ip/`에서 별도로 확인한다.
