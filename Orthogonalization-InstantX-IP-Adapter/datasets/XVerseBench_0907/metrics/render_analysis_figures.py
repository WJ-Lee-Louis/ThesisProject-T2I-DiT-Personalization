#!/usr/bin/env python3
"""Render publication-ready tables and figures for the six-condition experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mpl_colors
from matplotlib import font_manager
from PIL import Image


CONDITIONS = [
    "baseline",
    "ssg",
    "source_projection_ssg",
    "target_projection_ssg",
    "st_matched_projection_ssg",
    "st_all_projection_ssg",
]
COMPARISON_CONDITIONS = [
    "ssg",
    "source_projection_ssg",
    "target_projection_ssg",
    "st_matched_projection_ssg",
    "st_all_projection_ssg",
]
QUALITATIVE_CONDITIONS = ["st_matched_projection_ssg", "st_all_projection_ssg"]
LABELS = {
    "baseline": "Baseline",
    "ssg": "SSG",
    "source_projection_ssg": "Source + SSG",
    "target_projection_ssg": "Target + SSG",
    "st_matched_projection_ssg": "ST-matched + SSG",
    "st_all_projection_ssg": "ST-all + SSG",
}
SHORT_LABELS = {
    "ssg": "SSG",
    "source_projection_ssg": "Source",
    "target_projection_ssg": "Target",
    "st_matched_projection_ssg": "ST-Matched",
    "st_all_projection_ssg": "ST-All",
}
PALETTE = {
    "ssg": "#7A7A7A",
    "source_projection_ssg": "#4C78A8",
    "target_projection_ssg": "#F58518",
    "st_matched_projection_ssg": "#54A24B",
    "st_all_projection_ssg": "#B279A2",
}
MARKERS = {
    "ssg": "o",
    "source_projection_ssg": "s",
    "target_projection_ssg": "D",
    "st_matched_projection_ssg": "^",
    "st_all_projection_ssg": "P",
}
BENCHMARKS = ("subset", "custom")
PAPER_BENCHMARK_LABELS = {"subset": "Official Subset", "custom": "Custom Benchmark"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_style() -> None:
    candidates = [
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("C:/Windows/Fonts/NanumGothic.ttf"),
    ]
    for path in candidates:
        if path.exists():
            font_manager.fontManager.addfont(path)
            family = font_manager.FontProperties(fname=path).get_name()
            plt.rcParams["font.family"] = family
            break
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFAFA",
            "axes.edgecolor": "#B8B8B8",
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "font.size": 10,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def configure_publication_style() -> None:
    """Use a restrained serif style for paper-ready quantitative figures."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 0.8,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "font.size": 10,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def load_condition_summaries(experiment_root: Path) -> dict[str, dict[str, dict]]:
    result: dict[str, dict[str, dict]] = {}
    for benchmark in BENCHMARKS:
        path = experiment_root / "evaluation" / "summaries" / benchmark / "summary_by_condition.csv"
        rows = read_csv(path)
        parsed = {}
        for row in rows:
            parsed[row["condition"]] = {
                "n": int(row["n"]),
                "aes": float(row["aes"]),
                "dpg": float(row["dpg_with_dependency"]),
                "ip": float(row["ip_similarity"]),
            }
        result[benchmark] = parsed
    return result


def render_main_table(summaries: dict, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(16.0, 5.4))
    ax.axis("off")
    method_labels = {
        "baseline": "Baseline",
        "ssg": "SSG",
        "source_projection_ssg": "Source Projection + SSG",
        "target_projection_ssg": "Target Projection + SSG",
        "st_matched_projection_ssg": "ST-Matched Projection + SSG",
        "st_all_projection_ssg": "ST-All Projection + SSG",
    }
    columns = [
        "Method",
        "DPG↑",
        "ΔDPG",
        "IP-Sim↑",
        "ΔIP",
        "AES↑",
        "ΔAES",
        "DPG↑",
        "ΔDPG",
        "IP-Sim↑",
        "ΔIP",
        "AES↑",
        "ΔAES",
    ]
    rows = []
    for condition in CONDITIONS:
        row = [method_labels[condition]]
        for benchmark in ("subset", "custom"):
            values = summaries[benchmark][condition]
            base = summaries[benchmark]["baseline"]
            row.extend(
                [
                    f"{values['dpg']:.2f}",
                    "—" if condition == "baseline" else f"{values['dpg'] - base['dpg']:+.2f}",
                    f"{values['ip']:.2f}",
                    "—" if condition == "baseline" else f"{values['ip'] - base['ip']:+.2f}",
                    f"{values['aes']:.2f}",
                    "—" if condition == "baseline" else f"{values['aes'] - base['aes']:+.2f}",
                ]
            )
        rows.append(row)

    col_widths = [0.235] + [0.06375] * 12
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        colWidths=col_widths,
        bbox=[0.0, 0.15, 1.0, 0.66],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.7)
    table.scale(1, 1.15)

    for (row, col), cell in table.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("white")
        cell.set_linewidth(0)
        cell.PAD = 0.08
        if row == 0:
            cell.get_text().set_fontweight("bold")
            cell.visible_edges = "B"
            cell.set_edgecolor("black")
            cell.set_linewidth(0.9)
        if col == 0:
            cell.get_text().set_ha("left")
        if row == len(CONDITIONS):
            cell.visible_edges = "B"
            cell.set_edgecolor("black")
            cell.set_linewidth(1.1)

    for benchmark_index, benchmark in enumerate(("subset", "custom")):
        data = summaries[benchmark]
        start_col = 1 + benchmark_index * 6
        for offset, key in ((0, "dpg"), (2, "ip"), (4, "aes")):
            best_condition = max(CONDITIONS, key=lambda item: data[item][key])
            table[(CONDITIONS.index(best_condition) + 1, start_col + offset)].get_text().set_fontweight("bold")
        for offset, key in ((1, "dpg"), (3, "ip"), (5, "aes")):
            best_condition = max(
                COMPARISON_CONDITIONS,
                key=lambda item: data[item][key] - data["baseline"][key],
            )
            table[(CONDITIONS.index(best_condition) + 1, start_col + offset)].get_text().set_fontweight("bold")

    ax.text(
        0,
        0.965,
        "Table 1: Quantitative results across six generation conditions on XVerseBench.",
        transform=ax.transAxes,
        fontsize=16,
        va="top",
    )
    ax.plot([0, 1], [0.90, 0.90], transform=ax.transAxes, color="black", linewidth=1.1, clip_on=False)
    ax.text(0.426, 0.855, "Official Subset (n=116)", transform=ax.transAxes, ha="center", fontsize=13, fontweight="bold")
    ax.text(0.809, 0.855, "Custom Benchmark (n=90)", transform=ax.transAxes, ha="center", fontsize=13, fontweight="bold")
    ax.plot([0.235, 0.6175], [0.83, 0.83], transform=ax.transAxes, color="black", linewidth=0.7)
    ax.plot([0.6175, 1.0], [0.83, 0.83], transform=ax.transAxes, color="black", linewidth=0.7)
    ax.plot([0.235, 0.235], [0.15, 0.81], transform=ax.transAxes, color="black", linewidth=0.6)
    ax.plot([0.6175, 0.6175], [0.15, 0.81], transform=ax.transAxes, color="black", linewidth=0.6)
    ax.text(
        0,
        0.055,
        "DPG denotes the dependency-aware score. Δ reports paired changes from Baseline for matched record–seed pairs. Best results are in bold.",
        transform=ax.transAxes,
        fontsize=9.5,
        va="bottom",
    )
    fig.savefig(output, dpi=200)
    plt.close(fig)


def pareto_indices(points: list[tuple[float, float]]) -> list[int]:
    result = []
    for index, (x_value, y_value) in enumerate(points):
        dominated = any(
            other != index
            and other_x >= x_value
            and other_y >= y_value
            and (other_x > x_value or other_y > y_value)
            for other, (other_x, other_y) in enumerate(points)
        )
        if not dominated:
            result.append(index)
    return sorted(result, key=lambda index: points[index][0])


def render_tradeoff(summaries: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
    for ax, benchmark in zip(axes, BENCHMARKS, strict=True):
        data = summaries[benchmark]
        points = []
        for condition in CONDITIONS:
            x_value = data[condition]["ip"]
            y_value = data[condition]["dpg"]
            points.append((x_value, y_value))
            marker = "X" if condition == "baseline" else MARKERS[condition]
            edge_color = "#202020" if condition == "baseline" else PALETTE[condition]
            marker_size = 115 if condition != "ssg" else 145
            ax.scatter(
                x_value,
                y_value,
                s=marker_size,
                marker=marker,
                facecolor="white",
                edgecolor=edge_color,
                linewidth=2.2,
                zorder=4,
            )
            offsets_by_benchmark = {
                "subset": {
                    "baseline": (-54, 8),
                    "ssg": (-49, -22),
                    "source_projection_ssg": (11, 10),
                    "target_projection_ssg": (8, -19),
                    "st_matched_projection_ssg": (9, 9),
                    "st_all_projection_ssg": (9, 9),
                },
                "custom": {
                    "baseline": (9, -19),
                    "ssg": (-51, 8),
                    "source_projection_ssg": (9, 9),
                    "target_projection_ssg": (9, -19),
                    "st_matched_projection_ssg": (9, 9),
                    "st_all_projection_ssg": (9, 9),
                },
            }
            point_label = "Baseline" if condition == "baseline" else SHORT_LABELS[condition]
            ax.annotate(
                point_label,
                (x_value, y_value),
                xytext=offsets_by_benchmark[benchmark][condition],
                textcoords="offset points",
                fontsize=9,
                color=edge_color,
                arrowprops={"arrowstyle": "-", "color": edge_color, "lw": 0.6},
            )

        frontier = pareto_indices(points)
        ax.plot(
            [points[index][0] for index in frontier],
            [points[index][1] for index in frontier],
            color="#252525",
            linewidth=1.5,
            linestyle="--",
            alpha=0.7,
            zorder=2,
            label="2D Pareto frontier",
        )
        ax.margins(x=0.12, y=0.10)
        panel = "(a)" if benchmark == "subset" else "(b)"
        ax.set_title(f"{panel} {PAPER_BENCHMARK_LABELS[benchmark]}", pad=8)
        ax.set_xlabel("IP-Sim Score  (higher is better)")
        ax.set_ylabel("Dependency-Aware DPG Score  (higher is better)")
        ax.grid(True, color="#D8D8D8", linewidth=0.55, alpha=0.65)
        ax.tick_params(direction="in", top=True, right=True)

    fig.text(
        0.08,
        0.018,
        "Figure 1: Absolute-score trade-off between compositional fidelity and identity preservation. "
        "Dashed lines connect the 2D Pareto frontier; AES is reported separately in Table 1.",
        ha="left",
        fontsize=10,
        color="black",
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.18, wspace=0.24)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def render_composition_heatmap(experiment_root: Path, output: Path) -> None:
    data_by_benchmark = {}
    maxima = []
    minima = []
    for benchmark in BENCHMARKS:
        path = experiment_root / "evaluation" / "summaries" / benchmark / "paired_delta_vs_baseline.csv"
        rows = read_csv(path)
        compositions = ["O+A", "O+B", "O+attribute", "O+A+B"] if benchmark == "subset" else ["O+A", "O+B", "O+A+B"]
        lookup = {(row["condition"], row["composition"]): row for row in rows}
        values = np.full((len(COMPARISON_CONDITIONS), len(compositions)), np.nan)
        counts = np.zeros_like(values, dtype=int)
        for row_index, condition in enumerate(COMPARISON_CONDITIONS):
            for column_index, composition in enumerate(compositions):
                row = lookup.get((condition, composition))
                if row:
                    values[row_index, column_index] = float(row["delta_dpg_with_dependency"])
                    counts[row_index, column_index] = int(row["n"])
        data_by_benchmark[benchmark] = (compositions, values, counts)
        maxima.append(float(np.nanmax(values)))
        minima.append(float(np.nanmin(values)))

    limit = max(max(maxima), abs(min(minima)), 1.0)
    norm = mpl_colors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.0), gridspec_kw={"width_ratios": [1.15, 1]})
    for ax, benchmark in zip(axes, BENCHMARKS, strict=True):
        compositions, values, counts = data_by_benchmark[benchmark]
        ax.imshow(values, cmap="RdYlGn", norm=norm, aspect="auto")
        ax.set_xticks(range(len(compositions)), compositions)
        ax.set_yticks(range(len(COMPARISON_CONDITIONS)), [SHORT_LABELS[item] for item in COMPARISON_CONDITIONS])
        panel = "(a)" if benchmark == "subset" else "(b)"
        ax.set_title(f"{panel} {PAPER_BENCHMARK_LABELS[benchmark]}", pad=8)
        ax.set_xlabel("Composition")
        ax.tick_params(length=0)
        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                value = values[row_index, column_index]
                if np.isnan(value):
                    continue
                text_color = "white" if abs(value) > limit * 0.62 else "#222222"
                ax.text(
                    column_index,
                    row_index,
                    f"{value:+.1f}\n(n={counts[row_index, column_index]})",
                    ha="center",
                    va="center",
                    fontsize=9.5,
                    color=text_color,
                    fontweight="bold" if value == np.nanmax(values) else "normal",
                )
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.text(
        0.14,
        0.018,
        "Figure 2: Composition-wise paired ΔDPG. Values in parentheses denote paired sample counts; "
        "the Official Subset is compositionally imbalanced.",
        ha="left",
        fontsize=10,
        color="black",
    )
    fig.subplots_adjust(left=0.12, right=0.98, top=0.92, bottom=0.18, wspace=0.22)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def index_generated_images(experiment_root: Path, benchmark: str) -> dict[tuple[str, str, int], Path]:
    index = {}
    for condition in CONDITIONS:
        root = experiment_root / "generation" / benchmark / condition
        manifest = read_json(root / "generated_images_manifest.json")
        for image in manifest["images"]:
            key = (condition, image["record_id"], int(image["seed"]))
            index[key] = root / image["image_path"]
    return index


def index_references(dataset_root: Path, benchmark: str) -> dict[str, Path]:
    manifest = read_json(dataset_root / f"{benchmark}_bench" / "benchmark_manifest.json")
    return {
        record["record_id"]: dataset_root / record["reference_path"]
        for record in manifest["records"]
    }


def select_qualitative_cases(experiment_root: Path, benchmark: str) -> list[dict]:
    rows = read_csv(experiment_root / "evaluation" / "summaries" / benchmark / "per_image_scores.csv")
    baseline = {
        (row["record_id"], int(row["seed"])): row
        for row in rows
        if row["condition"] == "baseline"
    }
    candidates = []
    for row in rows:
        if row["condition"] not in QUALITATIVE_CONDITIONS:
            continue
        base = baseline[(row["record_id"], int(row["seed"]))]
        item = dict(row)
        item["seed_int"] = int(row["seed"])
        item["base_dpg"] = float(base["dpg_with_dependency"])
        item["base_ip"] = float(base["ip_similarity"])
        item["dpg"] = float(row["dpg_with_dependency"])
        item["ip"] = float(row["ip_similarity"])
        item["delta_dpg"] = item["dpg"] - item["base_dpg"]
        item["delta_ip"] = item["ip"] - item["base_ip"]
        candidates.append(item)

    rank = {condition: index for index, condition in enumerate(QUALITATIVE_CONDITIONS)}

    def tie_key(item: dict) -> tuple[int, str, int]:
        return rank[item["condition"]], item["record_id"], item["seed_int"]

    max_dpg = sorted(candidates, key=lambda item: (-item["delta_dpg"], *tie_key(item)))[0]
    max_ip_drop = sorted(candidates, key=lambda item: (item["delta_ip"], *tie_key(item)))[0]
    median = float(np.median([item["delta_dpg"] for item in candidates]))
    median_case = sorted(
        candidates,
        key=lambda item: (abs(item["delta_dpg"] - median), *tie_key(item)),
    )[0]
    return [
        {**max_dpg, "case_label": "Maximum DPG gain"},
        {**max_ip_drop, "case_label": "Largest IP-Sim regression"},
        {**median_case, "case_label": f"Median case (median ΔDPG={median:+.1f})"},
    ]


def show_image(ax, path: Path, *, mask: bool = False) -> None:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L" if mask else "RGB"))
    if mask:
        ax.imshow(array, cmap="gray", vmin=0, vmax=255)
    else:
        ax.imshow(array)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#D0D0D0")
        spine.set_linewidth(0.8)


def render_qualitative_grid(experiment_root: Path, dataset_root: Path, benchmark: str, output: Path) -> list[dict]:
    cases = select_qualitative_cases(experiment_root, benchmark)
    generated = index_generated_images(experiment_root, benchmark)
    references = index_references(dataset_root, benchmark)
    columns = [
        "Reference",
        "Baseline",
        "Compared Condition",
        "Baseline Mask",
        "Compared Mask",
        "Compared Foreground Crop",
    ]
    fig, axes = plt.subplots(3, 6, figsize=(17.5, 10.2))
    figure_number = "3(a)" if benchmark == "subset" else "3(b)"
    fig.suptitle(
        f"Figure {figure_number}: Qualitative Comparison on the {PAPER_BENCHMARK_LABELS[benchmark]}",
        fontsize=16,
        fontweight="normal",
        x=0.025,
        ha="left",
        y=0.99,
    )
    selected_manifest = []
    row_notes = []
    for row_index, case in enumerate(cases):
        record_id = case["record_id"]
        seed = case["seed_int"]
        condition = case["condition"]
        baseline_image = generated[("baseline", record_id, seed)]
        comparison_image = generated[(condition, record_id, seed)]
        ip_root = experiment_root / "evaluation" / "ip" / "gen" / benchmark
        baseline_artifact = ip_root / "baseline" / record_id / f"seed_{seed}"
        comparison_artifact = ip_root / condition / record_id / f"seed_{seed}"
        paths = [
            references[record_id],
            baseline_image,
            comparison_image,
            baseline_artifact / "candidate_00_mask.png",
            comparison_artifact / "candidate_00_mask.png",
            comparison_artifact / "candidate_00_crop.png",
        ]
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(path)
        for column_index, (title, path) in enumerate(zip(columns, paths, strict=True)):
            ax = axes[row_index, column_index]
            show_image(ax, path, mask=column_index in (3, 4))

        prompt = textwrap.fill(case["prompt"], width=56)
        row_note = (
            f"{case['case_label']}  |  {SHORT_LABELS[condition]}  |  {record_id}, seed {seed}\n"
            f"DPG {case['base_dpg']:.1f} → {case['dpg']:.1f} ({case['delta_dpg']:+.1f})  |  "
            f"IP {case['base_ip']:.2f} → {case['ip']:.2f} ({case['delta_ip']:+.2f})\n{prompt}"
        )
        row_notes.append(row_note)
        selected_manifest.append(
            {
                "case": case["case_label"],
                "condition": condition,
                "record_id": record_id,
                "seed": seed,
                "prompt": case["prompt"],
                "delta_dpg": case["delta_dpg"],
                "delta_ip": case["delta_ip"],
                "paths": [str(path.relative_to(experiment_root)) if path.is_relative_to(experiment_root) else str(path) for path in paths],
            }
        )

    fig.subplots_adjust(left=0.025, right=0.99, top=0.80, bottom=0.025, wspace=0.06, hspace=0.55)
    for column_index, title in enumerate(columns):
        position = axes[0, column_index].get_position()
        fig.text(
            (position.x0 + position.x1) / 2,
            0.855,
            title,
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
        )
    for row_index, row_note in enumerate(row_notes):
        position = axes[row_index, 0].get_position()
        fig.text(0.025, position.y1 + 0.006, row_note, fontsize=8.8, va="bottom", ha="left")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return selected_manifest


def write_index(output_root: Path, source_paths: list[Path], outputs: list[Path], qualitative: dict) -> None:
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {str(path): sha256(path) for path in source_paths},
        "outputs": {
            path.name: {
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
                "dimensions": list(Image.open(path).size),
            }
            for path in outputs
        },
        "qualitative_selection": qualitative,
    }
    (output_root / "figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    readme = """# 주요 결과 시각화

| 파일 | 내용 |
|---|---|
| `01_main_results_table.png` | subset/custom 조건별 DPG·IP-Sim·AES와 paired delta |
| `02_tradeoff_scatter.png` | ΔIP–ΔDPG trade-off, ΔAES 색상, 2D Pareto frontier |
| `03_composition_dpg_heatmap.png` | 조건·composition별 dependency-aware paired ΔDPG와 표본 수 |
| `04_qualitative_subset.png` | subset의 최대 DPG 향상·최대 IP 하락·중앙값 사례와 SAM2 산출물 |
| `05_qualitative_custom.png` | custom의 최대 DPG 향상·최대 IP 하락·중앙값 사례와 SAM2 산출물 |
| `figure_manifest.json` | 입력/출력 SHA-256, 이미지 크기, 정성 사례 선택 metadata |

DPG는 두 benchmark 모두 `dpg_with_dependency`를 사용한다. Δ는 동일 record·seed의 baseline 대비 차이다.
정성 grid는 분석 가이드에 따라 ST-matched와 ST-all 후보에서 결정론적으로 선택한다.

재생성:

```powershell
python datasets/XVerseBench_0907/metrics/render_analysis_figures.py
```
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    dataset_root = Path(__file__).resolve().parents[1]
    default_experiment = dataset_root / "experiments" / "animal_compositional_control_6cond_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=default_experiment)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_root = args.experiment_root.resolve()
    dataset_root = experiment_root.parents[1]
    output_root = (args.output_root or experiment_root / "tables").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = load_condition_summaries(experiment_root)
    outputs = [
        output_root / "01_main_results_table.png",
        output_root / "02_tradeoff_scatter.png",
        output_root / "03_composition_dpg_heatmap.png",
        output_root / "04_qualitative_subset.png",
        output_root / "05_qualitative_custom.png",
    ]
    configure_publication_style()
    render_main_table(summaries, outputs[0])
    render_tradeoff(summaries, outputs[1])
    render_composition_heatmap(experiment_root, outputs[2])
    qualitative = {
        "subset": render_qualitative_grid(experiment_root, dataset_root, "subset", outputs[3]),
        "custom": render_qualitative_grid(experiment_root, dataset_root, "custom", outputs[4]),
    }
    source_paths = [
        experiment_root / "evaluation" / "summaries" / benchmark / filename
        for benchmark in BENCHMARKS
        for filename in ("summary_by_condition.csv", "paired_delta_vs_baseline.csv", "per_image_scores.csv")
    ]
    write_index(output_root, source_paths, outputs, qualitative)
    print(f"Rendered {len(outputs)} figures to {output_root}")


if __name__ == "__main__":
    main()
