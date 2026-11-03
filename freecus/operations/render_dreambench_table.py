"""Update the supplied DreamBench table with verified FreeCus subject-macro scores."""
import csv
import json
from pathlib import Path


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from freecus.manifest import sha256

    base = Path(__file__).resolve().parents[1]
    source = base / "outputs/metrics/dreambench_scores.json"
    result = json.loads(source.read_text(encoding="utf-8"))
    summary = result["summary"]["dreambench/freecus"]
    if summary["cases"] != 750 or summary["subjects"] != 30 or result["evaluation_selection"]["name"] != "all":
        raise ValueError("Expected full DreamBench: 750 cases / 30 references")
    metrics = ["CLIP-I", "CLIP-T", "DINO-I"]
    freecus = summary["subject_macro"]
    rows = [
        ("InstantX-IP-Adapter (baseline)", [0.8319, 0.2939, 0.6900]),
        ("Ours (scale 0.7)", [0.8022, 0.3123, 0.6548]),
        ("Ours (scale best*)", [0.8144, 0.3078, 0.6726]),
        ("FreeCus", [freecus[m] for m in metrics]),
        ("UNO [T]", [0.8356, 0.3039, 0.7604]),
        ("USO [T]", [0.8301, 0.3178, 0.7686]),
    ]
    out = base / "outputs/figures"
    out.mkdir(parents=True, exist_ok=True)
    stem = out / "dreambench_method_comparison_freecus_updated"
    plt.rcParams.update({"font.family": "DejaVu Serif", "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(15, 6.44), dpi=200, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 3000)
    ax.set_ylim(1288, 0)
    ax.axis("off")

    def text(x, y, value, size=15, color="black", align="left"):
        ax.text(x, y, value, fontsize=size, color=color, ha=align, va="center")

    text(132, 79, "DreamBench", 25)
    text(132, 156, "Method comparison", 15, "#4a4a4a")
    for y, width in [(248, 1.7), (357, 0.85), (934, 1.7)]:
        ax.plot([132, 2868], [y, y], color="black", linewidth=width)
    text(160, 304, "Method", 15)
    columns = [1780, 2254, 2728]
    for x, metric in zip(columns, metrics):
        text(x, 304, metric + " ↑", 15, align="center")
    text(160, 417, "Overall (750 images / method)", 14)
    for y, (method, values) in zip([492, 570, 648, 726, 804, 882], rows):
        text(160, y, method, 15)
        for x, value in zip(columns, values):
            text(x, y, f"{value:.4f}", 15, align="center")
    notes = [
        "Subject macro average; single selected reference; higher is better.",
        "FreeCus: newly evaluated; 1024 × 1024; seed 42; 750 images / 30 subjects.",
        "Baseline and Ours scale 0.7: historical screenshot values (4-decimal precision).",
        "* best row uses scale 0.8 as requested; this is a provisional choice, not an optimized scale.",
        "[T] tuning-based (UNO, USO). External values copied from the supplied comparison PNG.",
    ]
    for y, note in zip([1004, 1055, 1106, 1157, 1208], notes):
        text(160, y, note, 10.7, "#4a4a4a")
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(stem.with_suffix("." + extension), dpi=200, facecolor="white")
    plt.close(fig)
    with stem.with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Method", *metrics])
        writer.writerows((method, *(f"{v:.4f}" for v in values)) for method, values in rows)
    provenance = dict(freecus_source=str(source.resolve()), freecus_source_sha256=sha256(source),
                      aggregation="subject_macro", freecus_full_precision=freecus,
                      other_rows_source="Values transcribed unchanged from the user-supplied figure; not recomputed",
                      preserved_scale_best_note="scale 0.8; provisional, not optimized", notes=notes)
    stem.with_suffix(".json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    print(stem.with_suffix(".png").resolve())
    print(json.dumps({metric: f"{freecus[metric]:.4f}" for metric in metrics}))


if __name__ == "__main__":
    main()
