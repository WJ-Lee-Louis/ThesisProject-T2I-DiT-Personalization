#!/usr/bin/env python3
"""Build one reference-and-six-condition comparison sheet per evaluated case."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

from PIL import Image, ImageDraw, ImageFont, ImageOps


DATASET = Path(__file__).resolve().parents[1]
EXPERIMENT = DATASET / "experiments" / "animal_compositional_control_6cond_v1"
DEFAULT_OUTPUT = EXPERIMENT / "tables" / "comparison_grids"
CONDITIONS = (
    ("baseline", "Baseline"),
    ("ssg", "SSG"),
    ("source_projection_ssg", "Source + SSG"),
    ("target_projection_ssg", "Target + SSG"),
    ("st_matched_projection_ssg", "ST-matched + SSG"),
    ("st_all_projection_ssg", "ST-all + SSG"),
)
METRICS = (
    ("dpg_with_dependency", "DPG"),
    ("ip_similarity", "IP-Sim"),
    ("aes", "AES"),
)
EXPECTED = {"subset": (29, 116, 4), "custom": (15, 90, 1)}
BG = "#F5F6F8"
INK = "#17212D"
MUTED = "#637183"
LINE = "#DCE1E7"
GREEN = "#147D57"
RED = "#B64B49"
CARD = 340
IMAGE = 314
GAP = 14
MARGIN = 26
CARD_H = 474
HEADER_H = 138
WIDTH = 2 * MARGIN + 4 * CARD + 3 * GAP
HEIGHT = HEADER_H + 2 * CARD_H + GAP + MARGIN


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    stem = "segoeuib.ttf" if bold else "segoeui.ttf"
    for path in (
        Path("C:/Windows/Fonts") / stem,
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


F_TITLE = font(27, True)
F_BODY = font(19)
F_LABEL = font(20, True)
F_SCORE = font(19)
F_SCORE_BOLD = font(19, True)
F_SMALL = font(16)


def text_width(draw: ImageDraw.ImageDraw, value: str, face: ImageFont.FreeTypeFont) -> int:
    return int(draw.textbbox((0, 0), value, font=face)[2])


def wrap(draw: ImageDraw.ImageDraw, value: str, face: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    lines: list[str] = []
    line = ""
    for word in value.split():
        candidate = f"{line} {word}".strip()
        if line and text_width(draw, candidate, face) > max_width:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines


def read_scores(benchmark: str) -> list[dict[str, str]]:
    path = EXPERIMENT / "evaluation" / "summaries" / benchmark / "per_image_scores.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def score(row: dict[str, str], metric: str) -> float:
    value = float(row[metric])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {metric}: {row['record_id']} {row['condition']}")
    return value


def image_path(benchmark: str, row: dict[str, str]) -> Path:
    return (
        EXPERIMENT
        / "generation"
        / benchmark
        / row["condition"]
        / "images"
        / row["record_id"]
        / f"seed_{int(row['seed']):04d}.png"
    )


def validate(benchmark: str, rows: list[dict[str, str]]) -> list[tuple[tuple[str, int], dict[str, dict[str, str]]]]:
    grouped: dict[tuple[str, int], dict[str, dict[str, str]]] = defaultdict(dict)
    refs: dict[str, set[str]] = defaultdict(set)
    seeds: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        key = (row["record_id"], int(row["seed"]))
        condition = row["condition"]
        if condition in grouped[key]:
            raise ValueError(f"Duplicate score row: {benchmark} {key} {condition}")
        grouped[key][condition] = row
        refs[row["reference_filename"]].add(row["record_id"])
        seeds[row["record_id"]].add(int(row["seed"]))
        for metric, _ in METRICS:
            score(row, metric)
        for path in (image_path(benchmark, row), DATASET / f"{benchmark}_bench" / row["reference_filename"]):
            if not path.is_file():
                raise FileNotFoundError(path)
    reference_count, case_count, seeds_per_record = EXPECTED[benchmark]
    if len(refs) != reference_count or len(grouped) != case_count:
        raise ValueError(f"{benchmark}: expected {reference_count} refs/{case_count} cases, got {len(refs)}/{len(grouped)}")
    if any(len(v) != seeds_per_record for v in seeds.values()):
        raise ValueError(f"{benchmark}: wrong seed count per record")
    names = {name for name, _ in CONDITIONS}
    for key, by_condition in grouped.items():
        if set(by_condition) != names:
            raise ValueError(f"{benchmark} {key}: missing/extra conditions {set(by_condition) ^ names}")
        if len({row["prompt"] for row in by_condition.values()}) != 1:
            raise ValueError(f"{benchmark} {key}: prompts differ across conditions")
        if len({row["reference_filename"] for row in by_condition.values()}) != 1:
            raise ValueError(f"{benchmark} {key}: reference differs across conditions")
    if benchmark == "custom" and any(len(v) != 6 for v in refs.values()):
        raise ValueError("custom: expected six distinct records per reference")
    return sorted(grouped.items())


def framed_image(canvas: Image.Image, path: Path, box: tuple[int, int, int, int]) -> None:
    with Image.open(path) as source:
        source = ImageOps.exif_transpose(source).convert("RGB")
        source.thumbnail((box[2] - box[0], box[3] - box[1]), Image.Resampling.LANCZOS)
        x = box[0] + (box[2] - box[0] - source.width) // 2
        y = box[1] + (box[3] - box[1] - source.height) // 2
        canvas.paste(source, (x, y))


def card_base(draw: ImageDraw.ImageDraw, x: int, y: int, height: int = CARD_H) -> None:
    draw.rounded_rectangle((x, y, x + CARD, y + height), radius=10, fill="white", outline=LINE, width=1)


def draw_metrics(draw: ImageDraw.ImageDraw, x: int, y: int, row: dict[str, str], baseline: dict[str, str]) -> None:
    for index, (metric, label) in enumerate(METRICS):
        raw = score(row, metric)
        delta = raw - score(baseline, metric)
        yy = y + index * 31
        draw.text((x, yy), label, font=F_SCORE_BOLD, fill=INK)
        draw.text((x + 84, yy), f"{raw:.2f}", font=F_SCORE, fill=INK)
        if row["condition"] == "baseline":
            delta_text, color = "+0.00", MUTED
        else:
            delta_text = f"{delta:+.2f}"
            color = GREEN if delta > 0.005 else RED if delta < -0.005 else MUTED
        draw.text((x + 204, yy), delta_text, font=F_SCORE_BOLD, fill=color)


def render(benchmark: str, key: tuple[str, int], rows: dict[str, dict[str, str]], destination: Path) -> None:
    record_id, seed = key
    baseline = rows["baseline"]
    canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((MARGIN, 19), f"{benchmark.upper()}  /  {record_id}  /  seed {seed}", font=F_TITLE, fill=INK)
    prompt_lines = wrap(draw, f"Prompt: {baseline['prompt']}", F_BODY, WIDTH - 2 * MARGIN)
    for i, line in enumerate(prompt_lines[:2]):
        draw.text((MARGIN, 61 + i * 25), line, font=F_BODY, fill=INK)
    draw.text((MARGIN, 115), "Dependency-aware DPG  |  Scores: 0-100  |  Delta vs paired baseline", font=F_SMALL, fill=MUTED)

    ref_x, top_y = MARGIN, HEADER_H
    card_base(draw, ref_x, top_y, 2 * CARD_H + GAP)
    draw.text((ref_x + 13, top_y + 12), "REFERENCE", font=F_LABEL, fill=INK)
    ref_box = (ref_x + 13, top_y + 46, ref_x + 13 + IMAGE, top_y + 46 + IMAGE)
    draw.rectangle(ref_box, fill=BG)
    framed_image(canvas, DATASET / f"{benchmark}_bench" / baseline["reference_filename"], ref_box)
    draw.text((ref_x + 13, top_y + 380), baseline["reference_filename"], font=F_BODY, fill=INK)
    draw.text((ref_x + 13, top_y + 420), f"Composition: {baseline['composition']}", font=F_SMALL, fill=MUTED)
    draw.line((ref_x + 13, top_y + 464, ref_x + CARD - 13, top_y + 464), fill=LINE, width=1)
    for i, line in enumerate(("Reading the cards", "Each card uses the same reference,", "prompt and seed.", "Green delta: higher than baseline", "Red delta: lower than baseline", "Metric values are per image.")):
        draw.text((ref_x + 13, top_y + 489 + i * 28), line, font=F_SMALL if i else F_LABEL, fill=INK if i == 0 else MUTED)

    for index, (condition, label) in enumerate(CONDITIONS):
        col, row_index = index % 3, index // 3
        x = MARGIN + CARD + GAP + col * (CARD + GAP)
        y = HEADER_H + row_index * (CARD_H + GAP)
        card_base(draw, x, y)
        draw.text((x + 13, y + 11), label, font=F_LABEL, fill=INK)
        image_box = (x + 13, y + 44, x + 13 + IMAGE, y + 44 + IMAGE)
        draw.rectangle(image_box, fill=BG)
        framed_image(canvas, image_path(benchmark, rows[condition]), image_box)
        draw.line((x + 13, y + 367, x + CARD - 13, y + 367), fill=LINE, width=1)
        draw_metrics(draw, x + 13, y + 376, rows[condition], baseline)

    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, "JPEG", quality=90, subsampling=0, optimize=True)


def url(value: str) -> str:
    return quote(value.replace(os.sep, "/"), safe="/._-+")


def write_index(output: Path, manifest: list[dict[str, object]]) -> None:
    data = json.dumps(manifest, ensure_ascii=False).replace("</", "<\\/")
    page = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Six-condition comparison grids</title><style>
body{font-family:Segoe UI,Arial,sans-serif;background:#f5f6f8;color:#17212d;margin:0 auto;max-width:1400px;padding:24px}
h1{font-size:27px;margin:0 0 8px}p{color:#637183;margin:0 0 18px}.controls{display:flex;gap:12px;flex-wrap:wrap;margin:18px 0}
input,select{font:inherit;border:1px solid #cbd4dc;border-radius:6px;padding:8px;background:white}input{min-width:260px}
table{border-collapse:collapse;width:100%;background:white;font-size:14px}th,td{padding:10px;border-bottom:1px solid #e3e8ec;text-align:left;vertical-align:middle}th{position:sticky;top:0;background:#eaf0f4}
tr:hover{background:#f4faf7}img{width:54px;height:54px;object-fit:contain;background:#eee;border-radius:4px}a{color:#175b8e;font-weight:600}
.plus{color:#147d57}.minus{color:#b64b49}.small{color:#637183;font-size:12px}
</style></head><body><h1>Six-condition comparison grids</h1>
<p>One grid per paired reference-prompt-seed case. Scores are per-image; deltas use the exact same case's baseline. DPG is dependency-aware. Click a record to open its full-size grid.</p>
<div class="controls"><select id="bench"><option value="all">Both benchmarks</option><option value="subset">Official subset</option><option value="custom">Custom benchmark</option></select>
<input id="search" placeholder="Search record, reference or prompt"><select id="condition"></select><select id="metric"><option value="dpg">DPG</option><option value="ip">IP-Sim</option><option value="aes">AES</option></select>
<select id="sort"><option value="desc">Largest gain first</option><option value="asc">Largest decline first</option><option value="record">Record order</option></select></div>
<p id="count"></p><table><thead><tr><th>Reference</th><th>Record / seed</th><th>Prompt</th><th>Selected score</th><th>Delta vs baseline</th><th>Grid</th></tr></thead><tbody id="rows"></tbody></table>
<script>const data=__DATA__;
const names=[['ssg','SSG'],['source_projection_ssg','Source + SSG'],['target_projection_ssg','Target + SSG'],['st_matched_projection_ssg','ST-matched + SSG'],['st_all_projection_ssg','ST-all + SSG']];
const byId=id=>document.getElementById(id), esc=s=>String(s).replace(/[&<>\"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}[c]));
byId('condition').innerHTML=names.map(([v,n])=>`<option value="${v}">${n}</option>`).join('');
function render(){const bench=byId('bench').value,query=byId('search').value.toLowerCase(),condition=byId('condition').value,metric=byId('metric').value,sort=byId('sort').value;
let items=data.filter(r=>(bench==='all'||r.benchmark===bench)&&`${r.record_id} ${r.reference_filename} ${r.prompt} ${r.seed}`.toLowerCase().includes(query));
if(sort!=='record')items.sort((a,b)=>(a.scores[condition][metric]-a.scores.baseline[metric]-(b.scores[condition][metric]-b.scores.baseline[metric]))*(sort==='desc'?-1:1));
byId('count').textContent=`${items.length} grids · subset 116 · custom 90`;
byId('rows').innerHTML=items.map(r=>{const val=r.scores[condition][metric],delta=val-r.scores.baseline[metric],color=delta>0.005?'plus':delta< -0.005?'minus':'';
return `<tr><td><img loading="lazy" src="${esc(r.reference_url)}" alt=""></td><td>${esc(r.benchmark)}<br><strong>${esc(r.record_id)}</strong><br><span class="small">seed ${r.seed}</span></td><td>${esc(r.prompt)}</td><td>${val.toFixed(2)}</td><td class="${color}">${delta>=0?'+':''}${delta.toFixed(2)}</td><td><a href="${esc(r.grid_url)}">Open grid</a></td></tr>`}).join('')}
['bench','search','condition','metric','sort'].forEach(id=>byId(id).addEventListener('input',render));render();</script></body></html>"""
    (output / "index.html").write_text(page.replace("__DATA__", data), encoding="utf-8")


def write_csv(output: Path, manifest: list[dict[str, object]]) -> None:
    columns = ["benchmark", "record_id", "seed", "reference_filename", "prompt", "composition", "grid_path"]
    for condition, _ in CONDITIONS:
        for metric, _ in METRICS:
            columns.extend((f"{condition}_{metric}", f"{condition}_delta_{metric}"))
    with (output / "grid_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for item in manifest:
            row = {field: item.get(field, "") for field in columns}
            scores = item["scores"]
            for condition, _ in CONDITIONS:
                for metric, _ in METRICS:
                    short = {"dpg_with_dependency": "dpg", "ip_similarity": "ip", "aes": "aes"}[metric]
                    row[f"{condition}_{metric}"] = scores[condition][short]
                    row[f"{condition}_delta_{metric}"] = scores[condition][short] - scores["baseline"][short]
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    cases = {benchmark: validate(benchmark, read_scores(benchmark)) for benchmark in EXPECTED}
    manifest: list[dict[str, object]] = []
    for benchmark, records in cases.items():
        for index, (key, rows) in enumerate(records, 1):
            record_id, seed = key
            filename = f"{record_id}_seed_{seed:04d}.jpg"
            destination = output / benchmark / filename
            render(benchmark, key, rows, destination)
            baseline = rows["baseline"]
            reference = DATASET / f"{benchmark}_bench" / baseline["reference_filename"]
            manifest.append({
                "benchmark": benchmark,
                "record_id": record_id,
                "seed": seed,
                "reference_filename": baseline["reference_filename"],
                "prompt": baseline["prompt"],
                "composition": baseline["composition"],
                "grid_path": str(destination.relative_to(output)).replace(os.sep, "/"),
                "grid_url": url(str(destination.relative_to(output))),
                "reference_url": url(os.path.relpath(reference, output)),
                "scores": {
                    condition: {
                        "dpg": score(rows[condition], "dpg_with_dependency"),
                        "ip": score(rows[condition], "ip_similarity"),
                        "aes": score(rows[condition], "aes"),
                    }
                    for condition, _ in CONDITIONS
                },
            })
            if index % 25 == 0 or index == len(records):
                print(f"{benchmark}: {index}/{len(records)}")
    write_csv(output, manifest)
    write_index(output, manifest)
    print(f"Wrote {len(manifest)} grids and index to {output}")


if __name__ == "__main__":
    main()
