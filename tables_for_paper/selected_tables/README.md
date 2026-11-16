# Selected paper tables

Four PNG tables use compact booktabs-style rules, Times New Roman, and metric
order **CLIP-I, DINO-I, CLIP-T**. Values are cosine scores, rounded to four
decimals; all three metrics are higher-is-better. There are no PDF or preview files.

1. `01_dreambench_plus_comparison.png`: pooled Subject and Style, side by side.
   Subject is our grouping of Animal + Human + Object, equivalent to the earlier
   Non-Style subset. It averages the 130 references (1,170 cases); Style averages the 20
   references (180 cases). FreeCus and FreeGraftor Style results are `-`.
2. `02_dreambench_comparison.png`: overall subject macro, 30 references / 750 cases.
3. `03_ablation.png`: paired DreamBench and DreamBench++ overall results. All
   rows use scale **0.7**, matching the existing ablation experiments. Rows are
   baseline, source only, target only, source + target, gate only, and all three.
   Source caption / Target prompt checks indicate the corresponding projection
   is enabled; disabled components use red crosses. A cross does not mean captioning/prompt
   inputs are absent from the generation pipeline. Marks are drawn as vector shapes before
   rasterization so the body text can retain Times New Roman.
4. `04_scale_sensitivity.png`: paired overall benchmark results. IP-Adapter
   scales are 0.7/0.8/0.9; VIP-Adapter scales are 0.6/0.7/0.8/0.9/1.0.
   VIP-Adapter rows appear first, followed by IP-Adapter rows. Ablation marks
   use blue (#0000FF) checks and red (#FF0000) crosses; enabled components use
   a tapered check with a heavy short arm.

In method comparison tables, **VIP-Adapter** is scale **0.8**, while
**VIP-Adapter (0.7)** is the requested alternate result. IP-Adapter is the
InstantX baseline at scale 0.7. XLabs rows are excluded. VIP-Adapter names are bold;
values are not bolded or ranked. Method resolutions remain those of the original
experiments and are not changed by this formatting operation.

Aggregation is subject macro, matching the existing tables. Non-Style is pooled,
not an equal-weight mean of animal/human/object class means. Baseline scale 0.7
and VIP-Adapter scale 0.7 use the existing historical values retained at four
decimals. Other VIP-Adapter scales and ablations use stored RunPod subject-macro
summaries; comparison methods use the audited `../method_metrics.json` export.
No model inference, metric evaluation or new scale optimization was performed.

Run `python tables_for_paper/selected_tables/build_selected_tables.py` from the
repository root with `requirements.txt` installed. The generator reads the
existing repository JSON files, writes these four PNGs, and records source SHA-256
hashes and displayed rows in `table_audit.json`. Output resolution is 288 DPI.

Suggested caption definition: “Subject denotes the pooled animal, human, and
object categories of DreamBench++, while Style denotes its style category.”
The official dataset categories remain unchanged. See `terminology_review.md`
for the literature supporting this display terminology. Internal `non_style`
keys are preserved to retain the original aggregation and source mapping.

## TB / TF comparison variants

`build_comparison_variants.py` additionally creates eight PNGs without replacing
the four original tables:

| Benchmark | TB, scale 0.7 | TF, scale 0.7 | TB, scale 0.8 | TF, scale 0.8 |
|---|---|---|---|---|
| DreamBench++ Subject / Style | `01-TB-0.7.png` | `01-TF-0.7.png` | `01-TB-0.8.png` | `01-TF-0.8.png` |
| DreamBench | `02-TB-0.7.png` | `02-TF-0.7.png` | `02-TB-0.8.png` | `02-TF-0.8.png` |

TB compares UNO, DreamO, XVerse, USO and MOSAIC with one VIP-Adapter setting.
TF compares Diptych Prompting, FreeCus and FreeGraftor with one VIP-Adapter setting.
IP-Adapter and XLabs are excluded from all eight variants. VIP-Adapter appears
in a separate last row; inclusion in a TF comparison does not classify its
inherited IP-Adapter dependency as training-free. The explicit scale is shown in
the VIP-Adapter row in 0.7 files. Scale-0.8 variants use `VIP-Adapter(Ours)`.

Within each PNG and each subset/metric column, the highest score is **bold** and
the second-highest distinct score is underlined. Rank displayed four-decimal
values to respect the historical precision of scale 0.7. Tied values share dense
ranks. Missing `-` cells are not ranked. Style in the TF variants has only two
eligible methods (Diptych and VIP-Adapter). VIP-Adapter method names remain bold
independently of numeric ranking. Full displayed values, cell ranks and source
hashes are saved in `comparison_variants_audit.json`.

Run: `python tables_for_paper/selected_tables/build_comparison_variants.py`.

## Combined scale-0.8 tables with category column

`build_grouped_tables.py` adds `01-0.8.png` (DreamBench++ Subject / Style) and
`02-0.8.png` (DreamBench). IP-Adapter and scale 0.7 are excluded. A leftmost
column headed `Training Dependency` groups Training-based and Training-free baselines.
VIP-Adapter 0.8 appears independently in the last row with a blank group cell,
separated by a horizontal rule. This placement does not remove its inherited
personalization-trained IP-Adapter dependency; the proposed refinement itself
requires no new training.
Ranks are computed globally across both categories within each metric column,
using the same bold/underline convention and four-decimal tie policy as the
split variants. Existing PNGs are preserved. `grouped_tables_audit.json` records
the displayed values, categories and ranks.

Run: `python tables_for_paper/selected_tables/build_grouped_tables.py`.
