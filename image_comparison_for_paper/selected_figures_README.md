# Selected paper figures

`build_selected_figures.py` creates the nine requested figures in `selected_figures/`.
Existing full comparison grids and source images are preserved.

- Top row: UNO, DreamO, XVerse, USO, MOSAIC.
- Bottom row: VIP-Adapter (Ours), IP-Adapter, Diptych Prompting, FreeCus, FreeGraftor.
- Both VIP-Adapter and (Ours) use Times New Roman Bold. Entire prompt sentences
  (including quotation marks) use Times New Roman Italic with the original colors.
  Other text, including the Prompt: label, uses regular Times New Roman.
- Ours scales: 016-21 0.8, 015-21 0.7, 014-21 0.8, 013-21 0.8,
  177-08 0.7, 166-05 0.7, 164-03 0.7, 103-02 0.6, 043-04 0.7.
- IP-Adapter always uses `baseline_scale0.7`.
- Style cases without FreeCus or FreeGraftor results display `(impossible)`.
- Blue (#0070C0): preserved class or benchmark style label.
- Red (#FF0000): explicitly requested changes, including scene content in style cases.
- Black: articles, connecting phrases and general descriptions.
- Original benchmark prompts are retained verbatim, including `A ancient` in 177-08.
- Images retain their complete field of view and aspect ratio; no crops or image edits.
- Each case has one final PNG, rendered with Times New Roman at 4752 × 2560
  pixels and 288 DPI. No PDF or preview files are written.
- `figure_manifest.json` records input paths, image hashes, scales and colored spans.

To reproduce on this Mac, install `selected_figures_requirements.txt` in a virtual
environment and run `python build_selected_figures.py`. The font path is defined
at the top of the script. No network access is required after installation.

The original 2,100 comparison PNGs in `dreamBench/` and `dreambench_plus/`
remain local and are excluded by the repository `.gitignore`. Their generator
`build_grids.py`, `case_mapping.json` and `generation_report.json` are kept in Git
alongside the nine selected PNGs and their generator. Source images referenced
by the manifests must be available locally to regenerate either set of grids.
