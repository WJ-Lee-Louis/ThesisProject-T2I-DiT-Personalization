# Ablation image grids

Two PNG figures, each containing one horizontal row in this order:
Reference, IP-Adapter, w/Source Proj., w/Target Proj.,
w/Source+Target Proj., w/SSG, Ours.

Method labels are above the images. The quoted reference identity uses larger
text on two lines (monochromatic / ink sketch), aligned with the prompt band.
The divider starts at the second column. The Prompt label and sentence form
one closely spaced group centered over columns two through seven.

| Case | Ours scale | Other generated comparison images |
| --- | --- | --- |
| 175-06 | 0.7 | scale 0.7 |
| 175-05 | 0.6 | scale 0.7 |

The 175-05 figure therefore compares different scales; it should not be
described as an ablation with scale held constant. Scales are documented here
and in `figure_manifest.json`, without adding extra labels inside the figures.

The shared reference is the DreamBench++ style image whose execution-record
identity term is `monochromatic ink sketch`. Blue denotes this preserved style
identity; red denotes requested subject, attributes, and scene changes; black
denotes generic language. Text uses Times New Roman, with bold Ours and italic
prompt sentences. Source images are fitted without cropping or modification.

## Reproduce

From this directory, using Python 3.9+:

```sh
python3 -m pip install Pillow==11.3.0 reportlab==4.4.5 PyMuPDF==1.26.5
python3 build_ablation_figures.py
```

The script imports `../build_selected_figures.py` and reads
`../case_mapping.json`. It requires the original image directories and the
Times New Roman fonts in macOS Supplemental fonts. Each output is 5276 × 1400
pixels. Rendering uses an in-memory PDF; no PDF or preview file is saved.
The manifest records source paths, hashes, and available execution metadata.
