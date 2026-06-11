# Dataset notes

Checked for reusable reference-image data around XLabs and Black Forest Labs FLUX repositories.

## XLabs `x-flux`

Repository: https://github.com/XLabs-AI/x-flux

Relevant files:

- `image_datasets/dataset.py`
- `image_datasets/canny_dataset.py`
- `assets/example_images/statue.jpg`

Finding:

- `image_datasets/` contains PyTorch dataset loader code, not a downloadable image dataset.
- The README documents an expected local training format:

```text
images/
  1.png
  1.json
  2.png
  2.json
```

- `assets/example_images/statue.jpg` is only a small demo reference image for smoke testing IP-Adapter inference.

Recommendation:

- Do not treat XLabs `image_datasets/` as a research dataset.
- Do not download the full repo just for reference images.
- Use `statue.jpg` only as a quick functionality check if needed.

## Black Forest Labs `flux`

Repository: https://github.com/black-forest-labs/flux

Relevant files:

- `assets/cup.png`
- `assets/cup_mask.png`
- `assets/robot.webp`
- `assets/docs/*.png`
- generated grids such as `assets/grid.jpg`, `assets/dev_grid.jpg`

Finding:

- These are documentation/demo assets, masks, control examples, and generated grids.
- They are not organized as a personalization reference-image benchmark.
- They do not provide identity-consistent multi-image subjects or clear pose/identity splits.

Recommendation:

- Do not use BFL repo assets as the main reference-image dataset for personalization research.
- They are acceptable only for smoke tests or pipeline debugging.

## Better direction for personalization experiments

For identity-pose disentanglement, prefer a dataset with subject identity and pose variation. Useful properties:

- multiple images per identity
- visible pose variation
- clean license for research use
- consent/ethics documentation for human faces, if using people
- metadata or captions if text-conditioned evaluation is needed

Good next-step candidates to evaluate separately include subject-driven personalization benchmarks such as DreamBooth-style datasets or licensed face/person datasets, depending on whether the research needs real human identity, objects, or stylized characters.
