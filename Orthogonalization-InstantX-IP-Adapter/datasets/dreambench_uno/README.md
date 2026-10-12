# DreamBench — UNO single-subject references

Selection: [a563432dcfb4caf98df4368facb8102bf97d376c](https://raw.githubusercontent.com/bytedance/UNO/a563432dcfb4caf98df4368facb8102bf97d376c/datasets/dreambench_singleip.json). DreamBooth submodule: `4f887af7970a06fc0cd3adaa1d0b368547d6a1d0`.

30 subjects × one original image, verified against the pinned DreamBooth Git blob. Original filenames and subject folders are preserved under `images/`.

`manifest.json` contains labels, image hashes and 25 UNO prompts per subject. `evaluation_750.json` contains 750 ready-to-use rows; image paths are relative to the project root. This reproduces UNO image selection and prompt text, not UNO model preprocessing.
