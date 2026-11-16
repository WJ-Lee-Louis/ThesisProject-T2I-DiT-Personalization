# Subject / Style terminology decision

Reviewed on 2026-11-16. Decision: use **DreamBench++ Subject** and
**DreamBench++ Style** in the selected method comparison PNG. Subject is an
explicitly defined pooled evaluation subset, not an official renamed category.
Its membership and metric values remain identical to the previous Non-Style
subset: 45 animal + 20 human + 65 object references, 1,170 cases in total.

## Primary sources

- [DreamBooth official dataset](https://github.com/google/dreambooth/blob/main/README.md):
  uses “subjects” to encompass both live subjects and objects (9 and 21,
  respectively). This supports using subject as a broader term than live subject.
- [DreamBench++ paper, §2.2 and Table 1](https://arxiv.org/html/2406.16855v2):
  distinguishes live subject, object, and style image categories; live subject
  comprises animal and human. Reference image categories differ from prompt
  categories, which include photorealistic, style, and imaginative prompts.
- [DreamBench++ official repository](https://github.com/yuangpeng/dreambench_plus):
  explicitly documents live subject (animal/human), object, and style categories.
  Consequently, Subject = Animal + Human + Object must be defined by our caption;
  it should not be described as the benchmark's official two-way split.
- [USO paper, abstract and §1](https://arxiv.org/html/2508.18966v1):
  explicitly distinguishes subject-driven and style-driven generation, and
  unifies them in one personalization framework. This supports the conceptual
  pairing of Subject and Style, but does not establish our exact dataset split.
- [DreamO, §3.4](https://doi.org/10.1145/3757377.3763956):
  distinguishes identity, subject-driven, and style-driven customization data.
  This also shows that task terminology varies between methods, so explicit
  subset definitions are preferable to claiming universal naming conventions.

## Caption wording

“Subject denotes the pooled animal, human, and object categories of DreamBench++,
while Style denotes its style category. Results are averaged over reference
instances within each subset.”

The grouping is based on the **reference concept**, not on whether the requested
output is stylized. Subject examples can still have stylization prompts. Avoid
replacing Subject with Live Subject (which excludes objects), Identity (which can
suggest face-specific evaluation), or Photorealistic (a different partition).

Only the displayed group header changes. Keep internal `non_style` identifiers,
reference-macro aggregation, numerical values, and the Style exclusions for
FreeCus/FreeGraftor unchanged.
