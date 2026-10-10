# Legacy benchmark component annotations

This directory contains benchmark annotation data for comparison and offline
inspection. These records are legacy annotations, not validated responses from
the current explicit-identity LLM pipeline. Their `identity_hint` field is audit
metadata and is not used to infer the preservation concept in the current pipeline.

## Retained data

- `dreambench_components.jsonl`: 750 records.
- `dreambench_plus_non_style_components.jsonl`: 1,170 records.
- `xversebench_selected_components.jsonl`: 60 records.
- `validation_summary.json`: counts and span-validation results for those records.

The redundant Markdown tables, old identity-inference prompt/schema, and duplicate
runtime prompt/schema files have been removed. Runtime prompts and schemas are
maintained in `orthogonalization_instantx_ip_adapter/semantic_components.py`.

For the current interface, see [the semantic pipeline](../../docs/semantic_pipeline.md)
and [the shared module prompts](../../docs/prompts/llm_modules_share.md).

## Legacy annotation generator

`python scripts/build_prompt_component_dataset.py` regenerates the benchmark
annotations and their Markdown renderings. It is an offline legacy utility,
not part of captioning, component analysis, or inference.
