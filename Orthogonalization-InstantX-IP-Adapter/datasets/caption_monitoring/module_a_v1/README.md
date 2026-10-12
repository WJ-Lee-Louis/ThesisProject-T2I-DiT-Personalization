# Module A caption monitoring — v1

DreamBench 30 images selected by UNO, plus all 150 DreamBench++ references.
The identity input is the original DreamBench class name or the first-line English DreamBench++ label, without adding an article. Subject IDs are separate from these labels.

- `references.json`: all 180 inputs, original image hashes, labels, dimensions and benchmark prompts. **Target prompts are metadata only and are never sent to the caption API.**
- `system_prompt.txt`: the exact English prompt extracted from `docs/prompts/module_a_caption_system_prompt.md` for this run.
- `run.json`: prompt/input hashes, requested model, start time and retry policy.
- `records/*.json`: independently saved first responses, raw JSON text, caption, validation status, image provenance and elapsed time.
- `captions.json` / `captions.jsonl`: ordered aggregate results, refreshed after each completed call.
- [Monitoring report](../../../docs/module_a_caption_monitoring_v1.md): all reference images and unedited captions, with manual review checklists.

The requested API model is `gpt-6-luna`. This monitor uses the project's `ResponsesJSONProvider` with strict `CAPTION_SCHEMA`; it reads the reviewed prompt document directly rather than the older runtime caption prompt. Images use the existing provider's RGB PNG conversion without resizing. API credentials are read from `SEMANTIC_LLM_API_KEY` and are not persisted.

There are no automatic validation or transport retries. `PASS` means JSON shape and exact identity validation passed, not that visual accuracy or prompt adherence has been established. Invalid responses and transport errors remain visible. A resumed run skips all recorded cases, including failures, to preserve the first-response audit. Existing successful captions are never regenerated automatically.

From the project root:

```powershell
# Prepare and validate references; no LLM requests.
python scripts/monitor_reference_captions.py

# Generate only missing first responses; paid API calls.
python scripts/monitor_reference_captions.py --run --workers 4
```

Re-running the script regenerates aggregate files and the Markdown report. Keep manual annotations in a separate copy of the report. A changed prompt, model or input manifest requires a separate run directory rather than mixing results with this run.
