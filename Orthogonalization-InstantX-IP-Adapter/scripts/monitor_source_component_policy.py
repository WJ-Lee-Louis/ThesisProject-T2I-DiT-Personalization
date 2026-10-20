# Legacy partition research tool; not used by the joint local generation path.
"""Small first-response integration audit using previously generated captions."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from orthogonalization_instantx_ip_adapter.llm_provider import ResponsesJSONProvider
from orthogonalization_instantx_ip_adapter.semantic_components import SemanticLLM
from orthogonalization_instantx_ip_adapter.source_component_prompt import SOURCE_COMPONENT_SYSTEM_PROMPT

OUT = ROOT / 'datasets/component_monitoring/source_policy_v1'
CASES = [
    ('dog_location', 'dreambench/dog', 'a dog standing on a beach'),
    ('dog_hat', 'dreambench/dog', 'a dog wearing a red hat'),
    ('dog_recolor', 'dreambench/dog', 'a dog with entirely purple fur'),
    ('dog_rendering', 'dreambench/dog', 'a dog depicted in watercolor style'),
    ('backpack_shape', 'dreambench/backpack', 'a cube shaped backpack'),
    ('backpack_same_color', 'dreambench/backpack', 'a magenta backpack on a table'),
    ('style_content', 'dreambench_plus/style/16', 'a castle in watercolor sketch style'),
    ('scene_content', 'dreambench_plus/style/02', 'a city during sunset'),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = json.loads((ROOT/'datasets/caption_monitoring/module_a_v1/captions.json').read_text(encoding='utf-8'))['records']
    references = {r['id']: r for r in records}
    prompt_hash = hashlib.sha256(SOURCE_COMPONENT_SYSTEM_PROMPT.encode()).hexdigest()
    snapshot = OUT/'system_prompt.txt'
    if snapshot.exists() and snapshot.read_text(encoding='utf-8') != SOURCE_COMPONENT_SYSTEM_PROMPT:
        raise ValueError('Prompt changed; preserve this audit and choose another run directory')
    snapshot.write_text(SOURCE_COMPONENT_SYSTEM_PROMPT, encoding='utf-8')
    provider = ResponsesJSONProvider.from_env('gpt-6-luna')

    def run(case):
        name, reference, target = case
        path = OUT/f'{name}.json'
        original = references[reference]
        payload = dict(source_caption=original['source_caption'], target_prompt=target,
                       identity_term=original['identity_term'])
        if path.exists():
            old = json.loads(path.read_text(encoding='utf-8'))
            if old['prompt_sha256'] != prompt_hash or old['input'] != payload:
                raise ValueError('Audit cache differs; choose another run directory')
            return old
        record = dict(case=name, reference=reference, input=payload, model=provider.model,
                      started_at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
                      prompt_sha256=prompt_hash, attempts=[])

        class Recording:
            def complete(self, **kwargs):
                attempt = {}
                record['attempts'].append(attempt)
                try:
                    text = provider.complete(**kwargs)
                    attempt['raw_response'] = text
                    return text
                except Exception as error:
                    attempt['error'] = str(error)
                    raise

        try:
            analysis = SemanticLLM(Recording(), max_retries=0).analyze_source(**payload)
            record.update(status='PASS', output=analysis.to_dict(),
                          all_removed=analysis.prompt_for_subset([]))
        except Exception as error:
            record.update(status='FAIL', error=f'{type(error).__name__}: {error}')
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        return record

    results = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(run, c) for c in CASES]):
            row = future.result()
            results[row['case']] = row
            print(row['case'], row['status'], flush=True)
    lines = ['# Source B target-aware policy: API smoke audit', '',
             'Model: `gpt-6-luna`. Eight held-out inputs using saved Module A captions. '
             'No images, target decomposition, embeddings or image generation were requested. '
             'One response per case; no retries. PASS means structural validation only.', '',
             f'Prompt SHA-256: `{prompt_hash}`', '']
    for name, _, _ in CASES:
        row = results[name]
        lines += [f'## {name}: {row["status"]}', '', '```json',
                  json.dumps(row['input'], ensure_ascii=False, indent=2), '```', '', '```json',
                  row['attempts'][0].get('raw_response', json.dumps(row.get('error'))), '```', '']
        if row.get('all_removed'):
            lines += ['All removal components deleted:', '', row['all_removed'], '']
    (ROOT/'docs/source_component_policy_api_audit.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
