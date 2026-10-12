"""Run ten real DreamBench++ style cases, saving each LLM attempt and contrast."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from orthogonalization_instantx_ip_adapter.llm_provider import ResponsesJSONProvider
from orthogonalization_instantx_ip_adapter.semantic_components import SemanticLLM, SemanticConditioning, identity_spans
from orthogonalization_instantx_ip_adapter.component_directions import component_contrast_plan

SELECTION = [
    ('01', 8, 'Style inserted inside a location noun phrase'),
    ('02', 1, 'Generic wrapper, temporal identity and related golden appearance'),
    ('04', 4, 'Variable vibrant qualifier interrupts rendering carrier'),
    ('05', 8, 'Prefix style modifier and imaginative scene'),
    ('06', 6, 'Long recovering-forest action, ash and environmental identity'),
    ('09', 6, 'Sharp qualifier inside low-poly material description'),
    ('10', 1, 'Delicate qualifier and style-specific brush strokes'),
    ('13', 7, 'Non-object preservation concept is the grammatical subject'),
    ('16', 8, 'Article-bearing style carrier and glowing-tree attributes'),
    ('18', 6, 'Style inside tablecloth location plus abstract mood clause'),
]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def cases():
    base = ROOT / 'datasets/XVerseBench/dreambench_plus_raw/extracted/data'
    result = []
    for sid, index, reason in SELECTION:
        caption_file = base / 'captions/style' / f'{sid}.txt'
        lines = [s.strip() for s in caption_file.read_text(encoding='utf-8').splitlines() if s.strip()]
        identity_spans(lines[index], lines[0])
        result.append(dict(case_id=f'{sid}-{index:02d}', identity_term=lines[0],
            target_prompt=lines[index], reference_image=str(base / 'images/style' / f'{sid}.jpg'),
            caption_file=str(caption_file), selection_reason=reason))
    return result


class RecordingProvider:
    def __init__(self, provider, output):
        self.provider, self.output = provider, output
        self.case_id = self.stage = ''

    def complete(self, **kwargs):
        start = time.monotonic()
        record = dict(case_id=self.case_id, stage=self.stage,
            timestamp=datetime.now(timezone.utc).isoformat(), payload=kwargs['payload'],
            feedback=kwargs.get('feedback'), model=self.provider.model)
        try:
            response = self.provider.complete(**kwargs)
            record['response'] = response
            return response
        except Exception as error:
            record['error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            record['elapsed_seconds'] = time.monotonic() - start
            with (self.output / 'llm_attempts.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')


def audit(conditioning):
    records = []
    for origin, analysis in conditioning.selected:
        assert analysis.prompt_for_subset(range(len(analysis.components))) == analysis.text
        fixed = [*analysis.preserved_spans, *(analysis.retained_spans or ()), *(analysis.neutral_spans or ())]
        records.append(dict(origin=origin, component_count=len(analysis.components),
            all_removed_prompt=analysis.prompt_for_subset([]), fixed_spans=fixed))
    plan = component_contrast_plan(conditioning)
    originals = dict(conditioning.selected)
    for name, pairs in plan:
        analysis = originals[name.split(':')[0]]
        fixed = [*analysis.preserved_spans, *(analysis.retained_spans or ()), *(analysis.neutral_spans or ())]
        for pair in pairs:
            for prompt in pair:
                for span in fixed:
                    assert ' '.join(span.split()) in prompt, (name, span, prompt)
    prompts = list(dict.fromkeys(prompt for _, pairs in plan for pair in pairs for prompt in pair))
    return dict(strategy=conditioning.strategy, input_direction_count=len(plan),
        unique_text_context_count=len(prompts), contrast_pair_count=sum(len(p) for _, p in plan),
        analyses=records, contrasts=[dict(direction=name, pairs=pairs) for name, pairs in plan])


def analyze(output, model):
    selected = cases()
    save(output / 'selection.json', selected)
    provider = RecordingProvider(ResponsesJSONProvider.from_env(model), output)
    llm = SemanticLLM(provider)
    statuses = []
    for case in selected:
        provider.case_id = case['case_id']
        directory = output / case['case_id']
        if all((directory / f'conditioning_{s}.json').exists() for s in ('local', 'factorial')):
            for strategy in ('local', 'factorial'):
                cached = SemanticConditioning.from_dict(json.loads((directory / f'conditioning_{strategy}.json').read_text(encoding='utf-8')))
                if cached.source is None or cached.source.conditioned_on_target != case['target_prompt']:
                    raise ValueError('Legacy or mismatched source cache; use a new output directory')
            print(case['case_id'], 'cached analysis', flush=True)
            statuses.append(dict(case_id=case['case_id'], status='cached'))
            continue
        record = dict(case)
        try:
            print(case['case_id'], 'caption', flush=True)
            provider.stage = 'caption'
            probe = output / 'probe_caption.txt'
            if case['case_id'] == '01-08' and probe.exists():
                source_caption = probe.read_text(encoding='utf-8')
                record['caption_from_probe'] = True
            else:
                source_caption = llm.caption(Path(case['reference_image']).read_bytes(), case['identity_term'])
            record['source_caption'] = source_caption
            save(directory / 'progress.json', record)
            print(case['case_id'], 'source analysis', flush=True)
            provider.stage = 'source_analysis'
            source = llm.analyze_source(source_caption, case['target_prompt'], case['identity_term'])
            save(directory / 'source_analysis.json', source.to_dict())
            print(case['case_id'], 'target analysis', flush=True)
            provider.stage = 'target_analysis'
            target = llm.analyze_target(case['target_prompt'], case['identity_term'])
            save(directory / 'target_analysis.json', target.to_dict())
            for strategy in ('local', 'factorial'):
                conditioning = SemanticConditioning(target, source, 'source_target', strategy)
                save(directory / f'conditioning_{strategy}.json', conditioning.to_dict())
                save(directory / f'audit_{strategy}.json', audit(conditioning))
            record['status'] = 'analysis_passed'
            print(case['case_id'], 'PASS', len(source.components), len(target.components), flush=True)
        except Exception as error:
            record['status'] = 'analysis_failed'
            record['error'] = f'{type(error).__name__}: {error}'
            print(case['case_id'], record['error'], flush=True)
        save(directory / 'status.json', record)
        statuses.append(record)
        save(output / 'analysis_summary.json', statuses)
    return statuses


def generate(output):
    from semantic_monitor_modal import app, run_cases
    requests = []
    for case in cases():
        directory = output / case['case_id']
        if all((directory / f'conditioning_{s}.json').exists() for s in ('local', 'factorial')):
            requests.append(dict(case_id=case['case_id'],
                image_bytes=Path(case['reference_image']).read_bytes(),
                local=json.loads((directory / 'conditioning_local.json').read_text(encoding='utf-8')),
                factorial=json.loads((directory / 'conditioning_factorial.json').read_text(encoding='utf-8'))))
    events = []
    if not requests:
        raise ValueError('No successfully analyzed cases to generate')
    with app.run():
        for event in run_cases.remote_gen(requests):
            png = event.pop('png_bytes', None)
            if 'case_id' in event:
                directory = output / event['case_id']
                if png is not None:
                    (directory / f"{event['strategy']}.png").write_bytes(png)
                save(directory / f"gpu_{event.get('strategy', 'failure')}.json", event)
            events.append(event)
            save(output / 'gpu_summary.json', events)
            print('GPU', event.get('case_id', ''), event.get('strategy', ''), event['status'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('analysis', 'gpu', 'all'), default='all')
    parser.add_argument('--model', default=None)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/semantic_monitor_20261010')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage in ('analysis', 'all'):
        analyze(args.output_dir, args.model)
    if args.stage in ('gpu', 'all'):
        generate(args.output_dir)
