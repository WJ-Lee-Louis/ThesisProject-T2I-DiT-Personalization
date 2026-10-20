"""Explicit analysis/GPU monitor for joint local contrasts; preserves legacy run directories."""
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
from orthogonalization_instantx_ip_adapter.semantic_contrasts import (SemanticLLM, SemanticContrastConditioning,
    ReferenceFingerprint, identity_occurrences as identity_spans, prepare_semantic_conditioning, parse_json_object)
from orthogonalization_instantx_ip_adapter.contrast_directions import contrast_diagnostics

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
        self.case_id = ''
        self.model = provider.model

    def complete(self, **kwargs):
        start = time.monotonic()
        record = dict(case_id=self.case_id, stage='caption' if kwargs.get('image_bytes') else 'contrast',
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
    return contrast_diagnostics(conditioning)


def check_run(output):
    marker = output / 'joint_local_run.json'
    if output.exists() and any(output.iterdir()) and not marker.exists():
        raise ValueError('Use a new output directory; legacy runs must not be overwritten')
    if marker.exists() and parse_json_object(marker.read_text(encoding='utf-8')) != {"schema_version": 4}:
        raise ValueError('Unsupported monitor run')
    save(marker, {"schema_version": 4})


def analyze(output, model):
    check_run(output)
    selected = cases()
    save(output / 'selection.json', selected)
    provider = None
    statuses = []
    for case in selected:
        directory = output / case['case_id']
        cache = directory / 'conditioning.contrasts.json'
        image_bytes = Path(case['reference_image']).read_bytes()
        if cache.exists():
            cached = SemanticContrastConditioning.from_dict(parse_json_object(cache.read_text(encoding='utf-8')))
            cached.validate_reuse(identity_term=case['identity_term'], target_prompt=case['target_prompt'],
                                  reference=ReferenceFingerprint.from_image(image_bytes), model=model)
            statuses.append(dict(case_id=case['case_id'], status='cached'))
            continue
        record = dict(case)
        try:
            if provider is None:
                provider = RecordingProvider(ResponsesJSONProvider.from_env(model), output)
            provider.case_id = case['case_id']
            conditioning = prepare_semantic_conditioning(llm=SemanticLLM(provider),
                identity_term=case['identity_term'], target_prompt=case['target_prompt'],
                reference_image_bytes=image_bytes)
            save(cache, conditioning.to_dict())
            save(directory / 'audit_local.json', audit(conditioning))
            record.update(status='analysis_passed', source_caption=conditioning.caption.text, preserve=list(conditioning.caption.preserve))
        except Exception as error:
            record.update(status='analysis_failed', error=f'{type(error).__name__}: {error}')
            if getattr(error, 'trace', None) is not None:
                record['failure_trace'] = error.trace.to_dict()
        save(directory / 'status.json', record)
        statuses.append(record)
    save(output / 'analysis_summary.json', statuses)
    return statuses


def generate(output):
    from semantic_monitor_modal import app, run_cases
    check_run(output)
    requests = []
    for case in cases():
        cache = output / case['case_id'] / 'conditioning.contrasts.json'
        if cache.exists():
            conditioning = SemanticContrastConditioning.from_dict(parse_json_object(cache.read_text(encoding='utf-8')))
            image_bytes = Path(case['reference_image']).read_bytes()
            conditioning.validate_reuse(identity_term=case['identity_term'], target_prompt=case['target_prompt'],
                                        reference=ReferenceFingerprint.from_image(image_bytes))
            requests.append(dict(case_id=case['case_id'], image_bytes=image_bytes,
                                 contrast_analysis_payload=conditioning.to_dict()))
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
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/semantic_local_v3')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage in ('analysis', 'all'):
        analyze(args.output_dir, args.model)
    if args.stage in ('gpu', 'all'):
        generate(args.output_dir)
