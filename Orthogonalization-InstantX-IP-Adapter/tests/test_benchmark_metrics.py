"""Selected-reference pairing, score aggregation, and preflight integrity guards."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

SPEC = importlib.util.spec_from_file_location('benchmark_metrics', Path(__file__).resolve().parents[1] / 'scripts/benchmark_metrics.py')
metrics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metrics)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(metrics, 'ROOT', tmp_path)
    Image.new('RGB', (32, 32), 'red').save(tmp_path / 'selected.png')
    Image.new('RGB', (32, 32), 'blue').save(tmp_path / 'not-selected.png')
    metrics.write(tmp_path / 'cache.json', {'cached': True})
    rows = [dict(case_id=f'{i:04}', benchmark='dreambench' if i < 750 else 'dreambench_plus',
                 image_path='selected.png', reference_id='dreambench/cat' if i < 750 else 'dreambench_plus/live_subject/animal/00', conditioning_path='cache.json',
                 target_prompt=f'original target {i}', status='READY') for i in range(2100)]
    manifest = tmp_path / 'manifest.json'
    metrics.write(manifest, rows)
    return metrics.prepare(manifest, tmp_path / 'images')


def make_result(record):
    image = Path(record['generated'])
    image.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (1024, 1024), 'green').save(image)
    settings = dict(condition=record['condition'], width=1024, height=1024, seed=42)
    inputs = dict(row=record['row'], settings=settings, cache={'cached': True},
                  image_sha256=record['reference_sha256'])
    signature = hashlib.sha256(json.dumps(inputs, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    enabled = record['condition'] == 'source_target_ssg'
    meta = dict(row=record['row'], settings=settings, job_signature=signature,
                png_sha256=metrics.sha(image), diagnostics=dict(mask_single_text_rows=enabled,
                orthogonalization=dict(enabled=enabled)))
    metrics.write(image.with_suffix('.json'), meta)
    return image.with_suffix('.json'), meta


def test_exact_reference_no_sibling_average_and_only_two_conditions(prepared):
    assert len(prepared['records']) == 4200
    assert {Path(r['reference']).name for r in prepared['records']} == {'selected.png'}
    assert {r['condition'] for r in prepared['records']} == {'baseline', 'source_target_ssg'}
    assert prepared['records'][0]['row']['target_prompt'] == 'original target 0'
    assert all(c['missing'] == c['expected'] for c in metrics.availability(prepared).values())


def test_changed_reference_rejected_before_model_loading(prepared):
    Path(prepared['records'][0]['reference']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='differs'):
        metrics.validate_pairs(prepared)


def test_changed_prompt_in_plan_rejected(prepared):
    changed = copy.deepcopy(prepared)
    changed['records'][0]['row']['target_prompt'] = 'source caption accidentally substituted'
    with pytest.raises(ValueError, match='differs'):
        metrics.validate_pairs(changed)


def test_missing_generated_image_is_not_silently_skipped(prepared):
    with pytest.raises(FileNotFoundError):
        metrics.validate_pairs(prepared)


def test_png_corruption_rejected(prepared):
    meta_path, _ = make_result(prepared['records'][0])
    meta_path.with_suffix('.png').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='hash mismatch'):
        metrics.validate_pairs(prepared)


def test_wrong_ssg_flag_rejected(prepared):
    make_result(prepared['records'][0])
    path, meta = make_result(prepared['records'][1])
    meta['diagnostics']['mask_single_text_rows'] = False
    metrics.write(path, meta)
    with pytest.raises(ValueError, match='Projection/SSG'):
        metrics.validate_pairs(prepared)


def test_case_and_subject_means_are_separate():
    scores = [dict(benchmark='dreambench', condition='baseline', reference_id=ref,
                   **{'CLIP-T': score, 'CLIP-I': score, 'DINO-I': score})
              for ref, score in [('dreambench/cat', 1.), ('dreambench/cat', 1.), ('dreambench/dog', 0.)]]
    result = metrics.summarize(scores)['dreambench/baseline']
    assert result['subject_macro']['DINO-I'] == .5
    assert result['case_micro']['DINO-I'] == pytest.approx(2 / 3)


def test_duplicate_cases_rejected(prepared):
    manifest = Path(prepared['manifest'])
    rows = metrics.read(manifest)
    rows[1]['case_id'] = rows[0]['case_id']
    metrics.write(manifest, rows)
    with pytest.raises(ValueError, match='unique'):
        metrics.prepare(manifest, manifest.parent / 'images')


@pytest.mark.parametrize('ref,expected', [
    ('dreambench/dog8', 'animal'), ('dreambench/backpack_dog', 'object'),
    ('dreambench/bear_plushie', 'object'), ('dreambench/wolf_plushie', 'object'),
    ('dreambench_plus/live_subject/human/00', 'human'), ('dreambench_plus/style/00', 'style'),
])
def test_reference_category_not_prompt_contents(ref, expected):
    assert metrics.category(dict(benchmark=ref.split('/')[0], reference_id=ref,
                                 target_prompt='a dog beside a human, watercolor style')) == expected


def test_class_macro_is_not_the_benchmark_reference_mean():
    scores = [dict(benchmark='dreambench', condition='baseline', reference_id=ref,
                   **{m: value for m in metrics.METRICS}) for ref, value in
              [('dreambench/cat', 1.), ('dreambench/backpack', 0.), ('dreambench/can', 0.)]]
    result = metrics.summarize(scores)['dreambench/baseline']
    assert result['subject_macro']['DINO-I'] == pytest.approx(1 / 3)
    assert result['class_macro']['DINO-I'] == .5
    assert set(result['by_class']) == {'animal', 'object'}
    assert result['by_class']['object']['subjects'] == 2


def test_plus_non_style_does_not_include_style():
    scores = [dict(benchmark='dreambench_plus', condition='baseline', reference_id=ref,
                   **{m: value for m in metrics.METRICS}) for ref, value in
              [('dreambench_plus/live_subject/human/00', .8), ('dreambench_plus/style/00', .2)]]
    result = metrics.summarize(scores)['dreambench_plus/baseline']
    assert result['non_style']['subject_macro']['DINO-I'] == .8
    assert result['non_style']['cases'] == 1


@pytest.fixture
def score_parts(tmp_path):
    plan = dict(protocol=metrics.PROTOCOL, records=[])
    files = []
    for benchmark, ref in [('dreambench', 'dreambench/cat'), ('dreambench_plus', 'dreambench_plus/style/00')]:
        scores = []
        for condition in metrics.CONDITIONS:
            row = dict(benchmark=benchmark, reference_id=ref, case_id=benchmark + '-00', target_prompt='original')
            label = metrics.category(row)
            plan['records'].append(dict(row=row, condition=condition, category=label, reference_sha256='same-reference'))
            scores.append(dict(row, condition=condition, category=label, reference_sha256='same-reference',
                               **{m: .4 if condition == 'baseline' else .5 for m in metrics.METRICS}))
        part = tmp_path / (benchmark + '.json')
        metrics.write(part, dict(protocol=metrics.PROTOCOL, device='cuda', versions={'torch': '2.7.1'},
                                 device_name='Tesla T4', scores=scores))
        files.append(part)
    return plan, files, tmp_path / 'merged.json'


def test_merge_case_scores_and_method_delta(score_parts):
    plan, parts, output = score_parts
    metrics.merge_scores(plan, parts, output)
    result = metrics.read(output)
    assert len(result['scores']) == 4
    assert result['method_minus_baseline']['dreambench']['by_class']['animal']['DINO-I'] == pytest.approx(.1)
    assert len(result['summary']) == 4


@pytest.mark.parametrize('failure', ['duplicate', 'missing', 'environment', 'reference'])
def test_merge_rejects_invalid_parts(score_parts, failure):
    plan, parts, output = score_parts
    if failure == 'duplicate':
        parts.append(parts[0])
    elif failure == 'missing':
        parts.pop()
    else:
        part = metrics.read(parts[1])
        if failure == 'environment':
            part['device_name'] = 'different GPU'
        else:
            part['scores'][0]['reference_sha256'] = 'other-reference'
        metrics.write(parts[1], part)
    with pytest.raises(ValueError):
        metrics.merge_scores(plan, parts, output)
    assert not output.exists()
