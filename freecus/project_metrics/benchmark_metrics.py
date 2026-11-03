"""Prepare and score exact selected-reference CLIP/DINO pairs; no generation imports."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ('baseline', 'source_target_ssg')
ACCOUNT_BENCHMARKS = {'A': 'dreambench_plus', 'B': 'dreambench'}
METRICS = ('CLIP-T', 'CLIP-I', 'DINO-I')
# Explicit local reporting taxonomy. Plush toys and backpack_dog are objects.
DREAMBENCH_CLASSES = {
    'animal': {'cat', 'cat2', 'dog', 'dog2', 'dog3', 'dog5', 'dog6', 'dog7', 'dog8'},
    'object': {'backpack', 'backpack_dog', 'bear_plushie', 'berry_bowl', 'can', 'candle',
               'clock', 'colorful_sneaker', 'duck_toy', 'fancy_boot', 'grey_sloth_plushie',
               'monster_toy', 'pink_sunglasses', 'poop_emoji', 'rc_car', 'red_cartoon',
               'robot_toy', 'shiny_sneaker', 'teapot', 'vase', 'wolf_plushie'},
}
PROTOCOL = dict(
    name='uno-selected-reference-v1',
    uno_revision='a563432dcfb4caf98df4368facb8102bf97d376c',
    clip_model='ViT-B/32',
    clip_revision='d05afc436d78f1c48dc0dbf8e5980a9d471f35f6',
    dino_model='dino_vits16',
    dino_revision='7c446df5b9f45747937fb0d72314eb9f7b66930a',
    reference_policy='exact manifest image_path; one reference per generated image',
    clip_preprocess='RGB; bicubic resize shorter edge 224; center crop 224; CLIP normalization',
    dino_preprocess='RGB; bicubic resize shorter edge 256; center crop 224; ImageNet normalization',
    clip_t='max(0, cosine(generated, original target_prompt)); weight=1; no prefix; truncate=True',
    image_metrics='cosine(generated, selected reference); no clipping; no logit scale',
    score_scale=1,
    aggregation='per benchmark, condition and class: reference macro and case micro; class macro separately',
    category_policy='DreamBench explicit animal/object map; DreamBench++ reference collection category',
    generated_format='original PNG; no JPEG export',
)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def category(row):
    """Classify the reference concept, not objects mentioned in target prompts."""
    ref = row['reference_id']
    if row['benchmark'] == 'dreambench':
        for label, subjects in DREAMBENCH_CLASSES.items():
            if ref in {'dreambench/' + s for s in subjects}:
                return label
    elif row['benchmark'] == 'dreambench_plus':
        parts = ref.split('/')
        if len(parts) == 4 and parts[:2] == ['dreambench_plus', 'live_subject'] and parts[2] in ('animal', 'human'):
            return parts[2]
        if len(parts) == 3 and parts[0] == 'dreambench_plus' and parts[1] in ('object', 'style'):
            return parts[1]
    raise ValueError(f'Unknown reference category: {ref}')


def prepare(manifest, images, root=None):
    root = ROOT if root is None else root
    rows = read(manifest)
    expected = {'dreambench': 750, 'dreambench_plus': 1350}
    if Counter(r['benchmark'] for r in rows) != expected or len({r['case_id'] for r in rows}) != 2100:
        raise ValueError('Expected unique 750 DreamBench + 1350 DreamBench++ cases')
    records, refs = [], {}
    for row in rows:
        if row['status'] != 'READY' or not row['target_prompt'].strip():
            raise ValueError(f'Invalid case: {row["case_id"]}')
        ref = (root / row['image_path']).resolve()
        if str(ref) not in refs:
            refs[str(ref)] = sha(ref)
        ref_hash = refs[str(ref)]
        for condition in CONDITIONS:
            image = (images / condition / row['benchmark'] / (row['case_id'] + '.png')).resolve()
            records.append(dict(condition=condition, category=category(row), row=row, reference=str(ref),
                                reference_sha256=ref_hash, generated=str(image)))
    return dict(schema_version=1, protocol=PROTOCOL, manifest=str(manifest.resolve()),
                manifest_sha256=sha(manifest), records=records)


def availability(plan):
    counts = defaultdict(lambda: dict(expected=0, local_pairs=0, missing=0))
    for r in plan['records']:
        count = counts[r['condition'] + '/' + r['row']['benchmark']]
        count['expected'] += 1
        image = Path(r['generated'])
        count['local_pairs' if image.is_file() and image.with_suffix('.json').is_file() else 'missing'] += 1
    return dict(counts)


def validate_pairs(plan, account=None):
    """Fail before model loading if any case is missing, changed, or from another run."""
    from PIL import Image
    if plan['protocol'] != PROTOCOL or sha(plan['manifest']) != plan['manifest_sha256']:
        raise ValueError('Protocol or generation manifest changed; prepare again')
    expected = prepare(Path(plan['manifest']), Path(plan['records'][0]['generated']).parents[2])
    if expected != plan:
        raise ValueError('Pair plan differs from the manifest or reference bytes; prepare again')
    checked_refs, paired = set(), {}
    for r in plan['records']:
        if account is not None and r['row']['benchmark'] != ACCOUNT_BENCHMARKS[account]:
            continue
        image, ref = Path(r['generated']), Path(r['reference'])
        if ref not in checked_refs:
            if sha(ref) != r['reference_sha256']:
                raise ValueError(f'Reference changed: {ref}')
            with Image.open(ref) as opened:
                opened.verify()
            checked_refs.add(ref)
        meta = read(image.with_suffix('.json'))
        settings = meta['settings']
        if meta['row'] != r['row'] or settings['condition'] != r['condition']:
            raise ValueError(f'Case/condition mismatch: {image}')
        if (settings['width'], settings['height']) != (1024, 1024):
            raise ValueError(f'Unexpected generation resolution: {image}')
        cache = read(ROOT / r['row']['conditioning_path'])
        sig_input = dict(row=r['row'], settings=settings, cache=cache, image_sha256=r['reference_sha256'])
        signature = hashlib.sha256(json.dumps(sig_input, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if signature != meta['job_signature'] or sha(image) != meta['png_sha256']:
            raise ValueError(f'Signature or PNG hash mismatch: {image}')
        diag = meta['diagnostics']
        enabled = r['condition'] == 'source_target_ssg'
        if diag['mask_single_text_rows'] != enabled or diag['orthogonalization']['enabled'] != enabled:
            raise ValueError(f'Projection/SSG mismatch: {image}')
        comparable = {k: v for k, v in settings.items() if k != 'condition'}
        key = r['row']['case_id']
        if paired.setdefault(key, comparable) != comparable:
            raise ValueError(f'Unpaired generation settings: {key}')
        with Image.open(image) as opened:
            opened.verify()
            if opened.size != (1024, 1024):
                raise ValueError(f'Unexpected PNG dimensions: {image}')


def summarize(scores):
    groups = defaultdict(list)
    for r in scores:
        groups[r['benchmark'], r['condition']].append(r)
    def aggregate(rows):
        subjects = defaultdict(list)
        for row in rows:
            subjects[row['reference_id']].append(row)
        return dict(
            cases=len(rows), subjects=len(subjects),
            subject_macro={m: statistics.mean(statistics.mean(r[m] for r in rs) for rs in subjects.values())
                           for m in METRICS},
            case_micro={m: statistics.mean(r[m] for r in rows) for m in METRICS})

    summary = {}
    for (benchmark, condition), rows in sorted(groups.items()):
        classes = defaultdict(list)
        for row in rows:
            label = category(row)
            if row.get('category', label) != label:
                raise ValueError('Score category differs from reference category')
            classes[label].append(row)
        result = aggregate(rows)
        result['by_class'] = {c: aggregate(rs) for c, rs in sorted(classes.items())}
        # This gives every PRESENT class equal weight, unlike the main reference macro.
        result['class_macro'] = {m: statistics.mean(c['subject_macro'][m] for c in result['by_class'].values())
                                 for m in METRICS}
        if benchmark == 'dreambench_plus':
            non_style = [r for r in rows if category(r) != 'style']
            if non_style:
                result['non_style'] = aggregate(non_style)
        summary[benchmark + '/' + condition] = result
    return summary


def merge_scores(plan, parts, destination):
    """Merge case scores on CPU, refusing missing/duplicate cases or mixed environments."""
    expected = {(r['row']['benchmark'], r['condition'], r['row']['case_id']): r for r in plan['records']}
    scores, seen, environments = [], set(), []
    for part in parts:
        result = read(part)
        if result['protocol'] != plan['protocol']:
            raise ValueError(f'Mixed metric protocols: {part}')
        environment = {k: result[k] for k in ('device', 'versions', 'device_name')}
        if environments and environments[0] != environment:
            raise ValueError('Different metric devices or library versions; rerun in a common environment')
        environments.append(environment)
        for score in result['scores']:
            key = (score['benchmark'], score['condition'], score['case_id'])
            if key not in expected or key in seen:
                raise ValueError(f'Unexpected or duplicate score: {key}')
            pair = expected[key]
            for field in ('reference_sha256', 'target_prompt', 'reference_id', 'category'):
                wanted = pair['row'][field] if field in ('target_prompt', 'reference_id') else pair[field]
                if score[field] != wanted:
                    raise ValueError(f'Score input mismatch: {key} / {field}')
            if not all(math.isfinite(score[m]) for m in METRICS):
                raise ValueError(f'Non-finite score: {key}')
            seen.add(key)
            scores.append(score)
    if seen != set(expected):
        raise ValueError(f'Missing {len(set(expected) - seen)} case scores; no partial final summary')
    scores.sort(key=lambda r: (r['benchmark'], r['case_id'], r['condition']))
    summary = summarize(scores)
    deltas = {}
    for benchmark in sorted({r['benchmark'] for r in scores}):
        baseline = summary[benchmark + '/baseline']
        method = summary[benchmark + '/source_target_ssg']
        deltas[benchmark] = dict(
            subject_macro={m: method['subject_macro'][m] - baseline['subject_macro'][m] for m in METRICS},
            by_class={c: {m: method['by_class'][c]['subject_macro'][m] - baseline['by_class'][c]['subject_macro'][m]
                          for m in METRICS} for c in baseline['by_class']})
    write(destination, dict(protocol=plan['protocol'], environment=environments[0],
                           summary=summary, method_minus_baseline=deltas, scores=scores))


def select_records(plan, account=None, pilot_only=False):
    records = [r for r in plan['records'] if account is None or r['row']['benchmark'] == ACCOUNT_BENCHMARKS[account]]
    if not pilot_only:
        return records
    cases = {}
    for r in records:
        cases.setdefault((r['row']['benchmark'], r['category']), r['row']['case_id'])
    return [r for r in records if r['row']['case_id'] == cases[r['row']['benchmark'], r['category']]]


def cached_scores(checkpoint, records, environment):
    if checkpoint is None or not Path(checkpoint).exists():
        return []
    result = read(checkpoint)
    if result['protocol'] != PROTOCOL or any(result[k] != v for k, v in environment.items()):
        raise ValueError('Checkpoint metric protocol/device/versions changed')
    expected = {(r['condition'], r['row']['case_id']): r for r in records}
    seen = set()
    for score in result['scores']:
        key = score['condition'], score['case_id']
        if key not in expected or key in seen:
            raise ValueError('Unexpected or duplicate checkpoint score')
        pair = expected[key]
        for field in ('reference_sha256', 'target_prompt', 'reference_id', 'category'):
            wanted = pair['row'][field] if field in ('target_prompt', 'reference_id') else pair[field]
            if score[field] != wanted:
                raise ValueError(f'Checkpoint input changed: {key}/{field}')
        if score['generated_sha256'] != sha(pair['generated']) or not all(math.isfinite(score[m]) for m in METRICS):
            raise ValueError('Checkpoint image changed or invalid score')
        seen.add(key)
    return result['scores']


def evaluate(plan, destination, device, batch_size, account=None, checkpoint=None, commit=None, pilot_only=False,
             validator=None):
    (validator or validate_pairs)(plan, account)
    import clip
    import numpy as np
    from PIL import Image
    import torch
    from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize, Resize, ToTensor
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use a separate CUDA evaluator or explicitly --device cpu')
    environment = dict(device=device, device_name=torch.cuda.get_device_name() if device == 'cuda' else 'CPU',
                       versions={p: version(p) for p in ('torch', 'torchvision', 'numpy', 'Pillow', 'clip')})
    all_records = select_records(plan, account)
    scores = cached_scores(checkpoint, all_records, environment)
    seen = {(r['condition'], r['case_id']) for r in scores}
    records = [r for r in select_records(plan, account, pilot_only)
               if (r['condition'], r['row']['case_id']) not in seen]

    def publish():
        result = dict(protocol=PROTOCOL, account=account, **environment,
                      summary=summarize(scores), scores=scores)
        if checkpoint is not None:
            write(checkpoint, result)
        if commit is not None:
            commit(result)
        return result

    if not records:
        result = publish()
        if not pilot_only:
            write(destination, result)
        return result
    clip_model, clip_transform = clip.load(PROTOCOL['clip_model'], device=device, jit=False,
                                            download_root=os.environ.get('CLIP_DOWNLOAD_ROOT'))
    clip_model.eval()
    dino = torch.hub.load('facebookresearch/dino:' + PROTOCOL['dino_revision'],
                          PROTOCOL['dino_model'], pretrained=True, trust_repo=True).to(device).eval()
    dino_transform = Compose([Resize(256, interpolation=InterpolationMode.BICUBIC), CenterCrop(224),
                              ToTensor(), Normalize((.485, .456, .406), (.229, .224, .225))])

    def images(paths, transform, encode, half=False):
        output = []
        for start in range(0, len(paths), batch_size):
            batch = []
            for path in paths[start:start + batch_size]:
                with Image.open(path) as opened:
                    batch.append(transform(opened.convert('RGB')))
            tensor = torch.stack(batch).to(device)
            output.append(encode(tensor.half() if half else tensor).cpu().numpy())
        return np.concatenate(output)

    def normalize(features):
        return features / np.sqrt(np.sum(features ** 2, axis=1, keepdims=True))

    paths = list(dict.fromkeys(r['reference'] for r in records))
    ref_indices = {p: i for i, p in enumerate(paths)}
    with torch.inference_mode():
        cr = normalize(images(paths, clip_transform, clip_model.encode_image, device == 'cuda'))
        dr = normalize(images(paths, dino_transform, dino))
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            generated = [r['generated'] for r in batch]
            ci = normalize(images(generated, clip_transform, clip_model.encode_image, device == 'cuda'))
            di = normalize(images(generated, dino_transform, dino))
            ct = normalize(clip_model.encode_text(clip.tokenize(
                [r['row']['target_prompt'] for r in batch], truncate=True).to(device)).cpu().numpy())
            ix = [ref_indices[r['reference']] for r in batch]
            values = (np.maximum(np.sum(ci * ct, axis=1), 0),
                      np.sum(ci * cr[ix], axis=1), np.sum(di * dr[ix], axis=1))
            if not all(np.isfinite(v).all() for v in values):
                raise ValueError('Non-finite metric score')
            for i, r in enumerate(batch):
                scores.append(dict(condition=r['condition'], benchmark=r['row']['benchmark'],
                                   category=r['category'],
                                   case_id=r['row']['case_id'], reference_id=r['row']['reference_id'],
                                   reference=r['reference'], reference_sha256=r['reference_sha256'],
                                   generated=r['generated'], generated_sha256=sha(r['generated']),
                                   target_prompt=r['row']['target_prompt'],
                                   **{m: float(v[i]) for m, v in zip(('CLIP-T', 'CLIP-I', 'DINO-I'), values)}))
            publish()
            print(f'COMMITTED SCORES {len(scores)}/{len(all_records)}', flush=True)
    result = publish()
    if not pilot_only:
        if len(scores) != len(all_records):
            raise ValueError('Incomplete metric result')
        write(destination, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'validate', 'evaluate', 'merge'))
    parser.add_argument('--manifest', type=Path, default=ROOT / 'outputs/semantic_benchmark_gpt56_sol/generation_manifest.json')
    parser.add_argument('--images', type=Path, default=ROOT / 'outputs/benchmark_images_v1')
    parser.add_argument('--plan', type=Path, default=ROOT / 'outputs/benchmark_metrics_v1/pairs.json')
    parser.add_argument('--scores', type=Path, default=ROOT / 'outputs/benchmark_metrics_v1/scores.json')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--account', choices=('A', 'B'), help='A: DreamBench++; B: DreamBench. Selects evaluation data only, not Modal credentials.')
    parser.add_argument('--parts', nargs='+', type=Path, help='Case score JSON files to merge on CPU')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch-size must be positive')
    if args.command == 'prepare':
        plan = prepare(args.manifest, args.images)
        write(args.plan, plan)
        print(json.dumps(dict(plan=str(args.plan), protocol=PROTOCOL, local_availability=availability(plan)), indent=2))
    elif args.command == 'validate':
        validate_pairs(read(args.plan))
        print('Validated all 4200 selected-reference pairs; no models loaded')
    elif args.command == 'merge':
        if not args.parts:
            parser.error('merge requires --parts')
        if args.scores.exists():
            raise FileExistsError('Scores already exist; select a new --scores path')
        merge_scores(read(args.plan), args.parts, args.scores)
        print(f'Merged all 4200 case scores: {args.scores}')
    else:
        if args.scores.exists():
            raise FileExistsError('Scores already exist; select a new --scores path')
        evaluate(read(args.plan), args.scores, args.device, args.batch_size, args.account)


if __name__ == '__main__':
    main()
