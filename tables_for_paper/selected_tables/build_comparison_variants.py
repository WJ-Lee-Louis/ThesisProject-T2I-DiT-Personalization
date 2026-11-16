"""Create the eight TB/TF × scale comparison PNGs without changing original tables.

Run with the same environment as build_selected_tables.py.
"""
import hashlib
import json

from build_selected_tables import HERE, METRICS, SOURCES, cells, render, values

COHORTS = {
    'TB': [('UNO', 'uno'), ('DreamO', 'dreamo'), ('XVerse', 'xverse'),
           ('USO', 'uso'), ('MOSAIC', 'mosaic')],
    'TF': [('Diptych Prompting', 'diptych'), ('FreeCus', 'freecus'),
           ('FreeGraftor', 'freegrafor2')],
}


def rank_cells(rows):
    """Dense ranks at displayed precision; tied cells share the same rank."""
    ranks = [[None]*len(row) for row in rows]
    for col in range(1, len(rows[0])):
        distinct = sorted({float(row[col]) for row in rows if row[col] != '-'}, reverse=True)
        for index, row in enumerate(rows):
            if row[col] != '-':
                ranks[index][col] = distinct.index(float(row[col]))+1
    return ranks


def main():
    original_paths = [HERE/name for name in ['01_dreambench_plus_comparison.png',
                      '02_dreambench_comparison.png', '03_ablation.png',
                      '04_scale_sensitivity.png', 'table_audit.json']]
    originals = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in original_paths}
    artifacts = []
    for number, bench in [('01', 'dreambench_plus'), ('02', 'dreambench')]:
        for scale in ['0.7', '0.8']:
            for cohort, methods in COHORTS.items():
                ours_label = 'VIP-Adapter(Ours)' if scale == '0.8' else f'VIP-Adapter ({scale})'
                selected = methods + [(ours_label, 'ours_'+scale)]
                groups = ['DreamBench++ Subject', 'DreamBench++ Style'] if number == '01' else ['DreamBench']
                subsets = ['non_style', 'style'] if number == '01' else ['overall']
                rows = []
                records = []
                for label, key in selected:
                    row = [label]
                    raw = []
                    for subset in subsets:
                        record = values(key, bench, subset)
                        row += cells(record)
                        raw += [record[m] for m in METRICS]
                    rows.append(row)
                    records.append({'label': label, 'method_key': key, 'source_values': raw})
                assert all('IP-Adapter' != r[0] and 'XLabs' not in r[0] for r in rows)
                assert sum(r[0].startswith('VIP-Adapter') for r in rows) == 1
                ranks = rank_cells(rows)
                name = f'{number}-{cohort}-{scale}.png'
                result = render(name, ['Method'], [139], groups, rows,
                                separators=[len(methods)], cell_ranks=ranks)
                result.update({'comparison_group': cohort, 'ours_scale': scale,
                               'cell_ranks': ranks, 'source_records': records})
                artifacts.append(result)
    assert len(artifacts) == 8
    for path, previous in originals.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == previous, path
    (HERE/'comparison_variants_audit.json').write_text(json.dumps({
        'metric_order': METRICS,
        'ranking_scope': 'Within each PNG and each metric/subset column; higher is better.',
        'ranking_precision': 'Displayed four-decimal values; consistent with historical scale-0.7 precision.',
        'tie_policy': 'Dense ranks: all equal best values are bold; all equal second-distinct values are underlined.',
        'missing_policy': 'Missing values displayed as - and excluded from ranking.',
        'group_definition': 'TB comparison models depend on personalization-trained weights; TF comparison models do not require personalization-specific training. VIP-Adapter is an independently listed refinement of a pretrained IP-Adapter, not claimed to be free of that dependency.',
        'source_sha256': SOURCES, 'original_tables_preserved': True,
        'artifacts': artifacts
    }, ensure_ascii=False, indent=2)+'\n')
    print('Created and verified eight comparison PNGs; original tables unchanged.')
    for item in artifacts:
        print(item['filename'])


if __name__ == '__main__':
    main()
