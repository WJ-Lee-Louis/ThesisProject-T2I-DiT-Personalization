"""Add 01-0.8 and 02-0.8 with a merged training-category column."""
import hashlib
import json

from build_comparison_variants import COHORTS, rank_cells
from build_selected_tables import HERE, METRICS, SOURCES, cells, render, values


def main():
    preserved = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.png')
                 if p.name not in ['01-0.8.png', '02-0.8.png']}
    methods = COHORTS['TB'] + COHORTS['TF'] + [('VIP-Adapter(Ours)', 'ours_0.8')]
    artifacts = []
    for number, bench in [('01', 'dreambench_plus'), ('02', 'dreambench')]:
        groups = ['DreamBench++ Subject', 'DreamBench++ Style'] if number == '01' else ['DreamBench']
        subsets = ['non_style', 'style'] if number == '01' else ['overall']
        value_rows = []
        for label, key in methods:
            row = [label]
            for subset in subsets:
                row += cells(values(key, bench, subset))
            value_rows.append(row)
        assert len(value_rows) == 9
        assert all(row[0] != 'IP-Adapter' and '0.7' not in row[0] for row in value_rows)
        ranks = [[None]+rank for rank in rank_cells(value_rows)]
        rows = [['']+row for row in value_rows]
        result = render(f'{number}-0.8.png', ['Training\nDependency', 'Method'], [85, 139], groups,
                        rows, separators=[5, 8], cell_ranks=ranks, method_column=1, method_indent=24,
                        merged_labels=[(0, 0, 4, 'Training-based'), (0, 5, 7, 'Training-free')])
        result['cell_ranks'] = ranks
        result['training_based_rows'] = list(range(5))
        result['training_free_rows'] = list(range(5, 8))
        result['independent_ours_rows'] = [8]
        artifacts.append(result)
    for path, old_hash in preserved.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == old_hash, path
    (HERE/'grouped_tables_audit.json').write_text(json.dumps({
        'ours_scale': 0.8, 'metric_order': METRICS,
        'ranking_scope': 'All eligible methods within each PNG metric/subset column, across both training categories.',
        'ranking_precision': 'Displayed four-decimal values; dense ranks, shared rank on ties; missing - is unranked.',
        'category_definition': 'Baseline groups reflect reliance on personalization-trained weights. VIP-Adapter appears independently in the last row with no group label; it retains the pretrained IP-Adapter dependency while introducing no new training.',
        'source_sha256': SOURCES, 'previous_pngs_preserved': True, 'artifacts': artifacts
    }, ensure_ascii=False, indent=2)+'\n')
    print('Created and verified 01-0.8.png and 02-0.8.png; all earlier PNGs preserved.')


if __name__ == '__main__':
    main()
