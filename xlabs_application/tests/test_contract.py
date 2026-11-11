import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import torch

from xlabs_application.config import ACCOUNTS, SETTINGS
from xlabs_application.inference import edit_embedding, validate_cache
from xlabs_application.io import extract, read, write
from xlabs_application.runtime import signature
from xlabs_application.cloud_control import plus_shards
from xlabs_application.vendor.contrast_directions import build_contrast_directions
from xlabs_application.vendor.semantic_contrasts import SemanticContrastConditioning

ORIGINAL = Path(__file__).resolve().parents[2] / 'Orthogonalization-InstantX-IP-Adapter'
CACHE = ORIGINAL / 'outputs/semantic_benchmark_gpt56_sol/conditioning'


def encode(texts):
    return torch.tensor([list(hashlib.sha256(t.encode()).digest()) for t in texts], dtype=torch.float32) - 127


class ContractTests(unittest.TestCase):
    def test_projected_geometry_and_norm(self):
        c = SemanticContrastConditioning.from_dict(json.loads((CACHE / '000-00.json').read_text(encoding='utf-8')))
        visual = torch.randn(1, 32)
        edited, diag = edit_embedding(visual, c, encode)
        self.assertGreater(diag['basis_rank'], 0)
        self.assertLess(diag['projection']['max_abs_basis_dot'][0], 1e-5)
        torch.testing.assert_close(edited.norm(), visual.norm())

    def test_strength_zero_is_exact_noop(self):
        c = SemanticContrastConditioning.from_dict(json.loads((CACHE / '000-00.json').read_text(encoding='utf-8')))
        v = torch.randn(1, 32)
        out, _ = edit_embedding(v, c, encode, strength=0)
        self.assertTrue(torch.equal(out, v))

    def test_fallback_empty_is_exact_noop(self):
        c = SemanticContrastConditioning.from_dict(json.loads((CACHE / '092-08.json').read_text(encoding='utf-8')))
        v = torch.randn(1, 32)
        out, diag = edit_embedding(v, c, encode)
        self.assertEqual(diag['basis_rank'], 0)
        self.assertTrue(diag['target_projection_skipped'])
        self.assertTrue(torch.equal(out, v))

    def test_second_fallback_retains_source_only(self):
        c = SemanticContrastConditioning.from_dict(json.loads((CACHE / '154-07.json').read_text(encoding='utf-8')))
        d = build_contrast_directions(c, encode)
        self.assertEqual(len(d.values), 1)
        self.assertTrue(d.names[0].startswith('source:'))

    def test_negative_sign_and_input_order(self):
        c = SemanticContrastConditioning.from_dict(json.loads((CACHE / '000-00.json').read_text(encoding='utf-8')))
        rows = build_contrast_directions(c, encode)
        p = c.analysis.source_caption
        n = c.analysis.source[0].negative
        ep, en = torch.nn.functional.normalize(encode([p, n]), dim=-1)
        expected = torch.nn.functional.normalize(ep - en, dim=0)
        torch.testing.assert_close(rows.values[0], expected)

    def test_condition_and_input_bound_signature(self):
        p = dict(code='x', models={}, settings=SETTINGS)
        row = dict(case_id='a', target_prompt='dog')
        self.assertNotEqual(signature(p, row, 'baseline'), signature(p, row, 'source_target'))
        self.assertNotEqual(signature(p, row, 'baseline'), signature(p, dict(row, target_prompt='cat'), 'baseline'))

    def test_zip_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / 'bad.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('../escape', 'x')
            with self.assertRaises(ValueError):
                extract(archive, Path(tmp) / 'out')

    def test_concurrent_atomic_status_publication(self):
        from concurrent.futures import ThreadPoolExecutor
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'status.json'
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda n: write(path, dict(value=n)), range(20)))
            self.assertIn(read(path)['value'], range(20))
            self.assertEqual(len(list(Path(tmp).glob('*.tmp'))), 0)

    def test_dreambench_plus_shards_cover_all_cases_and_keep_reference_pairs(self):
        rows = [dict(benchmark='dreambench_plus', reference_id=f'ref-{ref:03}', case_id=f'{ref}-{prompt}')
                for ref in range(150) for prompt in range(9)]
        shards = plus_shards(dict(rows=rows))
        self.assertEqual([len(cases) for cases in shards.values()], [675, 675])
        owner = {case: shard for shard, cases in shards.items() for case in cases}
        self.assertEqual(len(owner), 1350)
        for ref in range(150):
            self.assertEqual(len({owner[f'{ref}-{prompt}'] for prompt in range(9)}), 1)

    def test_dreambench_plus_rejects_duplicate_case_replacing_missing_case(self):
        rows = [dict(benchmark='dreambench_plus', reference_id=f'ref-{ref:03}', case_id=f'{ref}-{prompt}')
                for ref in range(150) for prompt in range(9)]
        rows[1]['case_id'] = rows[0]['case_id']
        with self.assertRaises(ValueError):
            plus_shards(dict(rows=rows))

    def test_recovery_never_submits_duplicate_work_when_generation_is_running(self):
        from contextlib import redirect_stdout
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch
        from xlabs_application import recovery
        modal = MagicMock()
        modal.Function.from_name.return_value.get_current_stats.return_value = SimpleNamespace(num_running_inputs=1)
        with patch.object(recovery, 'connect', return_value=modal), patch.object(
                recovery, 'current', return_value=({}, Path('unused'), {})), redirect_stdout(io.StringIO()):
            recovery.restart()
        modal.FunctionCall.from_id.assert_not_called()
        modal.Function.from_name.return_value.spawn.assert_not_called()


if __name__ == '__main__':
    unittest.main()
