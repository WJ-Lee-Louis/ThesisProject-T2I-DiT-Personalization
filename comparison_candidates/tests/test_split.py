import unittest
from collections import defaultdict
from freecus.manifest import PROJECT,load
from comparison_candidates.split_dbpp import assignments,SHARES


class SplitTests(unittest.TestCase):
    def test_partition_is_complete_disjoint_and_reference_indivisible(self):
        rows=load(PROJECT/'xverse/manifests/dreambench_plus_seed42.jsonl',PROJECT)
        parts=assignments(rows)
        ids=[c for subset in parts.values() for c in subset]
        self.assertEqual(len(ids),1350);self.assertEqual(len(set(ids)),1350)
        self.assertEqual(set(ids),{r['case_id'] for r in rows})
        ownership={c:workspace for workspace,cases in parts.items() for c in cases}
        refs=defaultdict(set)
        for r in rows:refs[r['reference_id']].add(ownership[r['case_id']])
        self.assertTrue(all(len(v)==1 for v in refs.values()))
        for workspace,part in SHARES.items():
            self.assertEqual(len(parts[workspace]),part['references']*9)
            self.assertTrue(any(r['category']=='style' and ownership[r['case_id']]==workspace for r in rows))


if __name__=='__main__':unittest.main()
