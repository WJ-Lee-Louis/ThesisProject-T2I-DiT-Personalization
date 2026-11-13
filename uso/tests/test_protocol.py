import json
import unittest
from pathlib import Path
from uso.experiment import BASE
from freecus.manifest import PROJECT,load,sha256
from freecus.evaluate import prepare,project_evaluator


class ProtocolTests(unittest.TestCase):
    def test_upstream_is_unchanged(self):
        info=json.loads((BASE/'UPSTREAM.json').read_text())
        self.assertEqual(info['commit'],'6587514aa3adf8e8f46e5f7e804239651d30b32d')
        for name,wanted in info['files'].items():
            self.assertEqual(sha256(BASE/'upstream'/name),wanted,name)

    def test_canonical_cohort_and_evaluator(self):
        for bench,n in [('dreambench',750),('dreambench_plus',1350)]:
            manifest=PROJECT/f'freecus/manifests/{bench}_project_seed42.jsonl'
            rows=load(manifest,PROJECT)
            self.assertEqual(len(rows),n)
            self.assertEqual({r['seed'] for r in rows},{42})
            if bench=='dreambench_plus':self.assertEqual(sum(r['category']=='style' for r in rows),180)
            plan=prepare(manifest,PROJECT,Path('unused'),project_evaluator(),condition='uso')
            self.assertEqual(len(plan['records']),n)
            self.assertEqual({r['condition'] for r in plan['records']},{'uso'})

    def test_pinned_settings(self):
        cfg=json.loads((BASE/'config.json').read_text())
        self.assertEqual((cfg['width'],cfg['height'],cfg['seed'],cfg['steps'],cfg['guidance']),
                         (1024,1024,42,25,4.0))
        lock=json.loads((BASE/'models.lock.json').read_text())
        self.assertEqual(set(lock),{'flux','t5','clip','lora','siglip'})
        self.assertTrue(all(len(m['revision'])==40 for m in lock.values()))


if __name__=='__main__':unittest.main()
