import copy
import json
from pathlib import Path
import unittest
from uno.experiment_512 import prerequisite_ready
from uno.protocol import validate_config
from freecus.evaluate import prepare,project_evaluator
from freecus.manifest import PROJECT


class Uno512Tests(unittest.TestCase):
    def test_only_generation_dimensions_change(self):
        old=json.loads((PROJECT/'uno/config.json').read_text())
        new=json.loads((PROJECT/'uno/config.512.json').read_text())
        validate_config(old);validate_config(new)
        new.update(width=1024,height=1024)
        self.assertEqual(old,new)

    def test_gate_requires_both_metrics_and_verified_exports(self):
        value=dict(status='complete',evaluations={},exports={})
        self.assertFalse(prerequisite_ready(value))
        for b,n in [('dreambench',750),('dreambench_plus',1170)]:
            value['evaluations'][b]=dict(coverage=dict(status='complete',expected=n,generated=n,missing=[]))
            value['exports'][b]=dict(verified=True,missing_case_ids=[])
        self.assertTrue(prerequisite_ready(value))
        broken=copy.deepcopy(value);broken['status']='failed'
        self.assertFalse(prerequisite_ready(broken))
        broken=copy.deepcopy(value);broken['exports']['dreambench_plus']['verified']=False
        self.assertFalse(prerequisite_ready(broken))

    def test_style_is_included_and_metric_label_is_uno(self):
        for bench,n in [('dreambench',750),('dreambench_plus',1350)]:
            p=prepare(PROJECT/f'freecus/manifests/{bench}_project_seed42.jsonl',PROJECT,
                      Path('unused'),project_evaluator(),condition='uno_512')
            self.assertEqual(len(p['records']),n)
            self.assertEqual({r['condition'] for r in p['records']},{'uno_512'})
            if bench=='dreambench_plus': self.assertEqual(sum(r['category']=='style' for r in p['records']),180)


if __name__=='__main__': unittest.main()
