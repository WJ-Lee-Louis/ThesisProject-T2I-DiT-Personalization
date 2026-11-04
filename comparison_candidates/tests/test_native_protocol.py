import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from comparison_candidates.evaluation import validate
from comparison_candidates.runtime import caption
from freecus.run import artifact_paths,write_json
from freecus.manifest import PROJECT,load,sha256


class NativeProtocolTests(unittest.TestCase):
    def test_cohort_ownership_and_original_prompt_matching(self):
        for method in ['diptych','xverse']:
            plan=json.loads((PROJECT/method/'outputs/plan.json').read_text())
            self.assertEqual(plan['profile'],'method-comparison' if method=='diptych' else 'method-comparison-dbpp')
            for benchmark in plan['plans']:
                rows=load(PROJECT/method/'manifests'/(benchmark['benchmark']+'_seed42.jsonl'),PROJECT)
                cases=[c for lane in benchmark['lanes'] for c in lane]
                self.assertEqual(len(cases),benchmark['cases'])
                self.assertEqual(len(set(cases)),len(cases))
                owner={c:i for i,lane in enumerate(benchmark['lanes']) for c in lane}
                reference_owners={}
                for row in rows:
                    self.assertIn(caption(row),row['prompt'])
                    self.assertEqual(reference_owners.setdefault(row['reference_id'],owner[row['case_id']]),owner[row['case_id']])

    def test_native_size_corruption_and_prompt_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';ref=root/'reference.png'
            Image.new('RGB',(80,60),'red').save(ref)
            manifest=root/'manifest.jsonl';manifest.write_text('{}\n')
            row=dict(id='subject/0_0',case_id='000-00',reference_id='subject',seed=42,prompt='a red object')
            paths=artifact_paths(run,row)
            for p in paths.values():p.parent.mkdir(parents=True,exist_ok=True)
            Image.new('RGB',(768,768),'red').save(paths['tgt_image'])
            paths['src_image'].write_bytes(ref.read_bytes());paths['text'].write_text(row['prompt'])
            info=dict(config=dict(method='diptych',seeds=[42],width=768,height=768),manifest_sha256=sha256(manifest),fingerprint='test')
            write_json(run/'run.json',info)
            def metadata():
                write_json(paths['metadata'],dict(**row,fingerprint='test',artifact_sha256={k:sha256(p) for k,p in paths.items() if k!='metadata'}))
            metadata()
            plan=dict(freecus_run=str(run),manifest=str(manifest),manifest_sha256=sha256(manifest),data_root=str(root),
                selection=dict(name='all'),records=[dict(condition='diptych',freecus_id=row['id'],
                reference=str(ref),reference_sha256=sha256(ref),generated=str(paths['tgt_image']),
                row=dict(case_id=row['case_id'],reference_id=row['reference_id'],target_prompt=row['prompt']))])
            with patch('comparison_candidates.evaluation.prepare',return_value=copy.deepcopy(plan)):
                validate(plan)
                paths['text'].write_text('tampered')
                with self.assertRaisesRegex(ValueError,'artifact'):validate(plan)
                paths['text'].write_text(row['prompt']);metadata()
                bad=json.loads(paths['metadata'].read_text());bad['prompt']='changed';write_json(paths['metadata'],bad)
                with self.assertRaisesRegex(ValueError,'identity'):validate(plan)
                Image.new('RGB',(1024,1024)).save(paths['tgt_image']);metadata()
                with self.assertRaisesRegex(ValueError,'size'):validate(plan)

    def test_saved_subset_checks_global_size_and_unknown_cases(self):
        canonical=dict(records=[dict(condition='xverse',freecus_id='a',row=dict(case_id='a')),
            dict(condition='xverse',freecus_id='b',row=dict(case_id='b'))],selection=dict(name='all'))
        plan=dict(freecus_run='unused',manifest='unused',manifest_sha256='test',data_root='unused',
            partition=dict(case_ids=['a'],global_cases=2),records=[canonical['records'][0]],selection=canonical['selection'])
        info=dict(config=dict(method='xverse',seeds=[42],width=768,height=768),manifest_sha256='test')
        with patch('pathlib.Path.read_text',return_value=json.dumps(info)), \
             patch('comparison_candidates.evaluation.sha256',return_value='test'), \
             patch('comparison_candidates.evaluation.project_evaluator',return_value=None), \
             patch('comparison_candidates.evaluation.prepare',side_effect=lambda *a:copy.deepcopy(canonical)), \
             patch('comparison_candidates.evaluation.artifact_paths',side_effect=RuntimeError('validated selection')):
            with self.assertRaisesRegex(RuntimeError,'validated selection'):validate(plan)
            plan['partition']['global_cases']=1350
            with self.assertRaisesRegex(ValueError,'partition'):validate(plan)
            plan['partition']=dict(case_ids=['unknown'],global_cases=2)
            with self.assertRaisesRegex(ValueError,'Unknown partition'):validate(plan)


if __name__=='__main__':unittest.main()
