import json
from pathlib import Path
import shutil
import tempfile
import unittest

from PIL import Image
from freecus.evaluate import prepare, project_evaluator, validate
from freecus.manifest import PROJECT, load, sha256
from freecus.operations.artifact_bundle import pack, verify
from freecus.run import artifact_paths, validate_config, write_json

BASE=PROJECT/'freecus'


class Resolution512Tests(unittest.TestCase):
    def test_separate_config_only_changes_resolution(self):
        original=json.loads((BASE/'config.json').read_text())
        experiment=json.loads((BASE/'config.512.json').read_text())
        self.assertEqual((original['width'],original['height']),(1024,1024))
        validate_config(experiment)
        experiment.update(width=1024,height=1024)
        self.assertEqual(experiment,original)
        experiment['height']=512
        with self.assertRaises(ValueError): validate_config(experiment)

    def test_partial_512_validation_and_lossless_export(self):
        manifest=BASE/'manifests/dreambench_project_seed42.jsonl'
        rows=load(manifest,PROJECT)
        row=rows[0]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);lane=root/'lane-0'
            cfg=json.loads((BASE/'config.512.json').read_text())
            info=dict(config=cfg,fingerprint='fixture',manifest_sha256=sha256(manifest),code={},
                      versions={},gpu='fixture',cuda=None)
            write_json(lane/'run.json',info)
            paths=artifact_paths(lane,row)
            for path in paths.values(): path.parent.mkdir(parents=True,exist_ok=True)
            Image.new('RGB',(512,512),'red').save(paths['tgt_image'])
            with Image.open(PROJECT/row['reference']) as ref: ref.convert('RGB').save(paths['src_image'])
            paths['text'].write_text(row['prompt'],encoding='utf-8')
            write_json(paths['metadata'],dict(**row,fingerprint='fixture',
                       artifact_sha256={k:sha256(p) for k,p in paths.items() if k!='metadata'}))
            plan=prepare(manifest,PROJECT,lane,project_evaluator(),'all',[row['case_id']])
            self.assertFalse(plan['selection']['complete'])
            self.assertEqual(len(plan['selection']['missing_case_ids']),749)
            self.assertEqual(len(plan['records']),1)
            validate(plan)
            index=pack([lane],manifest,PROJECT,root/'archives','freecus_512','dreambench','fixture',1,
                       resolution=(512,512))
            self.assertEqual(index['generation_resolution'],[512,512])
            self.assertEqual(len(index['missing_case_ids']),749)
            verify(root/'archives',index)
            Image.new('RGB',(1024,1024),'red').save(paths['tgt_image'])
            meta=json.loads(paths['metadata'].read_text())
            meta['artifact_sha256']['tgt_image']=sha256(paths['tgt_image'])
            write_json(paths['metadata'],meta)
            with self.assertRaisesRegex(ValueError,'resolution'): validate(plan)

    def test_canonical_style_cases_retained_for_exclusion_audit(self):
        manifest=BASE/'manifests/dreambench_plus_project_seed42.jsonl'
        full=prepare(manifest,PROJECT,Path('unused'),project_evaluator())
        self.assertEqual(len(full['records']),1350)
        self.assertEqual(sum(r['category']=='style' for r in full['records']),180)
        subset=prepare(manifest,PROJECT,Path('unused'),project_evaluator(),'dbpp-no-style',[])
        self.assertFalse(subset['selection']['complete'])
        self.assertEqual(subset['selection']['expected_cases'],1170)
        self.assertEqual(subset['selection']['excluded_cases'],180)
        self.assertEqual(len(subset['records']),0)


if __name__=='__main__': unittest.main()
