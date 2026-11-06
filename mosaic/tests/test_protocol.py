import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from mosaic.evaluation import validate
from freecus.run import artifact_paths, write_json
from freecus.manifest import PROJECT, load, sha256


class MosaicProtocolTests(unittest.TestCase):
    def test_native_size_corruption_and_prompt_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';ref=root/'reference.png'
            Image.new('RGB',(80,60),'red').save(ref)
            manifest=root/'manifest.jsonl';manifest.write_text('{}\n')
            row=dict(id='subject/0_0',case_id='000-00',reference_id='subject',seed=42,prompt='a red object')
            paths=artifact_paths(run,row)
            for p in paths.values():p.parent.mkdir(parents=True,exist_ok=True)
            Image.new('RGB',(512,512),'red').save(paths['tgt_image'])
            paths['src_image'].write_bytes(ref.read_bytes());paths['text'].write_text(row['prompt'])
            info=dict(config=dict(method='mosaic',seeds=[42],width=512,height=512),manifest_sha256=sha256(manifest),fingerprint='test')
            write_json(run/'run.json',info)
            def metadata():
                write_json(paths['metadata'],dict(**row,fingerprint='test',artifact_sha256={k:sha256(p) for k,p in paths.items() if k!='metadata'}))
            metadata()
            plan=dict(freecus_run=str(run),manifest=str(manifest),manifest_sha256=sha256(manifest),data_root=str(root),
                selection=dict(name='all'),records=[dict(condition='mosaic',freecus_id=row['id'],
                reference=str(ref),reference_sha256=sha256(ref),generated=str(paths['tgt_image']),
                row=dict(case_id=row['case_id'],reference_id=row['reference_id'],target_prompt=row['prompt']))])
            with patch('mosaic.evaluation.prepare',return_value=copy.deepcopy(plan)):
                validate(plan)
                paths['text'].write_text('tampered')
                with self.assertRaisesRegex(ValueError,'artifact'):validate(plan)
                paths['text'].write_text(row['prompt']);metadata()
                bad=json.loads(paths['metadata'].read_text());bad['prompt']='changed';write_json(paths['metadata'],bad)
                with self.assertRaisesRegex(ValueError,'identity'):validate(plan)
                Image.new('RGB',(1024,1024)).save(paths['tgt_image']);metadata()
                with self.assertRaisesRegex(ValueError,'size'):validate(plan)

    def test_canonical_manifests_and_original_references(self):
        all_ids = set()
        for benchmark, expected in [('dreambench',750),('dreambench_plus',1350)]:
            path = PROJECT/'mosaic/manifests'/f'{benchmark}_seed42.jsonl'
            self.assertEqual(path.read_bytes(), (PROJECT/'diptych/manifests'/path.name).read_bytes())
            rows = load(path, PROJECT)
            self.assertEqual(len(rows), expected)
            all_ids.update(r['case_id'] for r in rows)
            self.assertTrue(all(r['seed']==42 and r['reference_policy']=='selected_one' for r in rows))
        self.assertEqual(len(all_ids),2100)
