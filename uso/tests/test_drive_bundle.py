import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from PIL import Image
from freecus.operations.drive_bundle_512 import build


class DriveBundleTests(unittest.TestCase):
    def fixture(self, root, size):
        local, source = root / 'records', root / 'raw'
        local.mkdir()
        source.mkdir()
        row = dict(id='cat/0_0', case_id='case-cat', reference_id='cat', prompt='a cat in a jungle')
        (local / 'manifest.jsonl').write_text(json.dumps(row) + '\n', encoding='utf-8')
        image = io.BytesIO()
        Image.new('RGB', (size, size), 'red').save(image, format='PNG')
        data = image.getvalue()
        name = 'tgt_image/cat/0_0.png'
        archive = source / 'images-000.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr(name, data)
        index = dict(parts=[dict(file=archive.name, size_bytes=archive.stat().st_size,
                                 sha256=hashlib.sha256(archive.read_bytes()).hexdigest())],
                     members={name: dict(size=len(data), sha256=hashlib.sha256(data).hexdigest())})
        (source / 'bundle_index.json').write_text(json.dumps(index), encoding='utf-8')
        state = root / 'plans.json'
        state.write_text(json.dumps([dict(benchmark='dreambench', run_id='test', local=str(local),
                                          identity=dict(excluded_case_ids=[]))]), encoding='utf-8')
        return state, source, data

    def test_native_1024_bytes_and_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state, source, data = self.fixture(root, 1024)
            result = build('USO_1024', state, dict(dreambench=1), resolution=(1024, 1024),
                           destination=root / 'drive', source_folders=dict(dreambench=source))
            self.assertTrue(result['verified'])
            self.assertEqual(result['resolution'], [1024, 1024])
            with zipfile.ZipFile(result['archive']) as z:
                self.assertEqual(z.read('dreambench/case-cat.png'), data)
                self.assertIn('execution_records/upload_manifest.json', z.namelist())

    def test_wrong_resolution_is_not_resized(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state, source, _ = self.fixture(root, 512)
            with self.assertRaisesRegex(ValueError, 'resolution'):
                build('USO_1024', state, dict(dreambench=1), resolution=(1024, 1024),
                      destination=root / 'drive', source_folders=dict(dreambench=source))
            self.assertFalse((root / 'drive/USO_1024.zip').exists())

    def test_existing_512_default_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state, source, data = self.fixture(root, 512)
            result = build('FreeCus_512', state, dict(dreambench=1), destination=root / 'drive',
                           source_folders=dict(dreambench=source))
            self.assertEqual(result['resolution'], [512, 512])
            with zipfile.ZipFile(result['archive']) as z:
                self.assertEqual(z.read('dreambench/case-cat.png'), data)


if __name__ == '__main__':
    unittest.main()
