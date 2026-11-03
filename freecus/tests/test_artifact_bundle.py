import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from freecus.operations.artifact_bundle import pack, verify, restore, digest, safe_path


class ArtifactBundleTests(unittest.TestCase):
    def test_roundtrip_keeps_bytes_and_refuses_overwrite_or_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs, lane = root / "input", root / "lane-0"
            inputs.mkdir()
            Image.new("RGB", (1024, 1024), "red").save(inputs / "selected.png")
            sha = digest(inputs / "selected.png")
            rows = [dict(case_id=str(i), id="subject/"+str(i), seed=42, reference_id="subject",
                         prompt="a subject", reference="selected.png", reference_sha256=sha) for i in range(750)]
            manifest = root / "manifest.jsonl"
            manifest.write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
            info = dict(fingerprint="frozen", config=dict(width=1024,height=1024,seeds=[42]),
                        manifest_sha256=digest(manifest), code={}, versions={}, gpu="test", cuda=None)
            lane.mkdir()
            (lane / "run.json").write_text(json.dumps(info))
            meta = dict(rows[0], fingerprint="frozen", artifact_sha256={})
            for kind, ext in [("tgt_image","png"),("src_image","png"),("text","txt")]:
                p=lane/kind/(rows[0]["id"]+"."+ext);p.parent.mkdir(parents=True)
                p.write_bytes((inputs/"selected.png").read_bytes() if ext=="png" else b"a subject")
                meta["artifact_sha256"][kind]=digest(p)
            p=lane/"metadata/subject/0.json";p.parent.mkdir(parents=True);p.write_text(json.dumps(meta))
            archives=root/"archives"
            index=pack([lane],manifest,inputs,archives,"fixture","dreambench","test",1)
            verify(archives,index)
            restored=root/"restored"
            restore(archives,index,restored)
            for kind,ext in [("tgt_image","png"),("src_image","png"),("text","txt"),("metadata","json")]:
                self.assertEqual((lane/kind/('subject/0.'+ext)).read_bytes(),(restored/kind/('subject/0.'+ext)).read_bytes())
            with self.assertRaises(ValueError):restore(archives,index,restored)
            target=archives/index['parts'][0]['file']
            with target.open('r+b') as f:f.write(b'broken')
            with self.assertRaises(ValueError):verify(archives,index)

    def test_reject_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            for value in ['../outside','/absolute','a\\b']:
                with self.assertRaises(ValueError):safe_path(Path(tmp),value)


if __name__ == '__main__':unittest.main()
