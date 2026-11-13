import tempfile
import unittest
from pathlib import Path
from freecus.recovery_512 import verified_copy


class CacheRecoveryTests(unittest.TestCase):
    def test_verified_copy_and_idempotency(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'old.pth';dest=root/'new'/'dino.pth'
            source.write_bytes(b'fixture weights')
            first=verified_copy(source,dest)
            self.assertEqual(first,verified_copy(source,dest))
            self.assertEqual(source.read_bytes(),dest.read_bytes())
            self.assertTrue(first['verified'])
            self.assertFalse(first['tls_verification_disabled'])
            dest.write_bytes(b'different weights')
            with self.assertRaisesRegex(ValueError,'refusing overwrite'): verified_copy(source,dest)


if __name__=='__main__': unittest.main()
