import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

from pipeline import verify_train_data as v


def make(root, wrong_folder=False):
    T = 5
    d = dict(P=np.zeros((T, 46, 6), np.float32), B=np.zeros((T, 3), np.float32), cam=np.zeros((T, 2), np.float32),
             half=np.ones(T, np.int8), graded=np.ones(T, bool), y=np.zeros(T, np.int8),
             ball_xy=np.zeros((T, 2), np.float32))
    (root / 'train').mkdir(parents=True)
    np.savez(root / 'train' / ('DFL-MAT-B_e0.npz' if wrong_folder else 'DFL-MAT-A_e0.npz'), **d)
    (root / 'manifest.json').write_text(json.dumps({
        'roles': {'train': ['DFL-MAT-A'], 'dev': ['DFL-MAT-B'], 'test': ['DFL-MAT-C']},
        'files': {'train': ['DFL-MAT-B_e0.npz' if wrong_folder else 'DFL-MAT-A_e0.npz']}, 'release_hash': 'h'}))


class VerifyTests(unittest.TestCase):
    def test_ok_then_tamper(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            make(root)
            self.assertIn('checksums.json missing', v.verify(root)[0])
            v.write_checksums(root)
            self.assertEqual(v.verify(root), [])
            f = root / 'train' / 'DFL-MAT-A_e0.npz'
            f.write_bytes(f.read_bytes()[:-20])                      # truncated copy
            self.assertTrue(any('checksum mismatch' in p or 'unreadable' in p for p in v.verify(root)))

    def test_match_in_wrong_folder(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            make(root, wrong_folder=True)
            v.write_checksums(root)
            self.assertTrue(any('must not be in train' in p for p in v.verify(root)))


if __name__ == '__main__':
    unittest.main()
