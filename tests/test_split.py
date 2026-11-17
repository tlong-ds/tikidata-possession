import json
import tempfile
import unittest
from pathlib import Path

from pipeline import split as S


class SplitTests(unittest.TestCase):
    def test_official_roles_are_disjoint_and_complete(self):
        r = S.roles(S.default_split())
        groups = [set(r['train']), set(r['dev']), set(r['reserved']), set(r['target']), set(r['diagnostic'])]
        self.assertEqual(sum(len(g) for g in groups), len(set().union(*groups)))
        self.assertEqual(r['dev'], ['DFL-MAT-J03WN1'])
        self.assertEqual(r['reserved'], ['DFL-MAT-J03WR9'])
        self.assertEqual(len(r['train']), 5)
        self.assertEqual(set(r['train']) | set(r['dev']) | set(r['reserved']), set(S.DFL))

    def test_folds_never_touch_reserved_and_rotate_dev(self):
        s = S.default_split()
        self.assertEqual(len(s['folds']), 6)
        self.assertEqual({f['dev'] for f in s['folds']}, set(S.DFL) - {S.RESERVED})
        for f in s['folds']:
            self.assertNotIn(S.RESERVED, f['train'] + [f['dev']])
            self.assertNotIn(f['dev'], f['train'])
            self.assertEqual(len(f['train']), 5)
        flagged = [f['dev'] for f in s['folds'] if f['calibration_overlap']]
        self.assertEqual(flagged, [S.CALIBRATION])

    def test_external_train_is_opt_in_and_train_only(self):
        s = S.default_split()
        r = S.roles(s)
        for n in S.EXTERNAL_TRAIN:
            S.guard(n, 'train')
            self.assertNotIn(n, r['train'])
            self.assertNotIn(n, json.dumps(s))                  # split-v1 and split_hash remain unchanged.
            for purpose in ('dev', 'select', 'report', 'score', 'final', 'infer', 'diagnostic'):
                with self.assertRaises(ValueError):
                    S.guard(n, purpose)

    def test_guard_rules(self):
        s = S.default_split()
        S.guard('DFL-MAT-J03WOH', 'train', s)
        S.guard('DFL-MAT-J03WN1', 'dev', s)
        S.guard('DFL-MAT-J03WN1', 'select', s)
        S.guard(S.TARGET, 'infer', s)
        S.guard('Sample_Game_1', 'diagnostic', s)
        for purpose in ('train', 'dev', 'select', 'report', 'score', 'infer'):
            with self.assertRaises(ValueError):
                S.guard(S.RESERVED, purpose, s)
        with self.assertRaises(ValueError):
            S.guard('DFL-MAT-J03WN1', 'train', s)        # dev match is not trainable
        with self.assertRaises(ValueError):
            S.guard(S.TARGET, 'train', s)
        with self.assertRaises(ValueError):
            S.guard('Sample_Game_2', 'train', s)
        with self.assertRaises(ValueError):
            S.guard('DFL-MAT-J03WOH', 'bogus', s)

    def test_guard_follows_fold(self):
        s = S.default_split()
        S.guard('DFL-MAT-J03WN1', 'train', s, fold=0)     # fold 0 develops on WMX, so WN1 trains
        S.guard('DFL-MAT-J03WMX', 'dev', s, fold=0)
        with self.assertRaises(ValueError):
            S.guard('DFL-MAT-J03WMX', 'train', s, fold=0)
        with self.assertRaises(ValueError):
            S.roles(s, fold=99)

    def test_saved_split_roundtrip_and_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'split.json'
            p.write_text(json.dumps(S.default_split()))
            self.assertEqual(S.load_split(p, verify=False)['split_hash'], S.default_split()['split_hash'])
            data = json.loads(p.read_text())
            data['official']['dev'] = 'DFL-MAT-J03WR9'    # try to sneak the reserved match in as dev
            p.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                S.load_split(p, verify=False)

    def test_benchmark_and_audit_use_the_split(self):
        from pipeline import benchmark_baseline as bb, validate_preprocessing as vp
        self.assertNotIn(S.RESERVED, bb.ALLOWED)
        self.assertEqual(vp.ROLES[S.RESERVED], 'reserved_final_evaluation')
        self.assertEqual(vp.ROLES['DFL-MAT-J03WN1'], 'development')
        self.assertEqual(vp.ROLES[S.CALIBRATION], 'calibration/training')


if __name__ == '__main__':
    unittest.main()
