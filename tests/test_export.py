import io
import unittest
from contextlib import redirect_stdout
import numpy as np

from pipeline import devset, export, score_export
from pipeline.degrade import realization_seed
from pipeline.split import default_split
from test_feed import protocol_env


class ExportTests(unittest.TestCase):
    def build(self, d, release, competition=None):
        ds = d / 'devset'
        ds.mkdir()
        devset.build(default_split(), release, 4, ds)
        return export.export(d / 'td', release, ds, n_train=1, n_test=2, competition=competition)

    def test_layout_roles_and_keys(self):
        with protocol_env() as (d, release, pointer):
            files = self.build(d, release, competition=d / 'comp' / 'match.npz')
            td = d / 'td'
            self.assertEqual(len(files['train']), 5)
            self.assertEqual(len(files['dev/select']), 2)
            self.assertEqual(len(files['dev/report']), 2)
            self.assertEqual(files['test'], ['DFL-MAT-J03WR9_r00.npz', 'DFL-MAT-J03WR9_r01.npz'])
            self.assertFalse(any('J03WR9' in f or 'J03WN1' in f for f in files['train']))
            self.assertTrue(all('J03WN1' in f for f in files['dev/select'] + files['dev/report']))
            z = np.load(td / 'train' / files['train'][0])
            T = len(z['half'])
            self.assertEqual(z['P'].shape, (T, 46, 6))
            self.assertEqual(z['P'].dtype, np.float32)
            self.assertTrue(set(np.unique(z['y'])) <= {-1, 0, 1, 2})
            for f in (td / 'dev' / 'report' / files['dev/report'][0], td / 'test' / files['test'][0]):
                self.assertEqual(set(np.load(f).files) >= {'y', 'baseline', 'P', 'B'}, True)
            comp = np.load(d / 'comp' / 'match.npz')
            self.assertNotIn('y', comp.files)
            self.assertFalse((td / 'competition').exists())                  # stays outside train_data
            self.assertTrue((td / 'manifest.json').exists())

    def test_test_copies_never_coincide_with_dev_or_train(self):
        n = 'DFL-MAT-J03WR9'
        seeds = {realization_seed(p, 0, i, n) for p in ('train', 'dev', 'test') for i in range(5)}
        self.assertEqual(len(seeds), 15)

    def test_competition_view_is_invisible_to_the_cache_loader(self):
        from pipeline.schema import CACHE_DIR
        rel = export.COMPETITION.relative_to(CACHE_DIR)
        self.assertGreater(len(rel.parts), 2)            # load_match globs cache/<source>/<name>.npz only

    def test_refuses_to_overwrite_existing_export(self):
        with protocol_env() as (d, release, pointer):
            self.build(d, release)
            with self.assertRaises(FileExistsError):
                export.export(d / 'td', release, d / 'devset', n_train=1, n_test=1)

    def test_scorer_on_dev_and_test(self):
        with protocol_env() as (d, release, pointer):
            self.build(d, release)
            preds = d / 'preds'
            preds.mkdir()
            for sub in ('dev/select', 'test'):
                for f in (d / 'td' / sub).glob('*.npz'):
                    np.savez(preds / f.name, codes=np.load(f)['baseline'])
            for subset, note in (('select', 'tuning only'), ('test', 'run once')):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    score_export.main([str(preds), subset, str(d / 'td')])
                self.assertIn('+0.0000', buf.getvalue())
                self.assertIn(note, buf.getvalue())
            with self.assertRaises(SystemExit):
                score_export.main([str(preds), 'report', str(d / 'missing')])


if __name__ == '__main__':
    unittest.main()
