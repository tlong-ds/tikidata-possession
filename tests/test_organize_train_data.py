import tempfile
import unittest
from pathlib import Path

from pipeline.organize_train_data import organize


def old_layout(root):
    td = Path(root) / 'train_data'
    for d in ('idsse', 'metrica-sample-data/data', 'pff/FIFA World Cup 2022'):
        (td / d).mkdir(parents=True)
    (td / 'idsse' / 'match.xml').write_text('x')
    (td / 'README.md').write_text('readme')
    (td / '.DS_Store').write_text('')
    return td


class OrganizeTests(unittest.TestCase):
    def test_moves_datasets_and_creates_folders(self):
        with tempfile.TemporaryDirectory() as d:
            td = old_layout(d)
            moved, skipped = organize(td)
            self.assertEqual(moved, ['idsse', 'metrica-sample-data', 'pff'])
            self.assertEqual(skipped, [])
            self.assertEqual((td / 'raw' / 'idsse' / 'match.xml').read_text(), 'x')
            for f in ('raw', 'train', 'dev/select', 'dev/report', 'test'):
                self.assertTrue((td / f).is_dir(), f)
            self.assertEqual((td / 'README.md').read_text(), 'readme')       # untouched
            self.assertTrue((td / '.DS_Store').exists())
            self.assertFalse((td / 'idsse').exists())

    def test_idempotent_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as d:
            td = old_layout(d)
            organize(td)
            self.assertEqual(organize(td), ([], []))                         # second run: nothing to do
            (td / 'idsse').mkdir()                                           # a stray second copy
            (td / 'idsse' / 'other.xml').write_text('y')
            moved, skipped = organize(td)
            self.assertEqual((moved, skipped), ([], ['idsse']))
            self.assertEqual((td / 'raw' / 'idsse' / 'match.xml').read_text(), 'x')
            self.assertTrue((td / 'idsse' / 'other.xml').exists())

    def test_dry_run_changes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            td = old_layout(d)
            before = sorted(p.relative_to(td) for p in td.rglob('*'))
            moved, _ = organize(td, dry_run=True)
            self.assertEqual(moved, ['idsse', 'metrica-sample-data', 'pff'])
            self.assertEqual(sorted(p.relative_to(td) for p in td.rglob('*')), before)

    def test_missing_folder_is_a_clear_error(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit):
                organize(Path(d) / 'nope')


if __name__ == '__main__':
    unittest.main()
