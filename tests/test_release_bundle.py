import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

from pipeline import devset, organize_train_data, release_bundle as rb
from pipeline.split import default_split
from test_feed import protocol_env


def rewrite(src, dst, member, data):
    with zipfile.ZipFile(src) as a, zipfile.ZipFile(dst, 'w') as b:
        for n in a.namelist():
            b.writestr(n, data if n == member else a.read(n))


class BundleTests(unittest.TestCase):
    def make(self, d, release, pointer):
        ds = d / 'devset'
        ds.mkdir()
        devset.build(default_split(), release, 2, ds)
        out = d / 'release.zip'
        manifest, size = rb.pack(out, pointer, ds, with_checksums=False)
        return ds, out, manifest

    def test_roundtrip_restores_a_deleted_release(self):
        with protocol_env() as (d, release, pointer):
            ds, out, manifest = self.make(d, release, pointer)
            self.assertEqual(rb.check_release(pointer, ds), [])
            self.assertEqual(len(manifest['files']), 6)                     # pointer + 4 pinned files + devset
            shutil.rmtree(d / 'devset')
            for f in ('target.json', 'dfl.json', 'view-config.json', 'overrides.json', 'CURRENT.json'):
                (d / f).unlink()
            self.assertTrue(len(rb.check_release(pointer, ds)) >= 2)
            written, same = rb.unpack(out, devset=ds)
            self.assertEqual((len(written), len(same)), (6, 0))
            self.assertEqual(rb.check_release(pointer, ds), [])
            self.assertEqual(rb.unpack(out, devset=ds), ([], written))      # run again: nothing to do

    def test_corrupt_member_is_rejected_before_anything_is_written(self):
        with protocol_env() as (d, release, pointer):
            ds, out, manifest = self.make(d, release, pointer)
            bad = d / 'bad.zip'
            rewrite(out, bad, rb._rel(d / 'dfl.json'), b'{}')
            (d / 'dfl.json').unlink()
            with self.assertRaisesRegex(ValueError, 'corrupt'):
                rb.unpack(bad, devset=ds)
            self.assertFalse((d / 'dfl.json').exists())

    def test_never_silently_replaces_a_different_file(self):
        with protocol_env() as (d, release, pointer):
            ds, out, manifest = self.make(d, release, pointer)
            (d / 'overrides.json').write_text('{"x": {}}')
            with self.assertRaisesRegex(ValueError, 'already exist'):
                rb.unpack(out, devset=ds)
            self.assertEqual((d / 'overrides.json').read_text(), '{"x": {}}')
            rb.unpack(out, force=True, devset=ds)
            self.assertEqual((d / 'overrides.json').read_text(), '{}')

    def test_refuses_paths_outside_cache(self):
        with tempfile.TemporaryDirectory() as t:
            z = Path(t) / 'evil.zip'
            with zipfile.ZipFile(z, 'w') as f:
                f.writestr(rb.BUNDLE_KEY, json.dumps({'format': rb.FORMAT, 'release': 'cache/x', 'files': {
                    '../outside.txt': {'sha256': 'x', 'size': 1}}}))
                f.writestr('../outside.txt', 'x')
            with self.assertRaisesRegex(ValueError, 'outside cache'):
                rb.unpack(z)
            z2 = Path(t) / 'notbundle.zip'
            with zipfile.ZipFile(z2, 'w') as f:
                f.writestr('a.txt', 'x')
            with self.assertRaisesRegex(ValueError, 'not a release bundle'):
                rb.unpack(z2)

    def test_pack_refuses_a_devset_from_another_release(self):
        with protocol_env() as (d, release, pointer):
            ds, out, manifest = self.make(d, release, pointer)
            j = json.loads((ds / 'devset.json').read_text())
            j['release_hash'] = 'other'
            (ds / 'devset.json').write_text(json.dumps(j))
            with self.assertRaises(SystemExit):
                rb.pack(d / 'again.zip', pointer, ds, with_checksums=False)

    def test_check_names_each_missing_release_file(self):
        with tempfile.TemporaryDirectory() as t:
            problems = rb.check_release(Path(t) / 'CURRENT-v3.json', Path(t) / 'devset')
            self.assertEqual(len(problems), 2)
            self.assertIn('release pointer', problems[0])


class OrganizeKeepsChecksums(unittest.TestCase):
    def test_checksums_json_is_not_moved_into_raw(self):
        with tempfile.TemporaryDirectory() as t:
            td = Path(t) / 'train_data'
            (td / 'idsse').mkdir(parents=True)
            (td / 'checksums.json').write_text('{}')
            moved, _ = organize_train_data.organize(td)
            self.assertEqual(moved, ['idsse'])
            self.assertTrue((td / 'checksums.json').exists())


if __name__ == '__main__':
    unittest.main()
