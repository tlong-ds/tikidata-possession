import contextlib
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

from pipeline import feed, devset
from pipeline.degrade import realization_seed
from pipeline.quality import digest, fit_target_profile, save_profile
from pipeline.schema import ROOT
from pipeline.split import default_split
from pipeline.view import make_view, resolve_geometry
from test_degradation import fixture, parameters
from test_view import oriented, config


@contextlib.contextmanager
def protocol_env():
    """A tiny but complete release + patched loaders that serve a fixture for every DFL name."""
    with tempfile.TemporaryDirectory(dir=Path(ROOT) / 'cache') as d, contextlib.ExitStack() as stack:
        d = Path(d)
        profile = fit_target_profile(fixture(target=True))
        save_profile(profile, d / 'target.json')
        params = parameters(profile)
        (d / 'dfl.json').write_text(json.dumps(params))
        cfg = {'window': 5, 'polyorder': 2, 'max_speed_m_s': 30, 'target_profile_hash': profile['profile_hash']}
        cfg['config_hash'] = digest(cfg)
        (d / 'view-config.json').write_text(json.dumps(cfg))
        (d / 'overrides.json').write_text('{}')
        pointer = d / 'CURRENT.json'
        release = feed.write_current(d / 'target.json', d / 'dfl.json', d / 'view-config.json', d / 'overrides.json', pointer)

        def loader(name):
            m = oriented(400)
            m.name = name
            return m
        for target in ('pipeline.feed.load_match', 'pipeline.devset.load_match', 'pipeline.evaluate_dev.load_match'):
            stack.enter_context(patch(target, side_effect=loader))
        yield d, release, pointer


class SeedAndContractTests(unittest.TestCase):
    def test_realization_seeds_are_namespaced_and_stable(self):
        a = realization_seed('train', 0, 3, 'DFL-MAT-J03WOH')
        self.assertEqual(a, realization_seed('train', 0, 3, 'DFL-MAT-J03WOH'))
        self.assertNotEqual(a, realization_seed('dev', 0, 3, 'DFL-MAT-J03WOH'))
        self.assertNotEqual(a, realization_seed('train', 0, 4, 'DFL-MAT-J03WOH'))
        self.assertNotEqual(a, realization_seed('train', 1, 3, 'DFL-MAT-J03WOH'))
        with self.assertRaises(ValueError):
            realization_seed('bogus', 0, 0, 'x')

    def test_graded_defaults_to_in_half_frames_for_ungraded_sources(self):
        m = oriented(400)
        m.half[:20] = 0
        self.assertIsNone(m.graded)
        v = make_view(m, resolve_geometry(m), config())
        np.testing.assert_array_equal(v.graded, m.half > 0)
        m.graded = np.zeros(400, bool)
        m.graded[100:200] = True
        v = make_view(m, resolve_geometry(m), config())
        np.testing.assert_array_equal(v.graded, m.graded)


class FeedTests(unittest.TestCase):
    def test_release_roundtrip_and_tamper(self):
        with protocol_env() as (d, release, pointer):
            again = feed.load_release(pointer)
            self.assertEqual(again.config['config_hash'], release.config['config_hash'])
            (d / 'overrides.json').write_text('{"x": {}}')
            with self.assertRaises(ValueError):
                feed.load_release(pointer)
            (d / 'overrides.json').write_text('{}')
            ptr = json.loads(pointer.read_text())
            ptr['files']['view_config'] = 'cache/elsewhere.json'
            pointer.write_text(json.dumps(ptr))
            with self.assertRaises(ValueError):
                feed.load_release(pointer)

    def test_mismatched_profile_is_rejected(self):
        with protocol_env() as (d, release, pointer):
            params = json.loads((d / 'dfl.json').read_text())
            params['profile_hash'] = 'f' * 64
            (d / 'dfl2.json').write_text(json.dumps(params))
            with self.assertRaises(ValueError):
                feed.write_current(d / 'target.json', d / 'dfl2.json', d / 'view-config.json', d / 'overrides.json', d / 'C2.json')

    def test_train_view_is_deterministic_varies_by_epoch_and_is_guarded(self):
        with protocol_env() as (d, release, pointer):
            a, ta = feed.train_view('DFL-MAT-J03WOH', 1, 0, release=release)
            b, _ = feed.train_view('DFL-MAT-J03WOH', 1, 0, release=release)
            c, _ = feed.train_view('DFL-MAT-J03WOH', 2, 0, release=release)
            self.assertEqual(feed.view_hash(a), feed.view_hash(b))
            self.assertNotEqual(feed.view_hash(a), feed.view_hash(c))
            self.assertIn('ball_xy', ta)
            for bad in ('DFL-MAT-J03WR9', 'DFL-MAT-J03WN1', 'BAR-ATH_tactical'):
                with patch('pipeline.feed.load_match', side_effect=AssertionError('must not load')):
                    with self.assertRaises(ValueError):
                        feed.train_view(bad, 1, 0, release=release)

    def test_devset_is_frozen_split_into_subsets_and_detects_drift(self):
        with protocol_env() as (d, release, pointer):
            out = d / 'devset'
            out.mkdir()
            ds = devset.build(default_split(), release, 4, out)
            self.assertEqual(ds['select'], [0, 1])
            self.assertEqual(ds['report'], [2, 3])
            self.assertNotIn('DFL-MAT-J03WR9', ds['matches'])
            self.assertEqual(set(ds['matches']), set(default_split()['dfl']) - {'DFL-MAT-J03WR9'})
            rep = list(feed.dev_views(out, subset='report', release=release))
            self.assertEqual({r[1] for r in rep}, {'r02', 'r03'})
            self.assertEqual({r[0] for r in rep}, {'DFL-MAT-J03WN1'})        # official dev match only
            sel = list(feed.dev_views(out, subset='select', release=release))
            self.assertEqual({r[1] for r in sel}, {'r00', 'r01'})
            self.assertEqual([r[1] for r in feed.dev_views(out, subset='clean', release=release)], ['clean'])
            # Dev realizations are not the training ones.
            train, _ = feed.train_view('DFL-MAT-J03WOH', 0, 0, release=release)
            self.assertNotIn(feed.view_hash(train), {e['view_sha256'] for es in ds['matches'].values() for e in es})
            # Drift: any change to a recorded hash or the manifest is refused.
            tampered = copy.deepcopy(ds)
            tampered['matches']['DFL-MAT-J03WN1'][3]['view_sha256'] = '0' * 64
            (out / 'devset.json').write_text(json.dumps(tampered))
            with self.assertRaises(ValueError):
                list(feed.dev_views(out, subset='report', release=release))
            tampered['devset_hash'] = digest({k: v for k, v in tampered.items() if k != 'devset_hash'})
            (out / 'devset.json').write_text(json.dumps(tampered))
            with self.assertRaises(ValueError):
                list(feed.dev_views(out, subset='report', release=release))

    def test_fold_selects_other_dev_match(self):
        with protocol_env() as (d, release, pointer):
            out = d / 'devset'
            out.mkdir()
            devset.build(default_split(), release, 2, out)
            names = {r[0] for r in feed.dev_views(out, fold=0, subset='report', release=release)}
            self.assertEqual(names, {'DFL-MAT-J03WMX'})


if __name__ == '__main__':
    unittest.main()
