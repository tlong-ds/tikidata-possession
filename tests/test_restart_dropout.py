import copy
import unittest
import numpy as np

from pipeline.degrade import degrade_match
from pipeline.quality import fit_target_profile, profile_match, restart_frames, pre_restart_mask
from pipeline.quality_report import compare_profiles
from test_degradation import fixture, parameters

N = 4000                                      # fps 10 in the fixture: windows are 80 frames
ANCHORS = np.array([300, 700, 1100, 1500, 2300, 2700, 3100, 3500])    # half boundary at 2000


def target():
    m = fixture(N, target=True)
    t = np.arange(N)
    m.B[:] = np.array([.5, .5, 3.])
    window = pre_restart_mask(ANCHORS, m.half, 10)
    m.B[window & (t % 5 != 0)] = np.nan                 # ~20% visible just before restarts
    m.B[~window & (t % 40 < 8)] = np.nan                # ~80% visible elsewhere
    m.extras['tbl_tag_code'] = np.array(['SET PIECES'] * len(ANCHORS))
    m.extras['tbl_tag_start_frame'] = ANCHORS
    return m


def source():
    m = fixture(N)
    m.extras['tbl_ev_setpiece'] = np.ones(len(ANCHORS), bool)
    m.extras['tbl_ev_frame'] = ANCHORS
    return m


def pre_visibility(obs):
    mask = pre_restart_mask(ANCHORS, obs.half, 10)
    return float(np.isfinite(obs.B[mask, 0]).mean())


class RestartDropoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = fit_target_profile(target())
        cls.params = dict(parameters(cls.profile), restart_strength=1.)

    def test_target_metric_measures_the_cue(self):
        metric = self.profile['metrics']['ball_visible_pre_restart']
        self.assertAlmostEqual(metric['value'], .2, delta=.05)
        self.assertGreater(self.profile['metrics']['ball_visible']['value'], .6)

    def test_restart_sources_and_windows(self):
        np.testing.assert_array_equal(restart_frames(source()), ANCHORS)
        np.testing.assert_array_equal(restart_frames(target()), ANCHORS)
        self.assertEqual(len(restart_frames(fixture())), 0)
        half = np.r_[np.ones(100), np.full(100, 2)].astype(np.int8)
        mask = pre_restart_mask([110, 50, 5], half, 10)        # 110: window crosses halves; 5: before start
        self.assertFalse(mask.any())
        self.assertEqual(int(pre_restart_mask([190], half, 10).sum()), 80)

    def test_degraded_ball_loses_the_ball_before_restarts_and_conserves_budget(self):
        m = source()
        on = degrade_match(m, self.profile, self.params, 5)
        off = degrade_match(m, self.profile, dict(self.params, restart_strength=0.), 5)
        self.assertAlmostEqual(pre_visibility(on.observations), .2, delta=.1)
        self.assertGreater(pre_visibility(off.observations), .45)
        vis = lambda s: float(np.isfinite(s.observations.B[:, 0]).mean())
        self.assertAlmostEqual(vis(on), vis(off), delta=.03)           # same total dropout budget
        self.assertGreater(on.audit['ball_budget']['restart_preset_frames'], 0)
        self.assertEqual(off.audit['ball_budget']['restart_preset_frames'], 0)
        np.testing.assert_array_equal(on.observations.extras['restart_frames'], ANCHORS)
        # Source ball never reappears where it was already missing.
        self.assertFalse((np.isfinite(on.observations.B[:, 0]) & ~np.isfinite(m.B[:, 0])).any())

    def test_labels_do_not_influence_restart_dropout(self):
        m = source()
        other = copy.deepcopy(m)
        other.extras['y_true'][:] = 0
        a = degrade_match(m, self.profile, self.params, 9).observations.B
        b = degrade_match(other, self.profile, self.params, 9).observations.B
        np.testing.assert_equal(a, b)

    def test_old_profile_without_the_metric_is_unchanged(self):
        old = copy.deepcopy(self.profile)
        del old['metrics']['ball_visible_pre_restart']
        m = source()
        a = degrade_match(m, old, dict(self.params, restart_strength=1.), 3)
        b = degrade_match(m, old, dict(self.params, restart_strength=0.), 3)
        np.testing.assert_equal(a.observations.B, b.observations.B)
        np.testing.assert_equal(a.observations.P, b.observations.P)
        c = degrade_match(m, old, {k: v for k, v in self.params.items() if k != 'restart_strength'}, 3)
        np.testing.assert_equal(a.observations.B, c.observations.B)

    def test_gate_fails_without_the_cue_and_passes_with_it(self):
        m = source()
        clean = profile_match(m, include_sequences=False)
        on = profile_match(degrade_match(m, self.profile, self.params, 5).observations, include_sequences=False)
        off = profile_match(degrade_match(m, self.profile, dict(self.params, restart_strength=0.), 5).observations,
                            include_sequences=False)
        self.assertTrue(compare_profiles(self.profile, clean, on)['gates']['ball_visible_pre_restart']['passed'])
        self.assertFalse(compare_profiles(self.profile, clean, off)['gates']['ball_visible_pre_restart']['passed'])

    def test_gate_absent_for_sources_without_restarts_and_old_profiles(self):
        plain = fixture()
        clean = profile_match(plain, include_sequences=False)
        self.assertNotIn('ball_visible_pre_restart', compare_profiles(self.profile, clean, clean)['gates'])
        old = copy.deepcopy(self.profile)
        del old['metrics']['ball_visible_pre_restart']
        s = source()
        c = profile_match(s, include_sequences=False)
        self.assertNotIn('ball_visible_pre_restart', compare_profiles(old, c, c)['gates'])


if __name__ == '__main__':
    unittest.main()
