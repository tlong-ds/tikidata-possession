"""Test stage-1 block 1 (View)."""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from src.nap1.params import Params, ensure_params, load_fold_params
from src.nap1.view import View, create_view, from_feed, from_npz
from pipeline.load import load_match
from pipeline.view import View as PipelineView


class TestView(unittest.TestCase):
    def test_params_serialization(self):
        p = Params()
        d = p.to_dict()
        self.assertEqual(d["qa"], 15.0)
        p2 = Params.from_dict(d)
        self.assertEqual(p2.tau, 8.0)

    def test_ensure_params_defaults_to_official(self):
        off = load_fold_params(None)
        auto_p = ensure_params(None)
        direct_p = Params()
        prior_p = Params.hand_prior()

        self.assertEqual(auto_p.to_dict(), off.to_dict())
        self.assertEqual(direct_p.to_dict(), off.to_dict())
        self.assertAlmostEqual(auto_p.L_alive, 7.29)
        self.assertAlmostEqual(prior_p.L_alive, 20.0)
        self.assertNotEqual(auto_p.to_dict(), prior_p.to_dict())

    def test_view_shapes_and_flips(self):
        m = load_match("BAR-ATH_tactical")
        p = Params()
        v = create_view(m, p)

        self.assertIsInstance(v, View)
        self.assertEqual(v.P.shape, (m.n_frames, 46, 6))
        self.assertEqual(v.B.shape, (m.n_frames, 3))
        self.assertEqual(v.cam.shape, (m.n_frames, 2))
        self.assertEqual(v.fps, 25)
        self.assertEqual(v.pitch, (105.3, 68.0))

        # Attack direction: group 1 is on the left (-x) relative to group 2 (+x).
        for h in (1, 2):
            mask_h = (v.half == h)
            mean_p1_x = np.nanmean(v.P[mask_h, :24, 0])
            mean_p2_x = np.nanmean(v.P[mask_h, 24:, 0])
            self.assertLess(mean_p1_x, mean_p2_x, f"Hiệp {h}: nhóm 1 ({mean_p1_x}) phải < nhóm 2 ({mean_p2_x})")

    def test_boundary_velocity_criterion(self):
        m = load_match("BAR-ATH_tactical")
        p = Params()
        v = create_view(m, p)

        # Boundary-velocity requirements:
        # Tracks >=2 frames must have finite velocities at both ends.
        # A one-frame track may have NaN velocity.
        p_present = (v.P[:, :, 5] == 1.0)
        for player_idx in range(46):
            pres = p_present[:, player_idx]
            if not np.any(pres):
                continue
            diff = np.diff(pres.astype(int))
            starts = np.where(diff == 1)[0] + 1
            if pres[0]:
                starts = np.r_[0, starts]
            ends = np.where(diff == -1)[0]
            if pres[-1]:
                ends = np.r_[ends, len(pres) - 1]

            for s, e in zip(starts, ends):
                track_len = e - s + 1
                if track_len >= 2:
                    self.assertFalse(np.isnan(v.P[s, player_idx, 2]))
                    self.assertFalse(np.isnan(v.P[e, player_idx, 2]))
                    self.assertFalse(np.isnan(v.P[s, player_idx, 3]))
                    self.assertFalse(np.isnan(v.P[e, player_idx, 3]))

    def test_from_feed_fills_boundary_velocities_only(self):
        T = 12
        P = np.full((T, 46, 6), np.nan, dtype=np.float64)
        # Slot 0: five-frame track 0..4 with existing interior Savitzky-Golay velocities and NaN boundaries.
        P[0:5, 0, 0] = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
        P[0:5, 0, 1] = np.zeros(5)
        P[2, 0, 2] = 9.99  # Simulate existing interior Savitzky-Golay values; do not overwrite.
        P[2, 0, 3] = 0.0
        P[0:5, 0, 5] = 1.0

        # Slot 1: one-frame track at t=7 may retain NaN velocity.
        P[7, 1, 0] = 1.0
        P[7, 1, 1] = 2.0
        P[7, 1, 5] = 1.0

        half = np.ones(T, dtype=np.int8)
        pv = PipelineView(
            P=P,
            B=np.zeros((T, 3), dtype=np.float64),
            cam=np.zeros((T, 2), dtype=np.float64),
            camera_valid=np.ones(T, dtype=bool),
            half=half,
            frame_valid=np.ones(T, dtype=bool),
            graded=np.ones(T, dtype=bool),
            frame_indices=np.arange(T),
            meta={"fps": 25.0, "pitch": [105.0, 68.0], "attack_sign": {"1": 1}},
        )

        v = from_feed(pv)
        self.assertIsInstance(v, View)
        self.assertEqual(v.P[2, 0, 2], 9.99)
        self.assertFalse(np.isnan(v.P[0, 0, 2]))
        self.assertFalse(np.isnan(v.P[4, 0, 2]))
        self.assertAlmostEqual(v.P[0, 0, 2], 5.0)
        self.assertTrue(np.isnan(v.P[7, 1, 2]))

    def test_from_npz(self):
        T = 8
        P = np.full((T, 46, 6), np.nan, dtype=np.float32)
        P[0:4, 0, 0] = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float32)
        P[0:4, 0, 1] = np.zeros(4, dtype=np.float32)
        P[0:4, 0, 5] = 1.0
        z = {
            "P": P,
            "B": np.zeros((T, 3), dtype=np.float32),
            "cam": np.zeros((T, 2), dtype=np.float32),
            "half": np.ones(T, dtype=np.int8),
            "graded": np.ones(T, dtype=bool),
            "fps": np.array(25.0),
            "pitch": np.array([105.0, 68.0]),
        }
        v = from_npz(z)
        self.assertIsInstance(v, View)
        self.assertEqual(v.fps, 25)
        self.assertEqual(v.pitch, (105.0, 68.0))
        self.assertFalse(np.isnan(v.P[0, 0, 2]))
        self.assertFalse(np.isnan(v.P[3, 0, 2]))


if __name__ == "__main__":
    unittest.main()
