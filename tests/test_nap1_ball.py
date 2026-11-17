"""Test stage-1 block 2 (ball and touches)."""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from src.nap1.ball import estimate_ball
from src.nap1.params import Params
from src.nap1.view import create_view
from pipeline.load import load_match


class TestBall(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_match("BAR-ATH_tactical")
        cls.v = create_view(cls.m)

    def test_ball_estimation_default(self):
        p = Params(ball_gap_mode="2d_bounded")
        Bhat, K = estimate_ball(self.v, p)

        # 1. Check shape and dtype.
        self.assertEqual(Bhat.shape, (self.m.n_frames, 5))
        self.assertEqual(K.shape, (self.m.n_frames,))
        self.assertEqual(Bhat.dtype, np.float64)
        self.assertEqual(K.dtype, np.float64)

        # 2. Check Bhat values.
        # Column 2 (sigma) must be positive.
        self.assertTrue(np.all(Bhat[:, 2] > 0))
        # Column 3 (source): only 0 observed, 1 Kalman or 3 camera.
        unique_sources = set(np.unique(Bhat[:, 3]))
        self.assertTrue(unique_sources.issubset({0.0, 1.0, 3.0}))
        # Column 4 (confidence) lies in [0,1].
        self.assertTrue(np.all((Bhat[:, 4] >= 0.0) & (Bhat[:, 4] <= 1.0)))

        # 3. Check touch probability K.
        self.assertTrue(np.all((K >= 0.0) & (K <= 1.0)))
        # Touches must be detected.
        self.assertGreater(np.sum(K >= 0.5), 0)

    def test_ball_gap_modes(self):
        for mode in ("1d_forward", "rts_cov"):
            p = Params(ball_gap_mode=mode)
            Bhat, K = estimate_ball(self.v, p)
            self.assertEqual(Bhat.shape, (self.m.n_frames, 5))
            self.assertTrue(np.all(Bhat[:, 2] > 0))
            self.assertTrue(np.all((K >= 0.0) & (K <= 1.0)))

    def test_rts_cov_gap_uncertainty_grows_and_fuses_camera(self):
        # Ensure 2x2 RTS covariance during long gaps does not collapse to sigma_m,
        # and smoothly fuse camera center (source 3) when sig_k >sigma_cam.
        from src.nap1.view import View
        T = 600
        P = np.full((T, 46, 6), np.nan, dtype=np.float64)
        P[:, :, 5] = 0.0
        P[:, 0, 0] = -20.0
        P[:, 0, 1] = 0.0
        P[:, 0, 5] = 1.0

        B = np.zeros((T, 3), dtype=np.float64)
        B[:40, 0] = np.linspace(-10.0, -5.0, 40)
        B[:40, 2] = 1.0
        B[40:560, :2] = np.nan
        B[40:560, 2] = 0.0
        B[560:, 0] = np.linspace(5.0, 10.0, 40)
        B[560:, 2] = 1.0

        cam = np.zeros((T, 2), dtype=np.float64)
        cam[:, 0] = 2.0
        cam[:, 1] = 8.0

        v_syn = View(P=P, B=B, cam=cam, meta={"fps": 25, "pitch": (105.0, 68.0), "half": np.ones(T, dtype=np.int8)})
        p = Params(ball_gap_mode="rts_cov")
        Bhat, _ = estimate_ball(v_syn, p)
        mid_sig = float(Bhat[300, 2])
        self.assertGreater(mid_sig, 5.0, f"Độ bất định RTS giữa đoạn mất dài bị sập: {mid_sig}")
        self.assertEqual(Bhat[300, 3], 3.0)  # Fuse camera center (source 3) when sig_k >sigma_cam.
        self.assertGreater(float(Bhat[300, 1]), 1.0)  # Pull y toward camera center y=8.0.


if __name__ == "__main__":
    unittest.main()
