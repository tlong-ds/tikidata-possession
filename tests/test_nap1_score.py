"""Test stage-1 block 3: pitch control, dead ball and frame scoring."""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from src.nap1.ball import estimate_ball
from src.nap1.params import Params, load_fold_params
from src.nap1.score import ScoreDetails, score_frames, score_frames_detailed
from src.nap1.view import View, create_view
from pipeline.load import load_match


class TestScore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_match("BAR-ATH_tactical")
        cls.p = Params()
        cls.v = create_view(cls.m, cls.p)
        cls.Bhat, cls.K = estimate_ball(cls.v, cls.p)

    def test_score_frames_contract(self):
        # score_frames() MUST return exactly the two-element tuple (E,C).
        out = score_frames(self.v, self.Bhat, self.K, self.p)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        E, C = out
        self.assertEqual(E.shape, (self.m.n_frames, 3))
        self.assertEqual(C.shape, (self.m.n_frames, 2))
        self.assertEqual(E.dtype, np.float64)
        self.assertEqual(C.dtype, np.float64)
        self.assertTrue(np.all(np.isfinite(E)))
        self.assertTrue(np.all(np.isfinite(C)))
        self.assertTrue(np.all(E <= 0.0))
        self.assertTrue(np.all((C >= 0.0) & (C <= 1.0)))
        self.assertTrue(np.allclose(C[:, 0] + C[:, 1], 1.0))

    def test_score_frames_detailed(self):
        # score_frames_detailed() returns ScoreDetails(E,C,p1,D).
        details = score_frames_detailed(self.v, self.Bhat, self.K, self.p)
        self.assertIsInstance(details, ScoreDetails)
        self.assertEqual(details.E.shape, (self.m.n_frames, 3))
        self.assertEqual(details.C.shape, (self.m.n_frames, 2))
        self.assertEqual(details.p1.shape, (self.m.n_frames,))
        self.assertEqual(details.D.shape, (self.m.n_frames,))

        # Check value ranges and absence of NaN/Inf.
        self.assertTrue(np.all(np.isfinite(details.p1)))
        self.assertTrue(np.all(np.isfinite(details.D)))
        self.assertTrue(np.all((details.p1 >= 0.0) & (details.p1 <= 1.0)))
        self.assertTrue(np.all((details.D >= 0.0) & (details.D <= 1.0)))

        # Outside halves, D must be 1.0 (dead ball).
        outside = (self.v.half == 0)
        self.assertTrue(np.all(details.D[outside] == 1.0))

        # Mean dead-ball probability within halves must be reasonable.
        in_game = (self.v.half > 0)
        mean_D = np.mean(details.D[in_game])
        self.assertTrue(0.15 < mean_D < 0.65, f"Xác suất bóng chết {mean_D} ngoài khoảng kỳ vọng")

    def test_score_soft_boundary(self):
        p_soft = Params(w_dead_out=0.5)
        details_soft = score_frames_detailed(self.v, self.Bhat, self.K, p_soft)
        self.assertEqual(details_soft.D.shape, (self.m.n_frames,))
        self.assertTrue(np.all((details_soft.D >= 0.0) & (details_soft.D <= 1.0)))

    def test_empty_players_frame_is_finite(self):
        # No players (t1=inf, t0=inf) must still yield finite p1=0.5.
        T = 4
        P = np.full((T, 46, 6), np.nan, dtype=np.float64)
        P[:, :, 5] = 0.0
        # Frame 1 has only team 0, frame 2 only team 1; frames 0 and 3 have neither.
        P[1, 0, :2] = [0.0, 0.0]
        P[1, 0, 5] = 1.0
        P[2, 24, :2] = [0.0, 0.0]
        P[2, 24, 5] = 1.0
        B = np.zeros((T, 3), dtype=np.float64)
        cam = np.zeros((T, 2), dtype=np.float64)
        v_syn = View(P=P, B=B, cam=cam, meta={"fps": 25, "pitch": (105.0, 68.0), "half": np.ones(T, dtype=np.int8)})
        Bhat = np.zeros((T, 5), dtype=np.float64)
        Bhat[:, 2] = 0.2
        K = np.zeros(T, dtype=np.float64)
        details = score_frames_detailed(v_syn, Bhat, K, self.p)
        self.assertTrue(np.all(np.isfinite(details.p1)))
        self.assertTrue(np.all(np.isfinite(details.E)))
        self.assertAlmostEqual(details.p1[0], 0.5)
        self.assertGreater(details.p1[1], 0.9)
        self.assertLess(details.p1[2], 0.1)

    def test_score_frames_none_params_uses_official(self):
        E_none, C_none = score_frames(self.v, self.Bhat, self.K)
        E_off, C_off = score_frames(self.v, self.Bhat, self.K, load_fold_params(None))
        self.assertTrue(np.array_equal(E_none, E_off))
        self.assertTrue(np.array_equal(C_none, C_off))


if __name__ == "__main__":
    unittest.main()
