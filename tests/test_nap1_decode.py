"""Test stage-1 block 4 (IOHMM decoding)."""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from src.nap1.ball import estimate_ball
from src.nap1.decode import decode, codes_to_labels, to_eval_codes, to_submission_df
from src.nap1.params import Params, load_fold_params
from src.nap1.score import score_frames
from src.nap1.view import create_view
from pipeline.load import load_match


class TestDecode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_match("BAR-ATH_tactical")
        cls.p = Params()
        cls.v = create_view(cls.m, cls.p)
        cls.Bhat, cls.K = estimate_ball(cls.v, cls.p)
        cls.E, cls.C = score_frames(cls.v, cls.Bhat, cls.K, cls.p)

    def test_decode_pipeline_default(self):
        y_codes = decode(self.E, self.C, self.K, self.v.meta, self.p)

        # 1. Check shape and dtype (int8 codes).
        self.assertEqual(y_codes.shape, (self.m.n_frames,))
        self.assertEqual(y_codes.dtype, np.int8)

        # Convert to string labels.
        y_labels = codes_to_labels(y_codes)
        self.assertEqual(y_labels.shape, (self.m.n_frames,))
        self.assertEqual(y_labels.dtype, np.dtype("<U1"))

        # 2. Check the label set.
        unique_labels = set(np.unique(y_labels))
        self.assertTrue(unique_labels.issubset({"H", "A", "D", "X"}))
        self.assertTrue({"H", "A", "D"}.issubset(unique_labels))

        # 3. Never emit X on graded frames.
        graded = self.v.graded
        if graded is not None:
            graded_labels = set(np.unique(y_labels[graded]))
            self.assertNotIn("X", graded_labels, "Khung được chấm không được chứa nhãn X!")

        # 4. Check for label flickering: fewer than 2,000 spells.
        diffs = (y_labels[:-1] != y_labels[1:])
        n_spells = int(np.sum(diffs)) + 1
        self.assertLess(n_spells, 2000, f"Số lượt quá lớn ({n_spells})")

        # 5. Check to_submission_df and to_eval_codes.
        df = to_submission_df(y_labels)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), self.m.n_frames)
        self.assertIn("label", df.columns)

        eval_codes = to_eval_codes(y_codes)
        self.assertEqual(eval_codes.dtype, np.int8)
        self.assertTrue(np.isin(eval_codes, (0, 1, 2)).all())

    def test_decode_gating_modes(self):
        # Test both deadball_switch_gate modes: False (free exit) and True (gated).
        p_free = Params(deadball_switch_gate=False)
        y_free = decode(self.E, self.C, self.K, self.v.meta, p_free)
        self.assertEqual(y_free.shape, (self.m.n_frames,))

        p_gated = Params(deadball_switch_gate=True)
        y_gated = decode(self.E, self.C, self.K, self.v.meta, p_gated)
        self.assertEqual(y_gated.shape, (self.m.n_frames,))

    def test_decode_none_params_uses_official(self):
        y_none = decode(self.E, self.C, self.K, self.v.meta)
        y_off = decode(self.E, self.C, self.K, self.v.meta, load_fold_params(None))
        self.assertTrue(np.array_equal(y_none, y_off))


if __name__ == "__main__":
    unittest.main()
