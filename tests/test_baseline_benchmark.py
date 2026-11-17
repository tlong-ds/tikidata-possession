import copy
import unittest
from unittest.mock import patch
import numpy as np
from pipeline.benchmark_baseline import baseline_predictions,score_external,main
from pipeline.view import make_view,resolve_geometry
from test_view import oriented,config

class BaselineBenchmarkTests(unittest.TestCase):
    def fixture(self):
        m=oriented(40);m.name='DFL-MAT-J03WN1';m.P[:]=np.nan
        m.P[:,0,:2]=[.3,.7];m.P[:,24,:2]=[.6,.5];m.B[:,:2]=[.4,.5]
        m.extras['y_true'][:]=1
        return m
    def test_metric_distance_and_view_equivalence(self):
        m=self.fixture();pred=baseline_predictions(m)
        # Metric scaling changes the nearest team versus unscaled normalized distance.
        self.assertTrue((pred['baseline_raw']==0).all())
        v=make_view(m,resolve_geometry(m),config())
        for k,p in pred.items():np.testing.assert_equal(p,baseline_predictions(v)[k])
    def test_labels_never_change_predictions_and_missing_ball_no_vote(self):
        m=self.fixture();a=baseline_predictions(m);m.extras['y_true'][:]=0
        for k,p in a.items():np.testing.assert_equal(p,baseline_predictions(m)[k])
        m.B[:]=np.nan
        for p in baseline_predictions(m).values():self.assertTrue((p==2).all())
    def test_smoothing_never_crosses_half(self):
        m=self.fixture();m.P[20:,0,:2]=[.9,.9]
        p=baseline_predictions(m)['baseline_smooth125']
        self.assertTrue((p[:20]==0).all());self.assertTrue((p[20:]==1).all())
    def test_scorer_separates_adjacent_halves(self):
        m=self.fixture();m.extras['y_true'][:]=0
        r=score_external(m,baseline_predictions(m))['baseline_raw']
        self.assertEqual(r['macro_f1'],1/3)
        self.assertEqual(r['n_gt_spells'],2);self.assertEqual(r['spell_f1_mean'],1)
        self.assertEqual(r['n_graded'],40)
    def test_reserved_guard_precedes_label_access(self):
        m=self.fixture();m.name='DFL-MAT-J03WR9';m.extras={}
        with self.assertRaises(ValueError):score_external(m,{})
        with patch('pipeline.benchmark_baseline.load_match',side_effect=AssertionError('must not load reserved data')):
            with self.assertRaises(SystemExit):main(['--profile','x','--params','x','--matches',m.name,'--out','x'])
