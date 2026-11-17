"""feature.contest_geometry: declared team-swap, slot-permutation, vertical-flip and time-reversal
transforms, with no NaNs on degenerate inputs.
"""

from types import SimpleNamespace

import numpy as np

from src import steps
from src.steps.feature_contest_geometry import ContestGeometry, contest_geometry

FPS = 25.0
STEP = ContestGeometry()
XF = {n: i for i, n in enumerate(STEP.xf_names)}
XP = {n: i for i, n in enumerate(STEP.xp_names)}


def make(T=300, seed=0, S=46):
    rng = np.random.default_rng(seed)
    P = np.full((T, S, 6), np.nan)
    walk = np.cumsum(rng.normal(0, 0.3, (T, S, 2)), 0) + rng.uniform(-30, 30, (1, S, 2))
    P[..., :2] = walk
    P[1:, :, 2:4] = np.diff(walk, axis=0) * FPS
    P[0, :, 2:4] = P[1, :, 2:4]
    P[..., 4] = (np.arange(S) >= 24)[None, :]
    P[..., 5] = 1.0
    absent = rng.random((T, S)) < 0.05
    P[absent] = np.nan
    P[absent, 5] = 0.0
    B = np.zeros((T, 5))
    B[:, :2] = np.cumsum(rng.normal(0, 0.5, (T, 2)), 0)
    B[:, 2] = 1.0
    B[:, 3] = rng.choice([0, 1, 3], T)
    half = np.ones(T, int)
    half[:5] = 0
    half[150:] = 2
    return P, B, half


def run(P, B, half):
    return contest_geometry(P, B, half, FPS)


def test_layout_names_and_declarations():
    lay = steps.Layout([{"use": "feature.contest_geometry"}])
    assert lay.xf_names[-10:] == ["contest_geometry.%s" % n for n in STEP.xf_names]
    assert lay.xp_names[-3:] == ["contest_geometry.%s" % n for n in STEP.xp_names]
    assert len(lay.team_pairs_xf) == 4 + 5 and len(lay.reverse_negate_xp) == 3


def test_team_swap_equals_declared_pairs():
    P, B, half = make()
    a = run(P, B, half)
    P2, B2 = P.copy(), B.copy()
    P2[..., 4] = np.where(P[..., 5] > 0, 1 - P[..., 4], P[..., 4])
    for c in (0, 2):                    # Team swap = invert group and negate x, as in nap2.augment.
        P2[..., c] = -P[..., c]
    B2[:, 0] = -B[:, 0]
    b = run(P2, B2, half)
    perm = list(range(10))
    for i, j in STEP.team_pairs_xf:
        perm[XF[i]], perm[XF[j]] = XF[j], XF[i]
    np.testing.assert_allclose(b["xf"], a["xf"][:, perm], atol=1e-5)
    np.testing.assert_allclose(b["xp"], a["xp"], atol=1e-5)


def test_slot_shuffle_within_team_and_vertical_flip():
    P, B, half = make(seed=1)
    a = run(P, B, half)
    perm = np.r_[np.random.default_rng(3).permutation(24), 24 + np.random.default_rng(4).permutation(22)]
    b = run(P[:, perm], B, half)
    np.testing.assert_allclose(b["xf"], a["xf"], atol=1e-5)
    np.testing.assert_allclose(b["xp"], a["xp"][:, perm], atol=1e-5)
    P2, B2 = P.copy(), B.copy()
    P2[..., 1], P2[..., 3], B2[:, 1] = -P[..., 1], -P[..., 3], -B[:, 1]
    c = run(P2, B2, half)
    np.testing.assert_allclose(c["xf"], a["xf"], atol=1e-5)
    np.testing.assert_allclose(c["xp"], a["xp"], atol=1e-5)


def test_time_reversal_swaps_past_future_and_negates_approach():
    P, B, half = make(seed=2)
    a = run(P, B, half)
    P2 = P[::-1].copy()
    P2[..., 2:4] *= -1
    b = run(P2, B[::-1].copy(), half[::-1].copy())
    perm = list(range(10))
    for i, j in STEP.reverse_pairs_xf:
        perm[XF[i]], perm[XF[j]] = XF[j], XF[i]
    np.testing.assert_allclose(b["xf"], a["xf"][::-1][:, perm], atol=1e-5)
    sign = np.array([1, 1, -1])
    np.testing.assert_allclose(b["xp"], a["xp"][::-1] * sign, atol=1e-5)


def test_values_on_a_tiny_scene():
    T = 40
    P = np.full((T, 46, 6), np.nan)
    P[..., 5] = 0
    P[:, 0, :] = [3, 4, 0, 0, 0, 1]          # Group 1: 5 m from the ball.
    P[:, 1, :] = [0, 1, 0, 0, 0, 1]          # Group 1: 1 m from the ball.
    P[:, 30, :] = [-2, 0, 8, 0, 1, 1]        # Group 2: 2 m away, moving directly toward the ball at 8 m/s.
    B = np.zeros((T, 5))
    half = np.ones(T, int)
    o = run(P, B, half)
    assert np.allclose(o["xf"][:, XF["close1_h"]], np.exp(-1 / 5))
    assert np.allclose(o["xf"][:, XF["close2_h"]], np.exp(-5 / 5))
    assert np.allclose(o["xf"][:, XF["close2_a"]], 0)                       # Group 2 has only one player.
    assert np.allclose(o["xf"][:, XF["n5_h"]], 1 / 3)                       # Exactly 5 m is not <5 m.
    assert np.allclose(o["xp"][:, 30, XP["approach"]], 1.0)
    assert np.allclose(o["xp"][:, 1, XP["opp_close"]], np.exp(-np.hypot(2, 1) / 3))
    assert np.allclose(o["xp"][:, 5], 0)                                    # Absent slots are zero.
    assert np.allclose(o["xf"][:, 6:], 0)                                   # Stationary: no distance change.


def test_bad_inputs_stay_finite_and_zero():
    P, B, half = make(seed=5)
    for case in ("all_nan", "one_team", "src4", "nan_vel", "half0"):
        P2, B2, h2 = P.copy(), B.copy(), half.copy()
        if case == "all_nan":
            P2[:] = np.nan
            B2[:, :2] = np.nan
        elif case == "one_team":
            P2[:, 24:, 5] = 0
        elif case == "src4":
            B2[:, 3] = 4
        elif case == "nan_vel":
            P2[..., 2:4] = np.nan
        elif case == "half0":
            h2[:] = 0
        o = run(P2, B2, h2)
        assert np.isfinite(o["xf"]).all() and np.isfinite(o["xp"]).all(), case
        if case in ("all_nan", "src4", "half0"):
            assert not o["xf"].any() and not o["xp"][..., [0, 2]].any(), case
        if case == "one_team":
            assert not o["xf"][:, 1::2].any() and not o["xp"][..., XP["opp_close"]].any()
        if case == "nan_vel":
            assert not o["xp"][..., XP["approach"]].any()
    assert not run(P, B, half)["xf"][:5].any()                             # Outside halves: zero.


def test_track_moving_to_another_slot_after_one_empty_frame():
    """One player changes slots after an empty frame: team features remain continuous; dpast is zero exactly where counts change."""
    T = 60
    P = np.full((T, 46, 6), np.nan)
    P[..., 5] = 0
    xs = np.linspace(-10, -1, T)
    P[:, 2, :] = [20, 0, 0, 0, 0, 1]
    for slot, fr in ((3, slice(0, 30)), (7, slice(31, T))):
        P[fr, slot, :5] = 0
        P[fr, slot, 0] = xs[fr]
        P[fr, slot, 5] = 1
    B = np.zeros((T, 5))
    o = run(P, B, np.ones(T, int))
    L = 10
    dp = o["xf"][:, XF["dpast_h"]]
    assert dp[30] == 0                                                      # Empty frame changes the player count.
    np.testing.assert_allclose(dp[45], (abs(xs[45]) - abs(xs[45 - L])) / 5, atol=1e-5)   # Continue across two slots.
    assert np.isclose(dp[35], (abs(xs[35]) - abs(xs[25])) / 5, atol=1e-5)
