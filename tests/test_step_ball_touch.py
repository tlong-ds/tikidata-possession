"""feature.ball_touch: declared transforms, values on a small scene and no NaNs on malformed inputs."""

import numpy as np

from src import steps
from src.steps.feature_ball_touch import BallTouch, ball_touch

FPS = 25.0
STEP = BallTouch()
XF = {n: i for i, n in enumerate(STEP.xf_names)}


def make(T=300, seed=0, S=46):
    rng = np.random.default_rng(seed)
    P = np.full((T, S, 6), np.nan)
    walk = np.cumsum(rng.normal(0, 0.3, (T, S, 2)), 0) + rng.uniform(-30, 30, (1, S, 2))
    P[..., :2] = walk
    P[..., 4] = (np.arange(S) >= 24)[None, :]
    P[..., 5] = 1.0
    absent = rng.random((T, S)) < 0.05
    P[absent] = np.nan
    P[absent, 5] = 0.0
    B = np.zeros((T, 3))
    B[:, :2] = np.cumsum(rng.normal(0, 0.8, (T, 2)), 0)
    B[:, 2] = (rng.random(T) > 0.2).astype(float)
    B[B[:, 2] == 0, :2] = np.nan
    half = np.ones(T, int)
    half[:5] = 0
    half[150:] = 2
    return P, B, half


def run(P, B, half):
    return ball_touch(P, B, half, FPS)


def perm_of(pairs):
    perm = list(range(len(STEP.xf_names)))
    for i, j in pairs:
        perm[XF[i]], perm[XF[j]] = XF[j], XF[i]
    return perm


def test_layout_names_and_declarations():
    lay = steps.Layout([{"use": "feature.ball_touch"}])
    assert lay.xf_names[-9:] == ["ball_touch.%s" % n for n in STEP.xf_names]


def test_team_swap_slot_shuffle_vertical_flip():
    P, B, half = make()
    a = run(P, B, half)
    P2, B2 = P.copy(), B.copy()
    P2[..., 4] = np.where(P[..., 5] > 0, 1 - P[..., 4], P[..., 4])
    P2[..., 0], B2[:, 0] = -P[..., 0], -B[:, 0]
    np.testing.assert_allclose(run(P2, B2, half), a[:, perm_of(STEP.team_pairs_xf)], atol=1e-5)
    perm = np.r_[np.random.default_rng(3).permutation(24), 24 + np.random.default_rng(4).permutation(22)]
    np.testing.assert_allclose(run(P[:, perm], B, half), a, atol=1e-5)
    P3, B3 = P.copy(), B.copy()
    P3[..., 1], B3[:, 1] = -P[..., 1], -B[:, 1]
    np.testing.assert_allclose(run(P3, B3, half), a, atol=1e-5)


def test_time_reversal_swaps_speed_in_out():
    P, B, half = make(seed=2)
    a = run(P, B, half)
    b = run(P[::-1].copy(), B[::-1].copy(), half[::-1].copy())
    np.testing.assert_allclose(b, a[::-1][:, perm_of(STEP.reverse_pairs_xf)], atol=1e-5)


def test_bounce_off_one_team():
    T = 60
    P = np.full((T, 46, 6), np.nan)
    P[..., 5] = 0
    P[:, 0, :] = [10.0, 0.5, 0, 0, 0, 1]      # Group 1 is at the ball's turn point.
    P[:, 30, :] = [10.0, 20.0, 0, 0, 1, 1]    # Group 2 is 20 m away.
    B = np.zeros((T, 3))
    t = np.arange(T)
    B[:, 0] = np.where(t < 30, t * 10 / 25.0 - 2, 10 - (t - 30) * 10 / 25.0)   # Move at 10 m/s toward x=10, then reverse.
    B[:, 2] = 1
    half = np.ones(T, int)
    o = run(P, B, half)
    i = 30
    assert o[i, XF["kin_ok"]] == 1
    assert np.isclose(o[i, XF["kick"]], 2.0)                  # |(-10) - 10| / 10 = 2
    assert np.isclose(o[i, XF["turn_cos"]], -1.0)
    assert np.isclose(o[i, XF["speed_in"]], 0.5) and np.isclose(o[i, XF["speed_out"]], 0.5)
    assert o[i, XF["touch_h"]] > 0.5 and o[i, XF["touch_a"]] < 1e-4
    assert np.allclose(o[15, XF["kick"]], 0, atol=1e-5)       # Straight flight: no turn.
    assert o[i + 8, XF["touchw_h"]] == o[i, XF["touch_h"]]    # +-0.4 s window = +-10 frames.
    assert o[i + 11, XF["touchw_h"]] < o[i, XF["touch_h"]]


def test_bad_inputs_stay_finite():
    P, B, half = make(seed=5)
    for case in ("no_ball", "no_players", "half0"):
        P2, B2, h2 = P.copy(), B.copy(), half.copy()
        if case == "no_ball":
            B2[:, :2] = np.nan
            B2[:, 2] = 0
        elif case == "no_players":
            P2[:] = np.nan
            P2[..., 5] = 0
        else:
            h2[:] = 0
        o = run(P2, B2, h2)
        assert np.isfinite(o).all()
        if case != "no_players":
            assert np.allclose(o, 0)
        else:
            assert np.allclose(o[:, 5:], 0)
