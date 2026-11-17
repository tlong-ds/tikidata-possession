"""src.core.ball_smooth: change only visible coordinates, deterministic by (match, epoch, run), eliminating stationary repeats."""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d

from src.core import ball_smooth as bs

FPS = 25.0


def make(T=3000, seed=0):
    rng = np.random.default_rng(seed)
    xy = np.zeros((T, 2))
    t = 0
    while t < T:                      # Stationary ball followed by a kick: DFL-like repeated positions and a sharp turn.
        n = int(rng.integers(10, 60))
        v = rng.normal(0, 8, 2) if rng.random() < 0.5 else np.zeros(2)
        xy[t:t + n] = xy[t - 1] + np.outer(np.arange(1, min(n, T - t) + 1), v / FPS) if t else 0
        t += n
    half = np.ones(T, int)
    half[T // 2:] = 2
    half[:3] = 0
    B = np.c_[xy, (rng.random(T) > 0.4).astype(float)]
    B[B[:, 2] == 0, :2] = np.nan
    return {"B": B, "ball_xy": xy, "half": half, "fps": FPS}


def test_gauss_matches_scipy():
    x = np.random.default_rng(1).normal(size=(200, 2))
    np.testing.assert_allclose(bs.gauss(x, 3.0), gaussian_filter1d(x, 3.0, axis=0, mode="nearest"), atol=1e-6)


def test_check_cfg():
    with pytest.raises(ValueError):
        bs.check_cfg({"sigma": 0.2})
    with pytest.raises(ValueError):
        bs.check_cfg({"p": 1.5})
    with pytest.raises(ValueError):
        bs.check_cfg({"sig": [0.3, 0.1]})
    assert bs.check_cfg({})["p"] == bs.DEFAULT["p"]


def test_only_seen_changed_and_deterministic():
    raw = make()
    cfg = {"p": 1.0, "sig": 0.24, "noise_m": 0.4, "noise_s": 0.24}
    a, ia = bs.maybe_smooth(raw, cfg, "M", 3, 1)
    b, ib = bs.maybe_smooth(raw, cfg, "M", 3, 1)
    c, _ = bs.maybe_smooth(raw, cfg, "M", 4, 1)
    assert ia == ib and ia is not None
    np.testing.assert_array_equal(a["B"], b["B"])
    assert not np.array_equal(np.nan_to_num(a["B"]), np.nan_to_num(c["B"]))
    seen = raw["B"][:, 2] > 0
    np.testing.assert_array_equal(np.isnan(a["B"][:, 0]), np.isnan(raw["B"][:, 0]))
    np.testing.assert_array_equal(a["B"][:, 2], raw["B"][:, 2])
    assert np.isfinite(a["B"][seen, :2]).all()
    assert raw["B"] is not a["B"] and np.isnan(raw["B"][~seen, 0]).all()      # Do not mutate the input.


def test_no_repeats_and_softer_turns():
    raw = make(seed=2)
    a, _ = bs.maybe_smooth(raw, {"p": 1.0, "sig": 0.24, "noise_m": 0.4, "noise_s": 0.24}, "M", 0, 0)

    def rep_acc(B):
        x, s = np.nan_to_num(B[:, :2]), B[:, 2] > 0
        pair = s[1:] & s[:-1]
        step = np.hypot(*np.diff(x, axis=0).T)[pair]
        tri = pair[1:] & pair[:-1]
        acc = np.hypot(*(x[2:] - 2 * x[1:-1] + x[:-2]).T)[tri] * FPS ** 2
        return (step < 1e-9).mean(), np.quantile(acc, 0.99)

    r0, a0 = rep_acc(raw["B"])
    r1, a1 = rep_acc(a["B"])
    assert r0 > 0.2 and r1 == 0
    assert a1 < a0 / 3


def test_p_zero_and_missing_truth():
    raw = make()
    out, info = bs.maybe_smooth(raw, {"p": 0.0}, "M", 0, 0)
    assert out is raw and info is None
    r2 = {k: v for k, v in raw.items() if k != "ball_xy"}
    out, info = bs.maybe_smooth(r2, {"p": 1.0}, "M", 0, 0)
    assert out is r2 and info is None
    assert bs.maybe_smooth(raw, None, "M", 0, 0) == (raw, None)
