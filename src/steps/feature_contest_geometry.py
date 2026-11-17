"""feature.contest_geometry: contest geometry around stage-1 estimated ball B_hat.

Team-mean pooling cannot represent "who is nearest", so expose it directly:
xf, per frame and team k=h/a:
    close1_k / close2_k: exp(-d/5 m) for nearest / second-nearest player; absent=0 (very far)
    n5_k: number within 5 m / 3, capped at 2
    dpast_k: (d_min[t]-d_min[t-L])/5, clipped +-2; positive means retreat over the past 0.4 s
    dfut_k: (d_min[t]-d_min[t+L])/5, clipped +-2; positive means approach over the next 0.4 s
xp, per player:
    ball_close: exp(-d/5) to B_hat, including unseen ball
    opp_close: exp(-d/3) to the nearest opponent
    approach: velocity toward B_hat / 8 m/s, clipped +-2

Teams use group columns (0 group 1, 1 group 2), not slots. Unusable B_hat (NaN or source 4,
no observed ball for a full half) zeros all columns. Zero dpast/dfut when endpoint team counts
or B_hat sources differ, or the other endpoint lies outside the half.

Team swap with x flip: swap _h/_a, keep xp. Vertical flip: unchanged. Time reversal swaps
dpast/dfut (signs chosen accordingly) and negates approach.
"""

import numpy as np

from . import FeatureStep, register

TAU_BALL, TAU_OPP, R_NEAR, LAG_S = 5.0, 3.0, 5.0, 0.4
SRC_NONE = 4                    # src.nap1.ball: no observed ball in a half makes B_hat=(0,0) meaningless.
CHUNK = 8192                    # Frames per player-pair distance computation.


@register
class ContestGeometry(FeatureStep):
    name = "contest_geometry"
    xf_names = ("close1_h", "close1_a", "close2_h", "close2_a", "n5_h", "n5_a",
                "dpast_h", "dpast_a", "dfut_h", "dfut_a")
    xp_names = ("ball_close", "opp_close", "approach")
    team_pairs_xf = (("close1_h", "close1_a"), ("close2_h", "close2_a"), ("n5_h", "n5_a"),
                     ("dpast_h", "dpast_a"), ("dfut_h", "dfut_a"))
    reverse_pairs_xf = (("dpast_h", "dfut_h"), ("dpast_a", "dfut_a"))
    reverse_negate_xp = ("approach",)

    def compute(self, V, Bhat, raw):
        return contest_geometry(V.P, Bhat, np.asarray(V.meta["half"]), float(V.meta["fps"]))


def _exp_close(d, tau):
    with np.errstate(over="ignore", invalid="ignore"):
        return np.where(np.isfinite(d), np.exp(-d / tau), 0.0)


def contest_geometry(P, Bhat, half, fps):
    T, S = P.shape[:2]
    x, y = P[..., 0], P[..., 1]
    pres = (np.nan_to_num(P[..., 5]) > 0) & np.isfinite(x) & np.isfinite(y) & (half > 0)[:, None]
    grp = np.where(pres, np.round(np.nan_to_num(P[..., 4], nan=-1)), -1)
    teams = (pres & (grp == 0), pres & (grp == 1))
    bx, by = Bhat[:, 0], Bhat[:, 1]
    src = np.round(np.nan_to_num(Bhat[:, 3], nan=SRC_NONE)).astype(int)
    ball_ok = (half > 0) & np.isfinite(bx) & np.isfinite(by) & (src != SRC_NONE)
    x0, y0 = np.nan_to_num(x), np.nan_to_num(y)
    dx, dy = np.nan_to_num(bx)[:, None] - x0, np.nan_to_num(by)[:, None] - y0
    d = np.hypot(dx, dy)
    d_ok = pres & ball_ok[:, None]
    d_ball = np.where(d_ok, d, np.inf)

    xf = np.zeros((T, 10), np.float64)
    L = max(1, int(round(LAG_S * fps)))
    t = np.arange(T)
    for k, m in enumerate(teams):
        dk = np.where(m, d_ball, np.inf)
        two = np.sort(np.partition(dk, 1, axis=1)[:, :2], axis=1) if S >= 2 else np.full((T, 2), np.inf)
        xf[:, 0 + k] = _exp_close(two[:, 0], TAU_BALL)
        xf[:, 2 + k] = _exp_close(two[:, 1], TAU_BALL)
        xf[:, 4 + k] = np.clip((dk < R_NEAR).sum(1) / 3.0, 0, 2)
        dmin, n_k = two[:, 0], m.sum(1)
        for col, other in ((6 + k, t - L), (8 + k, t + L)):
            ok = (other >= 0) & (other < T)
            o = np.clip(other, 0, T - 1)
            ok &= (half[o] == half) & (half > 0) & (n_k[o] == n_k) & (src[o] == src)
            ok &= np.isfinite(dmin) & np.isfinite(dmin[o])
            with np.errstate(invalid="ignore"):
                v = (dmin - dmin[o]) / R_NEAR
            xf[:, col] = np.where(ok, np.clip(v, -2, 2), 0.0)

    xp = np.zeros((T, S, 3), np.float64)
    xp[..., 0] = _exp_close(d_ball, TAU_BALL)
    opp = np.full((T, S), np.inf)
    for a in range(0, T, CHUNK):
        b = min(T, a + CHUNK)
        dd = np.hypot(x0[a:b, :, None] - x0[a:b, None, :], y0[a:b, :, None] - y0[a:b, None, :])
        g = grp[a:b]
        rival = pres[a:b, :, None] & pres[a:b, None, :] & (g[:, :, None] != g[:, None, :])
        opp[a:b] = np.where(rival, dd, np.inf).min(2)
    xp[..., 1] = np.where(pres, _exp_close(opp, TAU_OPP), 0.0)
    vx, vy = np.nan_to_num(P[..., 2]), np.nan_to_num(P[..., 3])
    good = d_ok & (d > 0.1) & np.isfinite(P[..., 2]) & np.isfinite(P[..., 3])
    with np.errstate(invalid="ignore", divide="ignore"):
        toward = (vx * dx + vy * dy) / np.where(good, d, 1.0)
    xp[..., 2] = np.where(good, np.clip(toward / 8.0, -2, 2), 0.0)
    return {"xf": np.nan_to_num(xf).astype(np.float32), "xp": np.nan_to_num(xp).astype(np.float32)}
