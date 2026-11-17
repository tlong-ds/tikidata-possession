"""feature.ball_touch: observed-ball (V.B) touch evidence and the likely touching team.

Motivation (analysis/half_spells.md): many missed possessions are brief opposite-team touches
between same-team possessions, such as failed interceptions or rebounds. Kalman-smoothed B_hat
removes sharp turns; measure them directly on visible observations:

    v_in = (b[t]-b[t-L])/L, v_out = (b[t+L]-b[t])/L; lag L defaults to 0.2 s
    kin_ok: ball visible at t-L, t, t+L within one half; otherwise all columns are zero
    kick: |v_out-v_in| / 10 m/s, capped at 2
    turn_cos: velocity-angle cosine if both speeds >1 m/s; -1 for reversal, otherwise 0
    speed_in / speed_out: speed / 20 m/s, capped at 2
    touch_k: kick * exp(-d_k/1.5 m), with d_k the nearest team-k player's observed-ball distance
    touchw_k: max touch_k within +-0.4 s, allowing ball/player timing offsets

The target ball is smoothed, without repeated coordinates and with turns spread over ~0.4 s;
DFL repeats 36-49% of frames and turns sharply. At lag 0.2 s, target kick>0.5 peaks occur
4.7 times/visible-ball minute versus DFL 17-36; lag 0.4 s matches rates (18.9 vs 18-31).
Use lag=0.4 for submission.

Team swap (with x flip): swap touch_h/a and touchw_h/a; other features unchanged. Vertical
flip: unchanged. Time reversal: v_in <-> -v_out, hence swap speed_in/out; other features unchanged.
"""

import numpy as np

from . import FeatureStep, register

LAG_S, WIN_S = 0.2, 0.4
V_KICK, V_SPEED, MIN_V, TAU_TOUCH = 10.0, 20.0, 1.0, 1.5


@register
class BallTouch(FeatureStep):
    name = "ball_touch"
    xf_names = ("kin_ok", "kick", "turn_cos", "speed_in", "speed_out",
                "touch_h", "touch_a", "touchw_h", "touchw_a")
    team_pairs_xf = (("touch_h", "touch_a"), ("touchw_h", "touchw_a"))
    reverse_pairs_xf = (("speed_in", "speed_out"),)
    defaults = {"lag": LAG_S}

    def compute(self, V, Bhat, raw):
        return {"xf": ball_touch(V.P, V.B, np.asarray(V.meta["half"]), float(V.meta["fps"]), float(self.lag))}


def _window_max(a, w, half):
    """Maximum of a in [t-w,t+w], without crossing half boundaries."""
    out = a.copy()
    T = len(a)
    for s in range(1, w + 1):
        for sh in (s, -s):
            o = np.clip(np.arange(T) + sh, 0, T - 1)
            ok = (np.arange(T) + sh >= 0) & (np.arange(T) + sh < T) & (half[o] == half)
            out = np.where(ok, np.maximum(out, a[o]), out)
    return out


def ball_touch(P, B, half, fps, lag=None):
    T = len(B)
    L = max(1, int(round((LAG_S if lag is None else lag) * fps)))
    bx, by = B[:, 0], B[:, 1]
    seen = (half > 0) & np.isfinite(bx) & np.isfinite(by) & (np.nan_to_num(B[:, 2]) > 0)
    t = np.arange(T)
    p, n = np.clip(t - L, 0, T - 1), np.clip(t + L, 0, T - 1)
    ok = seen & seen[p] & seen[n] & (t - L >= 0) & (t + L < T) & (half[p] == half) & (half[n] == half)
    x0, y0 = np.nan_to_num(bx), np.nan_to_num(by)
    dt = L / fps
    vin = np.stack([x0 - x0[p], y0 - y0[p]], 1) / dt
    vout = np.stack([x0[n] - x0, y0[n] - y0], 1) / dt
    sin, sout = np.hypot(*vin.T), np.hypot(*vout.T)
    kick = np.clip(np.hypot(*(vout - vin).T) / V_KICK, 0, 2)
    both = ok & (sin > MIN_V) & (sout > MIN_V)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.where(both, (vin * vout).sum(1) / (sin * sout), 0.0)

    xf = np.zeros((T, 9), np.float64)
    xf[:, 0] = ok
    xf[:, 1] = np.where(ok, kick, 0)
    xf[:, 2] = np.clip(np.nan_to_num(cos), -1, 1)
    xf[:, 3] = np.where(ok, np.clip(sin / V_SPEED, 0, 2), 0)
    xf[:, 4] = np.where(ok, np.clip(sout / V_SPEED, 0, 2), 0)

    pres = (np.nan_to_num(P[..., 5]) > 0) & np.isfinite(P[..., 0]) & np.isfinite(P[..., 1])
    grp = np.where(pres, np.round(np.nan_to_num(P[..., 4], nan=-1)), -1)
    d = np.hypot(np.nan_to_num(P[..., 0]) - x0[:, None], np.nan_to_num(P[..., 1]) - y0[:, None])
    w = max(1, int(round(WIN_S * fps)))
    for k in (0, 1):
        dk = np.where(pres & (grp == k), d, np.inf).min(1)
        with np.errstate(over="ignore"):
            near = np.where(np.isfinite(dk), np.exp(-dk / TAU_TOUCH), 0.0)
        touch = np.where(ok, xf[:, 1] * near, 0.0)
        xf[:, 5 + k] = touch
        xf[:, 7 + k] = _window_max(touch, w, half)
    xf[half <= 0] = 0
    return np.nan_to_num(xf).astype(np.float32)
