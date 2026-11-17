"""Stage-2 network inputs and targets (plan section 3, features and shapes).

Use the same build_features() in training and inference:
    per player xp [T,46,9]: x, y, vx, vy, team, presence, dx, dy, ball distance
    per frame  xf [T,7]: B_hat x/y, ball observed, B_hat sigma, camera center x/y, pan speed
Fixed NORM scaling keeps values around [-1,1]. Empty slots are all zero with presence=0.
Compute dx/dy/distance to B_hat when sigma <= rel_sigma_max (default 0.5 m, observed ball only,
matching the "zero if unseen" plan); increase the threshold to include short interpolations.

Exclude ball height (absent in ATD), player roles/IDs and attack/defense status (label leakage).
"""

import numpy as np

NX, NY, NV, ND, NS = 52.5, 34.0, 8.0, 50.0, 20.0
N_PLAYER_FEAT = 9
N_FRAME_FEAT = 7
F_GROUP, F_PRESENT = 4, 5          # xp team and presence column indices used by model.py pooling.

FEAT_DEFAULTS = {"rel_sigma_max": 0.5, "pan_smooth": 13}

# Codes match evaluation/possession_metrics: 0=H, 1=A, 2=D, 3=X.
TOUCH_SIGMA_S = 0.25               # Blur touch targets with Gaussian sigma 0.25 s (~+-0.5 s at two sigma).
TOUCH_TOL = 12                     # Frames (0.48 s): touching-team assignment and near-change window.


def build_features(V, Bhat, cfg=None):
    c = dict(FEAT_DEFAULTS, **(cfg or {}))
    T, S = V.P.shape[:2]
    pres = V.P[..., 5] > 0
    x = np.where(pres, V.P[..., 0], 0.0)
    y = np.where(pres, V.P[..., 1], 0.0)
    vx = np.where(pres & np.isfinite(V.P[..., 2]), V.P[..., 2], 0.0)
    vy = np.where(pres & np.isfinite(V.P[..., 3]), V.P[..., 3], 0.0)
    rel = (Bhat[:, 2] <= c["rel_sigma_max"])[:, None] & pres
    dx = np.where(rel, x - Bhat[:, 0:1], 0.0)
    dy = np.where(rel, y - Bhat[:, 1:2], 0.0)
    dist = np.where(rel, np.hypot(dx, dy), 0.0)

    xp = np.empty((T, S, N_PLAYER_FEAT), dtype=np.float32)
    xp[..., 0] = x / NX
    xp[..., 1] = y / NY
    xp[..., 2] = np.clip(vx / NV, -2, 2)
    xp[..., 3] = np.clip(vy / NV, -2, 2)
    xp[..., 4] = np.where(pres, V.P[..., 4], 0.0)
    xp[..., 5] = pres
    xp[..., 6] = np.clip(dx / NX, -2, 2)
    xp[..., 7] = np.clip(dy / NY, -2, 2)
    xp[..., 8] = np.clip(dist / ND, 0, 3)

    cx, cy = V.cam[:, 0], V.cam[:, 1]
    has_cam = np.isfinite(cx) & np.isfinite(cy)
    fps = V.meta["fps"]
    pan = np.zeros(T)
    pan[1:] = np.hypot(np.diff(np.nan_to_num(cx)), np.diff(np.nan_to_num(cy))) * fps
    pan[1:][~(has_cam[1:] & has_cam[:-1])] = 0.0
    w = int(c["pan_smooth"])
    if w > 1:
        pan = np.convolve(pan, np.ones(w) / w, mode="same")
    xf = np.empty((T, N_FRAME_FEAT), dtype=np.float32)
    xf[:, 0] = np.clip(Bhat[:, 0] / NX, -1.5, 1.5)
    xf[:, 1] = np.clip(Bhat[:, 1] / NY, -1.5, 1.5)
    xf[:, 2] = V.B[:, 2] > 0
    xf[:, 3] = np.clip(Bhat[:, 2] / NS, 0, 3)
    xf[:, 4] = np.where(has_cam, np.clip(cx / NX, -1.5, 1.5), 0.0)
    xf[:, 5] = np.where(has_cam, np.clip(cy / NY, -1.5, 1.5), 0.0)
    xf[:, 6] = np.clip(pan / 10.0, 0, 3)
    return {"xp": xp, "xf": xf}


def _blur_touch(ind, fps):
    sig = TOUCH_SIGMA_S * fps
    k = np.arange(-int(3 * sig), int(3 * sig) + 1)
    ker = np.exp(-0.5 * (k / sig) ** 2)
    return np.minimum(np.convolve(ind.astype(np.float64), ker, mode="same"), 1.0)


def _nearest_event_team(touch_team_at, T, tol):
    """Team (0/1) of the nearest unambiguous touch within +-tol frames; -1 if absent."""
    idx = np.arange(T)
    has = touch_team_at >= 0
    prev = np.maximum.accumulate(np.where(has, idx, -10 ** 9))
    nxt = np.minimum.accumulate(np.where(has, idx, 10 ** 9)[::-1])[::-1]
    dp, dn = idx - prev, nxt - idx
    use_prev = dp <= dn
    near = np.where(use_prev, prev, nxt)
    dist = np.minimum(dp, dn)
    out = np.full(T, -1, dtype=np.int8)
    ok = dist <= tol
    out[ok] = touch_team_at[near[ok]]
    return out


def targets_from_arrays(y, touch_true, ball_xy, half, fps, w=None):
    """Frame targets from existing shared-data arrays (pipeline.feed).

    y [T] 0/1/2 (-1 excluded); touch_true [T] 0 / teams 1,2 / both 3; ball_xy [T,2] true ball
    in View metres (NaN if absent); w [T] optional frame weights, e.g. stage-3 pseudo-labels.
    Return y, blurred touch, team (-1 absent), scaled ball, ball_ok, near and w.
    """
    half = np.asarray(half)
    T = len(half)
    y = np.asarray(y).astype(np.int8).copy()
    y[(half <= 0) | (y < 0) | (y > 2)] = -1

    tt = np.zeros(T, dtype=np.int8) if touch_true is None else np.asarray(touch_true, dtype=np.int8)
    touch = _blur_touch(tt > 0, fps).astype(np.float32)
    team_at = np.full(T, -1, dtype=np.int8)
    team_at[tt == 1] = 0
    team_at[tt == 2] = 1                       # 3 = both teams touch in one frame: ambiguous team.
    team = _nearest_event_team(team_at, T, TOUCH_TOL)

    bxy = np.asarray(ball_xy, dtype=np.float64)
    ball_ok = np.isfinite(bxy).all(1)
    ball = np.zeros((T, 2), dtype=np.float32)
    ball[ball_ok, 0] = bxy[ball_ok, 0] / NX
    ball[ball_ok, 1] = bxy[ball_ok, 1] / NY

    change = np.zeros(T, bool)
    change[1:] = (y[1:] != y[:-1]) & (half[1:] == half[:-1])
    near = np.convolve(change.astype(float), np.ones(2 * TOUCH_TOL + 1), mode="same") > 0

    w = np.ones(T, dtype=np.float32) if w is None else np.asarray(w, dtype=np.float32).copy()
    w[y < 0] = 0.0
    return {"y": y, "touch": touch, "team": team, "ball": ball, "ball_ok": ball_ok, "near": near, "w": w}
