"""Block 3b: dead-ball detector.

Motivation: 66% of organiser-baseline errors concern dead-ball D; D occupies 45% of Bundesliga
time. Stoppage cues: low ball speed vb, few players within 5 m (n5), large nearest-player
distance dm (median 4.5 m dead vs 1.3 m live), and low mean player speed sl.
An ablation adds soft P(out) from pitch dimensions and ball uncertainty sigma_B.

Logistic regression:
  logit = w_dead_b + w_vb * vb + w_n5 * n5 + w_dm * dm + w_sl * sl + w_out * p_out
  D = 1 / (1 + exp(-logit))
"""

from typing import Any, Optional, Union

import numpy as np
from scipy.special import erf

from .params import Params, ensure_params


def _soft_out_of_bounds_prob(bx: np.ndarray, by: np.ndarray, sig_b: np.ndarray,
                             pitch_dims: tuple) -> np.ndarray:
    """Soft ball out-of-bounds probability P(out) from coordinates and uncertainty sigma_B."""
    L, W = pitch_dims
    sig = np.maximum(0.1, sig_b)
    # Gaussian CDF: 0.5 * (1 + erf(x / sqrt(2))).
    sqrt2 = np.sqrt(2.0)
    p_in_x = 0.5 * (erf((L / 2.0 - bx) / (sig * sqrt2)) - erf((-L / 2.0 - bx) / (sig * sqrt2)))
    p_in_y = 0.5 * (erf((W / 2.0 - by) / (sig * sqrt2)) - erf((-W / 2.0 - by) / (sig * sqrt2)))
    p_out = 1.0 - np.clip(p_in_x * p_in_y, 0.0, 1.0)
    return p_out


def dead_prob(V: Any, Bhat: np.ndarray, params: Optional[Union[Params, dict]] = None) -> np.ndarray:
    """Per-frame dead-ball probability D[T] in [0,1]."""
    p = ensure_params(params)
    T = V.n_frames if hasattr(V, "n_frames") else V.P.shape[0]
    fps = V.fps if hasattr(V, "fps") else int(V.meta.get("fps", 25))
    half = V.half if hasattr(V, "half") else V.meta["half"]

    D = np.ones(T, dtype=np.float64)  # Default to dead ball outside halves.

    in_game = (half == 1) | (half == 2)
    if not np.any(in_game):
        return D

    # 1. Ball speed vb (m/s)
    b_diff = np.diff(Bhat[:, :2], axis=0)
    vb = np.r_[0.0, np.hypot(b_diff[:, 0], b_diff[:, 1])] * fps

    # 2. Distances from the ball to all present players
    dx = V.P[:, :, 0] - Bhat[:, 0, None]
    dy = V.P[:, :, 1] - Bhat[:, 1, None]
    d2 = dx * dx + dy * dy
    present = (V.P[:, :, 5] == 1.0)
    d2[~present] = np.inf
    d_all = np.sqrt(d2)

    # 3. Player count within 5 m of the ball (n5)
    n5 = np.sum((d_all <= 5.0) & present, axis=1)

    # 4. Distance to the nearest player (dm)
    dm = np.nanmin(d_all, axis=1)
    dm[np.isinf(dm)] = 10.0  # Use 10 m when no players are present.

    # 5. Mean on-pitch player speed (sl)
    player_speed = np.hypot(V.P[:, :, 2], V.P[:, :, 3])
    player_speed[~present] = np.nan
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        sl = np.nanmean(player_speed, axis=1)
    sl = np.nan_to_num(sl, nan=1.5)

    # 6. Base logistic regression
    logit = (p.w_dead_b +
             p.w_dead_vb * vb +
             p.w_dead_n5 * n5 +
             p.w_dead_dm * dm +
             p.w_dead_sl * sl)

    # 7. Soft out-of-bounds probability, enabled when w_dead_out != 0
    if getattr(p, "w_dead_out", 0.0) != 0.0:
        pitch = getattr(V, "pitch", None)
        if pitch is None and hasattr(V, "meta"):
            pitch = V.meta.get("pitch", (105.0, 68.0))
        p_out = _soft_out_of_bounds_prob(Bhat[:, 0], Bhat[:, 1], Bhat[:, 2], pitch or (105.0, 68.0))
        logit += p.w_dead_out * p_out

    d_in_game = 1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0)))
    D[in_game] = d_in_game[in_game]

    return D
