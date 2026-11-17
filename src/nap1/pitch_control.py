"""Block 3a: pitch control, estimating which team controls the space at the ball.

Nearest-player ownership is only ~77% accurate. Pitch control uses arrival times:
  * Player arrival = reaction time + distance / maximum speed.
  * Evaluate synchronously at t by default (use_velocity=False), avoiding a mismatch between
    future player positions at t+react and the current ball at t.
  * Take each team's earliest arrival (t0 group 1, t1 group 2).
  * Convert t1-t0 to p1 with a sigmoid.
  * Shrink p1 toward neutral 0.5 when ball uncertainty sigma is large.
"""

from typing import Any, Optional, Union

import numpy as np

from pipeline.schema import N_TEAM1
from .params import Params, ensure_params


def pitch_control(V: Any, Bhat: np.ndarray, params: Optional[Union[Params, dict]] = None) -> np.ndarray:
    """Probability p1[T] in [0,1] that group 1 (home) controls the ball location."""
    p = ensure_params(params)
    T = V.n_frames if hasattr(V, "n_frames") else V.P.shape[0]
    half = V.half if hasattr(V, "half") else V.meta["half"]
    p1 = np.full(T, 0.5, dtype=np.float64)

    # Compute only within the two regular halves.
    in_game = (half == 1) | (half == 2)
    if not np.any(in_game):
        return p1

    react = p.react_time
    vmax = max(0.1, p.v_max)
    sigma_pc = max(0.01, p.sigma_pc)

    # 1. Predict player position after reaction time if use_velocity is enabled.
    px = V.P[:, :, 0]
    py = V.P[:, :, 1]
    vx = V.P[:, :, 2]
    vy = V.P[:, :, 3]
    present = (V.P[:, :, 5] == 1.0)

    if p.use_velocity:
        has_v = ~np.isnan(vx) & ~np.isnan(vy)
        ex = np.where(has_v, px + vx * react, px)
        ey = np.where(has_v, py + vy * react, py)
    else:
        ex = px
        ey = py

    # 2. Distance from player positions to the estimated ball position.
    bx = Bhat[:, 0, None]
    by = Bhat[:, 1, None]
    dist = np.hypot(ex - bx, ey - by)
    dist[~present] = np.inf

    # 3. Arrival time t_arr
    t_arr = react + dist / vmax

    # 4. Earliest arrival per team; safe when one or both teams have no present players.
    with np.errstate(invalid="ignore"):
        t0 = np.nanmin(t_arr[:, :N_TEAM1], axis=1)
        t1 = np.nanmin(t_arr[:, N_TEAM1:], axis=1)
        diff_t = t1 - t0

    diff_t = np.nan_to_num(diff_t, nan=0.0, posinf=30.0, neginf=-30.0)

    # 5. Sigmoid of time difference: t1 - t0 > 0 means group 1 arrives first.
    scale = (np.pi / np.sqrt(3.0)) / sigma_pc
    p1_raw = 1.0 / (1.0 + np.exp(-np.clip(scale * diff_t, -30.0, 30.0)))

    # 6. Adjust for ball uncertainty sigma_B:
    # Long gaps (large sigma) shrink p1 toward neutral 0.5.
    sigma_b = Bhat[:, 2]
    cert = np.exp(-np.maximum(0.0, sigma_b - p.sigma_m) / 8.0)
    p1_adjusted = 0.5 + (p1_raw - 0.5) * cert

    p1[in_game] = p1_adjusted[in_game]
    return p1
