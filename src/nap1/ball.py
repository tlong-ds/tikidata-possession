"""Block 2: ball estimation and touch detection.

  1. Kalman/RTS estimation of ball position (x,y) and velocity (vx,vy).
  2. Classify measurements using Mahalanobis distance d²:
     - d² <= tau: normal; update ball position.
     - d² > tau with a player within r: TOUCH (K=1); reset velocity and increase Q.
     - d² > tau without a nearby player: OUTLIER; reject measurement and retain inertial prediction.
     - Two consecutive deviations without a player: reacquire from measurement to avoid drift.
  3. Backward RTS smoothing uses future observations with exact 2x2 covariance.
  4. Long gaps support ball_gap_mode:
     - 'rts_cov' (selected on select): direct smoothed 2x2 covariance P_{t|T}.
     - '2d_bounded': min(t_prev,t_next) with two endpoints, one-sided at half boundaries.
     - '1d_forward': time since last observation.
     Fuse camera center V.cam according to uncertainty.
  5. Return K[T], softened with a +-2-frame kernel.
"""

from typing import Any, Optional, Tuple, Union

import numpy as np

from .params import Params, ensure_params


def _savgol_touch_kernel(win: int = 2) -> np.ndarray:
    """Triangular kernel softening touch probability around a detected touch frame."""
    k = np.arange(-win, win + 1)
    kernel = 1.0 - np.abs(k) / (win + 1.0)
    return kernel / np.max(kernel)  # Peak = 1.0.


def _kf_predict_1d(x: float, v: float, p11: float, p12: float, p22: float,
                   dt: float, q11: float, q12: float, q22: float) -> Tuple[float, float, float, float, float]:
    """Predict 1D state and covariance for one time step dt."""
    nx = x + dt * v
    nv = v
    np11 = p11 + 2.0 * dt * p12 + dt * dt * p22 + q11
    np12 = p12 + dt * p22 + q12
    np22 = p22 + q22
    return nx, nv, np11, np12, np22


def _kf_update_1d(xp: float, vp: float, p11: float, p12: float, p22: float,
                  z: float, R: float) -> Tuple[float, float, float, float, float, float, float]:
    """Update state and covariance with measurement z."""
    S = p11 + R
    K0 = p11 / S
    K1 = p12 / S
    nu = z - xp

    x = xp + K0 * nu
    v = vp + K1 * nu
    np11 = (1.0 - K0) * p11
    np12 = (1.0 - K0) * p12
    np22 = p22 - K1 * p12
    return x, v, np11, np12, np22, nu, S


def _smooth_rts_1d(x_f: np.ndarray, v_f: np.ndarray,
                   p11_f: np.ndarray, p12_f: np.ndarray, p22_f: np.ndarray,
                   x_p: np.ndarray, v_p: np.ndarray,
                   p11_p: np.ndarray, p12_p: np.ndarray, p22_p: np.ndarray,
                   dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward Rauch-Tung-Striebel (RTS) smoothing with exact 2x2 covariance.

    Update the full matrix P_{t|T} = P_{t|t} + G_t (P_{t+1|T} - P_{t+1|t}) G_t^T, preserving
    positive-definite gap uncertainty (a Brownian bridge between observed endpoints).
    """
    n = len(x_f)
    xs = np.empty(n, dtype=np.float64)
    vs = np.empty(n, dtype=np.float64)
    s11 = np.empty(n, dtype=np.float64)

    xs[-1] = x_f[-1]
    vs[-1] = v_f[-1]
    c11 = p11_f[-1]
    c12 = p12_f[-1]
    c22 = p22_f[-1]
    s11[-1] = max(1e-6, c11)

    for t in range(n - 2, -1, -1):
        m00 = p11_f[t] + dt * p12_f[t]
        m01 = p12_f[t]
        m10 = p12_f[t] + dt * p22_f[t]
        m11 = p22_f[t]

        A = p11_p[t + 1]
        B = p12_p[t + 1]
        C = p22_p[t + 1]
        det = A * C - B * B
        if abs(det) < 1e-12:
            det = 1e-12
        inv_det = 1.0 / det

        g00 = (m00 * C - m01 * B) * inv_det
        g01 = (-m00 * B + m01 * A) * inv_det
        g10 = (m10 * C - m11 * B) * inv_det
        g11 = (-m10 * B + m11 * A) * inv_det

        dx = xs[t + 1] - x_p[t + 1]
        dv = vs[t + 1] - v_p[t + 1]

        xs[t] = x_f[t] + g00 * dx + g01 * dv
        vs[t] = v_f[t] + g10 * dx + g11 * dv

        # Full 2x2 covariance update: P_s[t] = P_f[t] + G * (P_s[t+1] - P_p[t+1]) * G^T.
        d11 = c11 - p11_p[t + 1]
        d12 = c12 - p12_p[t + 1]
        d22 = c22 - p22_p[t + 1]

        gd00 = g00 * d11 + g01 * d12
        gd01 = g00 * d12 + g01 * d22
        gd10 = g10 * d11 + g11 * d12
        gd11 = g10 * d12 + g11 * d22

        c11 = max(1e-6, p11_f[t] + gd00 * g00 + gd01 * g01)
        c12 = p12_f[t] + gd00 * g10 + gd01 * g11
        c22 = max(1e-6, p22_f[t] + gd10 * g10 + gd11 * g11)
        s11[t] = c11

    return xs, vs, np.sqrt(s11)


def _compute_min_player_dist(z_x: np.ndarray, z_y: np.ndarray, ball_present: np.ndarray,
                             player_x: np.ndarray, player_y: np.ndarray,
                             player_present: np.ndarray) -> np.ndarray:
    """Minimum ball-to-present-player distance on frames with visible ball."""
    n = len(z_x)
    min_dist = np.full(n, np.inf, dtype=np.float64)
    b_idx = np.where(ball_present)[0]
    if len(b_idx) > 0:
        with np.errstate(invalid="ignore"):
            dx = player_x[b_idx] - z_x[b_idx, None]
            dy = player_y[b_idx] - z_y[b_idx, None]
            dist = np.where(player_present[b_idx], np.hypot(dx, dy), np.inf)
        min_dist[b_idx] = np.min(dist, axis=1)
    return min_dist


def _run_pass1_half(z_x: np.ndarray, z_y: np.ndarray, ball_present: np.ndarray,
                    min_dist_to_player: np.ndarray, params: Params,
                    fps: int) -> Tuple[np.ndarray, int]:
    """Pass-1 forward Kalman measurement flags: 0 normal, 1 touch, 2 outlier, 3 missing, 4 near touch, 5 reacquired."""
    n = len(z_x)
    dt = 1.0 / fps
    q_acc = params.qa
    q11 = (dt ** 4 / 4.0) * (q_acc ** 2)
    q12 = (dt ** 3 / 2.0) * (q_acc ** 2)
    q22 = (dt ** 2) * (q_acc ** 2)
    R = params.sigma_m * params.sigma_m

    near_player = min_dist_to_player <= params.touch_radius
    flag = np.zeros(n, dtype=np.int8)

    seen_indices = np.where(ball_present)[0]
    if len(seen_indices) == 0:
        flag[:] = 3
        return flag, -1

    first_seen = int(seen_indices[0])
    flag[:first_seen] = 3

    sx, vx = float(z_x[first_seen]), 0.0
    sy, vy = float(z_y[first_seen]), 0.0
    px11, px12, px22 = R, 0.0, 25.0
    py11, py12, py22 = R, 0.0, 25.0

    last_touch = -999
    consec_outliers = 0
    tau = params.tau
    min_gap = params.min_touch_gap

    for t in range(first_seen, n):
        if not ball_present[t]:
            flag[t] = 3
            sx, vx, px11, px12, px22 = _kf_predict_1d(sx, vx, px11, px12, px22, dt, q11, q12, q22)
            sy, vy, py11, py12, py22 = _kf_predict_1d(sy, vy, py11, py12, py22, dt, q11, q12, q22)
            continue

        xp, nxv, npx11, npx12, npx22 = _kf_predict_1d(sx, vx, px11, px12, px22, dt, q11, q12, q22)
        yp, nyv, npy11, npy12, npy22 = _kf_predict_1d(sy, vy, py11, py12, py22, dt, q11, q12, q22)

        Sx = npx11 + R
        Sy = npy11 + R
        zx = float(z_x[t])
        zy = float(z_y[t])
        nu_x = zx - xp
        nu_y = zy - yp
        d2 = (nu_x * nu_x) / Sx + (nu_y * nu_y) / Sy
        near = bool(near_player[t])

        if d2 > tau:
            if near and (t - last_touch >= min_gap):
                flag[t] = 1
                last_touch = t
                consec_outliers = 0
                Kx0 = npx11 / Sx
                Ky0 = npy11 / Sy
                sx = xp + Kx0 * nu_x
                vx = (zx - sx) / dt
                sy = yp + Ky0 * nu_y
                vy = (zy - sy) / dt
                px11, px12, px22 = npx11 * (1.0 - Kx0), 0.0, 60.0
                py11, py12, py22 = npy11 * (1.0 - Ky0), 0.0, 60.0
                continue
            elif not near:
                if consec_outliers >= 2:
                    flag[t] = 5
                    consec_outliers = 0
                    Kx0 = npx11 / Sx
                    Ky0 = npy11 / Sy
                    sx = xp + Kx0 * nu_x
                    vx = (zx - sx) / (3.0 * dt)
                    sy = yp + Ky0 * nu_y
                    vy = (zy - sy) / (3.0 * dt)
                    px11, px12, px22 = npx11 * (1.0 - Kx0), 0.0, 60.0
                    py11, py12, py22 = npy11 * (1.0 - Ky0), 0.0, 60.0
                    continue
                else:
                    flag[t] = 2
                    consec_outliers += 1
                    sx, vx, px11, px12, px22 = xp, nxv, npx11, npx12, npx22
                    sy, vy, py11, py12, py22 = yp, nyv, npy11, npy12, npy22
                    continue
            else:
                flag[t] = 4

        consec_outliers = 0
        sx, vx, px11, px12, px22, _, _ = _kf_update_1d(xp, nxv, npx11, npx12, npx22, zx, R)
        sy, vy, py11, py12, py22, _, _ = _kf_update_1d(yp, nyv, npy11, npy12, npy22, zy, R)

    return flag, first_seen


def _process_half(z_x: np.ndarray, z_y: np.ndarray, ball_present: np.ndarray,
                  player_x: np.ndarray, player_y: np.ndarray, player_present: np.ndarray,
                  cam_xy: np.ndarray, params: Params, fps: int) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate the ball and detect touches for one half."""
    n = len(z_x)
    dt = 1.0 / fps

    q_acc = params.qa
    q11 = (dt ** 4 / 4.0) * (q_acc ** 2)
    q12 = (dt ** 3 / 2.0) * (q_acc ** 2)
    q22 = (dt ** 2) * (q_acc ** 2)
    R = params.sigma_m * params.sigma_m

    min_dist_to_player = _compute_min_player_dist(
        z_x, z_y, ball_present, player_x, player_y, player_present
    )
    flag, first_seen = _run_pass1_half(z_x, z_y, ball_present, min_dist_to_player, params, fps)

    if first_seen < 0:
        B_empty = np.zeros((n, 5), dtype=np.float64)
        B_empty[:, 2] = 20.0
        B_empty[:, 3] = 4.0
        B_empty[:, 4] = 0.05
        return B_empty, np.zeros(n, dtype=np.float64)

    # --- PASS 2: Record forward filtering and apply bidirectional RTS smoothing ---
    x_p = np.empty(n, dtype=np.float64)
    v_p = np.empty(n, dtype=np.float64)
    px11_p = np.empty(n, dtype=np.float64)
    px12_p = np.empty(n, dtype=np.float64)
    px22_p = np.empty(n, dtype=np.float64)

    y_p = np.empty(n, dtype=np.float64)
    v_py = np.empty(n, dtype=np.float64)
    py11_p = np.empty(n, dtype=np.float64)
    py12_p = np.empty(n, dtype=np.float64)
    py22_p = np.empty(n, dtype=np.float64)

    x_f = np.empty(n, dtype=np.float64)
    v_f = np.empty(n, dtype=np.float64)
    px11_f = np.empty(n, dtype=np.float64)
    px12_f = np.empty(n, dtype=np.float64)
    px22_f = np.empty(n, dtype=np.float64)

    y_f = np.empty(n, dtype=np.float64)
    v_fy = np.empty(n, dtype=np.float64)
    py11_f = np.empty(n, dtype=np.float64)
    py12_f = np.empty(n, dtype=np.float64)
    py22_f = np.empty(n, dtype=np.float64)

    sx, vx = float(z_x[first_seen]), 0.0
    sy, vy = float(z_y[first_seen]), 0.0
    px11, px12, px22 = R, 0.0, 25.0
    py11, py12, py22 = R, 0.0, 25.0

    for t in range(first_seen):
        x_f[t], v_f[t] = sx, vx
        px11_f[t], px12_f[t], px22_f[t] = px11, px12, px22
        y_f[t], v_fy[t] = sy, vy
        py11_f[t], py12_f[t], py22_f[t] = py11, py12, py22

        x_p[t], v_p[t] = sx, vx
        px11_p[t], px12_p[t], px22_p[t] = px11, px12, px22
        y_p[t], v_py[t] = sy, vy
        py11_p[t], py12_p[t], py22_p[t] = py11, py12, py22

    for t in range(first_seen, n):
        boost = 4000.0 if flag[t] in (1, 5) else 1.0
        cur_q11 = q11 * boost
        cur_q12 = q12 * boost
        cur_q22 = q22 * boost

        xp, nxv, npx11, npx12, npx22 = _kf_predict_1d(sx, vx, px11, px12, px22, dt, cur_q11, cur_q12, cur_q22)
        yp, nyv, npy11, npy12, npy22 = _kf_predict_1d(sy, vy, py11, py12, py22, dt, cur_q11, cur_q12, cur_q22)

        x_p[t], v_p[t] = xp, nxv
        px11_p[t], px12_p[t], px22_p[t] = npx11, npx12, npx22
        y_p[t], v_py[t] = yp, nyv
        py11_p[t], py12_p[t], py22_p[t] = npy11, npy12, npy22

        if flag[t] in (2, 3):
            sx, vx, px11, px12, px22 = xp, nxv, npx11, npx12, npx22
            sy, vy, py11, py12, py22 = yp, nyv, npy11, npy12, npy22
        else:
            sx, vx, px11, px12, px22, _, _ = _kf_update_1d(xp, nxv, npx11, npx12, npx22, float(z_x[t]), R)
            sy, vy, py11, py12, py22, _, _ = _kf_update_1d(yp, nyv, npy11, npy12, npy22, float(z_y[t]), R)

        x_f[t], v_f[t] = sx, vx
        px11_f[t], px12_f[t], px22_f[t] = px11, px12, px22
        y_f[t], v_fy[t] = sy, vy
        py11_f[t], py12_f[t], py22_f[t] = py11, py12, py22

    xs, _, sig_sx = _smooth_rts_1d(x_f, v_f, px11_f, px12_f, px22_f,
                                   x_p, v_p, px11_p, px12_p, px22_p, dt)
    ys, _, sig_sy = _smooth_rts_1d(y_f, v_fy, py11_f, py12_f, py22_f,
                                   y_p, v_py, py11_p, py12_p, py22_p, dt)
    sig_k = np.sqrt(sig_sx * sig_sx + sig_sy * sig_sy)

    # --- PASS 3: Long ball gaps and camera-center fusion ---
    B_out = np.empty((n, 5), dtype=np.float64)
    B_out[:, 0] = xs
    B_out[:, 1] = ys
    B_out[:, 2] = sig_k
    B_out[:, 3] = 1.0

    obs_mask = (flag == 0) | (flag == 1) | (flag == 4)
    B_out[obs_mask, 2] = params.sigma_m
    B_out[obs_mask, 3] = 0.0

    # Compute ball uncertainty according to ball_gap_mode:
    gap_mode = getattr(params, "ball_gap_mode", "rts_cov")
    if gap_mode == "rts_cov":
        cur_sig_k_arr = np.maximum(params.sigma_m, sig_k)
    elif gap_mode == "1d_forward":
        gap_frames = np.zeros(n, dtype=np.int32)
        cur_gap = 0
        for t in range(n):
            cur_gap = 0 if obs_mask[t] else cur_gap + 1
            gap_frames[t] = cur_gap
        t_sec = gap_frames * dt
        cur_sig_k_arr = np.sqrt(params.sigma_m ** 2 + (params.v_sigma * t_sec) ** 2)
    else:  # "2d_bounded"
        idx = np.arange(n)
        prev_idx = np.maximum.accumulate(np.where(obs_mask, idx, -1))
        next_idx = np.minimum.accumulate(np.where(obs_mask, idx, n)[::-1])[::-1]
        has_p = prev_idx >= 0
        has_n = next_idx < n
        both = has_p & has_n
        only_p = has_p & ~has_n
        only_n = ~has_p & has_n
        neither = ~has_p & ~has_n

        t_sec = np.full(n, 30.0, dtype=np.float64)
        t_sec[both] = np.minimum(idx[both] - prev_idx[both], next_idx[both] - idx[both]) * dt
        t_sec[only_p] = (idx[only_p] - prev_idx[only_p]) * dt
        t_sec[only_n] = (next_idx[only_n] - idx[only_n]) * dt
        cur_sig_k_arr = np.sqrt(params.sigma_m ** 2 + (params.v_sigma * t_sec) ** 2)
        cur_sig_k_arr[neither] = params.sigma_cam * 2.0

    sig_cam = params.sigma_cam
    sig_cam2 = sig_cam * sig_cam
    cam_valid = ~np.isnan(cam_xy[:, 0]) & ~np.isnan(cam_xy[:, 1])

    gap_mask = ~obs_mask
    gap_no_cam = gap_mask & ~cam_valid
    B_out[gap_no_cam, 2] = cur_sig_k_arr[gap_no_cam]
    B_out[gap_no_cam, 3] = 1.0

    gap_cam = gap_mask & cam_valid
    if np.any(gap_cam):
        sig_k2 = cur_sig_k_arr[gap_cam] ** 2
        w_k = sig_cam2 / (sig_k2 + sig_cam2)
        w_cam = 1.0 - w_k
        B_out[gap_cam, 0] = w_k * xs[gap_cam] + w_cam * cam_xy[gap_cam, 0]
        B_out[gap_cam, 1] = w_k * ys[gap_cam] + w_cam * cam_xy[gap_cam, 1]
        B_out[gap_cam, 2] = 1.0 / np.sqrt(1.0 / sig_k2 + 1.0 / sig_cam2)
        B_out[gap_cam, 3] = np.where(w_cam > w_k, 3.0, 1.0)

    B_out[:, 4] = 1.0 / (1.0 + B_out[:, 2])

    # --- PASS 4: Touch probability K[T] ---
    K_raw = (flag == 1).astype(np.float64)
    kernel = _savgol_touch_kernel(params.touch_smooth_win)
    K_smooth = np.convolve(K_raw, kernel, mode="same")
    K_out = np.clip(K_smooth, 0.0, 1.0)

    return B_out, K_out


def detect_touches_raw(V: Any, params: Optional[Union[Params, dict]] = None) -> np.ndarray:
    """Fast pass 1 over a full match to obtain raw touch-frame flags (flag == 1)."""
    p = ensure_params(params)
    T = V.n_frames if hasattr(V, "n_frames") else V.P.shape[0]
    fps = V.fps if hasattr(V, "fps") else int(V.meta.get("fps", 25))
    half = V.half if hasattr(V, "half") else V.meta["half"]
    touches = np.zeros(T, dtype=bool)

    for h in sorted(set(half) - {0}):
        h_mask = (half == h)
        if not np.any(h_mask):
            continue
        h_idx = np.where(h_mask)[0]
        z_x = V.B[h_idx, 0]
        z_y = V.B[h_idx, 1]
        ball_present = (V.B[h_idx, 2] == 1.0)
        min_d = _compute_min_player_dist(
            z_x, z_y, ball_present,
            V.P[h_idx, :, 0], V.P[h_idx, :, 1], (V.P[h_idx, :, 5] == 1.0),
        )
        flag, _ = _run_pass1_half(z_x, z_y, ball_present, min_d, p, fps)
        touches[h_idx] = (flag == 1)
    return touches


def estimate_ball(V: Any, params: Optional[Union[Params, dict]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate ball trajectory Bhat and touch probability K for the full match."""
    p = ensure_params(params)
    T = V.n_frames if hasattr(V, "n_frames") else V.P.shape[0]
    fps = V.fps if hasattr(V, "fps") else int(V.meta.get("fps", 25))
    half = V.half if hasattr(V, "half") else V.meta["half"]

    Bhat = np.zeros((T, 5), dtype=np.float64)
    Bhat[:, 2] = 20.0
    Bhat[:, 3] = 1.0
    Bhat[:, 4] = 0.05

    K = np.zeros(T, dtype=np.float64)

    for h in sorted(set(half) - {0}):
        h_mask = (half == h)
        if not np.any(h_mask):
            continue

        h_idx = np.where(h_mask)[0]
        z_x = V.B[h_idx, 0]
        z_y = V.B[h_idx, 1]
        ball_present = (V.B[h_idx, 2] == 1.0)

        player_x = V.P[h_idx, :, 0]
        player_y = V.P[h_idx, :, 1]
        player_present = (V.P[h_idx, :, 5] == 1.0)

        cam_xy = V.cam[h_idx]

        b_half, k_half = _process_half(z_x, z_y, ball_present,
                                       player_x, player_y, player_present,
                                       cam_xy, p, fps)

        Bhat[h_idx] = b_half
        K[h_idx] = k_half

    return Bhat, K
