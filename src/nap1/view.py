"""Block 1: View, normalizing match data into a common working coordinate system.

  1. Centered metres: x in [-L/2,L/2], y in [-W/2,W/2].
  2. Group 1 (home) ALWAYS attacks +x in both halves. If it attacks -x, negate both axes
     for the ball, players and camera center.
  3. Segment contiguous non-NaN tracks, additionally splitting jumps > max_step_m.
  4. Velocities vx/vy: Savitzky-Golay on sufficiently long tracks, with centered/one-sided
     finite differences as fallback. Tracks >=2 frames have finite velocities; singletons may be NaN.
  5. Camera center: inverse-project through homography H or read extras["cam"].
  6. Adapters: load a Match directly or wrap a pipeline.feed View.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from pipeline.schema import Match, N_SLOTS, N_TEAM1
from .params import Params, ensure_params


@dataclass
class View:
    """Normalized copy of a match for all downstream computations."""
    P: np.ndarray             # [T, 46, 6] float64: x, y, vx, vy, group, present
    B: np.ndarray             # [T, 3] float64: x, y, present
    cam: np.ndarray           # [T, 2] float64: x_cam, y_cam (m)
    meta: Dict[str, Any]      # Metadata (fps, pitch, half, graded, flipped_halves, etc.).

    @property
    def n_frames(self) -> int:
        return self.P.shape[0]

    @property
    def pitch(self) -> Tuple[float, float]:
        return tuple(self.meta["pitch"])

    @property
    def fps(self) -> int:
        return int(self.meta.get("fps", 25))

    @property
    def half(self) -> np.ndarray:
        return self.meta["half"]

    @property
    def graded(self) -> Optional[np.ndarray]:
        return self.meta.get("graded")


def _savgol_deriv1_kernel(window: int, fps: int) -> np.ndarray:
    """Quadratic Savitzky-Golay first-derivative convolution kernel, equivalent to scipy savgol deriv=1."""
    k = window // 2
    t = np.arange(-k, k + 1, dtype=np.float64)
    denom = np.sum(t ** 2)
    return (t / denom) * fps


def _diff_1d_velocities(x: np.ndarray, fps: int) -> np.ndarray:
    """1D velocities for a contiguous [N] array, using finite differences:
    N>=2: centered interior differences and one-sided endpoints. N==1: NaN.
    """
    n = len(x)
    v = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return v
    if n == 2:
        val = (x[1] - x[0]) * fps
        v[0] = val
        v[1] = val
        return v

    # Use one-sided differences at both endpoints.
    v[0] = (x[1] - x[0]) * fps
    v[-1] = (x[-1] - x[-2]) * fps
    # Use centered differences at interior points.
    v[1:-1] = (x[2:] - x[:-2]) * (fps / 2.0)
    return v


def _compute_velocities(px: np.ndarray, py: np.ndarray, half: np.ndarray,
                        window: int, max_step_m: float, fps: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute vx/vy for all player slots:
    Length >= window: Savitzky-Golay interior, finite-difference edges.
    2 <= length < window: centered / one-sided differences. Singletons: NaN.
    """
    T, n_slots = px.shape
    vx = np.full((T, n_slots), np.nan, dtype=np.float64)
    vy = np.full((T, n_slots), np.nan, dtype=np.float64)

    k = window // 2
    kernel = _savgol_deriv1_kernel(window, fps)
    kernel_flip = kernel[::-1]

    for j in range(n_slots):
        valid = ~np.isnan(px[:, j]) & ~np.isnan(py[:, j])
        if not np.any(valid):
            continue

        # Find contiguous blocks with available data.
        diff = np.diff(valid.astype(np.int8))
        starts = np.where(diff == 1)[0] + 1
        if valid[0]:
            starts = np.r_[0, starts]
        ends = np.where(diff == -1)[0] + 1
        if valid[-1]:
            ends = np.r_[ends, T]

        for s, e in zip(starts, ends):
            # Never compute velocities across half boundaries.
            half_changes = np.where(np.diff(half[s:e]) != 0)[0] + 1
            h_starts = np.r_[0, half_changes]
            h_ends = np.r_[half_changes, e - s]

            for hs, he in zip(h_starts, h_ends):
                seg_s = s + hs
                seg_e = s + he
                seg_len = seg_e - seg_s
                if seg_len < 2:
                    continue  # A single frame has NaN velocity.

                x_seg = px[seg_s:seg_e, j]
                y_seg = py[seg_s:seg_e, j]

                # Split tracks when adjacent displacement exceeds max_step_m.
                dist_steps = np.hypot(np.diff(x_seg), np.diff(y_seg))
                jump_pts = np.where(dist_steps > max_step_m)[0] + 1
                sub_starts = np.r_[0, jump_pts]
                sub_ends = np.r_[jump_pts, seg_len]

                for ss, ee in zip(sub_starts, sub_ends):
                    sub_len = ee - ss
                    if sub_len < 2:
                        continue

                    sub_idx_s = seg_s + ss
                    sub_idx_e = seg_s + ee
                    xs = x_seg[ss:ee]
                    ys = y_seg[ss:ee]

                    # 1. Start with finite differences over the whole segment, including boundaries.
                    vx_sub = _diff_1d_velocities(xs, fps)
                    vy_sub = _diff_1d_velocities(ys, fps)

                    # 2. If segment length >= window, overwrite the interior with Savitzky-Golay.
                    if sub_len >= window:
                        sg_vx = np.convolve(xs, kernel_flip, mode="valid")
                        sg_vy = np.convolve(ys, kernel_flip, mode="valid")
                        vx_sub[k : sub_len - k] = sg_vx
                        vy_sub[k : sub_len - k] = sg_vy

                    vx[sub_idx_s:sub_idx_e, j] = vx_sub
                    vy[sub_idx_s:sub_idx_e, j] = vy_sub

    return vx, vy


def _extract_camera_centre(match: Match, pitch_dims: Tuple[float, float]) -> np.ndarray:
    """Inverse-project the image center through homography, or read extras['cam']."""
    T = match.n_frames
    L, W = pitch_dims

    if "cam" in match.extras:
        cam_n = np.asarray(match.extras["cam"], dtype=np.float64)
        cx = cam_n[:, 0] * L - L / 2.0
        cy = cam_n[:, 1] * W - W / 2.0
        return np.stack([cx, cy], axis=1)

    cam = np.full((T, 2), np.nan, dtype=np.float64)
    H = match.extras.get("H")
    valid = match.extras.get("H_valid")

    if H is not None and valid is not None:
        valid_mask = valid.astype(bool) & ~np.isnan(H[:, 0, 0])
        if np.any(valid_mask):
            try:
                Hi = np.linalg.inv(H[valid_mask])
                n_valid = int(np.sum(valid_mask))
                uv1 = np.empty((n_valid, 3), dtype=np.float64)
                uv1[:, 0] = 960.0
                uv1[:, 1] = 540.0
                uv1[:, 2] = 1.0

                p = np.einsum("nij,nj->ni", Hi, uv1)
                cx_norm = p[:, 0] / p[:, 2]
                cy_norm = p[:, 1] / p[:, 2]

                cx_m = cx_norm * L - L / 2.0
                cy_m = cy_norm * W - W / 2.0

                cam[valid_mask, 0] = cx_m
                cam[valid_mask, 1] = cy_m
            except Exception:
                pass

    return cam


def create_view(match: Match, params: Optional[Union[Params, dict]] = None) -> View:
    """Convert a raw Match to a View with normalized coordinates and velocities."""
    p = ensure_params(params)
    L, W = match.pitch
    T = match.n_frames
    fps = int(match.meta.get("fps", 25))

    px = match.P[:, :, 0] * L - L / 2.0
    py = match.P[:, :, 1] * W - W / 2.0
    p_present = (~np.isnan(px) & ~np.isnan(py)).astype(np.float64)

    bx = match.B[:, 0] * L - L / 2.0
    by = match.B[:, 1] * W - W / 2.0
    b_present = (~np.isnan(bx) & ~np.isnan(by)).astype(np.float64)

    cam = _extract_camera_centre(match, (L, W))

    # Normalize attack direction: group 1 always attacks +x.
    flipped_halves = []
    for h in sorted(set(match.half) - {0}):
        mask_h = (match.half == h)
        if not np.any(mask_h):
            continue

        needs_flip = False
        if match.source == "dfl" and "left_group" in match.meta:
            lg_dict = match.meta["left_group"]
            h_key = "firstHalf" if h == 1 else "secondHalf"
            lg = lg_dict.get(h_key)
            if lg == 2:
                needs_flip = True
        else:
            # Use group-1 per-frame median x to infer attack direction.
            p1_x = px[mask_h, :N_TEAM1]
            p2_x = px[mask_h, N_TEAM1:N_SLOTS]
            med_p1 = np.nanmedian(p1_x)
            med_p2 = np.nanmedian(p2_x)
            if med_p1 > med_p2:
                needs_flip = True

        if needs_flip:
            flipped_halves.append(int(h))
            px[mask_h] = -px[mask_h]
            py[mask_h] = -py[mask_h]
            bx[mask_h] = -bx[mask_h]
            by[mask_h] = -by[mask_h]
            cam[mask_h] = -cam[mask_h]

    # Compute vx, vy after attack-direction normalization.
    vx, vy = _compute_velocities(px, py, match.half, p.savgol_window,
                                 p.max_step_m, fps)

    group_ids = np.zeros(N_SLOTS, dtype=np.float64)
    group_ids[N_TEAM1:] = 1.0

    P_view = np.empty((T, N_SLOTS, 6), dtype=np.float64)
    P_view[:, :, 0] = px
    P_view[:, :, 1] = py
    P_view[:, :, 2] = vx
    P_view[:, :, 3] = vy
    P_view[:, :, 4] = group_ids[None, :]
    P_view[:, :, 5] = p_present

    B_view = np.empty((T, 3), dtype=np.float64)
    B_view[:, 0] = bx
    B_view[:, 1] = by
    B_view[:, 2] = b_present

    view_meta = dict(match.meta)
    view_meta["fps"] = fps
    view_meta["pitch"] = (float(L), float(W))
    view_meta["half"] = match.half
    view_meta["graded"] = match.graded
    view_meta["flipped_halves"] = flipped_halves
    view_meta["source"] = match.source
    view_meta["name"] = match.name

    return View(P=P_view, B=B_view, cam=cam, meta=view_meta)


# Bidirectional compatibility alias
make_view = create_view


def _fill_missing_velocities(P: np.ndarray, half: np.ndarray,
                             max_step_m: float, fps: int) -> np.ndarray:
    """Fill remaining pipeline.view NaN velocities at boundaries or short tracks (2 <= length < window)."""
    P_out = np.array(P, dtype=np.float64, copy=True)
    T, n_slots, _ = P_out.shape
    half_arr = np.asarray(half)

    for j in range(n_slots):
        valid = (P_out[:, j, 5] == 1.0) & np.isfinite(P_out[:, j, 0]) & np.isfinite(P_out[:, j, 1])
        if not np.any(valid):
            continue

        diff = np.diff(valid.astype(np.int8))
        starts = np.where(diff == 1)[0] + 1
        if valid[0]:
            starts = np.r_[0, starts]
        ends = np.where(diff == -1)[0] + 1
        if valid[-1]:
            ends = np.r_[ends, T]

        for s, e in zip(starts, ends):
            half_changes = np.where(np.diff(half_arr[s:e]) != 0)[0] + 1
            h_starts = np.r_[0, half_changes]
            h_ends = np.r_[half_changes, e - s]

            for hs, he in zip(h_starts, h_ends):
                seg_s = s + hs
                seg_e = s + he
                seg_len = seg_e - seg_s
                if seg_len < 2:
                    continue

                x_seg = P_out[seg_s:seg_e, j, 0]
                y_seg = P_out[seg_s:seg_e, j, 1]
                dist_steps = np.hypot(np.diff(x_seg), np.diff(y_seg))
                jump_pts = np.where(dist_steps > max_step_m)[0] + 1
                sub_starts = np.r_[0, jump_pts]
                sub_ends = np.r_[jump_pts, seg_len]

                for ss, ee in zip(sub_starts, sub_ends):
                    if ee - ss < 2:
                        continue
                    idx_s, idx_e = seg_s + ss, seg_s + ee
                    nan_vx = np.isnan(P_out[idx_s:idx_e, j, 2])
                    nan_vy = np.isnan(P_out[idx_s:idx_e, j, 3])
                    if np.any(nan_vx):
                        vx_fb = _diff_1d_velocities(x_seg[ss:ee], fps)
                        P_out[idx_s:idx_e, j, 2] = np.where(nan_vx, vx_fb, P_out[idx_s:idx_e, j, 2])
                    if np.any(nan_vy):
                        vy_fb = _diff_1d_velocities(y_seg[ss:ee], fps)
                        P_out[idx_s:idx_e, j, 3] = np.where(nan_vy, vy_fb, P_out[idx_s:idx_e, j, 3])
    return P_out


def from_feed(feed_view: Any, params: Optional[Union[Params, dict]] = None) -> View:
    """Adapt a pipeline.feed View (pipeline/view.py) to nap1 View.

    Preserve geometry, attack direction and Savitzky-Golay velocities, only filling remaining
    boundary / short-track NaNs with finite differences for tracks of at least two frames.
    """
    if isinstance(feed_view, View):
        return feed_view
    p = ensure_params(params)
    meta = dict(getattr(feed_view, "meta", {}))
    fps = int(meta.get("fps", 25))
    meta.setdefault("fps", fps)
    meta.setdefault("pitch", (105.0, 68.0))
    half = getattr(feed_view, "half", None)
    if half is not None:
        meta["half"] = half
    graded = getattr(feed_view, "graded", None)
    if graded is not None:
        meta["graded"] = graded
    if "flipped_halves" not in meta and "attack_sign" in meta:
        meta["flipped_halves"] = sorted(
            int(h) for h, s in meta["attack_sign"].items() if int(s) == -1
        )

    P_filled = _fill_missing_velocities(
        feed_view.P,
        half if half is not None else np.ones(feed_view.P.shape[0], dtype=np.int8),
        p.max_step_m,
        fps,
    )
    return View(
        P=P_filled,
        B=np.asarray(feed_view.B, dtype=np.float64),
        cam=np.asarray(feed_view.cam, dtype=np.float64),
        meta=meta,
    )



def from_npz(z: Any, params: Optional[Union[Params, dict]] = None) -> View:
    """Load an exported train_data/ .npz or competition View directly into nap1 View."""
    p = ensure_params(params)
    files = z.files if hasattr(z, "files") else z
    fps = int(np.asarray(z["fps"]).item()) if "fps" in files else 25
    pitch_arr = np.asarray(z["pitch"], dtype=np.float64) if "pitch" in files else np.array([105.0, 68.0])
    half = np.asarray(z["half"])
    graded = np.asarray(z["graded"], dtype=bool) if "graded" in files else (half > 0)
    P_raw = np.asarray(z["P"], dtype=np.float64)
    P_filled = _fill_missing_velocities(P_raw, half, p.max_step_m, fps)
    meta = {
        "fps": fps,
        "pitch": (float(pitch_arr[0]), float(pitch_arr[1])),
        "half": half,
        "graded": graded,
    }
    return View(
        P=P_filled,
        B=np.asarray(z["B"], dtype=np.float64),
        cam=np.asarray(z["cam"], dtype=np.float64),
        meta=meta,
    )
