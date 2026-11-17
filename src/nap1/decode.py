"""Block 4: two-level sequence decoding (four-state IOHMM + touch gate).

  1. Hidden states: 0 H (home live ownership), 1 A (away live ownership),
     2 D_H (dead after home), 3 D_A (dead after away).
  2. Touch gate permits H<->A changes when the opposite team has touch evidence within
     +-touch_window of t. Without a touch, only the small eps_gate background remains.
  3. Dead-ball exit ablation:
     - deadball_switch_gate=False (default): ungated exit with a 70/30 prior.
     - deadball_switch_gate=True: exit through the touch gate.
  4. Standard outputs: int8 codes 0 H, 1 A, 2 D, 3 X, matching possession_metrics / nap2.
     to_submission_df exports the challenge submission format.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .params import Params, ensure_params

H, A, D, X = 0, 1, 2, 3
NEG = -1e18


def _compute_touch_evidence(K: np.ndarray, C: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Per-team touch evidence within +-window frames."""
    T = len(K)
    raw_touch_0 = np.clip(np.asarray(K, dtype=np.float64) * np.asarray(C[:, 0], dtype=np.float64), 0.0, 1.0)
    raw_touch_1 = np.clip(np.asarray(K, dtype=np.float64) * np.asarray(C[:, 1], dtype=np.float64), 0.0, 1.0)

    kopp_0 = raw_touch_0.copy()
    kopp_1 = raw_touch_1.copy()

    for w in range(1, window + 1):
        kopp_0[w:] = np.maximum(kopp_0[w:], raw_touch_0[:-w])
        kopp_0[:-w] = np.maximum(kopp_0[:-w], raw_touch_0[w:])
        kopp_1[w:] = np.maximum(kopp_1[w:], raw_touch_1[:-w])
        kopp_1[:-w] = np.maximum(kopp_1[:-w], raw_touch_1[w:])

    return kopp_0, kopp_1


def _safe_log(val: float) -> float:
    return math.log(val) if val > 1e-12 else NEG


def _safe_log_arr(arr: np.ndarray) -> np.ndarray:
    out = np.full(arr.shape, NEG, dtype=np.float64)
    mask = arr > 1e-12
    out[mask] = np.log(arr[mask])
    return out


def _viterbi_half(E_half: np.ndarray, kopp_0: np.ndarray, kopp_1: np.ndarray,
                  p: Params, fps: int) -> np.ndarray:
    """Four-state Viterbi with log-transition terms vectorized before the DP loop."""
    n = len(E_half)
    if n == 0:
        return np.empty(0, dtype=np.int8)

    pa = max(0.0, 1.0 - 1.0 / (p.L_alive * fps))
    pd = max(0.0, 1.0 - 1.0 / (p.L_dead * fps))
    sd = p.share_dead
    ps = p.p_switch_after_dead
    eps_g = p.eps_gate
    lam = p.lambda_emit
    use_gate_on_dead_exit = getattr(p, "deadball_switch_gate", True)

    sw_a = 1.0 - pa
    sw_d = 1.0 - pd

    # Dead-ball exit probability
    p_d_keep = sw_d * (1.0 - ps)
    p_d_switch = sw_d * ps
    l_d_keep = _safe_log(p_d_keep)
    l_d_switch = _safe_log(p_d_switch)
    l_dd = _safe_log(pd)
    l_to_dead = _safe_log(sw_a * sd)
    sw_live = sw_a * (1.0 - sd)
    sw_dead_stay = 1.0 - sw_a * sd

    g0_arr = eps_g + (1.0 - eps_g) * kopp_0
    g1_arr = eps_g + (1.0 - eps_g) * kopp_1

    p_HA_arr = sw_live * g1_arr
    p_AH_arr = sw_live * g0_arr
    sHH = _safe_log_arr(sw_dead_stay - p_HA_arr).tolist()
    sAA = _safe_log_arr(sw_dead_stay - p_AH_arr).tolist()
    sHA = _safe_log_arr(p_HA_arr).tolist()
    sAH = _safe_log_arr(p_AH_arr).tolist()

    if use_gate_on_dead_exit:
        l_d_to_H_from_DH = _safe_log_arr(p_d_keep * g0_arr).tolist()
        l_d_to_H_from_DA = _safe_log_arr(p_d_switch * g0_arr).tolist()
        l_d_to_A_from_DH = _safe_log_arr(p_d_switch * g1_arr).tolist()
        l_d_to_A_from_DA = _safe_log_arr(p_d_keep * g1_arr).tolist()
    else:
        l_d_to_H_from_DH = [l_d_keep] * n
        l_d_to_H_from_DA = [l_d_switch] * n
        l_d_to_A_from_DH = [l_d_switch] * n
        l_d_to_A_from_DA = [l_d_keep] * n

    eH = (lam * E_half[:, 0]).tolist()
    eA = (lam * E_half[:, 1]).tolist()
    eD = (lam * E_half[:, 2]).tolist()

    bp = np.empty((n, 4), dtype=np.int8)
    dH, dA, dDH, dDA = eH[0], eA[0], eD[0], eD[0]

    for t in range(1, n):
        # Enter H from H, A, DH or DA.
        c0 = dH + sHH[t]
        c1 = dA + sAH[t]
        c2 = dDH + l_d_to_H_from_DH[t]
        c3 = dDA + l_d_to_H_from_DA[t]
        kH, mH = 0, c0
        if c1 > mH:
            kH, mH = 1, c1
        if c2 > mH:
            kH, mH = 2, c2
        if c3 > mH:
            kH, mH = 3, c3
        nH = mH + eH[t]

        # Enter A from H, A, DH or DA.
        c0 = dH + sHA[t]
        c1 = dA + sAA[t]
        c2 = dDH + l_d_to_A_from_DH[t]
        c3 = dDA + l_d_to_A_from_DA[t]
        kA, mA = 0, c0
        if c1 > mA:
            kA, mA = 1, c1
        if c2 > mA:
            kA, mA = 2, c2
        if c3 > mA:
            kA, mA = 3, c3
        nA = mA + eA[t]

        # Enter DH from H, or remain in DH.
        if dH + l_to_dead >= dDH + l_dd:
            kDH, nDH = 0, dH + l_to_dead + eD[t]
        else:
            kDH, nDH = 2, dDH + l_dd + eD[t]

        # Enter DA from A, or remain in DA.
        if dA + l_to_dead >= dDA + l_dd:
            kDA, nDA = 1, dA + l_to_dead + eD[t]
        else:
            kDA, nDA = 3, dDA + l_dd + eD[t]

        bp[t, 0], bp[t, 1], bp[t, 2], bp[t, 3] = kH, kA, kDH, kDA
        dH, dA, dDH, dDA = nH, nA, nDH, nDA

    s = int(np.argmax([dH, dA, dDH, dDA]))
    states = np.empty(n, dtype=np.int8)
    for t in range(n - 1, -1, -1):
        states[t] = s
        s = bp[t, s]

    # Map: 0 -> 0 (H), 1 -> 1 (A), 2 -> 2 (D), 3 -> 2 (D)
    to_code = np.array([H, A, D, D], dtype=np.int8)
    return to_code[states]


def decode(E: np.ndarray, C: np.ndarray, K: np.ndarray,
           meta: Dict[str, Any], params: Optional[Union[Params, dict]] = None,
           return_codes: bool = True) -> np.ndarray:
    """Decode full-match H/A/D/X labels.

    Args:
        E: log-probabilities [T,3]
        C: touching-team probabilities [T,2]
        K: touch detection [T]
        meta: metadata dict (half, graded, fps)
        params: Params object or dict
        return_codes: True for int8 codes (0 H, 1 A, 2 D, 3 X); False for strings H/A/D/X

    Returns:
        y: np.ndarray [T], int8 or string <U1
    """
    p = ensure_params(params)
    T = len(E)
    fps = int(meta.get("fps", 25))
    half = np.asarray(meta.get("half", np.ones(T, dtype=np.int8)))
    graded = meta.get("graded")

    kopp_0, kopp_1 = _compute_touch_evidence(K, C, p.touch_window)
    y_codes = np.full(T, X, dtype=np.int8)

    for h in sorted(set(half) - {0}):
        h_mask = (half == h)
        if not np.any(h_mask):
            continue

        h_idx = np.where(h_mask)[0]
        y_h = _viterbi_half(E[h_idx], kopp_0[h_idx], kopp_1[h_idx], p, fps)
        y_codes[h_idx] = y_h

    # Handle frames outside both halves (half == 0):
    # Assign D if graded == True outside a half; otherwise X.
    if graded is not None:
        graded_arr = np.asarray(graded, dtype=bool)
        outside_graded = (half <= 0) & graded_arr
        y_codes[outside_graded] = D
        outside_ungraded = (half <= 0) & (~graded_arr)
        y_codes[outside_ungraded] = X
    else:
        outside = (half <= 0)
        y_codes[outside] = X

    if return_codes:
        return y_codes
    return codes_to_labels(y_codes)


def to_eval_codes(y_codes: np.ndarray) -> np.ndarray:
    """Codes [0,1,2,3] -> {0,1,2} for pipeline.evaluate_dev, replacing outside-half X with D."""
    arr = np.asarray(y_codes, dtype=np.int8)
    return np.where(arr == X, D, arr).astype(np.int8)


def codes_to_labels(y_codes: np.ndarray) -> np.ndarray:
    """Codes [0,1,2,3] -> characters ['H','A','D','X']."""
    label_map = np.array(["H", "A", "D", "X"], dtype="<U1")
    return label_map[np.clip(y_codes, 0, 3)]


def to_submission_df(y_codes_or_labels: np.ndarray) -> pd.DataFrame:
    """Create the required two-column submission DataFrame: 'frame,label'."""
    if np.issubdtype(y_codes_or_labels.dtype, np.integer):
        labels = codes_to_labels(y_codes_or_labels)
    else:
        labels = y_codes_or_labels
    return pd.DataFrame({
        "frame": np.arange(len(labels), dtype=np.int64),
        "label": labels
    })
