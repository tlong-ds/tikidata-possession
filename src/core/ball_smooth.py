"""Make observed training balls resemble the smoothed BAR-ATH ball, which does not hold still.

The shared degradation retains true visible-ball coordinates: DFL repeats positions in 47-52%
of adjacent frames, with sharp turns (p99 acceleration ~430 m/s²); the target has no repeats,
p99 ~37 m/s², and turns spread over ~0.4 s. Approximation measured by scratchpad
analysis/ball_smooth_fit on 2026-11-15:

    b_obs = Gauss(b_true, sig) + Gauss(white_noise, noise_s), normalized to deviation noise_m

sig = 0.24 s, noise_m = 0.4 m, noise_s = 0.24 s match the statistics (turn peaks/min at lag 0.2 s:
6.6 vs 4.8; at 0.4 s: 15.4 vs 19.4; acceleration p50/p90/p99: 8/18/40 vs 6/18/37).
Only VISIBLE ball coordinates change; visibility masks, players and labels remain unchanged.

TOML [train.smooth_ball]: p (smoothing probability per training realization), sig, noise_m,
noise_s (each a scalar or [lo, hi], uniformly sampled per realization). Seeds depend on
(match name, epoch, run), making each realization reproducible.
"""

import hashlib

import numpy as np

DEFAULT = {"p": 0.5, "sig": [0.16, 0.32], "noise_m": [0.2, 0.6], "noise_s": [0.16, 0.32]}
KEYS = set(DEFAULT)


def check_cfg(cfg):
    bad = set(cfg) - KEYS
    if bad:
        raise ValueError("[train.smooth_ball] có khoá lạ %s (có: %s)" % (sorted(bad), sorted(KEYS)))
    out = dict(DEFAULT, **cfg)
    if not 0 <= float(out["p"]) <= 1:
        raise ValueError("[train.smooth_ball] p phải trong [0, 1]")
    for k in ("sig", "noise_m", "noise_s"):
        v = out[k]
        lo, hi = (v, v) if np.isscalar(v) else (v[0], v[1])
        if not 0 <= float(lo) <= float(hi):
            raise ValueError("[train.smooth_ball] %s phải là số >= 0 hoặc [lo, hi] với 0 <= lo <= hi" % k)
    return out


def gauss(x, sigma_f):
    """Gaussian filter along axis 0 with 'nearest' boundaries (as in scipy.ndimage.gaussian_filter1d, truncated at 4 sigma)."""
    if sigma_f <= 0:
        return x.copy()
    r = int(4 * sigma_f + 0.5)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_f) ** 2)
    k /= k.sum()
    pad = np.concatenate([np.repeat(x[:1], r, 0), x, np.repeat(x[-1:], r, 0)], 0)
    return np.stack([np.convolve(pad[:, j], k, mode="valid") for j in range(x.shape[1])], 1)


def smooth_ball(B, ball_xy, half, fps, sig, noise_m, noise_s, rng):
    out = np.array(B, dtype=np.float64, copy=True)
    seen = np.isfinite(out[:, 0]) & np.isfinite(out[:, 1]) & (np.nan_to_num(out[:, 2]) > 0)
    for h in np.unique(half[half > 0]):
        idx = np.where(half == h)[0]
        xy = np.asarray(ball_xy[idx], np.float64).copy()
        good = np.isfinite(xy).all(1)
        if good.sum() < 2:
            continue
        t = np.arange(len(idx))
        for j in range(2):
            xy[:, j] = np.interp(t, t[good], xy[good, j])
        s = gauss(xy, sig * fps)
        if noise_m > 0:
            n = gauss(rng.normal(0, 1, xy.shape), noise_s * fps)
            sd = n.std(0)
            s = s + n / np.where(sd > 0, sd, 1) * noise_m
        m = seen[idx]
        out[idx[m], :2] = s[m]
    return out.astype(np.asarray(B).dtype)


def _draw(v, rng):
    if np.isscalar(v):
        return float(v)
    return float(rng.uniform(float(v[0]), float(v[1])))


def maybe_smooth(raw, cfg, name, epoch, run):
    """raw (fresh_train dict) -> raw with B smoothed with probability p. Return (raw, info or None)."""
    if not cfg:
        return raw, None
    cfg = check_cfg(cfg)
    seed = int.from_bytes(hashlib.sha256(("%s|%d|%d" % (name, epoch, run)).encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    if rng.random() >= float(cfg["p"]) or "ball_xy" not in raw:
        return raw, None
    sig, nm, ns = _draw(cfg["sig"], rng), _draw(cfg["noise_m"], rng), _draw(cfg["noise_s"], rng)
    out = dict(raw)
    out["B"] = smooth_ball(raw["B"], raw["ball_xy"], np.asarray(raw["half"]), float(raw["fps"]), sig, nm, ns, rng)
    return out, {"sig": sig, "noise_m": nm, "noise_s": ns}
