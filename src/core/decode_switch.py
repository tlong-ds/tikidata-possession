"""Stage-3 decoder: four-state Viterbi as in src.nap1.decode, with switches from the switch head.

Measured motivation: src.nap1.decode's touch gate can only REDUCE switch probability below
sw_live ~0.0042/frame (L_alive 7.29 s); it cannot create a switch unsupported by E. Even true
touches at change points yield only ~0.56-0.58. Here, at frame t:

    p(H -> A) = clip(scale * s_A[t], floor, cap)
    p(A -> H) = clip(scale * s_H[t], floor, cap)
    p(live -> D) = 1 / (L_alive * fps) * share_dead       (as in stage 1)
    D -> k: stage-1 rule with g_k = eps_gate + (1 - eps_gate) * clip(gate_scale * s_k, 0, 1)

s[t] = (s_H, s_A) predicts switches to H / A. Strong evidence makes switching inexpensive even
when E is uncertain; E still determines dead ball. Output codes match src.nap1.decode:
0 = H, 1 = A, 2 = D, 3 = X (outside halves: D if graded, otherwise X).

decode_gate is the fallback: pass S through src.nap1.decode's touch gate (K = 1, C = S).
"""

import dataclasses
import math
from dataclasses import dataclass

import numpy as np

H, A, D, X = 0, 1, 2, 3
NEG = -1e18


@dataclass(frozen=True)
class SwitchParams:
    scale: float = 0.2           # Multiply by s to obtain the per-frame team-switch probability.
    floor: float = 1e-6          # Minimum per-frame switch probability without evidence.
    cap: float = 0.5             # Maximum per-frame switch probability.
    gate_scale: float = 1.0      # Gate dead-ball exits using s.
    window: int = 0              # Max-pool s over +-window frames (0 disables pooling).
    lambda_emit: float = 2.0     # E weight, as in nap2_decode_params.
    L_alive: float = 7.29        # Seconds; only used for p(live -> D).
    L_dead: float = 24.5
    share_dead: float = 0.234
    p_switch_after_dead: float = 0.693
    eps_gate: float = 0.01
    deadball_switch_gate: bool = True

    def to_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})


def default_params():
    """SwitchParams inherits L_*, share_dead, p_switch_after_dead, eps_gate and lambda_emit from stage 2."""
    from src.nap2 import shared
    p = shared.nap2_decode_params()
    return SwitchParams(lambda_emit=float(p.lambda_emit), L_alive=float(p.L_alive), L_dead=float(p.L_dead),
                        share_dead=float(p.share_dead), p_switch_after_dead=float(p.p_switch_after_dead),
                        eps_gate=float(p.eps_gate),
                        deadball_switch_gate=bool(getattr(p, "deadball_switch_gate", True)))


def ensure(params):
    if params is None:
        return default_params()
    if isinstance(params, SwitchParams):
        return params
    return dataclasses.replace(default_params(), **{k: v for k, v in dict(params).items()
                                                    if k in SwitchParams.__dataclass_fields__})


def _log(a):
    a = np.asarray(a, dtype=np.float64)
    out = np.full(a.shape, NEG)
    m = a > 1e-300
    out[m] = np.log(a[m])
    return out


def _maxpool(x, w):
    if w <= 0:
        return x
    y = x.copy()
    for k in range(1, w + 1):
        y[k:] = np.maximum(y[k:], x[:-k])
        y[:-k] = np.maximum(y[:-k], x[k:])
    return y


def _viterbi_half(E, sH, sA, p, fps):
    n = len(E)
    if n == 0:
        return np.empty(0, np.int8)
    sw_a = 1.0 / (p.L_alive * fps)
    sw_d = 1.0 / (p.L_dead * fps)
    p_dead = sw_a * p.share_dead
    l_to_dead = math.log(p_dead)
    l_dd = math.log(max(1e-300, 1.0 - sw_d))
    p_keep = sw_d * (1.0 - p.p_switch_after_dead)
    p_sw = sw_d * p.p_switch_after_dead

    hi = min(p.cap, 1.0 - p_dead - 1e-9)
    pHA = np.clip(p.scale * sA, p.floor, hi)
    pAH = np.clip(p.scale * sH, p.floor, hi)
    sHH, sAA = _log(1.0 - p_dead - pHA).tolist(), _log(1.0 - p_dead - pAH).tolist()
    sHA, sAH = _log(pHA).tolist(), _log(pAH).tolist()
    if p.deadball_switch_gate:
        g0 = p.eps_gate + (1.0 - p.eps_gate) * np.clip(p.gate_scale * sH, 0.0, 1.0)
        g1 = p.eps_gate + (1.0 - p.eps_gate) * np.clip(p.gate_scale * sA, 0.0, 1.0)
        dHH, dAH = _log(p_keep * g0).tolist(), _log(p_sw * g0).tolist()      # DH->H, DA->H
        dHA, dAA = _log(p_sw * g1).tolist(), _log(p_keep * g1).tolist()      # DH->A, DA->A
    else:
        dHH = dAA = [math.log(p_keep)] * n
        dAH = dHA = [math.log(p_sw)] * n

    lam = float(p.lambda_emit)
    eH, eA, eD = (lam * E[:, 0]).tolist(), (lam * E[:, 1]).tolist(), (lam * E[:, 2]).tolist()
    bp = np.empty((n, 4), np.int8)
    vH, vA, vDH, vDA = eH[0], eA[0], eD[0], eD[0]
    for t in range(1, n):
        c = (vH + sHH[t], vA + sAH[t], vDH + dHH[t], vDA + dAH[t])
        kH = max(range(4), key=c.__getitem__)
        nH = c[kH] + eH[t]
        c = (vH + sHA[t], vA + sAA[t], vDH + dHA[t], vDA + dAA[t])
        kA = max(range(4), key=c.__getitem__)
        nA = c[kA] + eA[t]
        if vH + l_to_dead >= vDH + l_dd:
            kDH, nDH = 0, vH + l_to_dead + eD[t]
        else:
            kDH, nDH = 2, vDH + l_dd + eD[t]
        if vA + l_to_dead >= vDA + l_dd:
            kDA, nDA = 1, vA + l_to_dead + eD[t]
        else:
            kDA, nDA = 3, vDA + l_dd + eD[t]
        bp[t, 0], bp[t, 1], bp[t, 2], bp[t, 3] = kH, kA, kDH, kDA
        vH, vA, vDH, vDA = nH, nA, nDH, nDA
    s = int(np.argmax([vH, vA, vDH, vDA]))
    st = np.empty(n, np.int8)
    for t in range(n - 1, -1, -1):
        st[t] = s
        s = bp[t, s]
    return np.array([H, A, D, D], np.int8)[st]


def decode(E, S, meta, params=None):
    """E [T,3] log P(H,A,D), S [T,2] switch-to-H / switch-to-A probabilities, meta {half, graded?, fps?}
    -> int8 [T] codes (0=H, 1=A, 2=D, 3=X), decoded per half as in src.nap1.decode.
    """
    p = ensure(params)
    E = np.asarray(E, np.float64)
    S = np.clip(np.asarray(S, np.float64), 0.0, 1.0)
    T = len(E)
    fps = int(meta.get("fps", 25))
    half = np.asarray(meta.get("half", np.ones(T, np.int8)))
    sH, sA = _maxpool(S[:, 0], p.window), _maxpool(S[:, 1], p.window)
    out = np.full(T, X, np.int8)
    for h in sorted(set(half.tolist()) - {0}):
        if h < 0:
            continue
        idx = np.flatnonzero(half == h)
        out[idx] = _viterbi_half(E[idx], sH[idx], sA[idx], p, fps)
    graded = meta.get("graded")
    if graded is not None:
        g = np.asarray(graded, bool)
        out[(half <= 0) & g] = D
        out[(half <= 0) & ~g] = X
    else:
        out[half <= 0] = X
    return out


def decode_gate(E, S, meta, params=None):
    """Fallback: S through src.nap1.decode's touch gate (K = 1, C = S); params is nap1 Params."""
    from src.nap1.decode import decode as nap1_decode
    from src.nap2 import shared
    S = np.clip(np.asarray(S, np.float64), 0.0, 1.0)
    return nap1_decode(E, S, np.ones(len(S)), meta, params if params is not None else shared.nap2_decode_params())


def to_eval_codes(codes):
    c = np.asarray(codes, np.int8)
    return np.where(c == X, D, c).astype(np.int8)


DEFAULT_GRID = {"scale": [0.05, 0.1, 0.2, 0.4], "lambda_emit": [1.0, 2.0], "floor": [1e-6, 1e-4]}


def grid_params(grid=None, base=None):
    """Cartesian product of the grid -> SwitchParams list (default: DEFAULT_GRID over default_params)."""
    import itertools
    base = ensure(base)
    grid = DEFAULT_GRID if grid is None else grid
    keys = list(grid)
    return [dataclasses.replace(base, **dict(zip(keys, v))) for v in itertools.product(*[grid[k] for k in keys])]
