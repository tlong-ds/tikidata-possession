"""Blend frame probabilities from multiple models before decoding (stage 3).

Measured motivation: models make different errors, so blending their probabilities before
stage 4 (src.nap1.decode) can outperform individual models. No weights are hard-coded:
`tune` selects them on dev/select; `apply` evaluates dev/report once.

Each data file (e.g. train_data/dev/select/DFL-MAT-J03WN1_r00.npz) has a SAME-NAMED .npz
in the source directory, containing:
    E       float [T,3]  calibrated log P(H, A, D); exp(E) sums to 1 per row
    C       float [T,2]  optional with K: touching-team probability, columns H (group 1), A
    K       float [T]    optional with C: touch probability
    S       float [T,2]  optional: switch-to-H / switch-to-A probabilities
    half    int8  [T]    optional copy of data half, for alignment checks
    source  str          source name, e.g. 'nap1' or 'nap2:official_s1/ep018,...'
Outside halves (half <= 0): E = log(1/3), C = 0.5, K = 0, S = 0, as in src.nap2.infer.combine.
Frame t corresponds to data frame t (25 fps, untrimmed).

Blends:
    blend_probs       weighted average of P(H, A, D) in probability space
    blend_team_dead   separately blend q = P(H | live) and pD = P(D), with separate weights
    blend_touch       weighted K*C evidence, supplied to the touch gate with K = 1

Used through blend.* steps (src/steps/blend_*.py) in `python -m src.exp run`:
export_* writes source probabilities; tune selects weights on dev/select.

Data rules: tune only on dev/select; dev/report is for reporting. train_data/test (reserved)
requires explicit allow_test=True and pipeline.split.guard(..., 'final'). Never read evaluation/gt.
"""

import dataclasses
import glob
import itertools
import json
import os
import time
from pathlib import Path

import numpy as np

H, A, D = 0, 1, 2
CLIP = 1e-8
LOG_THIRD = float(np.log(1.0 / 3.0))
SUBSETS = ("select", "report", "test")


# ---------------------------------------------------------------------------------------------
# Frame-probability files
# ---------------------------------------------------------------------------------------------

def _float(x, shape, name):
    a = np.asarray(x)
    if not np.issubdtype(a.dtype, np.floating):
        a = a.astype(np.float64)
    if a.shape != shape:
        raise ValueError("%s phải có shape %s, nhận %s" % (name, shape, a.shape))
    if not np.isfinite(a).all():
        raise ValueError("%s có giá trị không hữu hạn" % name)
    return a


def check_probs(E, C=None, K=None, S=None, half=None):
    """Validate frame-probability shapes and values; return arrays with normalized dtypes."""
    E = np.asarray(E)
    if E.ndim != 2 or E.shape[1] != 3:
        raise ValueError("E phải có shape [T,3], nhận %s" % (E.shape,))
    T = len(E)
    E = _float(E, (T, 3), "E")
    rs = np.exp(E.astype(np.float64)).sum(1)
    if np.abs(rs - 1).max() > 1e-3:
        raise ValueError("exp(E) không cộng lại bằng 1 (E phải là log-xác suất, không phải logit)")
    if (C is None) != (K is None):
        raise ValueError("C và K phải đi cùng nhau")
    if C is not None:
        C = _float(C, (T, 2), "C")
        K = _float(K, (T,), "K")
        if C.min() < -1e-6 or C.max() > 1 + 1e-6 or K.min() < -1e-6 or K.max() > 1 + 1e-6:
            raise ValueError("C, K phải nằm trong [0, 1]")
    if S is not None:
        S = _float(S, (T, 2), "S")
        if S.min() < -1e-6 or S.max() > 1 + 1e-6:
            raise ValueError("S phải nằm trong [0, 1]")
    if half is not None:
        half = np.asarray(half).astype(np.int8)
        if half.shape != (T,):
            raise ValueError("half phải có shape [T]")
    return E, C, K, S, half


def save_probs(path, E, source, C=None, K=None, S=None, half=None):
    """Write a frame-probability file in the format documented above."""
    E, C, K, S, half = check_probs(E, C, K, S, half)
    out = {"E": E, "source": np.array(str(source))}
    for k, v in (("C", C), ("K", K), ("S", S), ("half", half)):
        if v is not None:
            out[k] = v
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    return path


def load_probs(path):
    """Load a frame-probability file -> dict E, C, K, S, half (None if absent), source (str)."""
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    E, C, K, S, half = check_probs(d["E"], d.get("C"), d.get("K"), d.get("S"), d.get("half"))
    return {"E": E, "C": C, "K": K, "S": S, "half": half,
            "source": str(d["source"]) if "source" in d else Path(path).parent.name}


def prob_path(src_dir, data_file):
    """Source probability path for a data file: same filename in the source's directory."""
    return Path(src_dir) / Path(data_file).name


def outside(half, T):
    """Mask outside-half frames (half <= 0); None when half is unavailable."""
    if half is None:
        return np.zeros(T, bool)
    half = np.asarray(half)
    if half.shape != (T,):
        raise ValueError("half phải có shape [T]")
    return half <= 0


# ---------------------------------------------------------------------------------------------
# Blending operations
# ---------------------------------------------------------------------------------------------

def _weights(ws, what):
    w = np.asarray(ws, dtype=np.float64)
    if (w < 0).any():
        raise ValueError("trọng số %s không được âm" % what)
    if w.sum() <= 0:
        raise ValueError("tổng trọng số %s phải dương" % what)
    return w / w.sum()


def _norm_log(P, half=None):
    """Clip probabilities at 1e-8, renormalize rows and take logs; outside halves E = log(1/3)."""
    P = np.clip(np.asarray(P, dtype=np.float64), CLIP, None)
    E = np.log(P / P.sum(1, keepdims=True))
    E[outside(half, len(E))] = LOG_THIRD
    return E


def _single(items, nweights):
    """Index of the only source with positive weight across all components, or None."""
    live = [i for i, it in enumerate(items) if any(w > 0 for w in it[1:1 + nweights])]
    if len(live) == 1 and all(w > 0 for w in items[live[0]][1:1 + nweights]):
        return live[0]
    return None


def _as_is(E, half):
    E = np.array(E, dtype=np.float64)
    E[outside(half, len(E))] = LOG_THIRD
    return E


def blend_probs(items, half=None):
    """[(E_i, w_i)] -> E = log(weighted mean of exp(E_i)), clipped at 1e-8 and row-normalized.

    Normalize weights by their sum. If only one source has positive weight, return its exact E
    to reproduce that model's score.
    """
    items = list(items)
    if not items:
        raise ValueError("cần ít nhất một nguồn")
    one = _single(items, 1)
    if one is not None:
        return _as_is(items[one][0], half)
    w = _weights([it[1] for it in items], "probs")
    P = sum(wi * np.exp(np.asarray(E, dtype=np.float64)) for (E, _), wi in zip(items, w) if wi > 0)
    return _norm_log(P, half)


def team_dead(E):
    """E [T,3] -> (q = P(H | live), pD = P(D)); q = 0.5 when P(H) + P(A) = 0."""
    P = np.exp(np.asarray(E, dtype=np.float64))
    P = P / P.sum(1, keepdims=True)
    alive = P[:, 0] + P[:, 1]
    q = np.where(alive > 0, P[:, 0] / np.where(alive > 0, alive, 1.0), 0.5)
    return q, P[:, 2]


def from_team_dead(q, pD, half=None):
    """(q, pD) -> E = log [q*(1-pD), (1-q)*(1-pD), pD], clipped at 1e-8 and renormalized."""
    q = np.clip(np.asarray(q, dtype=np.float64), 0.0, 1.0)
    pD = np.clip(np.asarray(pD, dtype=np.float64), 0.0, 1.0)
    P = np.stack([q * (1 - pD), (1 - q) * (1 - pD), pD], 1)
    return _norm_log(P, half)


def blend_team_dead(items, half=None):
    """[(E_i, w_team_i, w_dead_i)] -> E, separately blending q = P(H | live) and pD.

    Normalize each weight set by its own sum. If just one source has positive weight in both
    components, return its exact E.
    """
    items = list(items)
    if not items:
        raise ValueError("cần ít nhất một nguồn")
    one = _single(items, 2)
    if one is not None:
        return _as_is(items[one][0], half)
    wt = _weights([it[1] for it in items], "team")
    wd = _weights([it[2] for it in items], "dead")
    T = len(items[0][0])
    q, pD = np.zeros(T), np.zeros(T)
    for (E, _, _), a, b in zip(items, wt, wd):
        if a == 0 and b == 0:
            continue
        qi, di = team_dead(E)
        q += a * qi
        pD += b * di
    return from_team_dead(q, pD, half)


def blend_touch(items, half=None):
    """[(K_i, C_i, w_i)] -> (K = 1, C = clip[0,1](sum_i w_i*K_i*C_i)).

    src.nap1.decode uses clip(K*C), so K = 1 and C = blended evidence supply the exact mixture.
    K_i = None means C_i already contains evidence (e.g. switch S). Weights are NOT normalized:
    a sum other than 1 widens or narrows the gate. Outside halves: C = 0.5, K = 0.
    """
    items = list(items)
    if not items:
        raise ValueError("cần ít nhất một nguồn")
    T = len(items[0][1])
    C = np.zeros((T, 2))
    for K, Ci, w in items:
        if w < 0:
            raise ValueError("trọng số touch không được âm")
        if w == 0:
            continue
        Ci = np.asarray(Ci, dtype=np.float64)
        k = 1.0 if K is None else np.asarray(K, dtype=np.float64)[:, None]
        C += w * k * Ci
    C = np.clip(C, 0.0, 1.0)
    K = np.ones(T)
    out = outside(half, T)
    C[out] = 0.5
    K[out] = 0.0
    return K, C


def touch_evidence(d):
    """Loaded-source (K, C) if available; otherwise (None, S) if S exists; otherwise None."""
    if d.get("C") is not None:
        return d["K"], d["C"]
    if d.get("S") is not None:
        return None, d["S"]
    return None


# ---------------------------------------------------------------------------------------------
# Blend specification and decoding
# ---------------------------------------------------------------------------------------------
# spec = {"mode": "team_dead", "team": {name: w}, "dead": {name: w}, "touch": {name: w}, "decode": {...}}
#     or {"mode": "probs", "probs": {name: w}, "touch": {name: w}, "decode": {...}}
# "decode" overrides the base stage-4 Params (e.g. {"lambda_emit": 1.5}).

def spec_kinds(mode):
    if mode == "team_dead":
        return ("team", "dead", "touch")
    if mode == "probs":
        return ("probs", "touch")
    raise ValueError("mode phải là 'team_dead' hoặc 'probs'")


def spec_sources(spec):
    """Names of sources with positive weights in spec."""
    names = []
    for kind in spec_kinds(spec.get("mode", "team_dead")):
        for n, w in spec.get(kind, {}).items():
            if w > 0 and n not in names:
                names.append(n)
    return names


def pure_spec(ref, mode="team_dead", decode=None):
    """Single-source reference specification."""
    s = {"mode": mode, "touch": {ref: 1.0}, "decode": dict(decode or {})}
    for kind in spec_kinds(mode)[:-1]:
        s[kind] = {ref: 1.0}
    return s


def blend_sources(srcs, spec, half=None):
    """srcs {name: load_probs dict} + spec -> (E, C, K) for src.nap1.decode."""
    mode = spec.get("mode", "team_dead")
    kinds = spec_kinds(mode)
    for kind in kinds:
        for n in spec.get(kind, {}):
            if n not in srcs:
                raise KeyError("spec dùng nguồn %r nhưng không có (có: %s)" % (n, ", ".join(srcs)))
    if mode == "probs":
        E = blend_probs([(srcs[n]["E"], w) for n, w in spec["probs"].items()], half)
    else:
        names = list(dict.fromkeys(list(spec.get("team", {})) + list(spec.get("dead", {}))))
        E = blend_team_dead([(srcs[n]["E"], spec.get("team", {}).get(n, 0.0),
                              spec.get("dead", {}).get(n, 0.0)) for n in names], half)
    items = []
    for n, w in spec.get("touch", {}).items():
        if w <= 0:
            continue
        ev = touch_evidence(srcs[n])
        if ev is None:
            raise ValueError("nguồn %r không có C/K hay S nên không dùng được cho touch" % n)
        items.append((ev[0], ev[1], w))
    if items:
        K, C = blend_touch(items, half)
    else:                                       # No touch evidence: only eps_gate remains.
        K, C = blend_touch([(None, np.zeros((len(E), 2)), 1.0)], half)
    return E, C, K


def decode_params(base="nap2"):
    """Base stage-4 Params: 'nap2' (src.nap2.shared.nap2_decode_params), 'nap1' (fitted stage-1
    configuration), or a JSON path overriding stage-1 settings.
    """
    from src.nap1.params import Params
    from src.nap2 import shared
    if isinstance(base, Params):
        return base
    if base in (None, "nap2"):
        return shared.nap2_decode_params()
    if base == "nap1":
        return shared.nap1_params()
    over = json.loads(Path(base).read_text(encoding="utf-8"))
    return Params.from_dict({**shared.nap1_params().to_dict(), **over})


def decode_codes(E, C, K, half, graded, params):
    """src.nap1.decode -> codes {0,1,2}; replace outside-half X with D as required by pipeline.score_export."""
    from src.nap1.decode import decode, to_eval_codes
    meta = {"half": np.asarray(half), "graded": np.asarray(graded, bool), "fps": 25}
    return to_eval_codes(decode(E, C, K, meta, params))


def n_spells(codes, half):
    """Number of H/A possessions within the two halves."""
    from pipeline.quality import segments
    n = 0
    for i, j in segments(np.asarray(half)):
        x = codes[i:j]
        n += int((np.isin(x, (H, A)) & np.r_[True, x[1:] != x[:-1]]).sum())
    return n


def label(spec, ref=None):
    """Short table label for a spec, omitting reference-source weights."""
    parts = []
    for kind in spec_kinds(spec.get("mode", "team_dead")):
        ws = {n: w for n, w in spec.get(kind, {}).items() if n != ref}
        if ws:
            parts.append("%s[%s]" % (kind, " ".join("%s=%g" % (n, w) for n, w in ws.items())))
    if spec.get("decode"):
        parts.append("decode[%s]" % " ".join("%s=%g" % kv for kv in sorted(spec["decode"].items())))
    return " ".join(parts) or "chỉ %s" % ref


# ---------------------------------------------------------------------------------------------
# Weight grid
# ---------------------------------------------------------------------------------------------
# grid = {"mode": "team_dead", "ref": "nap2",
#         "team": {"nap1": [0, .1, .2]}, "dead": {"nap1": [0, .2, .35]},
#         "touch": {"nap1": [0]},                       # optional; defaults to the reference source only
#         "decode": [{}, {"lambda_emit": 1.5}]}         # optional; defaults to [{}]
# Each component: reference weight = 1 - sum(other weights); skip negative combinations.

def expand_grid(grid, names):
    """-> [(spec, idx)], with grid-index coordinates idx used to find a plateau."""
    mode = grid.get("mode", "team_dead")
    kinds = spec_kinds(mode)
    ref = grid.get("ref") or names[0]
    if ref not in names:
        raise ValueError("nguồn mốc %r không có trong %s" % (ref, names))
    dims = []                                   # (kind, name, values)
    for kind in kinds:
        for n, vals in grid.get(kind, {}).items():
            if n == ref:
                raise ValueError("không đặt lưới cho nguồn mốc %r (nó nhận phần còn lại)" % ref)
            if n not in names:
                raise ValueError("lưới dùng nguồn %r không có trong %s" % (n, names))
            dims.append((kind, n, [float(v) for v in vals]))
    decodes = grid.get("decode") or [{}]
    out = []
    for idx in itertools.product(*[range(len(d[2])) for d in dims], range(len(decodes))):
        spec = {"mode": mode, "decode": dict(decodes[idx[-1]])}
        ok = True
        for kind in kinds:
            ws = {d[1]: d[2][i] for d, i in zip(dims, idx) if d[0] == kind}
            rest = 1.0 - sum(ws.values())
            if rest < -1e-9:
                ok = False
                break
            spec[kind] = {ref: round(max(rest, 0.0), 10), **ws}
        if ok:
            out.append((spec, tuple(idx)))
    return out


def parse_values(text):
    """'0:0.5:0.1' (inclusive endpoints) or '0,0.1,0.2' -> list of numbers."""
    text = text.strip()
    if ":" in text:
        a, b, s = (float(x) for x in text.split(":"))
        if s <= 0:
            raise ValueError("bước phải dương: %r" % text)
        return [round(v, 6) for v in np.arange(a, b + s / 2, s)]
    return [float(x) for x in text.split(",") if x.strip()]


def pick_plateau(rows, tol=0.002):
    """Choose the center of the plateau around the best score.

    A plateau contains connected grid cells with spellF1 >= best - tol; neighbors differ by at
    most one step in every coordinate. Choose the cell nearest the centroid, breaking ties by
    score. Non-grid rows (idx None, e.g. an added reference row) qualify only if they are best.
    """
    best = max(rows, key=lambda r: r["spell_f1"])
    if best.get("idx") is None:
        return best
    cand = [r for r in rows if r.get("idx") is not None and r["spell_f1"] >= best["spell_f1"] - tol]
    comp, todo = [best], [best]
    seen = {tuple(best["idx"])}
    while todo:
        r = todo.pop()
        for s in cand:
            k = tuple(s["idx"])
            if k not in seen and max(abs(a - b) for a, b in zip(r["idx"], s["idx"])) <= 1:
                seen.add(k)
                comp.append(s)
                todo.append(s)
    centre = np.mean([r["idx"] for r in comp], axis=0)
    return min(comp, key=lambda r: (round(float(np.linalg.norm(np.asarray(r["idx"]) - centre)), 9),
                                    -r["spell_f1"]))


# ---------------------------------------------------------------------------------------------
# Data files and match-split rules
# ---------------------------------------------------------------------------------------------

def match_of(path):
    """'DFL-MAT-J03WN1_r00.npz' -> 'DFL-MAT-J03WN1'."""
    return Path(path).stem.rsplit("_", 1)[0]


def guard_files(files, subset, allow_test=False):
    """Enforce file-split rules: select/report use dev; test requires an explicit flag."""
    from pipeline import split
    if subset not in SUBSETS:
        raise ValueError("subset phải là một trong %s" % (SUBSETS,))
    if subset == "test" and not allow_test:
        raise ValueError("train_data/test là trận dự trữ: chỉ mở khi truyền rõ allow_test (--allow-test)")
    for f in files:
        folder = Path(f).parent.name
        if folder in SUBSETS and folder != subset:
            raise ValueError("file %s thuộc %s, không phải %s" % (f, folder, subset))
        split.guard(match_of(f), "final" if subset == "test" else subset)


def data_files(subset, allow_test=False):
    """Subset data files after guard_files validation."""
    from src.nap2 import shared
    from pipeline.schema import TRAIN_DATA_DIR
    if subset == "test":
        guard_files([], subset, allow_test)
        files = sorted(glob.glob(str(Path(TRAIN_DATA_DIR) / "test" / "*.npz")))
    else:
        files = shared.dev_files(subset)
    guard_files(files, subset, allow_test)
    return files


def check_out_dir(path):
    """Never write into train_data/, cache/ or evaluation/gt/."""
    from pipeline.schema import CACHE_DIR, ROOT, TRAIN_DATA_DIR
    # Check both written and resolved paths: train_data/dev... may be a symlink in a worktree.
    # resolve() alone would point into the main checkout and bypass the check.
    p = Path(path).resolve()
    forms = {Path(os.path.abspath(path)), p}
    for bad in (TRAIN_DATA_DIR, CACHE_DIR, os.path.join(ROOT, "evaluation", "gt")):
        for b in {Path(os.path.abspath(bad)), Path(bad).resolve()}:
            if any(f == b or b in f.parents for f in forms):
                raise ValueError("không ghi vào %s" % bad)
    return p


def load_sources(src_dirs, data_file, T=None, half=None):
    """{name: directory} -> {name: probability dict} for one data file; verify frame count and half alignment."""
    out = {}
    for n, d in src_dirs.items():
        pth = prob_path(d, data_file)
        if not pth.is_file():
            raise FileNotFoundError("nguồn %r thiếu file %s" % (n, pth))
        s = load_probs(pth)
        if T is not None and len(s["E"]) != T:
            raise ValueError("nguồn %r: %s có %d khung, dữ liệu có %d" % (n, pth.name, len(s["E"]), T))
        if half is not None and s["half"] is not None and not np.array_equal(s["half"], half):
            raise ValueError("nguồn %r: half của %s không khớp dữ liệu" % (n, pth.name))
        out[n] = s
    return out


def _load_data(path):
    with np.load(path, allow_pickle=False) as z:
        half = np.asarray(z["half"])
        graded = np.asarray(z["graded"], bool) if "graded" in z.files else half > 0
        y = np.asarray(z["y"]) if "y" in z.files else None
    return y, half, graded


def _score(y, half, codes):
    from pipeline.score_export import score
    return score({"y": y, "half": half}, codes)


# ---------------------------------------------------------------------------------------------
# Weight tuning and application
# ---------------------------------------------------------------------------------------------

def _eval_file(args):
    data_file, src_dirs, specs, params = args
    y, half, graded = _load_data(data_file)
    srcs = load_sources(src_dirs, data_file, len(half), half)
    res = []
    for spec in specs:
        E, C, K = blend_sources(srcs, spec, half)
        p = dataclasses.replace(params, **spec.get("decode", {}))
        codes = decode_codes(E, C, K, half, graded, p)
        sp, mf = _score(y, half, codes)
        res.append((sp, mf, n_spells(codes, half)))
    return res


def _map(fn, jobs, workers):
    if workers <= 1 or len(jobs) <= 1:
        return [fn(j) for j in jobs]
    from multiprocessing import get_context
    with get_context("spawn").Pool(min(workers, len(jobs))) as pool:
        return pool.map(fn, jobs)


def evaluate_specs(files, sources, specs, params, workers=1):
    """Score every spec on every file -> [n_spec, n_file, 3] (spellF1, macroF1, possession count)."""
    jobs = [(f, dict(sources), list(specs), params) for f in files]
    return np.asarray(_map(_eval_file, jobs, workers), dtype=np.float64).transpose(1, 0, 2)


def _spec_key(spec):
    """Spec deduplication key, omitting zero-weight sources (e.g. nap1=0 equals the reference row)."""
    canon = {k: ({n: w for n, w in v.items() if w > 0} if k in ("team", "dead", "probs", "touch") else v)
             for k, v in spec.items()}
    return json.dumps(canon, sort_keys=True)


def tune(files, sources, grid, subset="select", params="nap2", tol=0.002, workers=1):
    """Search a weight grid, decode with src.nap1.decode, score with pipeline.score_export.score.

    files: data files (None = train_data/dev/<subset>); sources: {name: probability directory}.
    The first source is the reference unless grid specifies "ref". Only subset 'select' is allowed.
    Return (table, chosen): table is sorted by descending spellF1 and always includes a reference-only
    row flagged "baseline"; chosen is the plateau-center row within tol of the best score.
    """
    if subset != "select":
        raise ValueError("chỉ chỉnh trọng số trên dev/select (report để báo điểm, test để kiểm cuối)")
    files = list(files) if files is not None else data_files(subset)
    guard_files(files, subset)
    sources = dict(sources)
    names = list(sources)
    p0 = decode_params(params)
    ref = grid.get("ref") or names[0]
    mode = grid.get("mode", "team_dead")
    cand = expand_grid(dict(grid, ref=ref), names)
    base = pure_spec(ref, mode)
    specs = [base] + [s for s, _ in cand]
    idxs = [None] + [i for _, i in cand]
    res = evaluate_specs(files, sources, specs, p0, workers)
    ref_sp = res[0, :, 0]
    rows = []
    seen = {}
    for k, (spec, idx) in enumerate(zip(specs, idxs)):
        key = _spec_key(spec)
        if key in seen:                         # The reference spec matches a grid cell: retain its grid coordinates.
            if idx is not None:
                rows[seen[key]]["idx"] = list(idx)
            continue
        d = res[k, :, 0] - ref_sp
        seen[key] = len(rows)
        rows.append({"label": label(spec, ref), "spec": spec, "idx": None if idx is None else list(idx),
                     "baseline": k == 0, "spell_f1": float(res[k, :, 0].mean()),
                     "sd": float(res[k, :, 0].std(ddof=1)) if len(files) > 1 else 0.0,
                     "macro_f1": float(res[k, :, 1].mean()), "gain": float(d.mean()),
                     "wins": int((d > 1e-9).sum()), "spells": float(res[k, :, 2].mean()),
                     "per_file": [float(v) for v in res[k, :, 0]]})
    table = sorted(rows, key=lambda r: -r["spell_f1"])
    chosen = pick_plateau(table, tol)
    return table, chosen


def apply(files, sources, spec, out_dir, subset, params="nap2", allow_test=False, ref=None):
    """Apply one spec to subset files, writing <out_dir>/<filename>.npz with int8 [T] codes {0,1,2}.

    If labels exist, also score predictions, including a reference-only row `ref` for paired
    comparisons. Return a results dict.
    """
    files = list(files) if files is not None else data_files(subset, allow_test)
    guard_files(files, subset, allow_test)
    out = check_out_dir(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p0 = decode_params(params)
    p = dataclasses.replace(p0, **spec.get("decode", {}))
    need = spec_sources(spec)
    if ref is not None and ref not in need:
        need.append(ref)
    src = {n: sources[n] for n in need}
    rows = []
    for f in files:
        y, half, graded = _load_data(f)
        srcs = load_sources(src, f, len(half), half)
        E, C, K = blend_sources(srcs, spec, half)
        codes = decode_codes(E, C, K, half, graded, p)
        np.savez(out / Path(f).name, codes=codes.astype(np.int8),
                 blend=np.array(json.dumps({"spec": spec, "sources": {n: str(sources[n]) for n in need}})))
        row = {"file": Path(f).name, "spells": n_spells(codes, half)}
        if y is not None:
            row["spell_f1"], row["macro_f1"] = _score(y, half, codes)
            if ref is not None:
                bs = pure_spec(ref, spec.get("mode", "team_dead"), spec.get("decode"))
                Eb, Cb, Kb = blend_sources(srcs, bs, half)
                row["ref_spell_f1"] = _score(y, half, decode_codes(Eb, Cb, Kb, half, graded, p))[0]
        rows.append(row)
    result = {"subset": subset, "out": str(out), "spec": spec, "ref": ref, "files": rows}
    if rows and "spell_f1" in rows[0]:
        sp = np.array([r["spell_f1"] for r in rows])
        result["spell_f1"] = float(sp.mean())
        result["sd"] = float(sp.std(ddof=1)) if len(sp) > 1 else 0.0
        result["macro_f1"] = float(np.mean([r["macro_f1"] for r in rows]))
        if ref is not None:
            d = sp - np.array([r["ref_spell_f1"] for r in rows])
            result["ref_spell_f1"] = float(np.mean([r["ref_spell_f1"] for r in rows]))
            result["gain"] = float(d.mean())
            result["wins"] = int((d > 1e-9).sum())
    return result


# ---------------------------------------------------------------------------------------------
# Export stage-1 and stage-2 frame probabilities
# ---------------------------------------------------------------------------------------------

def mix_hold(p1, K, half, alpha, thr=0.5):
    """'New stage-1' variant: alpha * p1 at the last touch (K > thr) + (1-alpha) * current p1.

    Hold touch-frame p1 until the next touch within a half; before the first touch, retain p1.
    """
    from pipeline.quality import segments
    p1 = np.asarray(p1, dtype=np.float64)
    out = p1.copy()
    for i, j in segments(np.asarray(half)):
        idx = np.where(np.asarray(K[i:j]) > thr, np.arange(j - i), -1)
        last = np.maximum.accumulate(idx)
        held = np.where(last >= 0, p1[i:j][np.maximum(last, 0)], p1[i:j])
        out[i:j] = alpha * held + (1 - alpha) * p1[i:j]
    return out


def nap1_frame_probs(raw, hold=0.0):
    """src.nap2.shared.load_file dict -> stage-1 (E, C, K) via estimate_ball -> score_frames_detailed.

    hold > 0: replace p1 with mix_hold(p1, K, half, hold), then rebuild E and C from p1 and D.
    """
    from src.nap1.ball import estimate_ball
    from src.nap1.score import score_frames_detailed
    from src.nap2 import shared
    p = shared.nap1_params()
    V = shared.as_view(raw)
    half = np.asarray(V.meta["half"])
    Bhat, K = estimate_ball(V, p)
    det = score_frames_detailed(V, Bhat, K, p)
    if hold > 0:
        p1 = mix_hold(det.p1, K, half, hold)
        E = from_team_dead(p1, det.D, half)
        C = np.stack([p1, 1 - p1], 1)
    else:
        E, C = det.E.copy(), det.C.copy()
    out = half <= 0
    E[out], C[out] = LOG_THIRD, 0.5
    K = np.where(out, 0.0, np.clip(K, 0.0, 1.0))
    return E, np.clip(C, 0.0, 1.0), K


def _export_nap1_one(args):
    f, out_dir, hold, source = args
    from src.nap2 import shared
    t0 = time.time()
    raw = shared.load_file(f)
    E, C, K = nap1_frame_probs(raw, hold)
    save_probs(Path(out_dir) / Path(f).name, E, source, C=C, K=K, half=raw["half"])
    return Path(f).name, time.time() - t0


def export_nap1(files, out_dir, hold=0.0, workers=1):
    """Export stage-1 probabilities for each data file. Return [(filename, seconds)]."""
    out = check_out_dir(out_dir)
    source = "nap1" if hold <= 0 else "nap1-hold%g" % hold
    return _map(_export_nap1_one, [(f, str(out), hold, source) for f in files], workers)


def nap2_checkpoints(pattern):
    """Retained checkpoints in run directories matching a glob, from summary.json['kept'].

    Summary paths refer to the training host; use filenames to locate checkpoints within each
    run directory. Without summary.json, use all ep*.pt files in the directory.
    """
    runs = sorted(d for d in glob.glob(pattern) if os.path.isdir(d))
    if not runs:
        raise FileNotFoundError("không có thư mục run nào khớp %r" % pattern)
    out = []
    for d in runs:
        s = Path(d) / "summary.json"
        if s.is_file():
            kept = json.loads(s.read_text(encoding="utf-8")).get("kept", [])
            paths = [Path(d) / Path(k["path"].replace("\\", "/")).name for k in kept]
        else:
            paths = [Path(p) for p in sorted(glob.glob(os.path.join(d, "ep*.pt")))]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError("thiếu checkpoint: %s" % ", ".join(missing))
        if not paths:
            raise FileNotFoundError("thư mục %s không có checkpoint" % d)
        out += [str(p) for p in paths]
    return out


def load_nap2_checkpoint(path, device="cpu"):
    """Load a tensor/basic-type-only stage-2 checkpoint (weights_only=True), build Nap2Net and load strictly."""
    import torch
    from torch.torch_version import TorchVersion
    from src.nap2.model import build_model
    with torch.serialization.safe_globals([TorchVersion]):
        ck = torch.load(path, map_location=device, weights_only=True)
    model = build_model(ck["model_cfg"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


def nap2_frame_probs(raw, models, device="cpu"):
    """load_file dict + [(model, ck)] -> stage-2 (E, C, K), as in src.nap2.infer.score_frames_net."""
    import torch
    from src.nap1.ball import estimate_ball
    from src.nap2 import shared
    from src.nap2.features import build_features
    from src.nap2.infer import combine, forward_full
    V = shared.as_view(raw)
    half = np.asarray(V.meta["half"])
    Bhat, _ = estimate_ball(V, shared.nap1_params())
    feats, outs, temps = {}, [], []
    with torch.no_grad():
        for model, ck in models:
            key = json.dumps(ck.get("feat_cfg"), sort_keys=True)
            if key not in feats:
                feats[key] = build_features(V, Bhat, ck.get("feat_cfg"))
            x = feats[key]
            tc = ck["train_cfg"]
            outs.append(forward_full(model, x["xp"], x["xf"], half, tc["chunk"], tc["stride"], device))
            temps.append(float(ck.get("T", 1.0)))
    return combine(outs, temps, half)


def export_nap2(files, out_dir, ckpts, device="cpu", threads=4):
    """Export stage-2 checkpoint-ensemble probabilities for each data file."""
    import torch
    torch.set_num_threads(threads)
    from src.nap2 import shared
    out = check_out_dir(out_dir)
    models = [load_nap2_checkpoint(p, device) for p in ckpts]
    source = "nap2:" + ",".join("%s/%s" % (Path(p).parent.name, Path(p).stem) for p in ckpts)
    done = []
    for f in files:
        t0 = time.time()
        raw = shared.load_file(f)
        E, C, K = nap2_frame_probs(raw, models, device)
        save_probs(out / Path(f).name, E, source, C=C, K=K, half=raw["half"])
        done.append((Path(f).name, time.time() - t0))
        print("  %s %.0fs" % done[-1], flush=True)
    return done
