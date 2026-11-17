"""Shared evaluation tools for all models (numpy only; works without torch).

Select on dev/select, report dev/report once, compare models with paired realizations, and
break errors down by situation (visible / unseen ball, near a team change, fast ball, etc.).

Prediction directory: one <data filename>.npz per realization, containing:
    codes      int8[T]    0=H, 1=A, 2=D (required)
    team_prob  float[T]   pre-decoding P(H | live), optional for situation tables
    probs      float[T,3] P(H), P(A), P(D), optional alternative to team_prob
    p_dead     float[T]   per-frame P(D), optional

    python -m src.core.evaltools compare --subset select --pred nap1=DIR1 --pred nap2=DIR2
    python -m src.core.evaltools compare --subset select --pred smooth50=@baseline --pred nap1=DIR1
    python -m src.core.evaltools breakdown --subset select --pred DIR1

Ground truth (y, ball_xy) is only for situation masks and partial-oracle experiments, NEVER
model input. Do not read evaluation/gt/. Test requires the single-use --final flag.
"""
import argparse
import json
import math
import re
import sys
import warnings
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from pipeline.quality import segments
from pipeline.schema import TRAIN_DATA_DIR

H, A, D, X = 0, 1, 2, 3
FPS = 25
NEAR = 25            # +-1 s around a true team change.
FAST = 8.0           # m/s threshold for a fast-moving ball.
HI = 0.999           # "Certain" probability used in oracle experiments.
FOLDERS = {'select': 'dev/select', 'report': 'dev/report', 'test': 'test'}
PURPOSE = {'select': 'select', 'report': 'report', 'test': 'final'}
NOTE = {'select': 'chỉ để chọn, không báo cáo', 'report': 'số báo cáo, chạy một lần',
        'test': 'KIỂM TRA CUỐI, CHỈ MỘT LẦN'}
BASELINE = '@baseline'     # --pred NAME=@baseline: use the data file's existing baseline (smooth50) column.
EVAL_KEYS = ('y', 'half', 'graded', 'fps', 'pitch', 'B', 'ball_xy')
PRED_KEYS = ('codes', 'team_prob', 'probs', 'p_dead')


# ---------------------------------------------------------------- paired confidence intervals

# Two-sided t values for n-1 = 1..30 degrees of freedom, used without scipy.
_T_TABLE = {
    0.90: (6.314, 2.920, 2.353, 2.132, 2.015, 1.943, 1.895, 1.860, 1.833, 1.812, 1.796, 1.782, 1.771,
           1.761, 1.753, 1.746, 1.740, 1.734, 1.729, 1.725, 1.721, 1.717, 1.714, 1.711, 1.708, 1.706,
           1.703, 1.701, 1.699, 1.697),
    0.95: (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
           2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
           2.052, 2.048, 2.045, 2.042),
    0.99: (63.657, 9.925, 5.841, 4.604, 4.032, 3.707, 3.499, 3.355, 3.250, 3.169, 3.106, 3.055, 3.012,
           2.977, 2.947, 2.921, 2.898, 2.878, 2.861, 2.845, 2.831, 2.819, 2.807, 2.797, 2.787, 2.779,
           2.771, 2.763, 2.756, 2.750),
}


def _t_approx(level, df):
    """Cornish-Fisher approximation of t from the normal quantile (adequate for large df)."""
    from statistics import NormalDist
    z = NormalDist().inv_cdf(0.5 + level / 2)
    g1 = (z ** 3 + z) / 4
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96
    g3 = (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / 384
    g4 = (79 * z ** 9 + 776 * z ** 7 + 1482 * z ** 5 - 1920 * z ** 3 - 945 * z) / 92160
    return z + g1 / df + g2 / df ** 2 + g3 / df ** 3 + g4 / df ** 4


def t_crit(level, df, use_scipy=True):
    """Two-sided t at `level` and `df`: scipy if available, otherwise table (df<=30) or approximation."""
    if not 0 < level < 1 or df < 1:
        raise ValueError('cần 0 < level < 1 và df >= 1, nhận level=%r df=%r' % (level, df))
    if use_scipy:
        try:
            from scipy.stats import t
            return float(t.ppf(0.5 + level / 2, df))
        except ImportError:
            pass
    table = _T_TABLE.get(round(level, 6))
    if table is not None and df <= len(table):
        return table[int(df) - 1]
    return _t_approx(level, df)


def paired_ci(a, b, level=0.95, use_scipy=True):
    """Paired t confidence interval for per-realization differences b - a.

    Return dict(mean, sd, lo, hi, wins, losses, n); wins counts realizations where b exceeds a.
    Dev realizations are corruptions of the SAME match: this measures degradation variability,
    not uncertainty across matches.
    """
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    if a.shape != b.shape:
        raise ValueError('hai dãy điểm phải cùng số bản sao: %d vs %d' % (a.size, b.size))
    d = b - a
    n = int(d.size)
    out = {'mean': float('nan'), 'sd': float('nan'), 'lo': float('nan'), 'hi': float('nan'),
           'wins': int((d > 0).sum()), 'losses': int((d < 0).sum()), 'n': n}
    if n == 0:
        return out
    out['mean'] = float(d.mean())
    if n > 1:
        sd = float(d.std(ddof=1))
        half = t_crit(level, n - 1, use_scipy) * sd / math.sqrt(n)
        out.update(sd=sd, lo=out['mean'] - half, hi=out['mean'] + half)
    return out


# ---------------------------------------------------------------- load data / predictions

def match_of(path):
    """'DFL-MAT-J03WN1_r00.npz' -> 'DFL-MAT-J03WN1', stripping realization suffix _rNN / _eK."""
    return re.sub(r'_[re]\d+$', '', Path(path).stem)


def data_files(subset, root=None, final=False):
    """Subset data files after pipeline.split.guard validation.

    Reject subset='test' unless final=True: test is evaluated once, immediately before submission.
    """
    if subset not in FOLDERS:
        raise ValueError('subset phải thuộc %s, nhận %r' % (sorted(FOLDERS), subset))
    if subset == 'test' and not final:
        raise PermissionError('tập test chỉ dùng một lần: thêm --final (final=True) nếu thật sự muốn chạy')
    folder = Path(root or TRAIN_DATA_DIR) / FOLDERS[subset]
    files = sorted(folder.glob('*.npz'))
    if not files:
        raise FileNotFoundError('không có file .npz nào trong %s' % folder)
    from pipeline.split import guard
    for f in files:
        guard(match_of(f), PURPOSE[subset])
    return files


def load_eval(path, keys=EVAL_KEYS):
    """Read only arrays needed for scoring / situation masks; omit P to reduce memory use."""
    with np.load(path) as z:
        out = {k: np.asarray(z[k]) for k in keys if k in z.files}
    out['name'] = Path(path).name
    return out


def _team_prob_from(pred):
    if 'team_prob' in pred:
        return np.asarray(pred['team_prob'], float)
    if 'probs' in pred:
        p = np.asarray(pred['probs'], float)
        with np.errstate(invalid='ignore', divide='ignore'):
            return p[:, H] / (p[:, H] + p[:, A])
    return None


def load_pred(pred_dir, name):
    """Read <pred_dir>/<name>: dict with 'codes', optionally 'team_prob' (inferred from 'probs') and 'p_dead'."""
    path = Path(pred_dir) / Path(name).name
    if not path.is_file():
        raise FileNotFoundError('không có file dự đoán %s' % path)
    with np.load(path) as z:
        out = {k: np.asarray(z[k]) for k in PRED_KEYS if k in z.files}
    if 'codes' not in out:
        raise KeyError('%s không có mảng codes' % path)
    tp = _team_prob_from(out)
    if tp is not None:
        out['team_prob'] = tp
    return out


def load_pred_dir(pred_dir, files=None):
    """Load predictions from a directory, or exactly the requested `files` (raise if any are missing)."""
    d = Path(pred_dir)
    if files is None:
        names = sorted(p.name for p in d.glob('*.npz'))
    else:
        names = [Path(f).name for f in files]
    missing = [n for n in names if not (d / n).is_file()]
    if missing:
        raise FileNotFoundError('%s thiếu %d file dự đoán: %s' % (d, len(missing), ', '.join(missing[:5])))
    out = {}
    for n in names:
        with np.load(d / n) as z:
            if 'codes' not in z.files:
                raise KeyError('%s không có mảng codes' % (d / n))
            out[n] = np.asarray(z['codes'])
    return out


def write_pred_dir(out_dir, preds):
    """Write predictions in the documented format: preds maps filenames to codes or dict(codes=..., team_prob=...)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, v in preds.items():
        arrays = dict(v) if isinstance(v, Mapping) else {'codes': v}
        arrays['codes'] = np.asarray(arrays['codes'], np.int8)
        np.savez_compressed(out / Path(name).name, **{k: np.asarray(a) for k, a in arrays.items()})
    return out


def baseline_preds(files):
    """{filename: baseline} from dev/test data files themselves (smooth50)."""
    return {Path(f).name: load_eval(f, ('baseline',))['baseline'] for f in files}


def _as_codes(codes, n, name=''):
    c = np.asarray(codes)
    if c.shape != (n,):
        raise ValueError('%s: codes có dạng %s, dữ liệu có %d khung' % (name, c.shape, n))
    if c.size and (c.min() < 0 or c.max() > X):
        raise ValueError('%s: codes phải nằm trong 0..3 (0=H, 1=A, 2=D, 3=X)' % name)
    return c.astype(np.int8)


def _pick(codes_by_file, f, i):
    if callable(codes_by_file):
        return codes_by_file(f)
    if isinstance(codes_by_file, Mapping):
        p = Path(f)
        for k in (p.name, p.stem, str(f), p):
            if k in codes_by_file:
                return codes_by_file[k]
        raise KeyError('không có codes cho %s' % p.name)
    return codes_by_file[i]


# ---------------------------------------------------------------- scoring

def score_codes(files, codes_by_file):
    """Per-realization pipeline.score_export.score -> [n, 2] = (spellF1, macroF1).

    codes_by_file: dict keyed by filename, stem or path; list aligned with `files`; or function f -> codes.
    """
    from pipeline.score_export import score
    rows = []
    for i, f in enumerate(files):
        z = load_eval(f, ('y', 'half'))
        c = _as_codes(_pick(codes_by_file, f, i), len(z['y']), Path(f).name)
        rows.append(score(z, c))
    return np.asarray(rows, float).reshape(len(rows), 2)


def compare(files, preds, level=0.95):
    """Compare multiple models on identical realizations. preds: {model name: codes_by_file}; first is the reference.

    Return dict(files, level, models={name: dict(spellF1=[...], macroF1=[...], vs_first=... | None)}).
    """
    names = list(preds)
    if not names:
        raise ValueError('cần ít nhất một mô hình')
    scores = {n: score_codes(files, preds[n]) for n in names}
    ref = scores[names[0]]
    res = {'files': [Path(f).name for f in files], 'level': level, 'models': {}}
    for n in names:
        s = scores[n]
        vs = None if n == names[0] else {'spellF1': paired_ci(ref[:, 0], s[:, 0], level),
                                         'macroF1': paired_ci(ref[:, 1], s[:, 1], level)}
        res['models'][n] = {'spellF1': s[:, 0].tolist(), 'macroF1': s[:, 1].tolist(), 'vs_first': vs}
    return res


def _msd(v):
    v = np.asarray(v, float)
    return v.mean(), (v.std(ddof=1) if v.size > 1 else float('nan'))


def format_compare(res, title=''):
    names = list(res['models'])
    ref = names[0]
    lv = int(round(100 * res['level']))
    lines = [title] if title else []
    lines.append('%-14s %-17s %-17s %-28s %-9s %s' % (
        'model', 'spellF1', 'macroF1', 'd spellF1 vs %s [%d%% CI]' % (ref, lv), 'wins', 'd macroF1'))
    for n in names:
        m = res['models'][n]
        (s, ss), (f, fs) = _msd(m['spellF1']), _msd(m['macroF1'])
        row = '%-14s %.4f ± %.4f   %.4f ± %.4f   ' % (n, s, ss, f, fs)
        v = m['vs_first']
        if v is None:
            row += '%-28s %-9s %s' % ('(mốc)', '-', '-')
        else:
            c = v['spellF1']
            row += '%-28s %-9s %+.4f' % ('%+.4f [%+.4f, %+.4f]' % (c['mean'], c['lo'], c['hi']),
                                         '%d/%d' % (c['wins'], c['n']), v['macroF1']['mean'])
        lines.append(row)
    return '\n'.join(lines)


# ---------------------------------------------------------------- situations

def true_changes(y, half):
    """bool[T]: first frame of a new owner at H<->A changes; carry ownership across D within each half."""
    y = np.asarray(y)
    out = np.zeros(len(y), bool)
    for i, j in segments(np.asarray(half)):
        live = np.flatnonzero(np.isin(y[i:j], (H, A)))
        if live.size > 1:
            v = y[i:j][live]
            out[i + live[1:][v[1:] != v[:-1]]] = True
    return out


def near_change_mask(y, half, w=NEAR):
    """bool[T]: within +-w frames of a true team change, without crossing half boundaries."""
    half = np.asarray(half)
    ch = true_changes(y, half)
    out = np.zeros(len(ch), bool)
    for i, j in segments(half):
        for t in np.flatnonzero(ch[i:j]) + i:
            out[max(i, t - w):min(j, t + w + 1)] = True
    return out


def _moving_mean(x, w):
    """Centered moving average ignoring NaNs; average the available portion at boundaries."""
    h = w // 2
    ok = np.isfinite(x)
    c = np.r_[0.0, np.cumsum(np.where(ok, x, 0.0))]
    k = np.r_[0, np.cumsum(ok)]
    idx = np.arange(len(x))
    lo, hi = np.maximum(idx - h, 0), np.minimum(idx + h + 1, len(x))
    n = k[hi] - k[lo]
    return np.where(n > 0, (c[hi] - c[lo]) / np.maximum(n, 1), np.nan)


def ball_speed(ball_xy, half, fps=FPS, pitch=None, smooth=5):
    """True ball speed (m/s), moving-averaged over `smooth` frames, separately per half.

    If ball_xy is normalized to 0..1, scale by `pitch` dimensions.
    """
    xy = np.asarray(ball_xy, float)
    if pitch is not None and np.nanmax(np.abs(xy)) <= 1.0:
        xy = xy * np.asarray(pitch, float).ravel()[:2]
    spd = np.full(len(xy), np.nan)
    for i, j in segments(np.asarray(half)):
        if j - i < 2:
            continue
        s = np.hypot(*np.diff(xy[i:j], axis=0).T) * float(fps)
        spd[i:j] = _moving_mean(np.r_[s[0], s], smooth)
    return spd


def ball_seen(B):
    """bool[T]: observed ball position exists (B[:,2] > 0 and x is not NaN)."""
    B = np.asarray(B)
    seen = np.isfinite(B[:, 0])
    if B.shape[1] > 2:
        seen &= B[:, 2] > 0
    return seen


def situations(z, near=NEAR, fast=FAST):
    """Situation masks bool[T], based only on ground truth (y, ball_xy) and observed ball B.

    in_play: y is H/A. ball_seen / ball_unseen. near_change / far_from_change: within / outside
    +-`near` frames of a true change. ball_fast / ball_slow: true speed above / below `fast` m/s.
    half_1 / half_2: separate halves. All operations use pipeline.quality.segments without crossing halves.
    """
    y = np.asarray(z['y'])
    half = np.asarray(z['half'])
    fps = float(z['fps']) if 'fps' in z else FPS
    graded = half > 0
    m = {'in_play': graded & np.isin(y, (H, A))}
    if 'B' in z:
        seen = ball_seen(z['B'])
        m['ball_seen'] = graded & seen
        m['ball_unseen'] = graded & ~seen
    nc = near_change_mask(y, half, near)
    m['near_change'] = graded & nc
    m['far_from_change'] = graded & ~nc
    if 'ball_xy' in z:
        spd = ball_speed(z['ball_xy'], half, fps, z['pitch'] if 'pitch' in z else None)
        ok = np.isfinite(spd)
        m['ball_fast'] = graded & ok & (spd > fast)
        m['ball_slow'] = graded & ok & (spd <= fast)
    for h in np.unique(half[graded]):
        m['half_%d' % h] = half == h
    return m


def spell_count(codes, half):
    """Possession count: contiguous H or A runs within each half, matching the scorer."""
    codes = np.asarray(codes)
    n = 0
    for i, j in segments(np.asarray(half)):
        c = codes[i:j]
        live = np.isin(c, (H, A))
        n += int((live & np.r_[True, c[1:] != c[:-1]]).sum())
    return n


def _mean(x):
    x = np.asarray(x)
    return float(x.mean()) if x.size else float('nan')


def breakdown(z, codes=None, team_prob=None, sit=None, near=NEAR, fast=FAST):
    """Situation error table for one realization.

    team_prob: pre-decoding per-frame P(H), choosing H above 0.5. codes: decoded predictions.
    Return dict(rows=[...], rates={...}). Each row, on truly live frames:
      share            fraction of live frames in this situation
      acc_prob         team accuracy from team_prob
      acc_codes        codes == y, including predicted D as errors
      acc_codes_alive  team accuracy where codes is H/A
      err_share        fraction of wrong-team errors falling in this situation
    rates: in_play_to_D, D_to_in_play, wrong_team (by true class), frame_acc, predicted / true
    possession and team-change counts, spellF1, macroF1.
    """
    y = np.asarray(z['y'])
    half = np.asarray(z['half'])
    n_t = len(y)
    sit = situations(z, near, fast) if sit is None else sit
    live = sit['in_play']
    n_live = max(1, int(live.sum()))
    if codes is not None:
        codes = _as_codes(codes, n_t, z.get('name', ''))
        alive = np.isin(codes, (H, A))
        wrong = live & alive & (codes != y)
        n_wrong = max(1, int(wrong.sum()))
    if team_prob is not None:
        team_prob = np.asarray(team_prob, float)
        if team_prob.shape != (n_t,):
            raise ValueError('team_prob có dạng %s, dữ liệu có %d khung' % (team_prob.shape, n_t))
        prob_ok = np.where(team_prob > 0.5, H, A) == y
    rows = []
    for k in ['in_play'] + [k for k in sit if k != 'in_play']:
        mm = live & sit[k]
        r = {'situation': k, 'frames': int(mm.sum()), 'share': mm.sum() / n_live,
             'acc_prob': float('nan'), 'acc_codes': float('nan'), 'acc_codes_alive': float('nan'),
             'err_share': float('nan')}
        if team_prob is not None:
            r['acc_prob'] = _mean(prob_ok[mm])
        if codes is not None:
            r['acc_codes'] = _mean(codes[mm] == y[mm])
            r['acc_codes_alive'] = _mean(codes[mm & alive] == y[mm & alive])
            r['err_share'] = (wrong & sit[k]).sum() / n_wrong
        rows.append(r)
    rates = {}
    if codes is not None:
        from pipeline.score_export import score
        graded = half > 0
        dead = graded & (y == D)
        lab = graded & (y >= 0)
        sf, mf = score(z, codes)
        rates = {'spellF1': sf, 'macroF1': mf,
                 'in_play_to_D': (live & (codes == D)).sum() / n_live,
                 'D_to_in_play': (dead & alive).sum() / max(1, int(dead.sum())),
                 'wrong_team': wrong.sum() / n_live,
                 'frame_acc': _mean(codes[lab] == y[lab]),
                 'n_spells_true': spell_count(y, half), 'n_spells_pred': spell_count(codes, half),
                 'n_changes_true': int(true_changes(y, half).sum()),
                 'n_changes_pred': int(true_changes(codes, half).sum())}
    return {'rows': rows, 'rates': {k: float(v) for k, v in rates.items()}}


def _nanmean(v):
    v = np.asarray(v, float)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return float(np.nanmean(v)) if v.size else float('nan')


def mean_breakdown(bds):
    """Average realization-level breakdown tables, matching rows by situation name."""
    if not bds:
        raise ValueError('danh sách breakdown rỗng')
    names = [r['situation'] for r in bds[0]['rows']]
    rows = []
    for k in names:
        rs = [r for b in bds for r in b['rows'] if r['situation'] == k]
        rows.append({'situation': k, **{f: _nanmean([r[f] for r in rs]) for f in rs[0] if f != 'situation'}})
    rates = {f: _nanmean([b['rates'].get(f, np.nan) for b in bds]) for f in bds[0]['rates']}
    return {'rows': rows, 'rates': rates, 'n': len(bds)}


def format_breakdown(bd, title=''):
    pct = lambda v: '   -  ' if not np.isfinite(v) else '%5.1f%%' % (100 * v)
    lines = [title] if title else []
    lines.append('%-16s %9s %10s %10s %15s %14s' % (
        'situation', '% in-play', 'acc P(H)', 'acc codes', 'acc codes|H/A', '% team errors'))
    for r in bd['rows']:
        lines.append('%-16s %9s %10s %10s %15s %14s' % (
            r['situation'], pct(r['share']), pct(r['acc_prob']), pct(r['acc_codes']),
            pct(r['acc_codes_alive']), pct(r['err_share'])))
    q = bd['rates']
    if q:
        lines.append('spellF1 %.4f  macroF1 %.4f  frame acc %s' % (q['spellF1'], q['macroF1'], pct(q['frame_acc'])))
        lines.append('bóng sống -> D %s | D -> bóng sống %s | sai đội %s (trên khung bóng sống)' % (
            pct(q['in_play_to_D']), pct(q['D_to_in_play']), pct(q['wrong_team'])))
        lines.append('lượt kiểm soát đoán / thật: %.1f / %.1f | lần đổi đội H<->A đoán / thật: %.1f / %.1f' % (
            q['n_spells_pred'], q['n_spells_true'], q['n_changes_pred'], q['n_changes_true']))
    return '\n'.join(lines)


# ---------------------------------------------------------------- oracle experiments

def emissions(q, pD, eps=1e-12):
    """E[T,3] log H/A/D probabilities from q = P(H | live) and pD, normalized as in src.nap1.score."""
    q = np.asarray(q, float)
    pD = np.asarray(pD, float)
    pH = np.clip(q * (1 - pD), eps, 1 - eps)
    pA = np.clip((1 - q) * (1 - pD), eps, 1 - eps)
    pd_ = np.clip(pD, eps, 1 - eps)
    s = pH + pA + pd_
    return np.log(np.stack([pH / s, pA / s, pd_ / s], 1))


def with_true_team(q, y, hi=HI):
    """Replace q on true live frames with the true team (H -> hi, A -> 1-hi); leave other frames unchanged."""
    y = np.asarray(y)
    return np.where(y == H, hi, np.where(y == A, 1 - hi, np.asarray(q, float)))


def with_true_dead(pD, y, hi=HI):
    """Replace pD with true dead-ball status (D -> hi, H/A -> 1-hi); leave unlabelled frames unchanged."""
    y = np.asarray(y)
    return np.where(y == D, hi, np.where(np.isin(y, (H, A)), 1 - hi, np.asarray(pD, float)))


def true_switch_touch(y, half, C=None):
    """(K, C) with true touches at true change frames: K=1 there and C=new team; K=0 elsewhere.

    Retain supplied C on non-change frames, where K=0 makes it irrelevant to the touch gate.
    """
    y = np.asarray(y)
    sw = true_changes(y, half)
    K = sw.astype(float)
    C = np.full((len(y), 2), 0.5) if C is None else np.array(C, float, copy=True)
    C[sw & (y == H)] = (1.0, 0.0)
    C[sw & (y == A)] = (0.0, 1.0)
    return K, C


def decode_codes(E, C, K, z, params=None):
    """Decode (E, C, K) with src.nap1.decode.decode, then X -> D; defaults to fitted nap1 params."""
    from src.nap1.decode import decode, to_eval_codes
    if params is None:
        from src.nap1.params import load_fold_params
        params = load_fold_params(None)
    half = np.asarray(z['half'])
    meta = {'half': half, 'graded': np.asarray(z['graded']) if 'graded' in z else half > 0,
            'fps': int(float(z['fps'])) if 'fps' in z else FPS}
    return to_eval_codes(decode(E, C, K, meta, params))


def oracle_suite(z, q, pD, C, K, params=None):
    """Partial-oracle variants, decoded again: {name: codes}.

    base        unchanged model
    true_team   true team on live frames (E)
    true_dead   true dead-ball status (E)
    true_switch true touches at true change frames (K, C)
    """
    y = np.asarray(z['y'])
    Ks, Cs = true_switch_touch(y, z['half'], C)
    run = lambda q_, d_, C_, K_: decode_codes(emissions(q_, d_), C_, K_, z, params)
    return {'base': run(q, pD, C, K),
            'true_team': run(with_true_team(q, y), pD, C, K),
            'true_dead': run(q, with_true_dead(pD, y), C, K),
            'true_switch': run(q, pD, Cs, Ks)}


# ---------------------------------------------------------------- command line

def _parse_pred(spec):
    name, sep, path = spec.partition('=')
    if not sep:
        path = spec
        name = 'baseline' if spec == BASELINE else (Path(spec).name or spec)
    return name, path


def _preds_for(path, files):
    if path == BASELINE:
        return baseline_preds(files)
    return load_pred_dir(path, files)


def _files_or_exit(ap, a):
    if a.subset == 'test':
        if not a.final:
            ap.error('--subset test bị từ chối: test chỉ dùng MỘT LẦN, thêm --final nếu đây đúng là lần đó')
        print('CẢNH BÁO: đang chấm tập test (dùng một lần). Không chỉnh mô hình theo kết quả này.', file=sys.stderr)
    elif a.final:
        print('(--final chỉ có tác dụng với --subset test)', file=sys.stderr)
    try:
        return data_files(a.subset, a.data_root, final=a.final)
    except (FileNotFoundError, PermissionError, ValueError) as e:
        ap.error(str(e))


def _title(subset, files):
    return '%s: %d bản sao (%s) - %s' % (FOLDERS[subset], len(files), ', '.join(Path(f).stem for f in files),
                                       NOTE[subset])


def _cmd_compare(ap, a):
    files = _files_or_exit(ap, a)
    preds = {}
    for spec in a.pred:
        name, path = _parse_pred(spec)
        if name in preds:
            ap.error('trùng tên mô hình %r' % name)
        preds[name] = _preds_for(path, files)
    res = compare(files, preds, a.level)
    print(format_compare(res, _title(a.subset, files)))
    if a.per_copy:
        print('\nspellF1 từng bản sao:')
        for i, f in enumerate(res['files']):
            print('  %-24s ' % Path(f).stem + '  '.join('%s %.4f' % (n, m['spellF1'][i])
                                                        for n, m in res['models'].items()))
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps({'subset': a.subset, **res}, indent=1))
    return 0


def _cmd_breakdown(ap, a):
    files = _files_or_exit(ap, a)
    out = {}
    for spec in a.pred:
        name, path = _parse_pred(spec)
        base = baseline_preds(files) if path == BASELINE else None
        per = []
        for f in files:
            z = load_eval(f)
            pred = {'codes': base[f.name]} if base is not None else load_pred(path, f.name)
            bd = breakdown(z, pred['codes'], pred.get('team_prob'), near=a.near, fast=a.fast)
            per.append(bd)
            if a.per_copy:
                print(format_breakdown(bd, '%s / %s' % (name, f.stem)) + '\n')
        bd = mean_breakdown(per)
        print(format_breakdown(bd, '%s\nmô hình %s, trung bình trên các bản sao' % (
            _title(a.subset, files), name)) + '\n')
        out[name] = {'mean': bd, 'per_copy': per}
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps({'subset': a.subset, 'files': [f.name for f in files], 'models': out},
                                           indent=1, default=float))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m src.core.evaltools', description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for cmd, helptext in (('compare', 'so nhiều mô hình theo cặp trên cùng các bản sao'),
                          ('breakdown', 'bảng lỗi theo tình huống cho từng mô hình')):
        p = sub.add_parser(cmd, help=helptext)
        p.add_argument('--subset', choices=sorted(FOLDERS), default='select')
        p.add_argument('--pred', action='append', required=True,
                       help='NAME=DIR (thư mục <tên file>.npz chứa codes) hoặc NAME=@baseline; lặp lại được')
        p.add_argument('--final', action='store_true', help='bắt buộc khi --subset test (chỉ một lần)')
        p.add_argument('--data-root', default=None, help='thư mục thay cho train_data/ (để test)')
        p.add_argument('--per-copy', action='store_true', help='in thêm từng bản sao')
        p.add_argument('--json', default=None, help='ghi kết quả ra file JSON')
        if cmd == 'compare':
            p.add_argument('--level', type=float, default=0.95, help='mức tin cậy của khoảng t')
        else:
            p.add_argument('--near', type=int, default=NEAR, help='số khung quanh lần đổi đội thật')
            p.add_argument('--fast', type=float, default=FAST, help='ngưỡng m/s của bóng đang bay')
    a = ap.parse_args(argv)
    return {'compare': _cmd_compare, 'breakdown': _cmd_breakdown}[a.cmd](ap, a)


if __name__ == '__main__':
    raise SystemExit(main())
