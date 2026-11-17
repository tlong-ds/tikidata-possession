"""Additional stage-3 switchnet inputs, appended AFTER the stage-2 channels.

    xp [T,46,9+2]  stage-2 channels 0..8, followed by:
                   9  elapsed continuous presence
                   10 remaining continuous presence
    xf [T,7+14]    stage-2 channels 0..6, followed by:
                   7 time since ball seen; 8 time until seen again; 9 gap duration
                   10..12 stage-1 B_hat source (one-hot: observed / Kalman / camera)
                   13,14 H/A players vanished in the past 10 s, divided by 11
                   15,16 mean elapsed absence of those players, H/A
                   17,18 H/A players reappearing in the next 10 s, divided by 11
                   19,20 mean time until those players reappear, H/A

Encode durations as log1p(min(seconds, 60)) / log1p(60), in [0,1]. Compute separately per half;
outside-half values are 0. Future observations are allowed because evaluation is offline.

Vanishing players are derived from TEAM COUNTS, not slots: a slot is a track fragment that may
be split and reassigned within its team. On each count decrease, record an absence and its time;
on increase, match the most recent absence (last in, first out).

Team-swap augmentation swaps TEAM_PAIRS_XF. Time reversal swaps REVERSE_PAIRS_* (past <-> future).
Other additional columns are independent of pitch direction and team.
"""

import numpy as np

from src.nap1.ball import estimate_ball
from src.nap2 import shared
from src.nap2.features import N_FRAME_FEAT, N_PLAYER_FEAT, build_features, targets_from_arrays
from src.nap2.shared import runs_of

CAP_S = 60.0                 # Cap every time interval at 60 s.
RECENT_S = 10.0              # "Recently vanished" / "soon reappearing" means within 10 s.
FEAT_VERSION = "nap3-feats-1"

# Additional column indices
XP_AGE, XP_LEFT = 9, 10
XF_SINCE, XF_UNTIL, XF_GAP = 7, 8, 9
XF_SRC = (10, 11, 12)
XF_LOST_N, XF_LOST_T = (13, 14), (15, 16)
XF_SOON_N, XF_SOON_T = (17, 18), (19, 20)
N_XP_EXTRA, N_XF_EXTRA = 2, 14
N_PLAYER_FEAT3 = N_PLAYER_FEAT + N_XP_EXTRA      # 11
N_FRAME_FEAT3 = N_FRAME_FEAT + N_XF_EXTRA        # 21

# Column names for checkpoint input alignment; see src.steps.Layout.
XP_NAMES = ("x", "y", "vx", "vy", "group", "present", "dx", "dy", "dist", "age", "left")
XF_NAMES = ("ball_x", "ball_y", "ball_seen", "ball_sigma", "cam_x", "cam_y", "cam_pan",
            "since_ball", "until_ball", "ball_gap", "src_seen", "src_kalman", "src_camera",
            "lost_h", "lost_a", "lost_t_h", "lost_t_a", "soon_h", "soon_a", "soon_t_h", "soon_t_a")
NAP2_XP, NAP2_XF = XP_NAMES[:N_PLAYER_FEAT], XF_NAMES[:N_FRAME_FEAT]

TEAM_PAIRS_XF = ((13, 14), (15, 16), (17, 18), (19, 20))
REVERSE_PAIRS_XF = ((XF_SINCE, XF_UNTIL), (13, 17), (14, 18), (15, 19), (16, 20))
REVERSE_PAIRS_XP = ((XP_AGE, XP_LEFT),)
REVERSE_NEGATE_XP = (2, 3)   # Negate vx, vy when reversing time.
SRC_CODES = (0, 1, 3)        # src.nap1.ball source flags for B_hat: 0 observed, 1 Kalman, 3 camera.


def enc_time(sec):
    """Seconds -> log1p(min(seconds, 60)) / log1p(60), in [0,1]."""
    s = np.clip(np.asarray(sec, dtype=np.float64), 0.0, CAP_S)
    return np.log1p(s) / np.log1p(CAP_S)


def _since(seen):
    """Frames since last seen (0 if visible, -1 if never seen); seen [n] or [n,S]."""
    n = seen.shape[0]
    idx = np.arange(n).reshape((n,) + (1,) * (seen.ndim - 1))
    last = np.maximum.accumulate(np.where(seen, idx, -1), axis=0)
    return np.where(last >= 0, idx - last, -1)


def ball_timing(seen, fps):
    """(Time since seen, time until seen, gap duration) in seconds for ONE half."""
    seen = np.asarray(seen, bool)
    cap_f = CAP_S * fps
    s = _since(seen)
    u = _since(seen[::-1])[::-1]
    since = np.where(s >= 0, s, cap_f) / fps
    until = np.where(u >= 0, u, cap_f) / fps
    gap = np.where((s >= 0) & (u >= 0), s + u - 1, cap_f) / fps
    gap[seen] = 0.0
    return since, until, gap


def run_age(present, fps):
    """(Elapsed continuous presence, remaining presence) in seconds for ONE half, present [n,S].
    Boundary-touching runs are treated as long (capped at 60 s): their true start/end is unknown.
    """
    m = np.asarray(present, bool)
    n = m.shape[0]
    cap_f = CAP_S * fps

    def age(mm):
        idx = np.arange(n)[:, None]
        start = mm & ~np.vstack([np.zeros((1, mm.shape[1]), bool), mm[:-1]])
        st = np.maximum.accumulate(np.where(start, idx, -1), axis=0)
        a = np.where(st == 0, cap_f, idx - st).astype(np.float64)
        return np.where(mm, a, 0.0) / fps

    return age(m), age(m[::-1])[::-1]


def lost_units(n_present):
    """Match count decreases to the next count increases, last in, first out.
    Return [k,2] (disappearance frame, reappearance frame); unmatched reappearance = len.
    """
    n = np.asarray(n_present, dtype=np.int64)
    stack, out = [], []
    if len(n) > 1:
        d = np.diff(n)
        for t in np.flatnonzero(d) + 1:
            k = int(d[t - 1])
            if k < 0:
                stack.extend([int(t)] * (-k))
            else:
                for _ in range(min(k, len(stack))):
                    out.append((stack.pop(), int(t)))
    out += [(u, len(n)) for u in stack]
    return np.asarray(out, dtype=np.int64).reshape(-1, 2)


def missing_stats(n_present, fps, recent_s=RECENT_S):
    """(Count vanished in recent_s seconds, mean elapsed absence in seconds), per frame of ONE half."""
    n = len(n_present)
    units = lost_units(n_present)
    R, C = int(round(recent_s * fps)), int(round(CAP_S * fps))
    cnt_recent = np.zeros(n + 1)
    active = np.zeros(n + 1)
    a = np.zeros(n + 1)            # Coefficient of t
    b = np.zeros(n + 1)            # Constant term
    for u, p in units:
        np.add.at(cnt_recent, [u, min(p, u + R)], [1, -1])
        np.add.at(active, [u, p], [1, -1])
        e1 = min(p, u + C)
        np.add.at(a, [u, e1], [1, -1])
        np.add.at(b, [u, e1], [-u, u])
        if e1 < p:
            np.add.at(b, [e1, p], [C, -C])
    t = np.arange(n)
    cnt = np.cumsum(cnt_recent)[:n]
    act = np.cumsum(active)[:n]
    tot = np.cumsum(a)[:n] * t + np.cumsum(b)[:n]
    mean_s = np.where(act > 0, tot / np.maximum(act, 1), 0.0) / fps
    return cnt, mean_s, act


def extra_features(V, Bhat):
    """Additional stage-3 channels: {'xp': [T,46,2], 'xf': [T,14]}, float32, on the nap2 frame grid."""
    P = V.P
    T, S = P.shape[:2]
    fps = float(V.meta["fps"])
    half = np.asarray(V.meta["half"])
    pres = P[..., 5] > 0
    xp = np.zeros((T, S, N_XP_EXTRA), np.float32)
    xf = np.zeros((T, N_XF_EXTRA), np.float32)
    grp = np.zeros(S, bool)
    grp[24:] = True
    for a, b, h in runs_of(half):
        if h <= 0:
            continue
        seen = np.asarray(V.B[a:b, 2] > 0)
        since, until, gap = ball_timing(seen, fps)
        xf[a:b, 0] = enc_time(since)
        xf[a:b, 1] = enc_time(until)
        xf[a:b, 2] = np.where(seen, 0.0, enc_time(gap))
        src = np.round(Bhat[a:b, 3]).astype(int)
        for j, code in enumerate(SRC_CODES):
            xf[a:b, 3 + j] = src == code
        age, left = run_age(pres[a:b], fps)
        xp[a:b, :, 0] = np.where(pres[a:b], enc_time(age), 0.0)
        xp[a:b, :, 1] = np.where(pres[a:b], enc_time(left), 0.0)
        for k, g in enumerate((~grp, grp)):
            n_k = pres[a:b][:, g].sum(1)
            cnt, mean_s, act = missing_stats(n_k, fps)
            xf[a:b, 6 + k] = np.clip(cnt / 11.0, 0, 1.5)
            xf[a:b, 8 + k] = np.where(act > 0, enc_time(mean_s), 0.0)
            cnt_f, mean_f, act_f = missing_stats(n_k[::-1], fps)
            xf[a:b, 10 + k] = np.clip(cnt_f[::-1] / 11.0, 0, 1.5)
            xf[a:b, 12 + k] = np.where(act_f[::-1] > 0, enc_time(mean_f[::-1]), 0.0)
    return {"xp": xp, "xf": xf}


def build_all(V, Bhat):
    """Stage-2 features + extra channels: {'xp': [T,46,11], 'xf': [T,21]}, float32."""
    f = build_features(V, Bhat)
    ex = extra_features(V, Bhat)
    return {"xp": np.concatenate([f["xp"], ex["xp"]], -1), "xf": np.concatenate([f["xf"], ex["xf"]], -1)}


_LAYOUTS = {}


def get_layout(layout=None):
    """None | Layout | Layout.to_dict() | feature-spec list -> Layout, cached by spec."""
    from src.steps import Layout
    if isinstance(layout, Layout):
        return layout
    if layout is None:
        layout = []
    d = layout if isinstance(layout, dict) else {"features": list(layout)}
    key = repr(d.get("features", []))
    if key not in _LAYOUTS:
        _LAYOUTS[key] = Layout.from_dict(d)
    return _LAYOUTS[key]


def layout_spec(layout):
    """Representation suitable for sending to subprocesses."""
    return None if layout is None else get_layout(layout).to_dict()


def prepare_raw(raw, with_targets=True, layout=None):
    """src.nap2.shared.load_file / fresh_train dict -> arrays ready for windowing or inference,
    as in src.nap2.dataset.prepare_raw, plus stage-3 channels and layout feature-step columns.
    """
    lay = get_layout(layout)
    raw = lay.preprocess(raw)
    V = shared.as_view(raw)
    Bhat, K = estimate_ball(V, shared.nap1_params())
    f = build_all(V, Bhat)
    if lay.steps:
        ex = lay.extra(V, Bhat, raw)
        f = {"xp": np.concatenate([f["xp"], ex["xp"]], -1), "xf": np.concatenate([f["xf"], ex["xf"]], -1)}
    d = {"xp": f["xp"].astype(np.float16), "xf": f["xf"].astype(np.float32),
         "half": np.asarray(V.meta["half"]), "graded": np.asarray(V.meta["graded"], bool),
         "K_rule": K.astype(np.float32), "name": raw.get("name")}
    if with_targets and "y" in raw:
        d.update(targets_from_arrays(raw["y"], raw.get("touch_true"), raw["ball_xy"], raw["half"],
                                     V.meta["fps"], w=raw.get("w")))
        d["y_raw"] = np.asarray(raw["y"])
    for k in ("touch_true", "baseline"):
        if k in raw:
            d[k] = raw[k]
    return d


def prepare_file(path, max_frames=None, with_targets=True, layout=None):
    """Shared-data .npz file -> the same result as prepare_raw."""
    return prepare_raw(shared.load_file(path, max_frames), with_targets, layout)


def prepare_fresh(name, epoch, run, max_frames=None, layout=None, smooth=None):
    """NEW training-match degradation (feed.train_view, release v3, with the 'train' guard).
    smooth: [train.smooth_ball] configuration for target-like balls (src.core.ball_smooth).
    """
    raw = shared.fresh_train(name, epoch, run, max_frames)
    if smooth:
        from .ball_smooth import maybe_smooth
        raw, _ = maybe_smooth(raw, smooth, name, epoch, run)
    return prepare_raw(raw, layout=layout)


def prepare_item(item, max_frames=None, layout=None):
    """item: file path or ("fresh", match, epoch, run[, smooth]). Subprocess-compatible;
    layout is passed as the layout_spec dict.
    """
    if isinstance(item, (tuple, list)) and item and item[0] == "fresh":
        return prepare_fresh(item[1], item[2], item[3], max_frames, layout, item[4] if len(item) > 4 else None)
    return prepare_file(item, max_frames, layout=layout)
