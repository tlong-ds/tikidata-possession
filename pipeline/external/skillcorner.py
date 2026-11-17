"""SkillCorner opendata (A-League 2024/25, 10 fps broadcast tracking, MIT) -> cache/external/SKC-<id>.npz.

Raw data: train_data/raw/skillcorner-opendata (git clone github.com/SkillCorner/opendata; tracking
uses Git LFS, downloaded through media.githubusercontent.com and SHA-256-checked against LFS pointers).
Each data/matches/<id>/ has <id>_match.json (players, teams, periods, pitch dimensions),
<id>_tracking_extrapolated.jsonl and <id>_dynamic_events.csv.

Conversion:
  * 10 fps -> 25 fps by linear interpolation (gaps > 0.25 s become NaN). Centered metres ->
    Metrica / PFF normalization: x = (x + L/2) / L, y = (W/2 - y) / W. Speed: NaN.
  * Frames without tracking (camera cuts, ~27%): player / ball NaN, label -1 (excluded from learning).
  * Labels from player_possession events, then metrica.event_labels as in Metrica / PFF:
      - Each possession: team action from frame_start.
      - Throw-in / free kick / goal kick / corner start (*_reception: receiving team;
        *_interception: opposite team restarts): SET PIECE 1 s before frame_start; event_labels
        marks dead ball from the preceding last touch.
      - start_type 'unknown' near the center (|x|, |y| < 3 m) or first possession of a half:
        kickoff (SET PIECE).
      - end_type foul_suffered: whistle (FAULT RECEIVED) at frame_end.
      - End of half: END HALF at the period's end_frame.
  * Ball: metrica.dfl_like_ball, as in Metrica / PFF.

    python -m pipeline.external.skillcorner --ensure [--workers 4]
"""

import argparse
import csv
import json
import os
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ..schema import CACHE_DIR, N_SLOTS, N_TEAM1, RAW_DIR, save_match, validate_match
from . import metrica
from .pff import _interp
from .registry import SKILLCORNER

FPS = 25
SRC_FPS = 10
CONVERTER_VERSION = "skc-convert-v1"
LABEL_VERSION = "skc-events-v1"
RESTART_PREFIX = ("throw_in", "free_kick", "goal_kick", "corner")
RESTART_LEAD_S = 1.0
KICKOFF_R = 3.0


def raw_root():
    return os.environ.get("SKILLCORNER_RAW") or os.path.join(RAW_DIR, "skillcorner-opendata", "data", "matches")


def read_tracking(path, team_of, home_id, L, W):
    """-> t[N] (seconds, frame / 10), per[N], P_src[N,46,2] (NaN), ball[N,2], has[N] (tracking available)."""
    frames = []
    with open(path) as f:
        for line in f:
            frames.append(json.loads(line))
    N = max(d["frame"] for d in frames) + 1
    t = np.arange(N) / SRC_FPS
    per = np.zeros(N, np.int8)
    has = np.zeros(N, bool)
    ball = np.full((N, 2), np.nan)
    pos = {}
    for d in frames:
        i = d["frame"]
        if d.get("period") in (1, 2):
            per[i] = d["period"]
        pd_ = d.get("player_data") or []
        if pd_:
            has[i] = True
        for p in pd_:
            if p.get("x") is None or p.get("y") is None:
                continue
            pos.setdefault(int(p["player_id"]), {})[i] = ((p["x"] + L / 2) / L, (W / 2 - p["y"]) / W)
        b = d.get("ball_data") or {}
        if b.get("x") is not None and b.get("y") is not None:
            ball[i] = ((b["x"] + L / 2) / L, (W / 2 - b["y"]) / W)
    return t, per, has, pos, ball


def build_grid(per):
    """Per-half 25 fps grids: [(p, i0, i1, off, n)], where i0..i1 are the original 10 fps frames of half p."""
    grids, off = [], 0
    for p in (1, 2):
        idx = np.flatnonzero(per == p)
        if len(idx) < 2:
            continue
        i0, i1 = int(idx[0]), int(idx[-1])
        n = int(np.floor((i1 - i0) / SRC_FPS * FPS)) + 1
        grids.append((p, i0, i1, off, n))
        off += n
    return grids


def to_grid(grids, frame):
    """Original frame -> grid frame in its half; None outside the halves."""
    for p, i0, i1, off, n in grids:
        if i0 <= frame <= i1 + SRC_FPS:
            return off + min(n - 1, max(0, int(round((frame - i0) / SRC_FPS * FPS))))
    return None


def resample(t, per, has, pos, ball, team_of, home_id, grids):
    T = sum(g[4] for g in grids)
    tg = np.concatenate([t[i0] + np.arange(n) / FPS for p, i0, i1, off, n in grids])
    period = np.concatenate([np.full(n, p, np.int8) for p, i0, i1, off, n in grids])
    hasg = np.zeros(T, bool)
    for p, i0, i1, off, n in grids:
        src = np.arange(i0, i1 + 1)
        k = np.clip(np.round(tg[off:off + n] * SRC_FPS).astype(int), i0, i1)
        hasg[off:off + n] = has[k]
    tracks = {}
    for pid, d in pos.items():
        fr = np.array(sorted(d))
        v = np.array([d[f] for f in fr])
        tr = np.full((T, 2), np.nan)
        for p, i0, i1, off, n in grids:
            m = (fr >= i0) & (fr <= i1)
            if m.sum() >= 2:
                tr[off:off + n] = _interp(tg[off:off + n], t[fr[m]], v[m])
        if np.isfinite(tr[:, 0]).any():
            tracks[pid] = tr
    P = np.full((T, N_SLOTS, 3), np.nan, np.float32)
    for side, base, n_slots in ((0, 0, N_TEAM1), (1, N_TEAM1, N_SLOTS - N_TEAM1)):
        ks = [k for k in tracks if (team_of.get(k) == home_id) == (side == 0) and team_of.get(k) is not None]
        span = {}
        for k in ks:
            ok = np.flatnonzero(np.isfinite(tracks[k][:, 0]))
            span[k] = (int(ok[0]), int(ok[-1]))
        ks.sort(key=lambda k: (span[k][0], k))
        for k, s in zip(ks, metrica.first_fit([span[k] for k in ks], n_slots)):
            P[:, base + s, :2] = tracks[k]
    B = np.full((T, 3), np.nan, np.float32)
    for p, i0, i1, off, n in grids:
        fr = np.arange(i0, i1 + 1)
        m = np.isfinite(ball[fr, 0])
        if m.sum() >= 2:
            B[off:off + n, :2] = _interp(tg[off:off + n], t[fr[m]], ball[fr[m]])
    P[~hasg] = np.nan
    B[~hasg] = np.nan
    return P, B, period, hasg


def read_events(path, grids, home_id):
    rows = []
    first_in_period = set()
    lead = int(round(RESTART_LEAD_S * FPS))

    def add(team, typ, sub, period, sf, ef=None):
        rows.append({"team": team, "type": typ, "subtype": sub, "period": period, "start_frame": sf,
                     "end_frame": max(sf, ef if ef is not None else sf), "start_x": np.nan, "start_y": np.nan,
                     "end_x": np.nan, "end_y": np.nan})

    evs = [x for x in csv.DictReader(open(path)) if x["event_type"] == "player_possession"]
    evs.sort(key=lambda x: int(x["frame_start"]))
    for x in evs:
        try:
            p = int(x["period"])
            team = 1 if int(x["team_id"]) == home_id else 2
        except (TypeError, ValueError):
            continue
        sf, ef = to_grid(grids, int(x["frame_start"])), to_grid(grids, int(x["frame_end"]))
        if sf is None:
            continue
        ef = ef if ef is not None else sf
        st, et = x.get("start_type") or "", x.get("end_type") or ""
        other = 3 - team
        try:
            xs, ys = float(x["x_start"]), float(x["y_start"])
        except (TypeError, ValueError):
            xs = ys = np.nan
        if st.startswith(RESTART_PREFIX):
            kicker = team if st.endswith("_reception") else other
            add(kicker, "SET PIECE", "SKC-" + st, p, max(0, sf - lead))
        elif p not in first_in_period or (st == "unknown" and abs(xs) < KICKOFF_R and abs(ys) < KICKOFF_R):
            add(team, "SET PIECE", "SKC-KICKOFF", p, sf)
        first_in_period.add(p)
        add(team, "PASS" if et != "shot" else "SHOT", "SKC-%s-%s" % (st, et), p, sf, ef)
        if et == "foul_suffered":
            add(team, "FAULT RECEIVED", "SKC-FOUL", p, ef)
    for p, i0, i1, off, n in grids:
        add(0, "END", "END HALF", p, off + n - 1)
    rows.sort(key=lambda r: r["start_frame"])
    return metrica._event_table(rows)


def load_game(gid, raw=None, ball_params=None):
    d = os.path.join(raw or raw_root(), str(gid))
    meta = json.load(open(os.path.join(d, "%d_match.json" % gid)))
    L, W = float(meta.get("pitch_length") or 105), float(meta.get("pitch_width") or 68)
    home_id = int(meta["home_team"]["id"])
    team_of = {int(p["id"]): int(p["team_id"]) for p in meta.get("players") or []}
    t, per, has, pos, ball = read_tracking(os.path.join(d, "%d_tracking_extrapolated.jsonl" % gid), team_of, home_id, L, W)
    grids = build_grid(per)
    P, B, period, hasg = resample(t, per, has, pos, ball, team_of, home_id, grids)
    ev = read_events(os.path.join(d, "%d_dynamic_events.csv" % gid), grids, home_id)
    name = "SKC-%d" % gid
    m = metrica.build_match(P, B, period, ev, name, "dfl", raw_name=str(gid))
    y = m.extras["y_true"]
    y[(m.half > 0) & ~hasg] = -1                               # No tracking: exclude from learning.
    m.meta.update({"provider": "skillcorner", "pitch": [L, W], "provider_game": str(gid),
                   "coords": "chuẩn hoá 0-1 từ mét SkillCorner (gốc tâm sân): x=(x+L/2)/L, y=(W/2-y)/W",
                   "teams": {"group1": meta["home_team"]["name"], "group2": meta["away_team"]["name"]},
                   "label_source": "events (%s)" % LABEL_VERSION, "source_fps": SRC_FPS,
                   "no_tracking_frames": int(((m.half > 0) & ~hasg).sum())})
    bp = dict(metrica.BALL_DFL, **(ball_params or {}))
    seed = zlib.crc32(name.encode())
    m.B = metrica.dfl_like_ball(m.B, m.half, y, FPS, seed=seed, **bp)
    m.B[~hasg] = np.nan
    m.meta["ball_processing"] = {"kind": "dfl_like_ball", "seed": seed, **bp}
    m.meta["converter_version"] = CONVERTER_VERSION
    return m


def cache_file(name, cache_dir=None):
    return os.path.join(cache_dir or CACHE_DIR, metrica.CACHE_SUB, name + ".npz")


def _cache_ok(path):
    if not os.path.isfile(path):
        return False
    try:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(bytes(z["meta_json"]).decode("utf8"))
    except Exception:
        return False
    return meta.get("converter_version") == CONVERTER_VERSION


def _convert(args):
    name, path = args
    m = load_game(int(name.split("-")[1]))
    errs = validate_match(m)
    if errs:
        raise ValueError("%s không hợp lệ: %s" % (name, errs))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_match(m, path)
    y = m.extras["y_true"]
    inh = m.half > 0
    li = m.meta["label_info"]
    return "skc: %s T=%d no-track %.0f%% live %.0f%% (of tracked) stops %d restarts %d" % (
        name, m.n_frames, 100 * np.mean(y[inh] == -1), 100 * np.mean((y[inh & (y >= 0)] < 2)),
        li["stops"] + li["implicit_stops"], li["restarts"])


def ensure_cache(names=None, cache_dir=None, workers=1, log=print):
    names = list(names or SKILLCORNER)
    bad = [n for n in names if n not in SKILLCORNER]
    if bad:
        raise ValueError("không phải trận SkillCorner: %s" % bad)
    out = {n: cache_file(n, cache_dir) for n in names}
    todo = [(n, out[n]) for n in names if not _cache_ok(out[n])]
    if todo:
        if not os.path.isdir(raw_root()):
            raise FileNotFoundError("thiếu cache %s và không có dữ liệu gốc SkillCorner ở %s"
                                    % ([n for n, _ in todo], raw_root()))
        if workers > 1:
            with ProcessPoolExecutor(workers) as ex:
                for msg in ex.map(_convert, todo):
                    log(msg)
        else:
            for t in todo:
                log(_convert(t))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ensure", action="store_true")
    ap.add_argument("--games", nargs="+", default=SKILLCORNER)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    for n, p in ensure_cache(a.games, workers=a.workers).items():
        print(n, p, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
