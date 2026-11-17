"""PFF FC FIFA World Cup 2022 (.jsonl.bz2 tracking + .json events) -> DFL-compatible caches.

Raw data (obtained through PFF's form; not redistributed): train_data/raw/pff/FIFA World Cup 2022/
  Tracking Data/<id>.jsonl.bz2, Event Data/<id>.json, Metadata/<id>.json.

Conversion reuses pipeline.external.metrica for identical label and ball conventions:
  * ~29.97 fps tracking in centered metres -> 25 fps grid by videoTimeMs, with linear
    interpolation (gaps > 0.25 s become NaN). Use PFF smoothed positions (homePlayersSmoothed /
    ballsSmoothed, falling back to balls). Only halves 1 and 2; omit extra time and shootouts.
    Build a grid per half, then concatenate.
  * Players keyed by (team, jersey), packed first-fit as in io_dfl / Metrica (home 0..23, away 24..45).
  * Normalize x = (x + L/2) / L and y = (W/2 - y) / W (0 = top, as in DFL / Metrica). Speed: NaN.
  * PFF events -> Metrica-style table -> metrica.event_labels / build_match:
      setpieceType K T G C F P, including FIRSTKICKOFF / SECONDKICKOFF -> that team's SET PIECE;
      OUT (ball out, whistle W, goal H / A) -> BALL OUT (dead from that frame);
      END -> END HALF; PA / CR / CL -> PASS, SH -> SHOT, BC -> CARRY, IT -> RECOVERY (owning team);
      CH: challengeWinnerPlayerId, with team from Rosters/<id>.json -> winner's RECOVERY;
      unknown winner -> CHALLENGE. RE (rebound), TC (touch) -> CHALLENGE (retain ownership,
      matching the DFL loose-ball convention). Skip SUB and events without a team.
  * Ball: metrica.dfl_like_ball (interpolate live, hold dead, round to 1 cm), as in Metrica.

    python -m pipeline.external.pff --ensure [--games PFF-3812 ...] [--workers 8]
"""

import argparse
import bz2
import json
import os
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ..schema import CACHE_DIR, N_SLOTS, N_TEAM1, RAW_DIR, save_match, validate_match
from . import metrica
from .registry import PFF, PFF_IDS

FPS = 25
CONVERTER_VERSION = "pff-convert-v2"
LABEL_VERSION = "pff-events-v2"
MAX_GAP_S = 0.25
RESTART_SETPIECES = set("KTGCFP")
POSS_MAP = {"PA": "PASS", "CR": "PASS", "CL": "PASS", "SH": "SHOT", "BC": "CARRY", "IT": "RECOVERY",
            "CH": "CHALLENGE", "RE": "CHALLENGE", "TC": "CHALLENGE"}


def raw_root():
    return os.environ.get("PFF_RAW") or os.path.join(RAW_DIR, "pff", "FIFA World Cup 2022")


def _players(lst):
    return [(str(p.get("jerseyNum")), p.get("x"), p.get("y")) for p in (lst or [])
            if p and p.get("x") is not None and p.get("y") is not None]


def read_tracking(path):
    """-> t[N] (video seconds), period[N], rows of (frame_i, team(0/1), jersey, x, y), ball[N,2] (NaN if missing)."""
    ts, per, ball, rows = [], [], [], []
    last_t = None
    with bz2.open(path, "rt") as f:
        for line in f:
            d = json.loads(line)
            p = d.get("period")
            if p not in (1, 2) or d.get("videoTimeMs") is None:
                continue
            t = float(d["videoTimeMs"]) / 1000.0
            if last_t is not None and t <= last_t:              # Skip duplicate / decreasing timestamps.
                continue
            last_t = t
            i = len(ts)
            ts.append(t)
            per.append(p)
            for team, key in ((0, "homePlayers"), (1, "awayPlayers")):
                lst = d.get(key + "Smoothed")
                if lst is None:
                    lst = d.get(key)
                for j, x, y in _players(lst):
                    rows.append((i, team, j, float(x), float(y)))
            b = d.get("ballsSmoothed")
            if not (isinstance(b, dict) and b.get("x") is not None):
                bl = d.get("balls") or []
                b = bl[0] if bl and bl[0] and bl[0].get("x") is not None else None
            ball.append((float(b["x"]), float(b["y"])) if b else (np.nan, np.nan))
    return np.array(ts), np.array(per, np.int8), rows, np.array(ball, np.float64).reshape(-1, 2)


def _interp(tg, ts, v, max_gap=MAX_GAP_S):
    """Interpolate v at increasing ts onto tg; NaN outside the range or when bracketing samples are > max_gap apart."""
    out = np.full((len(tg),) + v.shape[1:], np.nan)
    if len(ts) < 2:
        return out
    k = np.searchsorted(ts, tg)
    ok = (k > 0) & (k < len(ts))
    k1 = np.clip(k, 1, len(ts) - 1)
    t0, t1 = ts[k1 - 1], ts[k1]
    ok &= (t1 - t0) <= max_gap
    w = np.where(t1 > t0, (tg - t0) / np.where(t1 > t0, t1 - t0, 1), 0.0)
    w = w.reshape((-1,) + (1,) * (v.ndim - 1))
    val = v[k1 - 1] * (1 - w) + v[k1] * w
    out[ok] = val[ok]
    exact = np.isin(tg, ts)                                      # Match the final sample exactly.
    if exact.any():
        out[exact] = v[np.searchsorted(ts, tg[exact])]
    return out


def resample(ts, per, rows, ball, L, W):
    """-> P[T,46,3] normalized to 0-1, B[T,3], period[T], and per-half grid[(t0, offset, n)]."""
    grids, chunks = [], []
    off = 0
    for p in (1, 2):
        idx = np.flatnonzero(per == p)
        if len(idx) < 2:
            continue
        t0, t1 = ts[idx[0]], ts[idx[-1]]
        n = int(np.floor((t1 - t0) * FPS)) + 1
        grids.append((p, t0, off, n))
        chunks.append(t0 + np.arange(n) / FPS)
        off += n
    tg = np.concatenate(chunks)
    T = len(tg)
    period = np.zeros(T, np.int8)
    for p, _, o, n in grids:
        period[o:o + n] = p
    # Players keyed by (team, jersey number).
    arr = np.array([(r[0], r[1]) for r in rows], np.int64).reshape(-1, 2)
    keys = [(r[1], r[2]) for r in rows]
    xs = np.array([r[3] for r in rows]); ys = np.array([r[4] for r in rows])
    by = {}
    for n_, k in enumerate(keys):
        by.setdefault(k, []).append(n_)
    tracks = {}
    for k, ids in by.items():
        ids = np.array(ids)
        fi = arr[ids, 0]
        v = np.stack([(xs[ids] + L / 2) / L, (W / 2 - ys[ids]) / W], 1)
        tr = np.full((T, 2), np.nan)
        for p, t0, o, n in grids:                               # Interpolate each half separately.
            m = per[fi] == p
            if m.sum() >= 2:
                tr[o:o + n] = _interp(tg[o:o + n], ts[fi[m]], v[m])
        if np.isfinite(tr[:, 0]).any():
            tracks[k] = tr
    P = np.full((T, N_SLOTS, 3), np.nan, np.float32)
    for team, base, n_slots in ((0, 0, N_TEAM1), (1, N_TEAM1, N_SLOTS - N_TEAM1)):
        ks = [k for k in tracks if k[0] == team]
        span = {k: (int(np.flatnonzero(np.isfinite(tracks[k][:, 0]))[0]),
                    int(np.flatnonzero(np.isfinite(tracks[k][:, 0]))[-1])) for k in ks}
        ks.sort(key=lambda k: (span[k][0], k[1]))
        for k, s in zip(ks, metrica.first_fit([span[k] for k in ks], n_slots)):
            P[:, base + s, :2] = tracks[k]
    bv = np.stack([(ball[:, 0] + L / 2) / L, (W / 2 - ball[:, 1]) / W], 1)
    B = np.full((T, 3), np.nan, np.float32)
    for p, t0, o, n in grids:
        m = (per == p) & np.isfinite(bv[:, 0])
        if m.sum() >= 2:
            B[o:o + n, :2] = _interp(tg[o:o + n], ts[m], bv[m])
    return P, B, period, grids


def read_events(path, grids, team_of=None, home_id=None):
    """Event Data/<id>.json -> Metrica-style event table on the 25 fps grid.
    team_of: {player_id: team_id} from Rosters; home_id: home team_id, used to resolve challenge winners.
    """
    ev = json.load(open(path))
    team_of = team_of or {}
    gmap = {p: (t0, o, n) for p, t0, o, n in grids}
    rows = []
    for x in ev:
        g = x.get("gameEvents") or {}
        p = g.get("period")
        if p not in gmap or x.get("startTime") is None:
            continue
        t0, o, n = gmap[p]
        sf = o + int(round((float(x["startTime"]) - t0) * FPS))
        et = x.get("endTime")
        ef = o + int(round((float(et) - t0) * FPS)) if et is not None else sf
        team = {True: 1, False: 2}.get(g.get("homeTeam"), 0)
        gt = g.get("gameEventType")
        sp = g.get("setpieceType")
        pe = (x.get("possessionEvents") or {}).get("possessionEventType")
        typ, sub = None, ""
        if gt == "OUT":
            typ, sub, ef = "BALL OUT", "OUT-%s" % (g.get("outType") or ""), sf
        elif gt == "END":
            typ, sub = "END", "END HALF"
        elif gt in ("FIRSTKICKOFF", "SECONDKICKOFF", "THIRDKICKOFF", "FOURTHKICKOFF") or (
                gt == "OTB" and sp in RESTART_SETPIECES):
            typ, sub = "SET PIECE", "SP-%s" % (sp or "K")
        elif gt == "OTB" and pe == "CH":
            w = (x.get("possessionEvents") or {}).get("challengeWinnerPlayerId")
            wt = team_of.get(int(w)) if w is not None else None
            if wt is not None and home_id is not None:
                typ, sub, team = "RECOVERY", "PFF-CH-WON", (1 if wt == home_id else 2)
            else:
                typ, sub = "CHALLENGE", "PFF-CH"
        elif gt == "OTB" and pe in POSS_MAP:
            typ, sub = POSS_MAP[pe], "PFF-%s" % pe
        if typ is None or (typ not in ("BALL OUT", "END") and team == 0):
            continue
        rows.append({"team": team, "type": typ, "subtype": sub, "period": p, "start_frame": sf,
                     "end_frame": max(ef, sf), "start_x": np.nan, "start_y": np.nan, "end_x": np.nan,
                     "end_y": np.nan})
    rows.sort(key=lambda r: (r["start_frame"],))
    return metrica._event_table(rows)


def load_game(gid, raw=None, ball_params=None):
    raw = raw or raw_root()
    meta = json.load(open(os.path.join(raw, "Metadata", "%d.json" % gid)))
    meta = meta[0] if isinstance(meta, list) else meta
    pitch = ((meta.get("stadium") or {}).get("pitches") or [{}])[0]
    L, W = float(pitch.get("length") or 105.0), float(pitch.get("width") or 68.0)
    ts, per, rows, ball = read_tracking(os.path.join(raw, "Tracking Data", "%d.jsonl.bz2" % gid))
    P, B, period, grids = resample(ts, per, rows, ball, L, W)
    team_of = {}
    rp = os.path.join(raw, "Rosters", "%d.json" % gid)
    if os.path.isfile(rp):
        for r in json.load(open(rp)):
            try:
                team_of[int(r["player"]["id"])] = int(r["team"]["id"])
            except (KeyError, TypeError, ValueError):
                pass
    home_id = int((meta.get("homeTeam") or {}).get("id")) if (meta.get("homeTeam") or {}).get("id") else None
    ev = read_events(os.path.join(raw, "Event Data", "%d.json" % gid), grids, team_of, home_id)
    name = "PFF-%d" % gid
    m = metrica.build_match(P, B, period, ev, name, "dfl", raw_name=str(gid))
    m.meta.update({"provider": "pff", "pitch": [L, W], "provider_game": str(gid),
                   "coords": "chuẩn hoá 0-1 từ mét PFF (gốc tâm sân): x=(x+L/2)/L, y=(W/2-y)/W",
                   "teams": {"group1": (meta.get("homeTeam") or {}).get("name"),
                             "group2": (meta.get("awayTeam") or {}).get("name")},
                   "label_source": "events (%s)" % LABEL_VERSION, "source_fps": meta.get("fps")})
    bp = dict(metrica.BALL_DFL, **(ball_params or {}))
    seed = zlib.crc32(name.encode())
    m.B = metrica.dfl_like_ball(m.B, m.half, m.extras["y_true"], FPS, seed=seed, **bp)
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
    li = m.meta["label_info"]
    y = m.extras["y_true"]
    return "pff: %s T=%d live %.0f%% stops %d restarts %d" % (name, m.n_frames, 100 * np.mean((y == 0) | (y == 1)),
                                                              li["stops"], li["restarts"])


def ensure_cache(names=None, cache_dir=None, workers=1, log=print):
    """Ensure cache/external/PFF-<id>.npz (requires train_data/raw/pff if uncached). Return {name: path}."""
    names = list(names or PFF)
    bad = [n for n in names if n not in PFF]
    if bad:
        raise ValueError("không phải trận PFF: %s" % bad)
    out = {n: cache_file(n, cache_dir) for n in names}
    todo = [(n, out[n]) for n in names if not _cache_ok(out[n])]
    if todo:
        if not os.path.isdir(raw_root()):
            raise FileNotFoundError("thiếu cache %s và không có dữ liệu gốc PFF ở %s" % ([n for n, _ in todo], raw_root()))
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
    ap.add_argument("--games", nargs="+", default=PFF)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    for n, p in ensure_cache(a.games, workers=a.workers).items():
        print(n, p, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
