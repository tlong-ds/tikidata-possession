"""DFL / Sportec (seven matches) -> ATD format.

Format conversion only, with no additional normalization:
  * Centered metres -> 0-1 using that match's pitch dimensions (x/PitchX + 0.5).
  * Players -> 46 slots (home = group 1 = 0..23, away = group 2 = 24..45), packed first-fit by
    presence intervals so substitutions reuse slots, as in ATD.
  * Exclude referees; retain DFL S unchanged as speed (km/h, not m/s). View converts a separate
    reference copy to m/s when fitting, without modifying caches.
  * NO direction flip: original per-half direction is stored in meta["left_group"].
Also includes source-specific labels (y_true), t_utc, Bz and event tables.
"""

import datetime as dt
import glob
import os

import numpy as np
from lxml import etree

from .schema import N_SLOTS, N_TEAM1, RAW_DIR, Match

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DFL_DIR = os.path.join(RAW_DIR, "idsse")
SECTIONS = ("firstHalf", "secondHalf")
TOUCH_KINDS = ("Play", "TacklingGame", "BallClaiming", "OtherBallAction")
SETPIECE_KINDS = ("ThrowIn", "CornerKick", "FreeKick", "GoalKick", "KickOff", "Penalty")


def match_ids():
    out = []
    for f in sorted(glob.glob(os.path.join(DFL_DIR, "DFL_02_01_matchinformation_*.xml"))):
        out.append(f.rsplit("_", 1)[-1][:-4])
    return out


def _file(prefix, mid):
    hits = glob.glob(os.path.join(DFL_DIR, "{}_*{}.xml".format(prefix, mid)))
    if len(hits) != 1:
        raise IOError("{} {}: tìm thấy {} file".format(prefix, mid, len(hits)))
    return hits[0]


def _ms(s):
    """Timezone-aware ISO-8601 -> integer epoch milliseconds."""
    return int(round(dt.datetime.fromisoformat(s).timestamp() * 1000))


def read_matchinfo(mid):
    root = etree.parse(_file("DFL_02_01_matchinformation", mid)).getroot()
    env = next(root.iter("Environment"))
    teams = {t.get("Role"): t.get("TeamId") for t in root.iter("Team") if t.get("Role") in ("home", "guest")}
    names = {t.get("Role"): t.get("TeamName") for t in root.iter("Team") if t.get("Role") in ("home", "guest")}
    return {"pitch": [float(env.get("PitchX")), float(env.get("PitchY"))],
            "home_id": teams["home"], "guest_id": teams["guest"],
            "home_name": names["home"], "guest_name": names["guest"]}


def read_positions(mid):
    """{section: {'ball': dict, 'persons': {(team_id, person_id): dict}}}"""
    out = {s: {"ball": None, "persons": {}} for s in SECTIONS}
    for _, el in etree.iterparse(_file("DFL_04_03_positions_raw_observed", mid), tag="FrameSet"):
        sec, tid, pid = el.get("GameSection"), el.get("TeamId"), el.get("PersonId")
        if tid == "referee":
            el.clear()
            continue
        fr = el.findall("Frame")
        rec = {"N": np.array([int(f.get("N")) for f in fr], dtype=np.int64),
               "x": np.array([float(f.get("X")) for f in fr]),
               "y": np.array([float(f.get("Y")) for f in fr]),
               "s": np.array([float(f.get("S")) if f.get("S") else np.nan for f in fr])}
        if tid == "BALL":
            rec["z"] = np.array([float(f.get("Z")) for f in fr])
            rec["status"] = np.array([int(f.get("BallStatus")) for f in fr], dtype=np.int8)
            rec["poss"] = np.array([int(f.get("BallPossession")) for f in fr], dtype=np.int8)
            rec["t"] = np.array([_ms(f.get("T")) for f in fr], dtype=np.int64)
            out[sec]["ball"] = rec
        else:
            out[sec]["persons"][(tid, pid)] = rec
        el.clear()
    return out


def pack_slots(intervals, n_slots):
    """First-fit: intervals [(first, last)] sorted by first -> slot indices (-1 if full)."""
    free_at = [-1] * n_slots      # Last frame for which the slot is occupied.
    out = []
    for first, last in intervals:
        for s in range(n_slots):
            if free_at[s] < first:
                free_at[s] = last
                out.append(s)
                break
        else:
            out.append(-1)
    return out


def read_events(mid, info):
    """Event table after removing <Delete/>: time, type, team (1/2/0), position, StartFrame if present."""
    group = {info["home_id"]: 1, info["guest_id"]: 2}
    rows, left = [], {}
    root = etree.parse(_file("DFL_03_02_events_raw", mid)).getroot()
    for ev in root.iter("Event"):
        kids = [c for c in ev if isinstance(c.tag, str)]
        if not kids or kids[0].tag == "Delete":
            continue
        top = kids[0]
        kind = top.tag
        sec = top.get("GameSection")
        if kind == "KickOff" and sec:
            left[sec] = group.get(top.get("TeamLeft"), 0)
        team = top.get("Team") or top.get("WinnerTeam") or top.get("TeamFouler")
        has_play = any(d.tag == "Play" for d in top.iter())
        if team is None and has_play:
            team = next(d.get("Team") for d in top.iter() if d.tag == "Play")
        sf = ev.get("StartFrame")
        rows.append({
            "t_ms": _ms(ev.get("EventTime")),
            "kind": kind,
            "team": group.get(team, 0),
            "x": float(ev.get("X-Position", "nan")), "y": float(ev.get("Y-Position", "nan")),
            "xs": float(ev.get("X-Source-Position", "nan")), "ys": float(ev.get("Y-Source-Position", "nan")),
            "start_n": int(sf) if sf else -1,
            "touch": kind in TOUCH_KINDS or (kind in SETPIECE_KINDS and has_play),
            "setpiece": kind in SETPIECE_KINDS,
        })
    tbl = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    tbl["t_ms"] = tbl["t_ms"].astype(np.int64)
    return tbl, left


def estimate_event_lag(ev, t_utc, B, half, pitch, max_ms=4000, step_ms=40):
    """Offset (ms) between EventTime and the frame clock, estimated separately per half.

    No global offset exists: measured offsets across seven matches range from -2.0 s to +1.5 s
    (the two halves of a match are close, within +-0.5 s). Minimize the median distance between
    event positions (X-Position, Y-Position, original corner-origin coordinates, unflipped) and
    ball positions at t + lag. Used for other-match analysis, respecting competition-label rules.
    Residual error is approximately 0.3-0.5 s.
    """
    px, py = pitch
    bx, by = B[:, 0].astype(np.float64) * px, B[:, 1].astype(np.float64) * py
    base = np.where(ev["touch"] & np.isfinite(ev["x"]))[0]
    f0 = np.clip(np.searchsorted(t_utc, ev["t_ms"][base]), 0, len(t_utc) - 1)
    lags = np.arange(-max_ms, max_ms + 1, step_ms)
    out = {}
    for h in (1, 2):
        sel = base[half[f0] == h]
        score = []
        for lag in lags:
            fr = np.clip(np.searchsorted(t_utc, ev["t_ms"][sel] + lag), 0, len(t_utc) - 1)
            score.append(np.nanmedian(np.hypot(bx[fr] - ev["x"][sel], by[fr] - ev["y"][sel])))
        score = np.array(score)
        out[h] = {"lag_ms": int(lags[score.argmin()]), "median_dist_m": float(score.min()),
                  "median_dist_at_0_m": float(score[lags == 0][0]), "n_events": int(len(sel))}
    return out


def event_frames(ev, t_utc, half, lag):
    """Nearest frame for each event after applying its half-specific lag."""
    f0 = np.clip(np.searchsorted(t_utc, ev["t_ms"]), 0, len(t_utc) - 1)
    lag_arr = np.where(half[f0] == 1, lag[1]["lag_ms"], lag[2]["lag_ms"])
    return np.clip(np.searchsorted(t_utc, ev["t_ms"] + lag_arr), 0, len(t_utc) - 1)


def load_dfl(mid):
    info = read_matchinfo(mid)
    pos = read_positions(mid)
    px, py = info["pitch"]

    # Timeline: concatenate the first half, then the second.
    ball_parts, offsets, half = [], {}, []
    n_tot = 0
    for h, sec in enumerate(SECTIONS, start=1):
        b = pos[sec]["ball"]
        offsets[sec] = n_tot
        n_tot += len(b["N"])
        ball_parts.append(b)
        half.append(np.full(len(b["N"]), h, dtype=np.int8))
        if not (np.diff(b["N"]) == 1).all():
            raise ValueError("{} {}: N của bóng không liên tục".format(mid, sec))
    T = n_tot
    P = np.full((T, N_SLOTS, 3), np.nan, dtype=np.float32)
    B = np.full((T, 3), np.nan, dtype=np.float32)
    Bz = np.full(T, np.nan, dtype=np.float32)
    status = np.zeros(T, dtype=np.int8)
    poss = np.zeros(T, dtype=np.int8)
    N = np.zeros(T, dtype=np.int64)
    t_utc = np.zeros(T, dtype=np.int64)
    for sec, b in zip(SECTIONS, ball_parts):
        o = offsets[sec]; n = len(b["N"]); sl = slice(o, o + n)
        B[sl, 0] = b["x"] / px + 0.5
        B[sl, 1] = b["y"] / py + 0.5
        B[sl, 2] = b["s"]
        Bz[sl] = b["z"]; status[sl] = b["status"]; poss[sl] = b["poss"]
        N[sl] = b["N"]; t_utc[sl] = b["t"]

    # Players -> slots, separately by team, using full-match presence intervals.
    persons = {}     # (team_id, person_id) -> {"first","last","segs":[(sec, rec)]}
    for sec in SECTIONS:
        for key, rec in pos[sec]["persons"].items():
            idx = offsets[sec] + (rec["N"] - pos[sec]["ball"]["N"][0])
            ok = (idx >= offsets[sec]) & (idx < offsets[sec] + len(pos[sec]["ball"]["N"]))
            if not ok.any():
                continue
            d = persons.setdefault(key, {"first": 10**9, "last": -1, "segs": []})
            d["first"] = min(d["first"], int(idx[ok].min()))
            d["last"] = max(d["last"], int(idx[ok].max()))
            d["segs"].append((idx[ok], rec, ok))
    slot_person = np.full((T, N_SLOTS), -1, dtype=np.int16)
    person_table = []
    for team_id, base in ((info["home_id"], 0), (info["guest_id"], N_TEAM1)):
        keys = sorted((k for k in persons if k[0] == team_id), key=lambda k: (persons[k]["first"], k[1]))
        n_sl = N_TEAM1 if base == 0 else N_SLOTS - N_TEAM1
        slots = pack_slots([(persons[k]["first"], persons[k]["last"]) for k in keys], n_sl)
        for k, s in zip(keys, slots):
            if s < 0:
                raise ValueError("{} {}: hơn {} người cùng lúc".format(mid, team_id, n_sl))
            pid = len(person_table)
            person_table.append(k[1])
            for idx, rec, ok in persons[k]["segs"]:
                P[idx, base + s, 0] = rec["x"][ok] / px + 0.5
                P[idx, base + s, 1] = rec["y"][ok] / py + 0.5
                P[idx, base + s, 2] = rec["s"][ok]
                slot_person[idx, base + s] = pid

    # Any live frame must name a possessor; otherwise "not home" would silently be labelled away.
    unnamed = (status != 0) & ~np.isin(poss, (1, 2))
    if unnamed.any():
        raise ValueError("{}: {} live frames without BallPossession 1/2".format(mid, int(unnamed.sum())))
    y_true = np.where(status == 0, 2, np.where(poss == 1, 0, 1)).astype(np.int8)   # 0=H 1=A 2=D

    ev, left = read_events(mid, info)
    half_all = np.concatenate(half)
    lag = estimate_event_lag(ev, t_utc, B, half_all, info["pitch"])
    ev_frame = event_frames(ev, t_utc, half_all, lag)
    # touch_true: 0 none, 1 / 2 touching group, 3 both in the same frame. Timing tolerance +-0.5 s.
    touch_true = np.zeros(T, dtype=np.int8)
    for f, team, is_touch in zip(ev_frame, ev["team"], ev["touch"]):
        if is_touch and team in (1, 2):
            touch_true[f] |= team
    ev["frame"] = ev_frame
    meta = {
        "fps": 25, "pitch": info["pitch"],
        "coords": "chuẩn hoá 0-1 theo kích thước sân của trận; mét = x*pitch[0], y*pitch[1]",
        "teams": {"group1": info["home_name"], "group2": info["guest_name"]},
        "section_offsets": offsets,
        "left_group": left,          # Group (1/2) on the left at each kickoff (original, unflipped).
        "persons": person_table,
        "labels": {"0": "H", "1": "A", "2": "D"},
        "event_lag": {str(k): v for k, v in lag.items()},
        "static_extras": [],
    }
    extras = {"y_true": y_true, "touch_true": touch_true, "Bz": Bz, "N": N, "t_utc": t_utc,
              "slot_person": slot_person}
    for k, v in ev.items():
        extras["tbl_ev_" + k] = v
    return Match(name=mid, source="dfl", P=P, B=B, half=half_all, meta=meta,
                 extras=extras)
