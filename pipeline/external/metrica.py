"""Metrica Sports sample data (github.com/metrica-sports/sample-data) -> DFL-compatible match caches.

Unlike pipeline/io_metrica.py (the old diagnostic converter with draft `y_draft` labels and source
"metrica"), this module implements the DFL match contract so the release degradation pipeline
(pipeline.degradation_example.prepare_epoch / feed.train_view) can be reused unchanged:

  * P [T,46,3], B [T,3] normalized to 0-1 (105 x 68 m pitch); speed = NaN (unavailable in Metrica;
    not read by View). Home = group 1 = slots 0..23; away = group 2 = slots 24..45. NO direction
    flip (as in io_dfl): meta["left_group"] records the team on the left at each half's kickoff;
    the "dfl" resolve_geometry branch reads this key. The y axis retains Metrica's convention
    (0 = top edge), matching the competition tracking, which Metrica also produced.
  * half: 1 / 2 from KICK OFF to END HALF or the last event + 2 s; 0 outside those intervals,
    matching DFL (tracking starts at kickoff and ends a few seconds after the whistle).
  * y_true (0 H / 1 A / 2 D, -1 outside halves) is derived from EVENTS with DFL label semantics
    (tracking BallStatus / BallPossession flags): D from ball-out / whistle (foul, offside, goal,
    end of half) to restart; otherwise the owning team, retaining the previous team through
    loose balls / challenges. See `event_labels` for the detailed rules.
  * touch_true (0, 1 / 2 = touching group, 3 = both) at touch-event start frames, matching
    io_dfl TOUCH_KINDS (pass, shot, ball lost, recovery, challenge won, set piece).
  * tbl_ev_* event arrays, including tbl_ev_frame + tbl_ev_setpiece as in DFL, so release-v3
    quality.restart_frames can find restarts.

The default source is "dfl" (the cache contract, allowing release dfl-strat.json parameters);
the actual provider is meta["provider"] = "metrica". Match names: METRICA-SG1/2/3.

    python -m pipeline.external.metrica --raw <sample-data/data directory> --out <cache directory>/dfl
    python -m pipeline.external.metrica --ensure      # pinned download, SHA-256 check, cache/external conversion

Training: `ensure_cache()` (called by src.core.train_switch for METRICA-* in [train] extra_train)
searches $METRICA_RAW, train_data/raw/metrica-sample-data/data and cache/raw/metrica-sample-data/data.
If absent, it fetches the public repository at SAMPLE_COMMIT and verifies all nine RAW_SHA256 hashes
before conversion. Downloaded files are only READ (pandas / json / xml); no downloaded code is executed.
"""

import argparse
import hashlib
import json
import os
import subprocess
import xml.etree.ElementTree as ET
import zlib

import numpy as np

from ..schema import CACHE_DIR, N_SLOTS, N_TEAM1, RAW_DIR, Match, save_match, validate_match

FPS = 25
PITCH = [105.0, 68.0]
GAMES = {"Sample_Game_1": "METRICA-SG1", "Sample_Game_2": "METRICA-SG2", "Sample_Game_3": "METRICA-SG3"}
LABEL_VERSION = "metrica-events-v1"
CONVERTER_VERSION = "metrica-convert-v2"        # Increment when output changes so ensure_cache reconverts.
CACHE_SUB = "external"                          # cache/external/METRICA-SG*.npz (load_match searches cache/*/<name>.npz).
SAMPLE_REPO = "https://github.com/metrica-sports/sample-data.git"
SAMPLE_COMMIT = "e706dd506b360d69d9d123d5b8026e7294b13996"
RAW_SHA256 = {
    "Sample_Game_1/Sample_Game_1_RawEventsData.csv": "097dda6345e17390dca095331458fd00e47960c480c76f4356fd4edf2575ef05",
    "Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv": "0a3df68a92af0290ede472e62de3b6489ddee29223da09bdf0cf76c8e0ab0618",
    "Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv": "0a3d0eff7785950379c587d3597b86a2d91c2492281b4652e1abb97c99b6b364",
    "Sample_Game_2/Sample_Game_2_RawEventsData.csv": "edf31a18599265b77a8baf150f2ce6d89456fb0324b62ea7a391229657a619ba",
    "Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv": "aa639f40762eade46c389020fe4458266b5ba14027b1e96e22e75db62803718b",
    "Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv": "3f9990ac204b07133e4202441d6fcbb386897c7c7f81e1bc3ce4d35bc8644694",
    "Sample_Game_3/Sample_Game_3_events.json": "28a53bfe9eaf32535362335553d517cdce0eabb588af6166009081753751f4e0",
    "Sample_Game_3/Sample_Game_3_metadata.xml": "009c7dcfcb66192014917ca1559c158f612353e390eafb85f8214b06a1c7dec1",
    "Sample_Game_3/Sample_Game_3_tracking.txt": "570e6ce82c2d93bc92723af331d93c31110bec1d6dc0be07c6d44ca3f981d658",
}

POSSESSION_TYPES = ("PASS", "SHOT", "CARRY", "RECOVERY", "BALL LOST", "SET PIECE")
REOPEN_TYPES = ("PASS", "SHOT", "CARRY", "BALL LOST")   # Actions that restart play when a SET PIECE event is missing.
TOUCH_TYPES = ("PASS", "SHOT", "RECOVERY", "BALL LOST", "SET PIECE")
END_MARGIN_S = 2.0


# ----------------------------------------------------------------------------- file reading

def _sub_tokens(s):
    """'HEAD-ON TARGET-GOAL' -> ('HEAD', 'ON TARGET', 'GOAL'). Metrica tokens contain spaces, not '-'."""
    if s is None or (isinstance(s, float) and np.isnan(s)) or s == "":
        return ()
    return tuple(t.strip() for t in str(s).split("-") if t.strip())


def _event_table(rows):
    keys = ("team", "type", "subtype", "period", "start_frame", "end_frame", "start_x", "start_y",
            "end_x", "end_y")
    out = {k: np.array([r[k] for r in rows]) if rows else np.zeros(0) for k in keys}
    out["team"] = out["team"].astype(np.int8)
    out["period"] = out["period"].astype(np.int8)
    out["start_frame"] = out["start_frame"].astype(np.int64)
    out["end_frame"] = out["end_frame"].astype(np.int64)
    for k in ("start_x", "start_y", "end_x", "end_y"):
        out[k] = out[k].astype(np.float64)
    out["type"] = out["type"].astype(str)
    out["subtype"] = out["subtype"].astype(str)
    return out


def read_events_csv(path):
    """RawEventsData.csv -> event table (zero-based frames; team 1 = Home, 2 = Away, 0 = unknown)."""
    import pandas as pd
    df = pd.read_csv(path)
    rows = []
    for r in df.itertuples(index=False):
        d = dict(zip(df.columns, r))
        sf = int(d["Start Frame"]) - 1
        ef = int(d["End Frame"]) - 1 if np.isfinite(d["End Frame"]) and int(d["End Frame"]) > 0 else sf
        rows.append({"team": {"Home": 1, "Away": 2}.get(d["Team"], 0), "type": str(d["Type"]),
                     "subtype": "-".join(_sub_tokens(d["Subtype"])), "period": int(d["Period"]),
                     "start_frame": sf, "end_frame": max(ef, sf),
                     "start_x": float(d["Start X"]), "start_y": float(d["Start Y"]),
                     "end_x": float(d["End X"]), "end_y": float(d["End Y"])})
    return _event_table(rows)


def read_events_json(path, home_id):
    """Sample_Game_3_events.json -> the same event table as read_events_csv."""
    with open(path) as f:
        data = json.load(f)["data"]

    def num(o, k):
        return np.nan if not o or o.get(k) is None else float(o[k])

    rows = []
    for e in data:
        team = e.get("team") or {}
        st = e.get("subtypes")
        st = [st] if isinstance(st, dict) else (st or [])
        sf = int(e["start"]["frame"]) - 1
        end = e.get("end") or {}
        ef = int(end["frame"]) - 1 if end.get("frame") else sf
        rows.append({"team": 0 if not team.get("id") else (1 if team["id"] == home_id else 2),
                     "type": e["type"]["name"], "subtype": "-".join(s["name"] for s in st),
                     "period": int(e["period"]), "start_frame": sf, "end_frame": max(ef, sf),
                     "start_x": num(e["start"], "x"), "start_y": num(e["start"], "y"),
                     "end_x": num(end, "x"), "end_y": num(end, "y")})
    return _event_table(rows)


def read_tracking_csv(path):
    """RawTrackingData_<Team>.csv -> (period[T], xy[T, n, 2], ball[T, 2]); frames must be consecutive from 1."""
    import pandas as pd
    df = pd.read_csv(path, header=None, skiprows=3, dtype=np.float64)
    period, frame = df[0].to_numpy(), df[1].to_numpy()
    if not np.array_equal(frame, np.arange(1, len(df) + 1)):
        raise ValueError(path + ": cột Frame không liên tục từ 1")
    xy = df.iloc[:, 3:].to_numpy()
    n = (xy.shape[1] - 2) // 2
    return period.astype(np.int8), xy[:, :2 * n].reshape(len(df), n, 2), xy[:, -2:]


def first_fit(intervals, n_slots):
    """Intervals [(first, last)] sorted by first -> slots (first-fit, as in io_dfl.pack_slots)."""
    free_at, out = [-1] * n_slots, []
    for first, last in intervals:
        for s in range(n_slots):
            if free_at[s] < first:
                free_at[s] = last
                out.append(s)
                break
        else:
            raise ValueError("hơn %d người cùng lúc" % n_slots)
    return out


def read_game3(d):
    """EPTS/FIFA (metadata.xml + tracking.txt) -> (P[T,46,3], B[T,3], period[T], home_id)."""
    root = ET.parse(os.path.join(d, "Sample_Game_3_metadata.xml")).getroot()
    score = next(root.iter("Score"))
    home_id = score.get("idLocalTeam")
    team_of = {p.get("id"): (0 if p.get("teamId") == home_id else 1) for p in root.iter("Player")}
    chan = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    params = {p.findtext("Name"): p.findtext("Value") for p in root.iter("ProviderParameter")}
    specs = []
    for sp in root.iter("DataFormatSpecification"):
        players = [chan[g.find("PlayerChannelRef").get("playerChannelId")] for g in sp.find("SplitRegister")
                   if g.find("PlayerChannelRef") is not None]
        specs.append((int(sp.get("startFrame")), int(sp.get("endFrame")), players))
    span = {}
    for s, e, players in specs:
        for pid in players:
            a, b = span.get(pid, (s, e))
            span[pid] = (min(a, s), max(b, e))
    slot = {}
    for t, base, n in ((0, 0, N_TEAM1), (1, N_TEAM1, N_SLOTS - N_TEAM1)):
        ids = sorted((p for p in span if team_of[p] == t), key=lambda p: (span[p][0], p))
        for pid, s in zip(ids, first_fit([span[p] for p in ids], n)):
            slot[pid] = base + s
    with open(os.path.join(d, "Sample_Game_3_tracking.txt")) as f:
        lines = f.read().splitlines()
    T = len(lines)
    P = np.full((T, N_SLOTS, 3), np.nan, dtype=np.float32)
    B = np.full((T, 3), np.nan, dtype=np.float32)
    for s, e, players in specs:
        cols = np.array([slot[p] for p in players])
        for i in range(s - 1, min(e, T)):
            fid, pl, ball = lines[i].split(":")
            if int(fid) != i + 1:
                raise ValueError("Game 3: frame_id lệch ở dòng %d" % (i + 1))
            v = np.array(pl.replace(";", ",").split(","), dtype=np.float64).reshape(-1, 2)
            P[i, cols, :2] = v
            B[i, :2] = np.array(ball.split(","), dtype=np.float64)[:2]
    period = np.zeros(T, np.int8)
    period[int(params["first_half_start"]) - 1:int(params["first_half_end"])] = 1
    period[int(params["second_half_start"]) - 1:int(params["second_half_end"])] = 2
    return P, B, period, home_id


def read_game(raw_dir, game):
    """-> (P[T,46,3], B[T,3], period[T], ev) for one sample match (raw_dir = .../sample-data/data)."""
    d = os.path.join(raw_dir, game)
    if game == "Sample_Game_3":
        P, B, period, home_id = read_game3(d)
        return P, B, period, read_events_json(os.path.join(d, game + "_events.json"), home_id)
    ph, xyh, bh = read_tracking_csv(os.path.join(d, game + "_RawTrackingData_Home_Team.csv"))
    pa, xya, _ = read_tracking_csv(os.path.join(d, game + "_RawTrackingData_Away_Team.csv"))
    if not np.array_equal(ph, pa):
        raise ValueError(game + ": cột Period của hai đội lệch nhau")
    if xyh.shape[1] > N_TEAM1 or xya.shape[1] > N_SLOTS - N_TEAM1:
        raise ValueError(game + ": quá nhiều cầu thủ")
    T = len(ph)
    P = np.full((T, N_SLOTS, 3), np.nan, dtype=np.float32)
    P[:, :xyh.shape[1], :2] = xyh
    P[:, N_TEAM1:N_TEAM1 + xya.shape[1], :2] = xya
    B = np.full((T, 3), np.nan, dtype=np.float32)
    B[:, :2] = bh
    return P, B, ph, read_events_csv(os.path.join(d, game + "_RawEventsData.csv"))


# ----------------------------------------------------------------------------- labels

def _classify(typ, tok):
    """-> (restart, stop_at, possession, reopen): stop_at = 'start' | 'end' | None."""
    restart = typ == "SET PIECE" or "GOAL KICK" in tok          # Metrica records goal kicks as PASS/BALL LOST GOAL KICK.
    stop = None
    if typ == "BALL OUT":
        stop = "end"                                           # end = frame when the ball crosses the line.
    elif typ == "SHOT" and ("OUT" in tok or "GOAL" in tok):
        stop = "end"                                           # Shot out of play / goal.
    elif typ == "FAULT RECEIVED" or "HAND BALL" in tok:
        stop = "start"                                         # Foul whistle.
    elif "OFFSIDE" in tok:
        stop = "end"                                           # Offside flag when a teammate receives the ball.
    elif "END HALF" in tok:
        stop = "start"
    return restart, stop, typ in POSSESSION_TYPES, typ in REOPEN_TYPES


def event_labels(ev, period, fps=FPS, end_margin_s=END_MARGIN_S):
    """Event table + period[T] from tracking -> (y[T] int8, half[T] int8, info).

    Each half starts at its first restart (KICK OFF) and follows a temporal state machine:
      * Restart (SET PIECE; PASS/BALL LOST with GOAL KICK): live ball owned by the restarting
        team from the start frame. If already live (no recorded stoppage), mark dead ball from
        the previous final-ball frame (`implicit_stops`).
      * Stoppage: BALL OUT (end frame = line crossing), SHOT with OUT / GOAL (end frame),
        FAULT RECEIVED and HAND BALL (start frame), OFFSIDE (pass end frame), END HALF.
      * Ownership: PASS, SHOT, CARRY, RECOVERY, BALL LOST, SET PIECE -> that team from the start
        frame. CHALLENGE, BALL OUT (last touching team), CARD and FAULT RECEIVED do not change
        ownership: loose balls / challenges retain the previous team, as required by the task.
      * PASS / SHOT / CARRY / BALL LOST during dead ball (missing SET PIECE) restarts play
        (`implicit_restarts`); RECOVERY during dead ball (keeper picking up the ball) does not.
    The half ends at END HALF or the last ball frame + end_margin_s; subsequent frames are D.
    """
    T = len(period)
    y = np.full(T, -1, np.int8)
    half = np.zeros(T, np.int8)
    info = {"implicit_stops": 0, "implicit_restarts": 0, "restarts": 0, "stops": 0, "switches": 0,
            "version": LABEL_VERSION}
    order = np.argsort(ev["start_frame"], kind="stable")
    margin = int(round(end_margin_s * fps))
    for p in (1, 2):
        rows = np.where(period == p)[0]
        if not len(rows):
            continue
        p_lo, p_hi = int(rows[0]), int(rows[-1]) + 1
        marks = []                                             # (frame, seq, kind, team, i)
        for seq, i in enumerate(order):
            if ev["period"][i] != p:
                continue
            restart, stop, poss, reopen = _classify(ev["type"][i], set(ev["subtype"][i].split("-")))
            sf, ef, tm = int(ev["start_frame"][i]), int(ev["end_frame"][i]), int(ev["team"][i])
            if restart and tm in (1, 2):
                marks.append((sf, 2 * seq, "restart", tm, i))
            elif poss and tm in (1, 2):
                marks.append((sf, 2 * seq, "reopen" if reopen else "action", tm, i))
            if stop:
                marks.append((sf if stop == "start" else ef, 2 * seq + 1, "stop", tm, i))
        marks.sort(key=lambda m: (m[0], m[1]))
        changes = []                                           # (frame, state)
        alive, team, started, last_ball, end_frame = False, None, False, None, None
        for f, _, kind, tm, i in marks:
            f = min(max(f, p_lo), p_hi - 1)
            is_end_half = "END HALF" in ev["subtype"][i].split("-")
            if kind == "restart" or (kind == "reopen" and not alive):
                if not started:
                    started, start_frame = True, f
                elif alive:
                    changes.append((min(last_ball, f) if last_ball is not None else f, 2))
                    info["implicit_stops"] += 1
                elif kind == "reopen":
                    info["implicit_restarts"] += 1
                if kind == "restart":
                    info["restarts"] += 1
                alive, team = True, tm
                changes.append((f, 0 if tm == 1 else 1))
            elif kind in ("action", "reopen"):
                if alive and tm != team:
                    team = tm
                    info["switches"] += 1
                    changes.append((f, 0 if tm == 1 else 1))
            elif kind == "stop" and started:
                if alive:
                    info["stops"] += 1
                    changes.append((f, 2))
                alive = False
                if is_end_half:
                    end_frame = f
                    break
            if started and kind != "stop":
                last_ball = min(max(last_ball or 0, int(ev["end_frame"][i]), f), p_hi - 1)
        if not started:
            continue
        if end_frame is None:                                  # End-of-half whistle: frames after the last ball frame are D.
            end_frame = min(max([last_ball or start_frame] + [c[0] for c in changes]), p_hi - 1)
            changes.append((end_frame, 2))
        hi = min(p_hi, end_frame + margin)
        lab = np.full(p_hi - p_lo, 2, np.int8)
        changes.sort(key=lambda c: c[0])                      # Stable sort: the later event wins when frames are equal.
        for (f, s), nxt in zip(changes, changes[1:] + [(p_hi, None)]):
            lab[f - p_lo:max(f, nxt[0]) - p_lo] = s
        half[start_frame:hi] = p
        y[start_frame:hi] = lab[start_frame - p_lo:hi - p_lo]
    return y, half, info


def touch_labels(ev, T):
    """touch_true[T]: team bit (1 / 2) at touch-event start frames; CHALLENGE counts only the WON side."""
    tt = np.zeros(T, np.int8)
    for typ, sub, tm, f in zip(ev["type"], ev["subtype"], ev["team"], ev["start_frame"]):
        won = typ == "CHALLENGE" and "WON" in sub.split("-")
        if (typ in TOUCH_TYPES or won) and tm in (1, 2) and 0 <= f < T:
            tt[f] |= tm
    return tt


def left_group(P, half):
    """{'firstHalf': g, 'secondHalf': g}: group (1/2) with the smaller mean x in each half's first second (unflipped)."""
    out = {}
    for h, key in ((1, "firstHalf"), (2, "secondHalf")):
        idx = np.flatnonzero(half == h)[:FPS]
        if not len(idx):
            continue
        x = P[idx, :, 0].astype(np.float64)
        m1, m2 = np.nanmean(x[:, :N_TEAM1]), np.nanmean(x[:, N_TEAM1:])
        out[key] = 1 if m1 < m2 else 2
    return out


def build_match(P, B, period, ev, name, source="dfl", raw_name=None):
    """Loaded arrays -> Match under the DFL cache contract (see the module docstring)."""
    T = len(period)
    y, half, info = event_labels(ev, period)
    restart = np.array([_classify(t, set(s.split("-")))[0] for t, s in zip(ev["type"], ev["subtype"])], bool)
    touch = np.array([(t in TOUCH_TYPES or (t == "CHALLENGE" and "WON" in s.split("-")))
                      for t, s in zip(ev["type"], ev["subtype"])], bool)
    meta = {"fps": FPS, "pitch": list(PITCH), "provider": "metrica", "provider_game": raw_name or name,
            "coords": "chuẩn hoá 0-1 (gốc Metrica: (0,0) góc trên trái); mét = x*pitch[0], y*pitch[1]",
            "speed_unit": "absent (NaN)",
            "teams": {"group1": "Home", "group2": "Away"},
            "left_group": left_group(P, half),
            "labels": {"0": "H", "1": "A", "2": "D", "-1": "ngoài hiệp"},
            "label_source": "events (%s)" % LABEL_VERSION, "label_info": info, "static_extras": []}
    extras = {"y_true": y, "touch_true": touch_labels(ev, T)}
    for k, v in ev.items():
        extras["tbl_ev_" + k] = v
    extras["tbl_ev_frame"] = np.clip(ev["start_frame"], 0, T - 1)
    extras["tbl_ev_setpiece"] = restart
    extras["tbl_ev_touch"] = touch
    return Match(name=name, source=source, P=P, B=B, half=half, meta=meta, extras=extras)


# ----------------------------------------------------------------------------- DFL-style ball

# Original DFL ball (cache/dfl, five training matches): visible throughout each half; during dead ball (y = D)
# coordinates stay at the last live-ball position, then jump to the restart (entry: 0 m; exit: median ~6 m).
# Live coordinates rarely repeat (0.4-1.1%), are rounded to 1 cm and have mild jitter (acceleration q50 ~14, q90 ~36 m/s²).
# Metrica ball: mostly missing during dead ball (0-14% visible), nearly linear when live (q50 ~0.9, q90 ~7).
# Without conversion, degradation sees less ball than the target and hides no more; the network learns "missing ball = D".
# dfl_like_ball converts Metrica to the original DFL ball convention before applying the same degradation as DFL.
BALL_DFL = {"sig": 0.0, "noise_m": 0.03, "noise_s": 0.04, "quant": 0.01}   # Selected with a 43-point grid; see analysis/metrica_data.md.


def _runs(mask):
    e = np.flatnonzero(np.diff(np.r_[0, np.asarray(mask, np.int8), 0]))
    return list(zip(e[::2], e[1::2]))


def dfl_like_ball(B, half, y, fps=FPS, sig=0.0, noise_m=0.0, noise_s=0.0, quant=0.01, seed=0):
    """Normalized B [T,3] -> original DFL-style ball. Only changes frames within a half (half > 0):
    * Live (y 0/1): linearly interpolate gaps within each contiguous live run, then optionally
      apply Gaussian smoothing sig (s) + Gaussian noise filtered by noise_s (s; 0 = white noise)
      with deviation noise_m (m), without crossing dead-ball boundaries.
    * Dead (y = D): hold the previous live position, or the next live position at the half's start.
    * Round to quant (m), matching DFL coordinates. Keep column 2 (speed) unchanged (Metrica: NaN).
    Labels only select the dead-ball holding convention, using the same information as DFL BallStatus.
    """
    from src.core.ball_smooth import gauss
    rng = np.random.default_rng(seed)
    scale = np.array(PITCH)
    out = np.array(B, dtype=np.float64, copy=True)
    xy = out[:, :2] * scale
    half = np.asarray(half)
    y = np.asarray(y)
    for h in np.unique(half[half > 0]):
        inh = half == h
        live = inh & (y >= 0) & (y < 2)
        for s, e in _runs(live):
            seg = xy[s:e]
            ok = np.isfinite(seg).all(1)
            if not ok.any():
                continue
            t = np.arange(e - s)
            seg = np.stack([np.interp(t, t[ok], seg[ok, j]) for j in range(2)], 1)
            if sig > 0:
                seg = gauss(seg, sig * fps)
            if noise_m > 0 and len(seg) > 1:
                n = rng.normal(0, 1, seg.shape)
                if noise_s > 0:
                    n = gauss(n, noise_s * fps)
                sd = n.std(0)
                seg = seg + n / np.where(sd > 0, sd, 1) * noise_m
            xy[s:e] = seg
        idx = np.flatnonzero(inh)
        good = live[idx] & np.isfinite(xy[idx]).all(1)
        if not good.any():
            continue
        pos = np.arange(len(idx))
        prev = np.maximum.accumulate(np.where(good, pos, -1))
        nxt = np.minimum.accumulate(np.where(good, pos, len(idx))[::-1])[::-1]
        src = np.where(prev >= 0, prev, nxt)
        dead = ~live[idx]
        xy[idx[dead]] = xy[idx[src[dead]]]
    if quant > 0:
        xy = np.round(xy / quant) * quant
    inside = half > 0
    out[inside, :2] = xy[inside] / scale
    return out.astype(np.asarray(B).dtype)


def load_game(raw_dir, game, source="dfl", ball="dfl", ball_params=None):
    """ball: "dfl" (default: dfl_like_ball with BALL_DFL, overridden by ball_params) or "raw" (original Metrica ball)."""
    P, B, period, ev = read_game(raw_dir, game)
    m = build_match(P, B, period, ev, GAMES.get(game, game), source, raw_name=game)
    if ball == "dfl":
        bp = dict(BALL_DFL, **(ball_params or {}))
        seed = zlib.crc32(m.name.encode())
        m.B = dfl_like_ball(m.B, m.half, m.extras["y_true"], FPS, seed=seed, **bp)
        m.meta["ball_processing"] = {"kind": "dfl_like_ball", "seed": seed, **bp}
    elif ball == "raw":
        m.meta["ball_processing"] = {"kind": "raw"}
    else:
        raise ValueError("ball phải là 'dfl' hoặc 'raw', nhận %r" % (ball,))
    m.meta["converter_version"] = CONVERTER_VERSION
    return m


# ----------------------------------------------------------------------------- download + training cache

def _sha(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_raw(data_dir, expected=None):
    """Verify raw sample-data/data files against SHA-256 hashes. Returns a list of errors (empty = match)."""
    bad = []
    for rel, want in (RAW_SHA256 if expected is None else expected).items():
        p = os.path.join(data_dir, rel)
        if not os.path.isfile(p):
            bad.append("thiếu " + rel)
        elif _sha(p) != want:
            bad.append("sha256 khác: " + rel)
    return bad


def raw_candidates():
    return [d for d in (os.environ.get("METRICA_RAW"), os.path.join(RAW_DIR, "metrica-sample-data", "data"),
                        os.path.join(CACHE_DIR, "raw", "metrica-sample-data", "data")) if d]


def fetch_raw(dst=None, commit=SAMPLE_COMMIT, repo=SAMPLE_REPO):
    """Fetch the pinned public repository commit into dst (depth 1); return data/. Never execute repository code."""
    dst = dst or os.path.join(CACHE_DIR, "raw", "metrica-sample-data")
    os.makedirs(dst, exist_ok=True)
    git = ["git", "-C", dst, "-c", "core.hooksPath=/dev/null"]
    if not os.path.isdir(os.path.join(dst, ".git")):
        subprocess.run(git + ["init", "-q"], check=True)
    subprocess.run(git + ["fetch", "-q", "--depth", "1", repo, commit], check=True)
    subprocess.run(git + ["checkout", "-q", "--force", "FETCH_HEAD"], check=True)
    return os.path.join(dst, "data")


def find_raw(fetch=True, expected=None):
    """Return the SHA-256-verified data/ directory; download if missing and fetch=True."""
    tried = []
    for d in raw_candidates():
        if os.path.isdir(d):
            bad = verify_raw(d, expected)
            if not bad:
                return d
            tried.append("%s: %s" % (d, bad[:3]))
    if fetch:
        d = fetch_raw()
        bad = verify_raw(d, expected)
        if not bad:
            return d
        tried.append("%s (vừa tải): %s" % (d, bad[:3]))
    raise FileNotFoundError("không có dữ liệu Metrica khớp sha256: " + "; ".join(tried or ["không thấy thư mục nào"]))


def cache_file(name, cache_dir=None):
    return os.path.join(cache_dir or CACHE_DIR, CACHE_SUB, name + ".npz")


def _cache_ok(path):
    if not os.path.isfile(path):
        return False
    try:
        with np.load(path, allow_pickle=False) as z:          # Read metadata only; do not decompress P / B.
            meta = json.loads(bytes(z["meta_json"]).decode("utf8"))
    except Exception:
        return False
    want = dict(BALL_DFL, kind="dfl_like_ball")
    got = {k: v for k, v in (meta.get("ball_processing") or {}).items() if k != "seed"}
    return meta.get("converter_version") == CONVERTER_VERSION and got == want


def ensure_cache(names=None, cache_dir=None, raw_dir=None, fetch=True, expected=None, log=print):
    """Ensure cache/external/<name>.npz exists with the current CONVERTER_VERSION (default: DFL-style ball). Return {name: path}."""
    inv = {v: k for k, v in GAMES.items()}
    names = list(names or GAMES.values())
    unknown = [n for n in names if n not in inv]
    if unknown:
        raise ValueError("không phải trận Metrica: %s" % unknown)
    out, todo = {}, []
    for n in names:
        p = cache_file(n, cache_dir)
        out[n] = p
        if not _cache_ok(p):
            todo.append(n)
    if todo:
        raw = raw_dir or find_raw(fetch=fetch, expected=expected)
        for n in todo:
            m = load_game(raw, inv[n])
            errs = validate_match(m)
            if errs:
                raise ValueError("%s không hợp lệ: %s" % (n, errs))
            os.makedirs(os.path.dirname(out[n]), exist_ok=True)
            save_match(m, out[n])
            log("metrica: %s -> %s (T=%d)" % (n, out[n], m.n_frames))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ensure", action="store_true", help="tạo cache/external/METRICA-SG*.npz (tải nếu cần)")
    ap.add_argument("--raw", help="thư mục data/ của metrica-sports/sample-data")
    ap.add_argument("--out", help="thư mục ghi <tên>.npz (vd. <cache>/dfl)")
    ap.add_argument("--games", nargs="+", default=list(GAMES))
    ap.add_argument("--source", default="dfl", choices=("dfl", "metrica"))
    ap.add_argument("--ball", default="dfl", choices=("dfl", "raw"),
                    help="dfl: đưa bóng về quy ước DFL gốc (dfl_like_ball); raw: giữ bóng Metrica")
    for k, v in BALL_DFL.items():
        ap.add_argument("--ball-" + k.replace("_", "-"), dest="ball_" + k, type=float, default=v)
    a = ap.parse_args(argv)
    if a.ensure:
        for n, p in ensure_cache([GAMES.get(g, g) for g in a.games], raw_dir=a.raw).items():
            print(n, p, flush=True)
        return 0
    if not (a.raw and a.out):
        ap.error("cần --raw và --out (hoặc --ensure)")
    bp = {k: getattr(a, "ball_" + k) for k in BALL_DFL}
    os.makedirs(a.out, exist_ok=True)
    for g in a.games:
        m = load_game(a.raw, g, a.source, a.ball, bp)
        errs = validate_match(m)
        if errs:
            raise SystemExit("%s không hợp lệ: %s" % (g, errs))
        path = save_match(m, os.path.join(a.out, m.name + ".npz"))
        print(g, "->", path, "T=%d" % m.n_frames, json.dumps(m.meta["label_info"]),
              "left_group", m.meta["left_group"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
