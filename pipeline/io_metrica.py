"""Metrica sample data (three matches) -> ATD format.

Coordinates are already 0-1 on a 105 x 68 m pitch, so only pack players into 46 slots: home =
group 1 (0..23), away = group 2 (24..45). Speed is NaN because Metrica does not provide it.
No direction flip.

Metrica has no possession flags. `y_draft` is an automatic event-derived DRAFT (0=H, 1=A, 2=D,
-1=no event), intended for team review at stage 1b (manual annotation of other matches is allowed).
"""

import json
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from .schema import N_SLOTS, N_TEAM1, RAW_DIR, Match

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(RAW_DIR, "metrica-sample-data", "data")
PITCH = [105.0, 68.0]
GAMES = ("Sample_Game_1", "Sample_Game_2", "Sample_Game_3")

EVENT_COLS = ("team", "type", "subtype", "period", "start_frame", "end_frame",
              "start_x", "start_y", "end_x", "end_y")


def _read_csv_tracking(path, n_players_max=N_TEAM1):
    df = pd.read_csv(path, header=None, skiprows=3, dtype=np.float64)
    period, frame = df[0].to_numpy(), df[1].to_numpy()
    xy = df.iloc[:, 3:].to_numpy()
    n_pl = (xy.shape[1] - 2) // 2
    if n_pl > n_players_max:
        raise ValueError("{}: {} cầu thủ > {}".format(path, n_pl, n_players_max))
    return period, frame, xy[:, :2 * n_pl].reshape(len(df), n_pl, 2), xy[:, -2:]


def _events_csv(path):
    df = pd.read_csv(path)
    return {
        "team": df["Team"].map({"Home": 1, "Away": 2}).fillna(0).to_numpy(np.int8),
        "type": df["Type"].fillna("").to_numpy(str), "subtype": df["Subtype"].fillna("").to_numpy(str),
        "period": df["Period"].to_numpy(np.int8),
        "start_frame": df["Start Frame"].to_numpy(np.int64) - 1,    # One-based frame -> array index.
        "end_frame": df["End Frame"].to_numpy(np.int64) - 1,
        "start_x": df["Start X"].to_numpy(float), "start_y": df["Start Y"].to_numpy(float),
        "end_x": df["End X"].to_numpy(float), "end_y": df["End Y"].to_numpy(float),
    }


def _events_json(path):
    d = json.load(open(path))["data"]
    def g(o, k):
        return np.nan if o is None or o.get(k) is None else float(o[k])
    rows = {"team": [], "type": [], "subtype": [], "period": [], "start_frame": [], "end_frame": [],
            "start_x": [], "start_y": [], "end_x": [], "end_y": []}
    for e in d:
        rows["team"].append(1 if e["team"]["id"].endswith("A") else 2)
        rows["type"].append(e["type"]["name"])
        st = e.get("subtypes")
        rows["subtype"].append((st[0] if isinstance(st, list) and st else st or {}).get("name", "")
                               if not isinstance(st, str) else st)
        rows["period"].append(e["period"])
        rows["start_frame"].append(e["start"]["frame"] - 1); rows["end_frame"].append(e["end"]["frame"] - 1)
        rows["start_x"].append(g(e["start"], "x")); rows["start_y"].append(g(e["start"], "y"))
        rows["end_x"].append(g(e["end"], "x")); rows["end_y"].append(g(e["end"], "y"))
    out = {k: np.array(v) for k, v in rows.items()}
    for k in ("team", "period"):
        out[k] = out[k].astype(np.int8)
    for k in ("start_frame", "end_frame"):
        out[k] = out[k].astype(np.int64)
    return out


def draft_labels(ev, T):
    """Draft H/A/D labels from events. Follow CHALLENGE.md: challenges / loose balls retain the
    previous team; D only during stoppages (BALL OUT, FAULT RECEIVED) until the next restart.
    """
    y = np.full(T, -1, dtype=np.int8)
    order = np.argsort(ev["start_frame"], kind="stable")
    state, since = 2, 0                      # Before the first event: play has not started (D).
    live_types = {"PASS", "SHOT", "CARRY", "RECOVERY"}
    for i in order:
        f = int(ev["start_frame"][i]); t = ev["type"][i]; tm = int(ev["team"][i])
        if t in live_types and tm in (1, 2):
            new = 0 if tm == 1 else 1
        elif t == "SET PIECE" and tm in (1, 2):
            new = 0 if tm == 1 else 1
        elif t in ("BALL OUT", "FAULT RECEIVED"):
            new = 2
        else:
            continue
        if new != state:
            y[since:f] = state; state, since = new, f
    y[since:] = state
    first = int(ev["start_frame"].min())
    y[:max(first, 0)] = -1
    return y


def load_game12(game):
    d = os.path.join(DIR, game)
    ph, fh, xyh, bh = _read_csv_tracking(os.path.join(d, game + "_RawTrackingData_Home_Team.csv"))
    pa, fa, xya, ba = _read_csv_tracking(os.path.join(d, game + "_RawTrackingData_Away_Team.csv"))
    if not (np.array_equal(fh, fa) and np.array_equal(ph, pa)):
        raise ValueError(game + ": khung home/away không khớp")
    T = len(fh)
    if not np.array_equal(fh, np.arange(1, T + 1)):
        raise ValueError(game + ": Frame không liên tục từ 1")
    P = np.full((T, N_SLOTS, 3), np.nan, dtype=np.float32)
    P[:, :xyh.shape[1], :2] = xyh
    P[:, N_TEAM1:N_TEAM1 + xya.shape[1], :2] = xya
    B = np.full((T, 3), np.nan, dtype=np.float32)
    B[:, :2] = bh
    ev = _events_csv(os.path.join(d, game + "_RawEventsData.csv"))
    return P, B, ph.astype(np.int8), ev


def load_game3():
    d = os.path.join(DIR, "Sample_Game_3")
    root = ET.parse(os.path.join(d, "Sample_Game_3_metadata.xml")).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    team_of_player = {p.get("id"): (0 if p.get("teamId") == team_ids[0] else 1) for p in root.iter("Player")}
    chan_player = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    specs = []
    for sp in root.iter("DataFormatSpecification"):
        players = []
        for grp in sp.find("SplitRegister"):
            ref = grp.find("PlayerChannelRef").get("playerChannelId")
            players.append(chan_player[ref])
        specs.append((int(sp.get("startFrame")), int(sp.get("endFrame")), players))
    raw = open(os.path.join(d, "Sample_Game_3_tracking.txt")).read().splitlines()
    T = len(raw)
    P = np.full((T, N_SLOTS, 3), np.nan, dtype=np.float32)
    B = np.full((T, 3), np.nan, dtype=np.float32)
    half = np.zeros(T, dtype=np.int8)
    params = {p.findtext("Name"): p.findtext("Value") for p in root.iter("ProviderParameter")}
    h1_end, h2_start, h2_end = (int(params["first_half_end"]), int(params["second_half_start"]),
                                int(params["second_half_end"]))
    half[:h1_end] = 1
    half[h2_start - 1:h2_end] = 2
    slot_of = []
    for s, e, players in specs:               # Each substitution creates a new player specification.
        cnt = [0, 0]; m = []
        for pid in players:
            t = team_of_player[pid]
            m.append(t * N_TEAM1 + cnt[t]); cnt[t] += 1
        slot_of.append((s, e, np.array(m)))
    for i, line in enumerate(raw):
        fid, pl, ball = line.split(":")
        if int(fid) != i + 1:
            raise ValueError("Game 3: frame_id lệch ở dòng {}".format(i + 1))
        spec = None
        for s, e, m in slot_of:
            if s <= i + 1 <= e:
                spec = m
                break
        vals = np.array(pl.replace(";", ",").split(","), dtype=np.float64).reshape(-1, 2)
        P[i, spec, 0] = vals[:, 0]
        P[i, spec, 1] = vals[:, 1]
        b = np.array(ball.split(","), dtype=np.float64)
        B[i, :2] = b
    ev = _events_json(os.path.join(d, "Sample_Game_3_events.json"))
    return P, B, half, ev


def load_metrica(game):
    if game in ("Sample_Game_1", "Sample_Game_2"):
        P, B, half, ev = load_game12(game)
    elif game == "Sample_Game_3":
        P, B, half, ev = load_game3()
    else:
        raise KeyError(game)
    T = len(P)
    meta = {"fps": 25, "pitch": PITCH,
            "coords": "chuẩn hoá 0-1; mét = x*pitch[0], y*pitch[1]; speed = NaN (Metrica không có)",
            "labels": {"0": "H", "1": "A", "2": "D", "-1": "chưa có"}, "static_extras": []}
    extras = {"y_draft": draft_labels(ev, T)}
    for k, v in ev.items():
        extras["tbl_ev_" + k] = v
    return Match(name=game, source="metrica", P=P, B=B, half=half, meta=meta, extras=extras)
