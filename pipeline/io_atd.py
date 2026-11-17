"""Read the competition ATD EXACTLY as supplied.

No file edits, interpolation, filtering, unit conversion or direction flips. A malformed line
(e.g. a truncated line missing slot 46 and the ball) leaves the whole frame NaN and is recorded
in meta["bad_frames"], matching starter/baseline.py, which skips it with `continue`.
Use float64 to match the baseline's `float(...)` values exactly.
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

from .schema import N_SLOTS, N_TEAM1, Match

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import games  # noqa: E402

VALUES_PER_LINE = N_SLOTS * 3 + 3


def team_of_slot(meta_path):
    """(Slot-wise 0/1 team array, (home name, away name)); the first declared team is home."""
    root = ET.parse(meta_path).getroot()
    teams = [(t.get("id"), t.findtext("Name")) for t in root.iter("Team")]
    if len(teams) != 2:
        raise ValueError("{}: cần 2 đội, thấy {}".format(meta_path, len(teams)))
    home_id = teams[0][0]
    slots = [p.get("teamId") for p in root.iter("Player")]
    arr = np.array([0 if t == home_id else 1 for t in slots], dtype=np.int8)
    return arr, (teams[0][1], teams[1][1])


def read_atd(raw_path, n_frames):
    """(P[T,46,3], B[T,3], bad_frames, id_mismatch). Index = video frame = frame_id - 1."""
    P = np.full((n_frames, N_SLOTS, 3), np.nan, dtype=np.float64)
    B = np.full((n_frames, 3), np.nan, dtype=np.float64)
    bad, mismatch = [], []
    with open(raw_path) as fh:
        for line_no, line in enumerate(fh):
            parts = line.rstrip("\n").split(":")
            try:
                fid = int(parts[0])
            except ValueError:
                bad.append({"line": line_no + 1, "reason": "frame_id không đọc được"})
                continue
            i = fid - 1
            if not 0 <= i < n_frames:
                bad.append({"line": line_no + 1, "frame_id": fid, "reason": "ngoài [1, T]"})
                continue
            if len(parts) != 3:
                bad.append({"line": line_no + 1, "frame_id": fid, "video_frame": i,
                            "reason": "{} phần thay vì 3 (dòng bị cắt?)".format(len(parts))})
                continue
            vals = np.array((parts[1] + ";" + parts[2]).replace(";", ",").split(","),
                            dtype=np.float64)
            if vals.size != VALUES_PER_LINE:
                bad.append({"line": line_no + 1, "frame_id": fid, "video_frame": i,
                            "reason": "{} giá trị thay vì {}".format(vals.size, VALUES_PER_LINE)})
                continue
            if fid != line_no + 1:
                mismatch.append(line_no + 1)
            P[i] = vals[:N_SLOTS * 3].reshape(N_SLOTS, 3)
            B[i] = vals[N_SLOTS * 3:]
    return P, B, bad, mismatch


def graded_and_half(n_frames, game=games.GAME):
    """Graded-frame and half masks from games.py (published boundaries, no label access)."""
    graded = np.zeros(n_frames, dtype=bool)
    half = np.zeros(n_frames, dtype=np.int8)
    blocks = []
    for h, first, last in games.half_video_blocks(game):
        graded[first:last + 1] = True
        half[first:last + 1] = h
        blocks.append([h, int(first), int(last)])
    return graded, half, blocks


def load_atd(game=games.GAME, source="tactical"):
    paths = games.require(["atd_meta", "atd_raw"], game, None, source)
    n = games.source_frames(source, game)
    P, B, bad, mismatch = read_atd(paths[1], n)
    tos, names = team_of_slot(paths[0])
    if not np.array_equal(tos[:N_SLOTS], np.array([0] * N_TEAM1 + [1] * (N_SLOTS - N_TEAM1))):
        raise ValueError("slot→đội trong .xml không phải 24/22 như giả định")
    graded, half, blocks = graded_and_half(n, game)
    meta = {
        "fps": games.GAMES[game]["fps"],
        "pitch": list(games.GAMES[game]["pitch"]),
        "coords": "chuẩn hoá 0-1, nguyên văn ATD; mét = x*pitch[0], y*pitch[1]",
        "teams": {"group1": names[0], "group2": names[1]},
        "half_blocks_video_frames": blocks,
        "bad_frames": bad,
        "frame_id_mismatch_lines": mismatch,
        "raw_file": os.path.relpath(paths[1], ROOT),
        "verbatim": True,
    }
    m = Match(name="{}_{}".format(game, source), source="atd", P=P, B=B, half=half,
              graded=graded, meta=meta)
    return m
