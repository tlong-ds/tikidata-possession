"""Pre-kickoff dead ball: assign D to graded frames before each half's kickoff.

Tracking may start before kickoff (target half 2 starts at frame 110808, kickoff near 113742).
The ball is not in play, but the network has not seen this interval in training and predicts H/A.

Locate kickoff from Metrica's AUTOMATIC Smart Tagging supplied by the organisers (DATA.md): each
half has a SET PIECES / "Kick Off (Start)" window of about 5 s. Choose its visible tracked ball
nearest the center spot; without visible ball, use the window midpoint. No labels, ground truth
or hard-coded frame indices. Skip intervals shorter than MIN_GAP (tracking starts at kickoff,
as in target half 1) to avoid changes caused by ball-position noise.
"""

import xml.etree.ElementTree as ET

import numpy as np

CODE_D = 2
KICKOFF_CODE = "set pieces"
KICKOFF_TYPE = "kick off (start)"
HALF_TAG = {"1st half": 1, "2nd half": 2}
MIN_GAP_S = 2.0


def kickoff_windows(xml_path, fps):
    """Smart Tagging XML -> {half: (start_frame, end_frame)} for Kick Off (Start); end is exclusive.
    Times are seconds from tactical-video frame 0 (DATA.md, "Time base -- verified").
    """
    out = {}
    for inst in ET.parse(xml_path).iter("instance"):
        if (inst.findtext("code") or "").strip().lower() != KICKOFF_CODE:
            continue
        tags = {(lab.findtext("group") or "").strip().lower(): (lab.findtext("text") or "").strip().lower()
                for lab in inst.findall("label")}
        if tags.get("type") != KICKOFF_TYPE or tags.get("half") not in HALF_TAG:
            continue
        half = HALF_TAG[tags["half"]]
        if half in out:
            raise ValueError("%s: hai thẻ Kick Off (Start) cho hiệp %d" % (xml_path, half))
        out[half] = (int(round(float(inst.findtext("start")) * fps)), int(round(float(inst.findtext("end")) * fps)))
    return out


def locate(B, window):
    """Kickoff frame: visible ball nearest centered View (0,0) within the window; midpoint if none."""
    s, e = window
    e = min(e, len(B))
    if e <= s:
        raise ValueError("cửa sổ giao bóng rỗng %r" % (window,))
    w = np.arange(s, e)
    ok = B[w, 2] > 0
    if not ok.any():
        return int(s + (e - s) // 2)
    r = np.hypot(B[w, 0], B[w, 1])[ok]
    return int(w[ok][np.argmin(r)])          # argmin chooses the earliest frame on ties, deterministically.


def apply(codes, half, graded, B, windows, fps):
    """Assign D to graded pre-kickoff frames in each half. Return (new codes, {half: info})."""
    codes = np.array(codes, copy=True)
    half = np.asarray(half)
    graded = np.asarray(graded, bool)
    info = {}
    for h, win in sorted(windows.items()):
        idx = np.flatnonzero(half == h)
        if not len(idx):
            continue
        first = int(idx[0])
        k = locate(B, win)
        n = 0
        if k - first >= MIN_GAP_S * fps:
            m = np.zeros(len(codes), bool)
            m[first:k] = True
            m &= graded & (half == h)
            n = int((codes[m] != CODE_D).sum())
            codes[m] = CODE_D
        info[str(h)] = {"half_first": first, "kickoff": k, "window": list(win), "changed_to_D": n}
    return codes, info
