"""Competition homography and Smart Tagging: read unchanged using the organisers' loaders."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "starter")):
    if p not in sys.path:
        sys.path.insert(0, p)
import load_homography  # noqa: E402
import load_smart_tagging  # noqa: E402


def load_h(source="tactical"):
    H, err, valid = load_homography.load(source=source)
    return H, err.astype(np.float64), valid


def tagging_table():
    """Array structure: start_frame, end_frame (half-open), string code, team, type, half."""
    ev = load_smart_tagging.load()
    return {
        "start_frame": np.array([e["start_frame"] for e in ev], dtype=np.int64),
        "end_frame": np.array([e["end_frame"] for e in ev], dtype=np.int64),
        "code": np.array([e["code"] or "" for e in ev]),
        "team": np.array([e["tags"].get("Team") or "" for e in ev]),
        "type": np.array([e["tags"].get("Type") or "" for e in ev]),
        "half": np.array([e["tags"].get("Half") or "" for e in ev]),
    }
