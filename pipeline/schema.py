"""Format contract: every source uses the competition ATD layout.

The competition ATD is read UNCHANGED (no edits, filtering, normalization or direction flips),
preserving reproduction of the four public-window baseline scores. Only other sources
(DFL, Metrica, PFF) are converted to this format.

Layout matching the original ATD:

    P[T, 46, 3]  x, y (normalized 0-1), speed (m/s); NaN = unseen
                 slots 0..23 = group 1 (home), slots 24..45 = group 2 (away)
    B[T, 3]      x, y (normalized 0-1), speed; NaN = unseen

    Speed units (column 2 of P and B) DIFFER by source: ATD = m/s, DFL = km/h (unchanged XML S),
    Metrica = NaN. Models MUST NOT read this column; use View vx/vy (m/s) from pipeline/feed.py.
    View.graded: ATD uses published graded boundaries; DFL/Metrica use (half > 0).
    View converts a reference copy when fitting; its output vx/vy is always m/s. Keep old caches unchanged.
    half[T]      0 = outside both halves, 1 / 2 = half
    graded[T]    True on graded frames (ATD only; from games.py, without reading labels)

Timeline: video frames (ATD: frame_id - 1), 25 fps. Convert to metres: x * pitch[0], y * pitch[1].
"""

import json
import os
from dataclasses import dataclass, field

import numpy as np

N_SLOTS = 46
N_TEAM1 = 24           # Slots 0..23 = group 1; 24..45 = group 2 (ATD: 1-24 Barcelona, 25-46 Athletic).
FPS = 25
TEAM_OF_SLOT = np.array([0] * N_TEAM1 + [1] * (N_SLOTS - N_TEAM1), dtype=np.int8)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.environ.get("MTAH_CACHE_DIR") or os.path.join(ROOT, "cache")
TRAIN_DATA_DIR = os.path.join(ROOT, "train_data")             # raw/ (original datasets), train/, dev/, test/
RAW_DIR = os.environ.get("MTAH_RAW_DIR") or os.path.join(TRAIN_DATA_DIR, "raw")


@dataclass
class Match:
    name: str
    source: str                      # atd | dfl | metrica
    P: np.ndarray                    # [T, 46, 3]
    B: np.ndarray                    # [T, 3]
    half: np.ndarray                 # [T] int8
    meta: dict
    graded: np.ndarray = None        # [T] bool, ATD only.
    extras: dict = field(default_factory=dict)   # Source-specific arrays (y_true, H, ...).

    @property
    def n_frames(self):
        return self.P.shape[0]

    @property
    def pitch(self):
        return tuple(self.meta["pitch"])


def validate_match(m):
    """Check shapes and types; return a list of errors (empty = pass)."""
    errs = []
    T = m.P.shape[0]
    if m.P.shape != (T, N_SLOTS, 3):
        errs.append("P shape {} != (T,{},3)".format(m.P.shape, N_SLOTS))
    if m.B.shape != (T, 3):
        errs.append("B shape {} != (T,3)".format(m.B.shape))
    if m.half.shape != (T,):
        errs.append("half shape {}".format(m.half.shape))
    if m.graded is not None and m.graded.shape != (T,):
        errs.append("graded shape {}".format(m.graded.shape))
    for k in ("fps", "pitch"):
        if k not in m.meta:
            errs.append("meta thiếu {}".format(k))
    for k, v in m.extras.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and k not in m.meta.get("static_extras", []) \
                and v.shape[0] != T and not k.startswith("tbl_"):
            errs.append("extras[{}] dài {} != T={}".format(k, v.shape[0], T))
    fin = m.P[np.isfinite(m.P[..., 0])]
    if fin.size and (m.P[..., :2][np.isfinite(m.P[..., :2])].min() < -0.2
                     or m.P[..., :2][np.isfinite(m.P[..., :2])].max() > 1.2):
        errs.append("toạ độ P ngoài khoảng 0-1 đáng kể (min/max {:.3f}/{:.3f})".format(
            np.nanmin(m.P[..., :2]), np.nanmax(m.P[..., :2])))
    return errs


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def save_match(m, path=None):
    path = path or os.path.join(CACHE_DIR, m.source, m.name + ".npz")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arrays = {"P": m.P, "B": m.B, "half": m.half}
    if m.graded is not None:
        arrays["graded"] = m.graded
    for k, v in m.extras.items():
        arrays["x_" + k] = np.asarray(v)
    arrays["meta_json"] = np.frombuffer(
        json.dumps({"name": m.name, "source": m.source, **m.meta}, default=_jsonable,
                   sort_keys=True).encode("utf8"), dtype=np.uint8)
    np.savez_compressed(path, **arrays)
    return path


def load_match_file(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(bytes(z["meta_json"]).decode("utf8"))
    extras = {k[2:]: z[k] for k in z.files if k.startswith("x_")}
    return Match(name=meta.pop("name"), source=meta.pop("source"), P=z["P"], B=z["B"],
                 half=z["half"], meta=meta, graded=z["graded"] if "graded" in z.files else None,
                 extras=extras)
