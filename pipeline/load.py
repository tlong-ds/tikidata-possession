"""Single loading entry point for downstream stages: load_match("BAR-ATH_tactical") / load_match("DFL-MAT-J03WMX")."""

import glob
import os

from .schema import CACHE_DIR, load_match_file


def available():
    return sorted(os.path.relpath(p, CACHE_DIR)[:-4] for p in glob.glob(os.path.join(CACHE_DIR, "*", "*.npz")))


def load_match(name):
    hits = glob.glob(os.path.join(CACHE_DIR, "*", name + ".npz"))
    if len(hits) != 1:
        raise FileNotFoundError("không thấy cache cho {!r}; có: {}. Chạy python -m pipeline.build_cache all"
                                .format(name, available()))
    return load_match_file(hits[0])
