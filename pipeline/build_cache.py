"""Build caches normalized to the ATD format.

    python -m pipeline.build_cache atd
    python -m pipeline.build_cache dfl [--match DFL-MAT-J03WMX ...]
    python -m pipeline.build_cache metrica
    python -m pipeline.build_cache all

Writes cache/<source>/<name>.npz and cache/MANIFEST.json (input SHA-256 hashes). No random steps.
"""

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

from . import schema
from .schema import save_match, validate_match

VERSION = "1"


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _manifest_update(entries):
    path = os.path.join(schema.CACHE_DIR, "MANIFEST.json")
    cur = json.load(open(path)) if os.path.isfile(path) else {"version": VERSION, "entries": {}}
    cur["version"] = VERSION
    cur["entries"].update(entries)
    os.makedirs(schema.CACHE_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cur, f, indent=1, sort_keys=True)


def build_atd():
    from . import io_atd, io_aux
    import games
    m = io_atd.load_atd()
    H, err, valid = io_aux.load_h()
    m.extras.update({"H": H, "H_reproj_err": err, "H_valid": valid})
    for k, v in io_aux.tagging_table().items():
        m.extras["tbl_tag_" + k] = v
    errs = validate_match(m)
    if errs:
        raise SystemExit("ATD không hợp lệ: {}".format(errs))
    path = save_match(m)
    paths = games.asset_paths()
    src = {k: sha256(p) for k, p in paths.items() if k != "video"}
    _manifest_update({m.source + "/" + m.name: {"file": os.path.relpath(path, schema.ROOT),
                                              "inputs_sha256": src,
                                              "bad_frames": m.meta["bad_frames"]}})
    return path


def _build_dfl_one(mid):
    from . import io_dfl
    m = io_dfl.load_dfl(mid)
    errs = validate_match(m)
    if errs:
        raise SystemExit("{} không hợp lệ: {}".format(mid, errs))
    path = save_match(m)
    src = {os.path.basename(f): sha256(f) for f in (
        io_dfl._file("DFL_02_01_matchinformation", mid), io_dfl._file("DFL_03_02_events_raw", mid),
        io_dfl._file("DFL_04_03_positions_raw_observed", mid))}
    return mid, {"file": os.path.relpath(path, schema.ROOT), "inputs_sha256": src,
                 "event_lag": m.meta["event_lag"]}


def build_dfl(matches=None, workers=3):
    from . import io_dfl
    ids = matches or io_dfl.match_ids()
    out = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for mid, entry in ex.map(_build_dfl_one, ids):
            out["dfl/" + mid] = entry
            print("dfl", mid, "ok", flush=True)
    _manifest_update(out)


def build_metrica():
    from . import io_metrica
    out = {}
    for g in io_metrica.GAMES:
        m = io_metrica.load_metrica(g)
        errs = validate_match(m)
        if errs:
            raise SystemExit("{} không hợp lệ: {}".format(g, errs))
        path = save_match(m)
        d = os.path.join(io_metrica.DIR, g)
        src = {f: sha256(os.path.join(d, f)) for f in sorted(os.listdir(d)) if not f.startswith(".")}
        out["metrica/" + g] = {"file": os.path.relpath(path, schema.ROOT), "inputs_sha256": src}
        print("metrica", g, "ok", flush=True)
    _manifest_update(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=["atd", "dfl", "metrica", "all"])
    ap.add_argument("--match", nargs="+")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args(argv)
    if a.what in ("atd", "all"):
        print("atd ->", build_atd())
    if a.what in ("metrica", "all"):
        build_metrica()
    if a.what in ("dfl", "all"):
        build_dfl(a.match, a.workers)


if __name__ == "__main__":
    sys.exit(main())
