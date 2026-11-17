"""Connect stage 2 to shared team data (pipeline/FEED.md, train_data/README.md).

All degradation and Views come from pipeline.feed, release v3:
  * train: NEW degradation every epoch via feed.train_view (fresh_train; needs cache/dfl and
           cache/degradation settings, included in the private Kaggle bundle).
           Fallback: rotate PREGENERATED cache/nap2/train/<match>_eXX.npz from export_train.
  * dev: train_data/dev/select selects checkpoints/temperature; dev/report is reporting only.
         Shared team realizations verified against checksums.json.
  * test: reserved train_data/test, never read by stage 2.
  * competition: cache/degradation/competition/BAR-ATH_tactical_view.npz, label-free inference only.

    python -m src.nap2.shared --export-train 8        # pregenerate training realizations (~5 minutes)

Keys match pipeline.export: P [T,46,6] (x/y metres, vx/vy m/s, team 0/1, presence), B [T,3],
cam [T,2], half, graded, fps, pitch, y (0/1/2, -1 excluded), ball_xy (metres).
Training files additionally contain touch_true (0 none, 1/2 team, 3 both).
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np

from pipeline.schema import CACHE_DIR, TRAIN_DATA_DIR

TRAIN_DIR = Path(os.environ.get("NAP2_TRAIN_DIR") or Path(CACHE_DIR) / "nap2" / "train")
RELEASE = Path(CACHE_DIR) / "degradation" / "CURRENT-v3.json"
DEV_DIR = Path(os.environ.get("NAP2_DEV_DIR") or Path(TRAIN_DATA_DIR) / "dev")
COMPETITION = Path(CACHE_DIR) / "degradation" / "competition" / "BAR-ATH_tactical_view.npz"
ARRAY_KEYS = ("P", "B", "cam", "half", "graded", "y", "ball_xy", "touch_true")


def load_file(path, max_frames=None):
    """One .npz -> array dict; optionally truncate to max_frames for a smoke check."""
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    T = len(d["half"])
    if max_frames and max_frames < T:
        d = {k: (v[:max_frames] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == T else v)
             for k, v in d.items()}
    d["name"] = Path(path).stem
    return d


_NAP1_PARAMS = None


def nap1_params():
    """Fitted stage-1 configuration (src/nap1/configs/params_official.json). Stage-2 calls to stage-1
    blocks 2/3/4 explicitly pass this configuration rather than relying on unfitted default values.
    """
    global _NAP1_PARAMS
    if _NAP1_PARAMS is None:
        from src.nap1.params import load_fold_params
        _NAP1_PARAMS = load_fold_params(None)
    return _NAP1_PARAMS


DECODE_CONFIG = Path(__file__).resolve().parent / "configs" / "decode.json"


def nap2_decode_params():
    """Stage-4 configuration AFTER the stage-2 NETWORK: stage-1 settings overridden by
    src/nap2/configs/decode.json, tuned on dev/select. Calibrated network E needs a larger emission weight.
    """
    import json
    from src.nap1.params import Params
    over = json.loads(DECODE_CONFIG.read_text(encoding="utf-8")) if DECODE_CONFIG.is_file() else {}
    return Params.from_dict({**nap1_params().to_dict(), **over})


def as_view(d):
    """load_file dict -> nap1 View via src.nap1.view.from_npz, identical to stage-1 loading."""
    from src.nap1.view import from_npz
    V = from_npz(d, nap1_params())
    V.meta["name"] = d.get("name")
    return V


def runs_of(values):
    """[(start, stop, value)] for contiguous equal-value runs, including out-of-half intervals."""
    v = np.asarray(values)
    if len(v) == 0:
        return []
    cut = np.flatnonzero(v[1:] != v[:-1]) + 1
    starts = np.concatenate(([0], cut))
    stops = np.concatenate((cut, [len(v)]))
    return [(int(a), int(b), v[a].item()) for a, b in zip(starts, stops)]


def train_files(names, train_dir=None):
    """{match: sorted degradation files} for training matches."""
    root = Path(train_dir or TRAIN_DIR)
    out = {n: sorted(glob.glob(str(root / (n + "_e*.npz")))) for n in names}
    missing = [n for n, f in out.items() if not f]
    if missing:
        raise SystemExit("Thiếu bản làm bẩn trong {} cho: {}. Chạy: python -m src.nap2.shared --export-train 8"
                         .format(root, ", ".join(missing)))
    return out


def dev_files(subset, match=None, dev_dir=None):
    """train_data/dev/<subset>/*.npz for select/report, optionally filtered by match."""
    if subset not in ("select", "report"):
        raise ValueError("subset phải là 'select' hoặc 'report'")
    files = sorted(glob.glob(str(Path(dev_dir or DEV_DIR) / subset / "*.npz")))
    if match:
        files = [f for f in files if Path(f).name.startswith(match + "_")]
    if not files:
        raise SystemExit("Không thấy file dev trong {}/{} (giải nén dữ liệu chung vào train_data/)".format(
            dev_dir or DEV_DIR, subset))
    return files


def release_hash():
    """Current data-release identifier, stored in checkpoints for training-data provenance."""
    import json
    for p in (Path(TRAIN_DATA_DIR) / "manifest.json", Path(DEV_DIR).parent / "manifest.json"):
        if p.is_file():
            return json.loads(p.read_text()).get("release_hash")
    return None


_RELEASE = None


def release():
    """Shared release-v3 degradation with per-file SHA-256 verification."""
    global _RELEASE
    if _RELEASE is None:
        from pipeline import feed
        if not RELEASE.is_file():
            raise SystemExit("Thiếu {} (giải nén release-v3.zip: python -m pipeline.prepare)".format(RELEASE))
        _RELEASE = feed.load_release(RELEASE)
    return _RELEASE


def release_files():
    """Repository-relative paths to CURRENT-v3.json and its four referenced files for Kaggle packaging."""
    import json
    from pipeline.schema import ROOT
    ptr = json.loads(RELEASE.read_text())
    return [os.path.relpath(RELEASE, ROOT).replace(os.sep, "/")] + list(ptr["files"].values())


def _as_dict(v, t):
    """pipeline.feed (View, targets) -> dict matching pipeline.export files, plus touch_true."""
    lab, ok = t["labels"], t["label_valid"]
    y = np.where(ok["y_true"], lab["y_true"], -1).astype(np.int8)
    tt = np.where(ok.get("touch_true", np.ones(len(y), bool)), lab.get("touch_true", 0), 0)
    return {"P": v.P.astype(np.float32), "B": v.B.astype(np.float32), "cam": v.cam.astype(np.float32),
            "half": v.half, "graded": v.graded, "fps": v.meta["fps"], "pitch": np.array(v.meta["pitch"]),
            "y": y, "ball_xy": t["ball_xy"].astype(np.float32), "touch_true": np.asarray(tt).astype(np.int8)}


FRESH_CACHE_ENV = "NAP3_FRESH_CACHE"


def release_tag():
    """First 12 SHA-256 characters of CURRENT-v3.json (release pointer with per-file hashes):
    cache-directory key that prevents a new release from reusing old degradation realizations.
    """
    import hashlib
    return hashlib.sha256(RELEASE.read_bytes()).hexdigest()[:12]


def fresh_cache_dirs():
    """Training-degradation cache directories from NAP3_FRESH_CACHE, separated by os.pathsep
    (e.g. read-only Kaggle input, then writable directory). Each contains
    <release_tag>/<match>_r<run>_e<epoch>.npz, or is flat with release_tag.txt. Unset disables caching.
    """
    raw = os.environ.get(FRESH_CACHE_ENV, "")
    tag, out = release_tag(), []
    for x in (Path(x) for x in raw.split(os.pathsep) if x.strip()):
        marker = x / "release_tag.txt"                 # Flat Kaggle-dataset directory: use only for the matching release.
        if marker.is_file():
            if marker.read_text().strip() == tag:
                out.append(x)
        else:
            out.append(x / tag)
    return out


def fresh_key(name, epoch, run=0):
    return "{}_r{}_e{:02d}".format(name, run, epoch)


def _fresh_from_cache(name, epoch, run):
    key = fresh_key(name, epoch, run) + ".npz"
    for d in fresh_cache_dirs():
        if (d / key).is_file():
            from pipeline.split import guard
            guard(name, "train")                   # As in feed.train_view, only training matches may be degraded for learning.
            try:
                return load_file(d / key)
            except Exception:                      # Ignore incomplete / corrupt files and regenerate degradation.
                continue
    return None


def _fresh_to_cache(d):
    """Write to the first writable directory via temporary-file rename, avoiding partial reads by other processes."""
    keep = {k: v for k, v in d.items() if k != "name"}
    for c in fresh_cache_dirs():
        try:
            c.mkdir(parents=True, exist_ok=True)
            dst = c / (d["name"] + ".npz")
            if dst.is_file():
                return dst
            tmp = c / (".%s.%d.tmp.npz" % (d["name"], os.getpid()))
            np.savez_compressed(tmp, **keep)       # ~49 MB per realization (~160 MB uncompressed).
            os.replace(tmp, dst)
            return dst
        except OSError:
            continue
    return None


def fresh_train(name, epoch, run=0, max_frames=None):
    """NEW training-match degradation via feed.train_view, seeded by run/epoch and distinct from dev.
    Return the load_file format. With NAP3_FRESH_CACHE, reuse / store the realization: identical
    (match, epoch, run) means identical seeds and supports sharing across experiments.
    """
    d = _fresh_from_cache(name, epoch, run)
    if d is None:
        from pipeline import feed
        v, t = feed.train_view(name, epoch, run=run, release=release())
        d = _as_dict(v, t)
        d["name"] = fresh_key(name, epoch, run)
        _fresh_to_cache(d)
    T = len(d["half"])
    if max_frames and max_frames < T:
        d = {k: (x[:max_frames] if isinstance(x, np.ndarray) and x.ndim >= 1 and len(x) == T else x)
             for k, x in d.items()}
    d["name"] = "{}_r{}_e{:02d}".format(name, run, epoch)
    return d


def export_train(n_realizations, run=0, out_dir=None):
    """Pregenerate n degradation realizations per training match, as a fallback described above."""
    from pipeline import feed, split
    out = Path(out_dir or TRAIN_DIR)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name in split.roles()["train"]:
        for e in range(n_realizations):
            path = out / "{}_e{:02d}.npz".format(name, e)
            if path.exists():
                continue
            v, t = feed.train_view(name, e, run=run, release=release())
            np.savez_compressed(path, **_as_dict(v, t))
            written.append(path.name)
            print("  ", path.name, flush=True)
    return written


def _fill_one(job):
    name, epoch, run = job
    fresh_train(name, epoch, run)
    return fresh_key(name, epoch, run)


def fill_fresh_cache(runs, epochs, workers=4):
    """Fill NAP3_FRESH_CACHE for training matches x epochs 0..epochs-1 x runs. run is the
    train_switch seed (experiments use 1,2,3; 3+7 training epochs correspond to indices 0..9).
    """
    from concurrent.futures import ProcessPoolExecutor
    from pipeline import split
    if not fresh_cache_dirs():
        raise SystemExit("đặt %s=<thư mục> trước" % FRESH_CACHE_ENV)
    jobs = [(n, e, r) for r in runs for e in range(epochs) for n in split.roles()["train"]
            if not any((d / (fresh_key(n, e, r) + ".npz")).is_file() for d in fresh_cache_dirs())]
    print("cần tạo %d bản (release %s, %s)" % (len(jobs), release_tag(), fresh_cache_dirs()[0]), flush=True)
    with ProcessPoolExecutor(max(1, workers)) as ex:
        for i, k in enumerate(ex.map(_fill_one, jobs), 1):
            print("  %d/%d %s" % (i, len(jobs), k), flush=True)
    d = fresh_cache_dirs()[0]
    d.mkdir(parents=True, exist_ok=True)
    (d / "release_tag.txt").write_text(release_tag() + "\n")      # Also support a flat directory.
    return len(jobs)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Dữ liệu chung cho nấc 2")
    ap.add_argument("--export-train", type=int, metavar="N", help="tạo N bản làm bẩn mỗi trận train")
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--fill-fresh-cache", type=int, metavar="EPOCHS",
                    help="tạo sẵn bản làm bẩn epoch 0..EPOCHS-1 vào NAP3_FRESH_CACHE (vd. 10)")
    ap.add_argument("--runs", type=int, nargs="+", default=[1, 2, 3], help="seed của train_switch")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    if a.export_train:
        w = export_train(a.export_train, a.run)
        print("đã ghi {} file vào {}".format(len(w), TRAIN_DIR))
    if a.fill_fresh_cache:
        fill_fresh_cache(a.runs, a.fill_fresh_cache, a.workers)


if __name__ == "__main__":
    main()
