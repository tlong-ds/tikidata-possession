"""Cross-validation scoring for folds k = 2, 3, 4: A = cv_a<k> (bt04_metrica configuration),
B = cv_f<k> (PFF pretraining then fine-tuning), COMBINED = A + B checkpoints with A's decoder
(as in final_ens). Five frozen dev realizations of the held-out match, both original and
target-like smoothed ball (analysis.smooth_dev.smoothed). Run from the nap3-dev worktree:

    PYTHONPATH=. python cv_score.py [k ...]
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

S = Path("/Users/bunnypro/Projects/more-than-a-hack-2027-nap3-dev/cache/cv")
ROOT = Path("/Users/bunnypro/Projects/more-than-a-hack-2027-nap3-dev")


def smooth_files(k):
    from analysis.smooth_dev import smoothed
    from src.nap2 import shared
    src = sorted((ROOT / "cache" / "folddev" / ("fold%d" % k) / "select").glob("*.npz"))
    out = S / "folddev_smooth" / ("fold%d" % k) / "select"
    out.mkdir(parents=True, exist_ok=True)
    res = []
    for i, f in enumerate(src):
        p = out / f.name
        if not p.is_file():
            raw = dict(np.load(f, allow_pickle=True))
            raw["B"] = smoothed(raw, i, "select")
            np.savez_compressed(p, **raw)
        res.append(p)
    return src, res


def export(k, exp, tag, files):
    W = S / "cvwork" / ("fold%d" % k)
    done = list((W / "raw" / tag).glob("%s_s*__*.npz" % exp))
    if len(done) >= 5 * len(list((ROOT / "artifacts" / "exp" / exp / "ckpt").glob("s*/ep*.pt"))):
        return
    ck = sorted(str(p) for p in (ROOT / "artifacts" / "exp" / exp / "ckpt").glob("s*/ep*.pt"))
    subprocess.run([sys.executable, "-m", "analysis.retune_smooth", "export", "--work", str(W), "--tag", tag,
                    "--ckpt", *ck, "--threads", "4", "--files", *map(str, files)], cwd=ROOT, check=True,
                   stdout=open(S / ("cv_export_%s_%s.log" % (exp, tag)), "w"), stderr=subprocess.STDOUT)


def main():
    from analysis.retune_smooth import Store, load_params, paired
    folds = [int(x) for x in sys.argv[1:]] or [2, 3, 4]
    out = {}
    for k in folds:
        A, B = "cv_a%d" % k, os.environ.get("BEXP", "cv_f") + str(k)
        orig, sm = smooth_files(k)
        for e in (A, B):
            export(k, e, "smooth", sm)
            export(k, e, "orig", orig)
        pool = ProcessPoolExecutor(4, mp_context=get_context("spawn"))
        st = Store(S / "cvwork" / ("fold%d" % k), ["smooth", "orig"], pool)
        keys = [x for x in st.files if x[0] == "smooth"] + [x for x in st.files if x[0] == "orig"]
        pick = lambda e: [c for c in st.cids if c.startswith(e + "/")]
        PA = load_params(ROOT / "artifacts" / "exp" / A / "decode_switch.json")
        PB = load_params(ROOT / "artifacts" / "exp" / B / "decode_switch.json")
        a = st.scores(pick(A), [PA], keys)[0]
        b = st.scores(pick(B), [PB], keys)[0]
        g = st.scores(pick(A) + pick(B), [PA], keys)[0]
        st.save()
        pool.shutdown()
        out[k] = {"A": a.tolist(), "B": b.tolist(), "ENS": g.tolist(),
                  "B_vs_A": {"smooth": paired(a[:5], b[:5]), "orig": paired(a[5:], b[5:])},
                  "ENS_vs_A": {"smooth": paired(a[:5], g[:5]), "orig": paired(a[5:], g[5:])}}
        print("fold %d  A mượt %.4f gốc %.4f | B mượt %.4f (%+.4f, %d/5) gốc %.4f (%+.4f) | GỘP mượt %.4f (%+.4f, %d/5) gốc %.4f (%+.4f)" % (
            k, a[:5].mean(), a[5:].mean(), b[:5].mean(), b[:5].mean() - a[:5].mean(), int((b[:5] > a[:5]).sum()),
            b[5:].mean(), b[5:].mean() - a[5:].mean(), g[:5].mean(), g[:5].mean() - a[:5].mean(),
            int((g[:5] > a[:5]).sum()), g[5:].mean(), g[5:].mean() - a[5:].mean()), flush=True)
    if len(out) > 1:
        for what in ("B", "ENS"):
            for part, sl in (("smooth", slice(0, 5)), ("orig", slice(5, 10))):
                d = np.concatenate([np.asarray(out[k][what])[sl] - np.asarray(out[k]["A"])[sl] for k in out])
                fm = [float((np.asarray(out[k][what])[sl] - np.asarray(out[k]["A"])[sl]).mean()) for k in out]
                print("%s - A %s: trung bình %+.4f, thắng %d/%d bản, theo fold %s" % (
                    what, part, d.mean(), int((d > 0).sum()), d.size, ["%+.4f" % x for x in fm]))
    (S / ("cv_scores_%s_%s.json" % (os.environ.get("BEXP", "cv_f"), "_".join(map(str, folds))))).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
