"""Check caches against published figures (deck/EDA/baseline).  python -m pipeline.checks"""

import json
import os
import sys

import numpy as np

from .load import load_match
from .schema import ROOT, validate_match

sys.path.insert(0, os.path.join(ROOT, "starter"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
FAILS = []


def check(name, ok, detail=""):
    print("{} {}  {}".format("PASS" if ok else "FAIL", name, detail))
    if not ok:
        FAILS.append(name)


def nearest_team(m, group_slots=24):
    """Team (0/1) of the nearest player, using starter/baseline.py rules; -1 when the ball is unseen."""
    L, W = m.pitch
    seen = np.isfinite(m.B[:, 0])
    dx = (m.P[:, :, 0].astype(np.float64) - m.B[:, 0, None]) * L
    dy = (m.P[:, :, 1].astype(np.float64) - m.B[:, 1, None]) * W
    d = dx * dx + dy * dy
    d[~np.isfinite(d)] = np.inf
    best = d.argmin(1)                                   # First occurrence of the minimum, as in the baseline.
    ok = seen & np.isfinite(d.min(1))
    return np.where(ok, (best >= group_slots).astype(np.int8), -1)


def check_atd():
    import baseline
    m = load_match("BAR-ATH_tactical")
    check("atd.hợp lệ", validate_match(m) == [])
    T = m.n_frames
    check("atd.T=199319", T == 199319, str(T))
    bad = m.meta["bad_frames"]
    check("atd.chỉ dòng 2 lỗi", [b["video_frame"] for b in bad] == [1],
          "bad_frames={}".format(bad))
    seen = np.isfinite(m.B[:, 0])
    check("atd.bóng thấy 56,9%", abs(seen.mean() - 0.569) < 0.001, "{:.4f}".format(seen.mean()))
    check("atd.khung chấm 158857", int(m.graded.sum()) == 158857, str(int(m.graded.sum())))
    check("atd.bóng thấy trên khung chấm 59,3%", abs(seen[m.graded].mean() - 0.593) < 0.001,
          "{:.4f}".format(seen[m.graded].mean()))
    n_obj = np.isfinite(m.P[:, :, 0]).sum(1)
    check("atd.TB 20,5 đối tượng/khung (trong hiệp)", abs(n_obj[m.graded].mean() - 20.5) < 0.3,
          "{:.2f}".format(n_obj[m.graded].mean()))
    check("atd.2,4% khung chấm có >22 đối tượng", abs((n_obj[m.graded] > 22).mean() - 0.024) < 0.002,
          "{:.4f}".format((n_obj[m.graded] > 22).mean()))
    H = m.extras["H_valid"]
    check("atd.homography valid 100%", bool(H.all()) and len(H) == T)
    check("atd.smart tagging 638 mục", len(m.extras["tbl_tag_code"]) == 638)
    # Baseline: labels reconstructed from the cache must match the shipped baseline files frame by frame.
    team = nearest_team(m)
    codes = team.astype(np.int8)       # baseline: HOME=0, AWAY=1, NOBODY=-1
    for name, w in (("raw", 0), ("smooth25", 25), ("smooth50", 50), ("smooth125", 125)):
        mine = baseline.smooth(codes, w)
        ref = np.loadtxt(os.path.join(ROOT, "evaluation", "results", "baseline_{}.csv".format(name)),
                         delimiter=",", skiprows=1, dtype=str)[:, 1]
        diff = int((mine != ref).sum())
        check("atd.baseline_{} trùng file ship".format(name), diff == 0, "{} khung khác".format(diff))


def check_dfl():
    stats = {r["match"]: r for r in json.load(open(os.path.join(ROOT, "eda", "dfl_stats.json")))["per_match"]}
    for mid, ref in sorted(stats.items()):
        m = load_match("DFL-MAT-" + mid)
        y = m.extras["y_true"]
        check("dfl.{} hợp lệ".format(mid), validate_match(m) == [])
        check("dfl.{} số khung".format(mid), m.n_frames == ref["frames"], str(m.n_frames))
        check("dfl.{} tỷ lệ D".format(mid), abs((y == 2).mean() - ref["dead_share"]) < 1e-3,
              "{:.4f} vs {:.4f}".format((y == 2).mean(), ref["dead_share"]))
        npos = 0
        for h in (1, 2):
            yy = y[m.half == h]
            lab = yy[(yy != 2)]
            ch = np.r_[True, yy[1:] != yy[:-1]]
            npos += int(((yy != 2) & ch).sum())
        check("dfl.{} số lượt".format(mid), npos == ref["n_possessions"], "{} vs {}".format(npos, ref["n_possessions"]))
        team = nearest_team(m)
        alive = y != 2
        acc = float((team[alive] == y[alive]).mean())
        check("dfl.{} gần-bóng-nhất".format(mid), abs(acc - ref["nearest_acc_alive"]) < 2e-3,
              "{:.4f} vs {:.4f}".format(acc, ref["nearest_acc_alive"]))
        n_team = [np.isfinite(m.P[:, :24, 0]).sum(1).max(), np.isfinite(m.P[:, 24:, 0]).sum(1).max()]
        check("dfl.{} tối đa ≤ 24/22 người".format(mid), n_team[0] <= 24 and n_team[1] <= 22, str(n_team))
        lag = m.meta["event_lag"]
        print("     τ hiệp1={} hiệp2={} ms (dist {:.1f}→{:.1f} m)".format(
            lag["1"]["lag_ms"], lag["2"]["lag_ms"], lag["1"]["median_dist_at_0_m"], lag["1"]["median_dist_m"]))


def check_metrica():
    for g, n_ev in (("Sample_Game_1", 1745), ("Sample_Game_2", 1935), ("Sample_Game_3", 3620)):
        m = load_match(g)
        check("metrica.{} hợp lệ".format(g), validate_match(m) == [])
        check("metrica.{} số event".format(g), len(m.extras["tbl_ev_type"]) == n_ev, str(len(m.extras["tbl_ev_type"])))
        check("metrica.{} 22 người/khung".format(g),
              abs(np.isfinite(m.P[:, :, 0]).sum(1).mean() - 22) < 0.5)


if __name__ == "__main__":
    for fn in (check_atd, check_dfl, check_metrica):
        fn()
    print("\n{} kiểm tra lỗi".format(len(FAILS)) if FAILS else "\nTất cả đạt")
    sys.exit(1 if FAILS else 0)
