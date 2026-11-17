"""Temperature scaling (stage-2 plan, section 5).

    p_T = softmax(logit / T), T* = argmin_T NLL(val), one scalar found by golden-section over log T.

    python -m src.nap2.calibrate --ckpt runs/nap2/official_s1/ep020.pt --file train_data/dev/select/DFL-MAT-J03WN1_r00.npz [--write]

Recompute T* on one frozen dev realization and print NLL / ECE before and after calibration.
"""

import argparse
import os
import math

import numpy as np


def log_softmax(z):
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def nll(logits, y, w=None, T=1.0):
    ok = y >= 0
    if w is not None:
        ok &= w > 0
    if not ok.any():
        return float("nan")
    lp = log_softmax(logits[ok] / T)
    ww = np.ones(ok.sum()) if w is None else w[ok]
    return float(-(lp[np.arange(len(lp)), y[ok]] * ww).sum() / ww.sum())


def fit_temperature(logits, y, w=None, lo=0.05, hi=20.0, iters=50):
    """T* minimizes NLL; NLL is unimodal in log T, so golden-section search suffices."""
    a, b = math.log(lo), math.log(hi)
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = nll(logits, y, w, math.exp(c)), nll(logits, y, w, math.exp(d))
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = nll(logits, y, w, math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = nll(logits, y, w, math.exp(d))
    return float(math.exp((a + b) / 2))


def ece(logits, y, w=None, T=1.0, n_bins=15):
    """Expected calibration error over maximum class probabilities."""
    ok = y >= 0
    if w is not None:
        ok &= w > 0
    p = np.exp(log_softmax(logits[ok] / T))
    conf = p.max(1)
    hit = (p.argmax(1) == y[ok]).astype(float)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    tot = 0.0
    for k in range(n_bins):
        m = bins == k
        if m.any():
            tot += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return float(tot)


def main(argv=None):
    import torch
    from src.nap2.dataset import prepare_file
    from src.nap2.infer import forward_full, load_checkpoint
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--file", required=True, help="một file dev, vd. train_data/dev/select/DFL-MAT-J03WN1_r00.npz")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--write", action="store_true", help="ghi T* mới vào checkpoint")
    a = ap.parse_args(argv)
    model, ck = load_checkpoint(a.ckpt, a.device)
    d = prepare_file(a.file, feat_cfg=ck["feat_cfg"])
    o = forward_full(model, d["xp"], d["xf"], d["half"], ck["train_cfg"]["chunk"],
                     ck["train_cfg"]["stride"], a.device)
    T = fit_temperature(o["had"], d["y"], d["w"])
    print("T* = {:.3f} (checkpoint đang có {:.3f})".format(T, ck.get("T", 1.0)))
    print("NLL {:.4f} -> {:.4f} | ECE {:.4f} -> {:.4f}".format(
        nll(o["had"], d["y"], d["w"]), nll(o["had"], d["y"], d["w"], T),
        ece(o["had"], d["y"], d["w"]), ece(o["had"], d["y"], d["w"], T)))
    if a.write:
        ck["T"] = T
        ck["T_source"] = {"files": [os.path.basename(a.file)]}
        torch.save(ck, a.ckpt)
        print("đã ghi vào", a.ckpt)


if __name__ == "__main__":
    main()
