"""Stage-2 inference: the same interface as stage-1 block 3, feeding the existing IOHMM decoder.

    from src.nap2.infer import score_frames_net
    E, C, K = score_frames_net(V, Bhat, K_rule, {"ckpt": ["a.pt", "b.pt"]})  # checkpoint ensemble

    python -m src.nap2.infer --ckpt runs/nap2/official_s1/ep020.pt --out evaluation/results/nap2.csv

E [T,3]: temperature-calibrated log H/A/D probabilities (T* saved in each checkpoint).
C [T,2]: touching-team probabilities; K [T]: touch probability, normally from the network;
touch_mode="max" combines it with rule-based block-2 K.
Infer half-overlapping training-length windows, retaining central outputs, separately per half
and stride phase so every 25 fps frame is predicted. Never read competition ground truth.
"""

import argparse
import os
import sys

import numpy as np
import torch

from src.nap2.shared import runs_of
from src.nap2.calibrate import log_softmax
from src.nap2.model import build_model

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ck["model_cfg"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


@torch.no_grad()
def forward_full(model, xp, xf, half, chunk, stride, device="cpu", batch_size=64):
    """Raw logits for all frames: had [T,3], touch [T], team [T,2], ball [T,2]."""
    T = len(half)
    out = {"had": np.zeros((T, 3), np.float32), "touch": np.full(T, -10.0, np.float32),
           "team": np.zeros((T, 2), np.float32), "ball": np.zeros((T, 2), np.float32)}
    jobs = []                                   # (window frame indices, keep start, keep stop)
    q = chunk // 4
    for a, b, h in runs_of(half):
        if h <= 0:
            continue
        for ph in range(stride):
            idx = np.arange(a + ph, b, stride)
            n = len(idx)
            if n == 0:
                continue
            if n <= chunk:
                jobs.append((idx, 0, n))
                continue
            starts = list(range(0, n - chunk + 1, chunk // 2))
            if starts[-1] != n - chunk:
                starts.append(n - chunk)
            for s in starts:
                lo = 0 if s == 0 else q
                hi = chunk if s + chunk == n else chunk - q
                jobs.append((idx[s:s + chunk], lo, hi))

    def run(group):
        x1 = torch.from_numpy(np.stack([xp[j[0]] for j in group])).float().to(device)
        x2 = torch.from_numpy(np.stack([xf[j[0]] for j in group])).float().to(device)
        o = model(x1, x2)
        o = {k: v.float().cpu().numpy() for k, v in o.items()}
        for i, (fr, lo, hi) in enumerate(group):
            for k in out:
                out[k][fr[lo:hi]] = o[k][i, lo:hi]

    full = [j for j in jobs if len(j[0]) == chunk]
    for i in range(0, len(full), batch_size):
        run(full[i:i + batch_size])
    for j in jobs:
        if len(j[0]) != chunk:
            run([j])
    return out


def combine(outputs, temps, half):
    """Combine checkpoints by averaging probabilities after applying each checkpoint's temperature."""
    probs = np.mean([np.exp(log_softmax(o["had"] / t)) for o, t in zip(outputs, temps)], axis=0)
    E = np.log(np.clip(probs, 1e-8, 1.0))
    K = np.mean([1 / (1 + np.exp(-o["touch"])) for o in outputs], axis=0)
    C = np.mean([np.exp(log_softmax(o["team"])) for o in outputs], axis=0)
    out_h = np.asarray(half) <= 0
    E[out_h] = np.log(1 / 3)
    C[out_h] = 0.5
    K[out_h] = 0.0
    return E, C, K


def combine_touch(K_rule, K_net, mode="net"):
    if mode == "net" or K_rule is None:
        return K_net
    if mode == "rule":
        return K_rule
    if mode == "max":
        return np.maximum(K_rule, K_net)
    raise ValueError(mode)


def score_frames_net(V, Bhat, K=None, params=None):
    """(E[T,3], C[T,2], K[T]). params: ckpt path or list, device, touch_mode."""
    from src.nap2.features import build_features
    p = dict({"device": "cpu", "touch_mode": "net"}, **(params or {}))
    paths = p["ckpt"] if isinstance(p["ckpt"], (list, tuple)) else [p["ckpt"]]
    outs, temps = [], []
    feats = None
    for path in paths:
        model, ck = load_checkpoint(path, p["device"])
        if feats is None:
            feats = build_features(V, Bhat, ck["feat_cfg"])
        tc = ck["train_cfg"]
        outs.append(forward_full(model, feats["xp"], feats["xf"], V.meta["half"], tc["chunk"],
                                 tc["stride"], p["device"]))
        temps.append(float(ck.get("T", 1.0)))
    E, C, K_net = combine(outs, temps, V.meta["half"])
    return E, C, combine_touch(K, K_net, p["touch_mode"])


def write_submission(path, codes, graded=None):
    """frame,label with H/A/D; X outside graded frames, or outside halves if graded is absent."""
    labels = np.array(["H", "A", "D", "X"])[np.asarray(codes)]
    if graded is not None:
        labels[~np.asarray(graded, bool)] = "X"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        f.write("frame,label\n")
        f.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
    return path


def main(argv=None):
    from src.nap1.ball import estimate_ball
    from src.nap1.decode import decode
    from src.nap2 import shared
    ap = argparse.ArgumentParser(description="Gán nhãn cả trận bằng nấc 2 (mạng + IOHMM) và ghi CSV")
    ap.add_argument("--input", default=str(shared.COMPETITION),
                    help="View của trận (mặc định đầu vào trận đề do pipeline.prepare tạo)")
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--touch-mode", default="net", choices=["net", "rule", "max"])
    a = ap.parse_args(argv)
    raw = shared.load_file(a.input)
    V = shared.as_view(raw)
    P1 = shared.nap1_params()
    Bhat, K_rule = estimate_ball(V, P1)
    E, C, K = score_frames_net(V, Bhat, K_rule, {"ckpt": a.ckpt, "device": a.device,
                                                "touch_mode": a.touch_mode})
    y = decode(E, C, K, V.meta, shared.nap2_decode_params())
    path = write_submission(a.out, y, V.meta["graded"])
    inh = np.asarray(V.meta["half"]) > 0
    n_spells = sum(1 for _, _, v in runs_of(np.where(inh, y, 3)) if v in (0, 1))
    print("đã ghi", path, "| tỷ lệ D trong hiệp {:.3f} | số lượt H/A {}".format((y[inh] == 2).mean(), n_spells))
    print("kiểm tra: python evaluation/validate_submission.py", path)


if __name__ == "__main__":
    sys.exit(main())
