"""Stage-3 switchnet inference: retain ALL heads, including switch, ensemble checkpoints,
decode, and export probabilities in src.core.blend format (E, C, K, S, half, source).

Usually called by exp.py. Manual example, evaluating dev/report ONCE after configuration selection:
    python -m src.core.infer_switch export --ckpt "artifacts/exp/baseline/ckpt/s*/ep*.pt" --decode-params artifacts/exp/baseline/decode_switch.json --subset report --out artifacts/exp/baseline/report

src.nap2.infer.forward_full discards heads other than had/touch/team/ball. This implementation
derives keys and sizes directly from network outputs. Always load checkpoints with weights_only=True.
"""

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.nap2.calibrate import log_softmax
from src.nap2.shared import runs_of

from . import decode_switch as dsw
from .net import build_model

FILL = {"touch": -10.0, "switch": -10.0}       # Outside-half logits


def load_checkpoint(path, device="cpu"):
    from torch.torch_version import TorchVersion
    with torch.serialization.safe_globals([TorchVersion]):
        ck = torch.load(path, map_location=device, weights_only=True)
    model = build_model(ck["model_cfg"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


def layout_of(cks):
    """Common checkpoint layout dict for input construction; all checkpoints must share it."""
    lays = [ck.get("layout") or {} for ck in cks]
    keys = {repr((l.get("xp_names"), l.get("xf_names"))) for l in lays}
    if len(keys) > 1:
        raise ValueError("các checkpoint có cột đầu vào khác nhau, không gộp được")
    return lays[0] or None


def _jobs(half, chunk, stride):
    """As in src.nap2.infer.forward_full: half-overlapping windows per half and stride phase, retaining centers."""
    jobs = []
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
    return jobs


@torch.no_grad()
def forward_full(model, xp, xf, half, chunk, stride, device="cpu", batch_size=64):
    """All-head logits at 25 fps: {key: [T,...]}; outside halves: 0, except touch/switch = -10."""
    T = len(half)
    out = {}

    def run(group):
        x1 = torch.from_numpy(np.stack([xp[j[0]] for j in group])).float().to(device)
        x2 = torch.from_numpy(np.stack([xf[j[0]] for j in group])).float().to(device)
        o = {k: v.float().cpu().numpy() for k, v in model(x1, x2).items()}
        for k, v in o.items():
            if k not in out:
                out[k] = np.full((T,) + v.shape[2:], FILL.get(k, 0.0), np.float32)
        for i, (fr, lo, hi) in enumerate(group):
            for k in o:
                out[k][fr[lo:hi]] = o[k][i, lo:hi]

    jobs = _jobs(half, chunk, stride)
    full = [j for j in jobs if len(j[0]) == chunk]
    for i in range(0, len(full), batch_size):
        run(full[i:i + batch_size])
    for j in jobs:
        if len(j[0]) != chunk:
            run([j])
    return out


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


def fit_switch_temperature(logits, target, mask, lo=0.02, hi=8.0, iters=40):
    """Positive T minimizing BCE(sigmoid(logit / T), target) on masked frames, by golden-section search over log T."""
    z = np.asarray(logits, np.float64)[mask]
    t = np.asarray(target, np.float64)[mask]
    if z.size == 0:
        return 1.0

    def bce(T):
        x = z / T
        return float(np.mean(np.logaddexp(0, x) - t * x))

    a, b = np.log(lo), np.log(hi)
    g = (np.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = bce(np.exp(c)), bce(np.exp(d))
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = bce(np.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = bce(np.exp(d))
    return float(np.exp((a + b) / 2))


def frame_switch_target(y, half, sigma_early_f=4.0, sigma_late_f=2.0):
    """Soft 25 fps switch targets for T_s calibration; same definition as src.core.switch_target, sigma in frames."""
    from .switch_target import switch_target
    yy = np.where(np.asarray(half) > 0, np.asarray(y), -1)
    out = np.zeros((len(yy), 2), np.float32)
    for a, b, h in runs_of(half):
        if h <= 0:
            continue
        t = switch_target(torch.from_numpy(yy[a:b].astype(np.int64))[None], sigma_early_f, sigma_late_f)
        out[a:b] = t[0].numpy()
    return out


def combine(outputs, temps, s_temps, half):
    """(E, C, K, S): E/C/K as in src.nap2.infer.combine; S averages sigmoid(switch / T_s), zero outside halves."""
    probs = np.mean([np.exp(log_softmax(o["had"] / t)) for o, t in zip(outputs, temps)], axis=0)
    E = np.log(np.clip(probs, 1e-8, 1.0))
    K = np.mean([sigmoid(o["touch"]) for o in outputs], axis=0)
    C = np.mean([np.exp(log_softmax(o["team"])) for o in outputs], axis=0)
    S = np.mean([sigmoid(o["switch"] / ts) for o, ts in zip(outputs, s_temps)], axis=0)
    out_h = np.asarray(half) <= 0
    E[out_h] = np.log(1 / 3)
    C[out_h] = 0.5
    K[out_h] = 0.0
    S[out_h] = 0.0
    return E, C, K, S


def run_models(prepared, models, device="cpu"):
    """prepared: src.core.feats.prepare_* dict -> checkpoint-ensembled (E, C, K, S)."""
    outs, temps, stemps = [], [], []
    for model, ck in models:
        tc = ck["train_cfg"]
        outs.append(forward_full(model, prepared["xp"], prepared["xf"], prepared["half"], tc["chunk"],
                                 tc["stride"], device))
        temps.append(float(ck.get("T", 1.0)))
        stemps.append(float(ck.get("T_switch", 1.0)))
    return combine(outs, temps, stemps, prepared["half"])


def decode_all(E, S, C, K, half, graded, params=None, how="switch"):
    """how: 'switch' (decode_switch) | 'gate' (S through nap1 gate) | 'nap1' (network C,K through src.nap1.decode)."""
    meta = {"half": half, "graded": graded, "fps": 25}
    if how == "switch":
        return dsw.to_eval_codes(dsw.decode(E, S, meta, params))
    if how == "gate":
        return dsw.to_eval_codes(dsw.decode_gate(E, S, meta, params))
    if how == "nap1":
        from src.nap1.decode import decode
        from src.nap2 import shared
        return dsw.to_eval_codes(decode(E, C, K, meta, params if params is not None else shared.nap2_decode_params()))
    raise ValueError(how)


def checkpoint_paths(specs):
    out = []
    for s in specs:
        hits = sorted(glob.glob(s))
        if not hits:
            raise FileNotFoundError("không có checkpoint khớp %r" % s)
        out += [h for h in hits if Path(h).name != "last.pt"]
    return out


def _decode_params_for(ckpts, override=None):
    if override:
        d = json.loads(Path(override).read_text()) if Path(override).is_file() else json.loads(override)
        return dsw.ensure(d.get("params", d))           # decode_switch.json stores values under "params".
    for p in ckpts:
        f = Path(p).parent / "decode_switch.json"
        if f.is_file():
            return dsw.ensure(json.loads(f.read_text())["params"])
    _, ck = load_checkpoint(ckpts[0])
    return dsw.ensure(ck.get("decode_switch"))


def main(argv=None):
    from src.core.blend import data_files, save_probs, check_out_dir
    from pipeline.score_export import score
    from . import feats
    ap = argparse.ArgumentParser(prog="python -m src.core.infer_switch", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tune", help="chỉnh lưới decode_switch cho một ensemble (có thể nhiều run) trên dev/select")
    t.add_argument("--ckpt", nargs="+", required=True, help="đường dẫn hoặc glob checkpoint")
    t.add_argument("--out", required=True, help="thư mục ghi decode_switch.json")
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--threads", type=int, default=4)
    for cmd in ("evaluate", "export"):
        p = sub.add_parser(cmd)
        p.add_argument("--ckpt", nargs="+", required=True, help="đường dẫn hoặc glob checkpoint")
        p.add_argument("--subset", default="select", choices=("select", "report", "test"))
        p.add_argument("--allow-test", action="store_true")
        p.add_argument("--decode-params", help="file/chuỗi JSON SwitchParams (mặc định decode_switch.json của run)")
        p.add_argument("--threads", type=int, default=4)
        p.add_argument("--out", required=(cmd == "export"))
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    ckpts = checkpoint_paths(a.ckpt)
    if a.cmd == "tune":
        from types import SimpleNamespace
        from .train_switch import final_tune
        out = check_out_dir(a.out)
        out.mkdir(parents=True, exist_ok=True)
        lay = layout_of([load_checkpoint(p)[1] for p in ckpts[:1]])
        vals = [feats.prepare_file(f, layout=lay) for f in data_files("select")]
        res = final_tune(ckpts, vals, SimpleNamespace(eval_workers=a.workers), out)
        print("đã ghi", out / "decode_switch.json", "| dùng lại: --decode-params", out / "decode_switch.json")
        return 0 if res else 1
    models = [load_checkpoint(p) for p in ckpts]
    lay = layout_of([ck for _, ck in models])
    params = _decode_params_for(ckpts, a.decode_params)
    files = data_files(a.subset, a.allow_test)
    print("%d checkpoint, %d file %s, decode_switch %s" % (len(ckpts), len(files), a.subset, params.to_dict()))
    rows = []
    for f in files:
        t0 = time.time()
        d = feats.prepare_file(f, layout=lay)
        E, C, K, S = run_models(d, models)
        if a.cmd == "export":
            out = check_out_dir(a.out)
            save_probs(out / Path(f).name, E, "nap3switch:" + ",".join(Path(p).stem for p in ckpts),
                       C=np.clip(C, 0, 1), K=np.clip(K, 0, 1), S=np.clip(S, 0, 1), half=d["half"])
        if "y_raw" in d:
            z = {"y": d["y_raw"], "half": d["half"]}
            r = {"file": Path(f).name}
            for how in ("switch", "gate", "nap1"):
                codes = decode_all(E, S, C, K, d["half"], d["graded"], params if how == "switch" else None, how)
                r[how] = float(score(z, codes)[0])
            rows.append(r)
            print("  %s switch %.4f · gate %.4f · nap1 %.4f (%.0fs)" % (
                r["file"], r["switch"], r["gate"], r["nap1"], time.time() - t0), flush=True)
    if rows:
        print("trung bình: " + " · ".join("%s %.4f" % (k, np.mean([r[k] for r in rows]))
                                         for k in ("switch", "gate", "nap1")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
