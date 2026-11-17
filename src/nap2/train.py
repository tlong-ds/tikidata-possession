"""Train stage 2 on shared team data (plan sections 2/4, pipeline/FEED.md).

    python -m src.nap2.train --seed 1              # official split: five DFL train matches, dev J03WN1
    python -m src.nap2.train --train A B C --val D # custom matches, still checked by pipeline.split.guard
    python -m src.nap2.train --quick               # CPU smoke check

Match roles come from pipeline.split.roles(); never access reserved J03WR9. Each epoch:
  1. Generate a NEW release-v3 degradation per match (pipeline.feed.train_view, seeded by
     --seed and epoch), or rotate pregenerated cache/nap2/train files with --train-source files.
     Run stage-1 block 2 and build features, optionally in parallel.
  2. Train shuffled chunks with team-swap / vertical-flip augmentation, AdamW and cosine decay.
  3. Evaluate shared FROZEN train_data/dev/select realizations: fit pooled T* by NLL, IOHMM
     decode, then compute spellF1 with pipeline.score_export.
Early-stop on mean dev/select spellF1, not loss. Retain --keep best checkpoints with T*, seed,
match lists, release ID, configuration and library versions. dev/report is only for final
reporting via nap2/compare.py on team/nap2, never used here.
"""

import argparse
import datetime
import json
import math
import os
import platform
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from src.nap1.decode import decode
from src.nap2 import shared
from src.nap2.calibrate import ece, fit_temperature, log_softmax, nll
from src.nap2.dataset import ChunkSampler, augment, prepare_item
from src.nap2.infer import forward_full
from src.nap2.losses import LOSS_DEFAULTS, compute_losses
from src.nap2.model import MODEL_DEFAULTS, build_model

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pm():
    p = os.path.join(ROOT, "evaluation")
    if p not in sys.path:
        sys.path.insert(0, p)
    import possession_metrics
    return possession_metrics


def team_score(y, half, codes):
    """Shared scorer (spellF1, macroF1), splitting halves as in pipeline.score_export."""
    from pipeline.score_export import score
    return score({"y": np.asarray(y), "half": np.asarray(half)}, np.asarray(codes, np.int8))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_many(items, workers, max_frames):
    """items: file paths or ("fresh", match, epoch, run); see src.nap2.dataset.prepare_item."""
    if workers <= 1 or len(items) == 1:
        return [prepare_item(it, max_frames) for it in items]
    with ProcessPoolExecutor(max_workers=min(workers, len(items))) as ex:
        return list(ex.map(prepare_item, items, [max_frames] * len(items)))


def item_name(it):
    return "{}_r{}_e{:02d}".format(it[1], it[3], it[2]) if isinstance(it, (tuple, list)) else Path(it).name


def touch_f1(K, touch_true, half, thr=0.5, tol=12, min_gap=8):
    """F1 of K>thr peaks against touch_true, within +-tol frames using greedy one-to-one matching."""
    T = len(K)
    order = np.argsort(-K)
    taken = np.zeros(T, bool)
    peaks = []
    for i in order:
        if K[i] < thr:
            break
        if half[i] <= 0 or taken[max(0, i - min_gap):i + min_gap + 1].any():
            continue
        taken[i] = True
        peaks.append(i)
    gt = np.flatnonzero(touch_true > 0)
    if len(gt) == 0 or len(peaks) == 0:
        return 0.0
    used = np.zeros(len(gt), bool)
    tp = 0
    for pk in sorted(peaks):
        j = np.searchsorted(gt, pk - tol)
        while j < len(gt) and gt[j] <= pk + tol:
            if not used[j]:
                used[j] = True
                tp += 1
                break
            j += 1
    prec, rec = tp / len(peaks), tp / len(gt)
    return 2 * prec * rec / (prec + rec) if tp else 0.0


def evaluate_one(o, d, T, decode_params=None):
    """Metrics for one prepared file with network outputs o and temperature T."""
    y, w = d["y"], d["w"]
    E = log_softmax(o["had"] / T)
    C = np.exp(log_softmax(o["team"]))
    K = 1 / (1 + np.exp(-o["touch"]))
    pred = decode(E, C, K, {"half": d["half"]}, decode_params if decode_params is not None else shared.nap1_params())
    spell, macro = team_score(d.get("y_raw", y), d["half"], np.where(pred == 3, 2, pred))
    r = pm().evaluate(np.where(y >= 0, y, 3).astype(np.int8), pred)
    alive = (y == 0) | (y == 1)
    out = {"spell_f1": float(spell), "macro_f1": float(macro),
           "n_pred_spells": r["n_pred_spells"], "n_gt_spells": r["n_gt_spells"],
           "d_recall": r["per_class"]["D"]["recall"],
           "frame_acc_alive": float((E[alive, :2].argmax(1) == y[alive]).mean()) if alive.any() else float("nan"),
           "nll_raw": nll(o["had"], y, w), "nll_T": nll(o["had"], y, w, T),
           "ece_raw": ece(o["had"], y, w), "ece_T": ece(o["had"], y, w, T)}
    if "touch_true" in d:
        out["touch_f1"] = touch_f1(K, d["touch_true"], d["half"])
    if "baseline" in d:
        out["smooth50_spell_f1"] = float(team_score(d.get("y_raw", y), d["half"], d["baseline"])[0])
    return out


def evaluate_many(model, ds, chunk, stride, device, decode_params=None, T=None):
    """Evaluate multiple files with one pooled-NLL T*, then average metrics across files."""
    outs = [forward_full(model, d["xp"], d["xf"], d["half"], chunk, stride, device) for d in ds]
    if T is None:
        T = fit_temperature(np.concatenate([o["had"] for o in outs]), np.concatenate([d["y"] for d in ds]),
                            np.concatenate([d["w"] for d in ds]))
    per = [evaluate_one(o, d, T, decode_params) for o, d in zip(outs, ds)]
    mean = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
    mean["spell_f1_sd"] = float(np.std([p["spell_f1"] for p in per], ddof=1)) if len(per) > 1 else 0.0
    mean["T"] = float(T)
    mean["files"] = [d["name"] for d in ds]
    mean["per_file_spell_f1"] = [round(p["spell_f1"], 4) for p in per]
    return mean


def lr_lambda(total, warmup):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        prog = min(1.0, (step - warmup) / max(1, total - warmup))
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * prog))
    return f


def now_iso():
    """Current timezone-aware date and time."""
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def versions():
    return {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Huấn luyện mạng nấc 2 trên dữ liệu chung của nhóm")
    ap.add_argument("--train", nargs="+", help="mặc định: pipeline.split.roles()['train']")
    ap.add_argument("--val", help="trận dev (mặc định: pipeline.split.roles()['dev'])")
    ap.add_argument("--train-source", default="fresh", choices=["fresh", "files"],
                    help="fresh: làm bẩn mới mỗi epoch (feed.train_view); files: bản tạo sẵn trong --train-dir")
    ap.add_argument("--train-dir", default=None, help="thư mục bản làm bẩn tạo sẵn (mặc định cache/nap2/train)")
    ap.add_argument("--dev-dir", default=None, help="thư mục dev chung (mặc định train_data/dev)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=128, help="số bước mỗi đoạn")
    ap.add_argument("--stride", type=int, default=2, help="lấy 1 khung mỗi stride khung (2 = 12,5 Hz)")
    ap.add_argument("--oversample", type=int, default=2, help="số lần mỗi khung xuất hiện mỗi epoch (đoạn chồng nhau)")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--keep", type=int, default=5)
    ap.add_argument("--regen-every", type=int, default=1, help="làm bẩn lại / đổi bản sau mỗi k epoch")
    ap.add_argument("--prep-workers", type=int, default=max(1, min(5, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--max-frames", type=int, default=None, help="cắt mỗi file còn n khung (chạy thử)")
    ap.add_argument("--max-chunks", type=int, default=None, help="tối đa số đoạn mỗi epoch (chạy thử)")
    ap.add_argument("--temporal", default=MODEL_DEFAULTS["temporal"], choices=["bigru", "tcn"])
    ap.add_argument("--w-touch", type=float, default=LOSS_DEFAULTS["w_touch"])
    ap.add_argument("--w-ball", type=float, default=LOSS_DEFAULTS["w_ball"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", default=None, help="đường dẫn last.pt để chạy tiếp")
    ap.add_argument("--quick", action="store_true", help="1 trận, 1 file dev, 3 epoch, 15.000 khung: kiểm tra đường ống")
    a = ap.parse_args(argv)
    if a.quick:
        a.epochs, a.max_frames, a.patience = 3, 15000, 3
        a.max_chunks = a.max_chunks or 200
    return a


def main(argv=None):
    from pipeline import split as team_split
    a = parse_args(argv)
    set_seed(a.seed)
    roles = team_split.roles()
    train_ids = list(a.train or roles["train"])
    dev_id = a.val or roles["dev"][0]
    for n in train_ids:
        team_split.guard(n, "train")
    team_split.guard(dev_id, "select")
    if a.quick:
        train_ids = train_ids[:1]
    if a.train_source == "fresh":
        shared.release()                                  # Stop immediately if degradation configuration files are missing.
        files = None
    else:
        files = shared.train_files(train_ids, a.train_dir)
    val_files = shared.dev_files("select", dev_id, a.dev_dir)[:1 if a.quick else None]
    tag = ("custom_s{}" if a.train or a.val else "official_s{}").format(a.seed)
    out_dir = a.out or os.path.join(ROOT, "runs", "nap2", tag + ("_quick" if a.quick else ""))
    os.makedirs(out_dir, exist_ok=True)
    dev = torch.device(a.device)
    model_cfg = dict(MODEL_DEFAULTS, temporal=a.temporal)
    loss_cfg = dict(LOSS_DEFAULTS, w_touch=a.w_touch, w_ball=a.w_ball)
    train_cfg = {k: v for k, v in vars(a).items()}
    started_at = now_iso()
    release = shared.release_hash()
    print("train:", train_ids, "| dev/select:", dev_id, "x", len(val_files), "| dữ liệu train:",
          "làm bẩn mới mỗi epoch" if files is None else {n: len(f) for n, f in files.items()},
          "| release:", (release or "?")[:12], "| thiết bị:", dev, "| bắt đầu:", started_at, flush=True)

    t0 = time.time()
    vals = prepare_many(val_files, a.prep_workers, a.max_frames)
    print("chuẩn bị dev/select {:.0f}s".format(time.time() - t0), flush=True)

    # files: seed-specific permutations give each seed a different realization order per match.
    order_rng = np.random.default_rng(10007 + a.seed)
    order = {} if files is None else {n: order_rng.permutation(len(files[n])) for n in train_ids}

    def epoch_paths(epoch):
        k = (epoch - 1) // max(1, a.regen_every)
        if files is None:
            return [("fresh", n, k, a.seed) for n in train_ids]
        return [files[n][order[n][k % len(files[n])]] for n in train_ids]

    model = build_model(model_cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    first_paths = epoch_paths(1)
    first = prepare_many(first_paths, a.prep_workers, a.max_frames)
    sampler = ChunkSampler(first, a.chunk, a.stride, a.oversample)
    steps_per_epoch = sampler.n_batches(a.batch)
    if a.max_chunks:
        steps_per_epoch = min(steps_per_epoch, -(-a.max_chunks // a.batch))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(a.epochs * steps_per_epoch, steps_per_epoch))
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    gen = torch.Generator().manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    start_epoch, kept, history, best, bad = 1, [], [], -1.0, 0
    if a.resume:
        st = torch.load(a.resume, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start_epoch, kept, history, best, bad = st["epoch"] + 1, st["kept"], st["history"], st["best"], st["bad"]
        print("chạy tiếp từ epoch", st["epoch"], flush=True)

    train_data, cur_paths = first, first_paths
    for epoch in range(start_epoch, a.epochs + 1):
        te = time.time()
        paths = epoch_paths(epoch)
        if paths != cur_paths:
            train_data, cur_paths = prepare_many(paths, a.prep_workers, a.max_frames), paths
            sampler = ChunkSampler(train_data, a.chunk, a.stride, a.oversample)
        t_prep = time.time() - te

        model.train()
        sums, nb = {}, 0
        for batch in sampler.batches(rng, a.batch, a.max_chunks):
            batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
            batch = augment(batch, gen)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                out = model(batch["xp"], batch["xf"])
            loss, parts = compute_losses(out, batch, loss_cfg)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v
            nb += 1
        t_train = time.time() - te - t_prep

        model.eval()
        metrics = evaluate_many(model, vals, a.chunk, a.stride, dev)
        t_all = time.time() - te
        rec = {"epoch": epoch, "loss": {k: v / max(nb, 1) for k, v in sums.items()}, "val": metrics,
               "train_data": [item_name(p) for p in cur_paths],
               "time_s": {"prep": round(t_prep, 1), "train": round(t_train, 1), "total": round(t_all, 1)},
               "lr": sched.get_last_lr()[0]}
        history.append(rec)
        with open(os.path.join(out_dir, "metrics.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        print("ep {:>2} | loss {:.3f} | dev/select spellF1 {:.4f}±{:.4f} (smooth50 {:.4f}) macroF1 {:.4f} "
              "acc_sống {:.3f} D-recall {:.3f} | lượt {:.0f}/{:.0f} | T* {:.2f} ECE {:.3f}->{:.3f} | "
              "{:.0f}s (chuẩn bị {:.0f}, học {:.0f})"
              .format(epoch, rec["loss"].get("total", float("nan")), metrics["spell_f1"], metrics["spell_f1_sd"],
                      metrics.get("smooth50_spell_f1", float("nan")), metrics["macro_f1"],
                      metrics["frame_acc_alive"], metrics["d_recall"], metrics["n_pred_spells"],
                      metrics["n_gt_spells"], metrics["T"], metrics["ece_raw"], metrics["ece_T"],
                      t_all, t_prep, t_train), flush=True)

        ck = {"model": model.state_dict(), "model_cfg": model_cfg, "feat_cfg": None,
              "loss_cfg": loss_cfg, "train_cfg": train_cfg,
              "T": metrics["T"], "T_source": {"subset": "dev/select", "files": metrics["files"]},
              "epoch": epoch, "seed": a.seed, "val_metrics": metrics,
              "matches": {"train": train_ids, "dev": dev_id}, "release_hash": release,
              "train_source": a.train_source, "versions": versions(), "started_at": started_at,
              "saved_at": now_iso()}
        path = os.path.join(out_dir, "ep{:03d}.pt".format(epoch))
        torch.save(ck, path)
        kept.append((metrics["spell_f1"], path))
        kept.sort(key=lambda t: -t[0])
        for _, pth in kept[a.keep:]:
            if os.path.exists(pth):
                os.remove(pth)
        kept = kept[:a.keep]

        if metrics["spell_f1"] > best + 1e-6:
            best, bad = metrics["spell_f1"], 0
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch, "kept": kept, "history": history, "best": best, "bad": bad},
                   os.path.join(out_dir, "last.pt"))
        if bad >= a.patience:
            print("dừng sớm: {} epoch không cải thiện spellF1 dev/select".format(bad), flush=True)
            break

    summary = {"tag": tag, "started_at": started_at, "finished_at": now_iso(),
               "matches": {"train": train_ids, "dev": dev_id}, "release_hash": release,
               "best_ckpt": kept[0][1] if kept else None, "best_val_spell_f1": best,
               "kept": [{"val_spell_f1": s, "path": p} for s, p in kept],
               "config": train_cfg, "model_cfg": model_cfg, "loss_cfg": loss_cfg, "versions": versions()}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print("xong. tốt nhất dev/select spellF1 {:.4f} -> {}".format(best, summary["best_ckpt"]), flush=True)
    return summary


if __name__ == "__main__":
    main()
