"""Train stage-3 switchnet on CPU/GPU: stage 2 + additional channels + switch head.

    python -m src.core.train_switch --quick                 # pipeline smoke check (~1-2 minutes)
    python -m src.core.train_switch --seed 1 --epochs 12 --init <nap2 ep018.pt> --tag s1

Reuse ChunkSampler, augment, compute_losses, fit_temperature and lr_lambda without modifying
src/nap2/. Stage-3 additions:
  * src.core.feats inputs, transformed consistently under team swap / time reversal (augment3).
  * Switch loss (src.core.switch_target), built from y AFTER augmentation.
  * Select checkpoints by dev/select spellF1 through the SAME decoder used at inference
    (src.core.decode_switch): small grid each epoch, then a full ensemble grid -> decode_switch.json.
Training uses pipeline.split.roles()['train'] with NEW degradation each epoch (feed.train_view,
release v3); dev/select selects models. Do not access dev/report, train_data/test or evaluation/gt.
Output: artifacts/nap3/switch_<tag>/ (gitignored): ep*.pt, last.pt, metrics.jsonl, summary.json,
decode_switch.json.
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

from src.nap2 import shared
from src.nap2.calibrate import fit_temperature
from src.nap2.dataset import ChunkSampler, augment
from src.nap2.losses import LOSS_DEFAULTS, compute_losses
from src.nap2.model import MODEL_DEFAULTS
from src.nap2.train import lr_lambda, now_iso, set_seed, versions

from . import decode_switch as dsw
from . import feats
from .infer_switch import combine, decode_all, fit_switch_temperature, forward_full, frame_switch_target
from . import freeze as frz
from .net import NAP3_DEFAULTS, NAP3_MODEL_KEYS, build_model, init_by_names, source_names
from .switch_target import near_change, switch_loss, switch_mask, switch_target

ROOT = Path(__file__).resolve().parents[2]
SWITCH_DEFAULTS = {"w_switch": 1.0, "pos_weight": 20.0, "sigma_early": 2.0, "sigma_late": 1.0,
                   "p_rev": 0.0, "rev_guard_s": 1.0}
EPOCH_GRID = {"scale": [0.1, 0.3], "lambda_emit": [2.0]}        # Small grid for checkpoint selection.


# ---------------------------------------------------------------- augmentation

def _swap_cols(x, mask, pairs):
    """Swap pairs of last-axis columns for samples selected by mask [B]."""
    if not pairs or not bool(mask.any()):
        return x
    x = x.clone()
    m = mask.view(-1, *([1] * (x.dim() - 2)))
    for i, j in pairs:
        a, b = x[..., i].clone(), x[..., j].clone()
        x[..., i] = torch.where(m, b, a)
        x[..., j] = torch.where(m, a, b)
    return x


def augment3(batch, gen, p_swap=0.5, p_flip=0.5, p_rev=0.0, rev_guard_steps=12, layout=None):
    """Stage-2 augmentation plus consistent extra-channel transforms; optional time reversal.

    Recreate stage-2 swap / vertical-flip masks from a copy of gen state (swap sampled first,
    then flip), and apply them to team-dependent xf pairs (TEAM_PAIRS_XF). Extra channels are
    vertically invariant, so the vertical flip does not change them.

    Time reversal (p_rev > 0, disabled by default) reverses all [B,W,...] arrays, negates vx/vy
    and swaps past/future pairs. Last-touch ownership conventions shift change boundaries when
    reversed, so set w=0 within +-rev_guard_steps of every reversed-sample change, excluding all
    auxiliary losses there as well. Return the batch with 'rev' (bool [B]).
    """
    lay = feats.get_layout(layout)
    B = batch["xp"].shape[0]
    g2 = torch.Generator()
    g2.set_state(gen.get_state())
    swap = torch.rand(B, generator=g2) < p_swap          # nap2.augment samples swap before flip.
    flip = torch.rand(B, generator=g2) < p_flip
    out = augment(batch, gen, p_swap, p_flip)
    dv = out["xf"].device
    swap, flip = swap.to(dv), flip.to(dv)
    out["xf"] = _swap_cols(out["xf"], swap, lay.team_pairs_xf)
    if lay.team_complement_xf or lay.team_negate_xf or lay.flip_negate_xf or lay.flip_negate_xp:
        xf, xp = out["xf"].clone(), out["xp"].clone()
        sw, fl = swap.view(-1, 1), flip.view(-1, 1)
        for c in lay.team_complement_xf:
            xf[..., c] = torch.where(sw, 1 - xf[..., c], xf[..., c])
        for c in lay.team_negate_xf:
            xf[..., c] = torch.where(sw, -xf[..., c], xf[..., c])
        for c in lay.flip_negate_xf:
            xf[..., c] = torch.where(fl, -xf[..., c], xf[..., c])
        for c in lay.flip_negate_xp:
            xp[..., c] = torch.where(fl.view(-1, 1, 1), -xp[..., c], xp[..., c])
        out["xf"], out["xp"] = xf, xp
    rev = torch.zeros(B, dtype=torch.bool)
    if p_rev > 0:
        rev = torch.rand(B, generator=gen) < p_rev
    rev = rev.to(out["xp"].device)
    if bool(rev.any()):
        W = out["xp"].shape[1]
        for k, v in list(out.items()):
            if torch.is_tensor(v) and v.dim() >= 2 and v.shape[0] == B and v.shape[1] == W:
                m = rev.view(-1, *([1] * (v.dim() - 1)))
                out[k] = torch.where(m, v.flip(1), v)
        xp = out["xp"].clone()
        m = rev.view(-1, 1, 1)
        for col in lay.reverse_negate_xp:
            xp[..., col] = torch.where(m, -xp[..., col], xp[..., col])
        out["xp"] = _swap_cols(xp, rev, lay.reverse_pairs_xp)
        out["xf"] = _swap_cols(out["xf"], rev, lay.reverse_pairs_xf)
        guard = near_change(out["y"], rev_guard_steps) & rev.view(-1, 1)
        out["w"] = torch.where(guard, torch.zeros_like(out["w"]), out["w"])
    out["rev"] = rev
    return out


def near_change_loss(had_logits, y, w, steps):
    """H/A/D cross-entropy only within +-steps of a label change (69% of team errors occur within
    1 s of changes). Weighted mean using w; zero when no change occurs in a window.
    """
    import torch.nn.functional as F
    y = y.long()
    m = (near_change(y, steps) & (y >= 0)).float() * w.float()
    ce = F.cross_entropy(had_logits.float().reshape(-1, had_logits.shape[-1]), y.clamp(min=0).reshape(-1),
                         reduction="none").reshape(y.shape)
    return (ce * m).sum() / m.sum().clamp(min=1e-6)


# ---------------------------------------------------------------- data

def prepare_many(items, workers, max_frames=None, layout=None):
    lay = feats.layout_spec(layout)
    if workers <= 1 or len(items) <= 1:
        return [feats.prepare_item(it, max_frames, lay) for it in items]
    with ProcessPoolExecutor(max_workers=min(workers, len(items)), mp_context=get_context("spawn")) as ex:
        return list(ex.map(feats.prepare_item, items, [max_frames] * len(items), [lay] * len(items)))


def item_name(it):
    if isinstance(it, (tuple, list)):
        return "%s:e%d:r%d" % (it[1], it[2], it[3])
    return Path(it).name


# ---------------------------------------------------------------- dev/select scoring

def _decode_job(args):
    E, S, C, K, half, graded, y, params, how = args
    from pipeline.score_export import score
    codes = decode_all(E, S, C, K, half, graded, params, how)
    from src.core.evaltools import spell_count
    return float(score({"y": y, "half": half}, codes)[0]), spell_count(codes, half)


def _map(fn, jobs, workers):
    if workers <= 1 or len(jobs) <= 1:
        return [fn(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=get_context("spawn")) as ex:
        return list(ex.map(fn, jobs, chunksize=max(1, len(jobs) // (4 * workers))))


def frame_probs(outs, vals, T=None, T_s=None):
    """Raw dev-file outputs -> pooled T and T_s, plus per-file (E, C, K, S)."""
    if T is None:
        T = fit_temperature(np.concatenate([o["had"] for o in outs]), np.concatenate([d["y"] for d in vals]),
                            np.concatenate([d["w"] for d in vals]))
    if T_s is None:
        logit = np.concatenate([o["switch"] for o in outs])
        tgt = np.concatenate([frame_switch_target(d["y_raw"], d["half"]) for d in vals])
        msk = np.concatenate([np.asarray(d["half"]) > 0 for d in vals])
        T_s = fit_switch_temperature(logit, tgt, msk)
    probs = [combine([o], [T], [T_s], d["half"]) for o, d in zip(outs, vals)]
    return float(T), float(T_s), probs


def score_grid(probs, vals, params_list, workers=1, with_nap1=True):
    """Mean file-level spellF1 for each SwitchParams, and src.nap1.decode if with_nap1.
    Return (rows [{params, spell_f1, per_file, spells}], nap1_row | None).
    """
    jobs, keys = [], []
    for pi, p in enumerate(params_list):
        for fi, ((E, C, K, S), d) in enumerate(zip(probs, vals)):
            jobs.append((E, S, C, K, d["half"], d["graded"], d["y_raw"], p, "switch"))
            keys.append(pi)
    if with_nap1:
        for (E, C, K, S), d in zip(probs, vals):
            jobs.append((E, S, C, K, d["half"], d["graded"], d["y_raw"], None, "nap1"))
            keys.append(-1)
    res = _map(_decode_job, jobs, workers)
    rows = []
    for pi, p in enumerate(params_list):
        r = [v for k, v in zip(keys, res) if k == pi]
        rows.append({"params": p.to_dict(), "spell_f1": float(np.mean([x[0] for x in r])),
                     "per_file": [round(x[0], 4) for x in r], "spells": float(np.mean([x[1] for x in r]))})
    nap1 = None
    if with_nap1:
        r = [v for k, v in zip(keys, res) if k == -1]
        nap1 = {"spell_f1": float(np.mean([x[0] for x in r])), "per_file": [round(x[0], 4) for x in r],
                "spells": float(np.mean([x[1] for x in r]))}
    rows.sort(key=lambda r: -r["spell_f1"])
    return rows, nap1


def evaluate_epoch(model, vals, chunk, stride, device, grid, workers):
    outs = [forward_full(model, d["xp"], d["xf"], d["half"], chunk, stride, device) for d in vals]
    T, T_s, probs = frame_probs(outs, vals)
    rows, nap1 = score_grid(probs, vals, dsw.grid_params(grid), workers)
    best = rows[0]
    return {"spell_f1": best["spell_f1"], "spell_f1_sd": float(np.std(best["per_file"], ddof=1))
            if len(best["per_file"]) > 1 else 0.0, "per_file_spell_f1": best["per_file"],
            "n_pred_spells": best["spells"], "decode_switch": best["params"],
            "nap1_decode_spell_f1": nap1["spell_f1"], "nap1_decode_spells": nap1["spells"],
            "T": T, "T_switch": T_s, "files": [d["name"] for d in vals]}


# ---------------------------------------------------------------- training loop

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Huấn luyện mạng nấc 3 (switchnet)")
    ap.add_argument("--train", nargs="+", help="mặc định pipeline.split.roles()['train']")
    ap.add_argument("--extra-train", nargs="+", help="thêm trận ngoài (pipeline.split.EXTERNAL_TRAIN, vd. METRICA-SG1)")
    ap.add_argument("--pseudo", nargs="+", default=[],
                    help="file nhãn giả (analysis/pseudo_label.py: y + w) thêm vào mọi epoch, không làm bẩn lại")
    ap.add_argument("--extra-keep", nargs="+", help="trận ngoài (trong --extra-train) dùng mọi epoch; chỉ phần còn lại xoay vòng --extra-per-epoch")
    ap.add_argument("--extra-repeat", type=int, default=1,
                    help="mỗi trận ngoài xuất hiện bấy nhiêu lần mỗi epoch (mỗi lần một bản làm bẩn khác): tăng trọng số")
    ap.add_argument("--extra-per-epoch", type=int, default=0,
                    help="mỗi epoch chỉ dùng N trận ngoài (xoay vòng theo seed; 0 = dùng hết)")
    ap.add_argument("--final", action="store_true",
                    help="train cuối sau khi chốt cấu hình: thêm trận dev vào train, không dừng sớm, giữ --keep epoch CUỐI "
                         "(không chọn theo dev vì dev đã nằm trong train; điểm dev in ra bị lộ, chỉ để xem)")
    ap.add_argument("--val", help="trận dev (mặc định roles()['dev'])")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=128, help="số bước mỗi đoạn train/suy luận (lưu trong checkpoint)")
    ap.add_argument("--par-tcn", type=int, default=0, help="nhánh TCN song song k tầng (0 = tắt), src.core.net")
    ap.add_argument("--par-player", action="store_true", help="nhánh BiGRU song song chỉ đọc cầu thủ, src.core.net")
    ap.add_argument("--player-attn", action="store_true",
                    help="thêm self-attention giữa các cầu thủ (+ token bóng) trong encoder (mặc định tắt)")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--oversample", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--keep", type=int, default=3)
    ap.add_argument("--init", help="checkpoint để khởi tạo: nấc 2 (lớp vào nới rộng, cột mới = 0) hoặc nấc 3 "
                    "(nạp đủ, để train tiếp). Bỏ trống = train từ đầu")
    ap.add_argument("--near-weight", type=float, default=0.0,
                    help="thêm loss H/A/D trên các bước gần lần đổi nhãn, nhân hệ số này (0 = tắt)")
    ap.add_argument("--near-steps", type=int, default=12, help="±số bước quanh lần đổi (12 bước = 1 s)")
    ap.add_argument("--w-switch", type=float, default=SWITCH_DEFAULTS["w_switch"])
    ap.add_argument("--pos-weight", type=float, default=SWITCH_DEFAULTS["pos_weight"])
    ap.add_argument("--sigma-early", type=float, default=SWITCH_DEFAULTS["sigma_early"])
    ap.add_argument("--sigma-late", type=float, default=SWITCH_DEFAULTS["sigma_late"])
    ap.add_argument("--p-rev", type=float, default=SWITCH_DEFAULTS["p_rev"], help="xác suất lật thời gian (0 = tắt)")
    ap.add_argument("--smooth-ball", default=None,
                    help="JSON của [train.smooth_ball] (src.core.ball_smooth): làm mượt bóng bản train như trận đề")
    ap.add_argument("--prep-workers", type=int, default=5)
    ap.add_argument("--eval-workers", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--max-chunks", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-final-tune", action="store_true", help="bỏ bước chỉnh lưới decode_switch đầy đủ cuối cùng")
    ap.add_argument("--quick", action="store_true", help="1 trận, 1 file dev, 2 epoch, 15.000 khung")
    ap.add_argument("--exp", help="file TOML của thí nghiệm (exps/*.toml): step, freeze, stage")
    a = ap.parse_args(argv)
    if a.quick:
        a.epochs, a.max_frames, a.patience = 2, 15000, 2
        a.max_chunks = a.max_chunks or 200
    return a


def load_init(path):
    """Stage-2 or stage-3 checkpoint -> (state_dict, model_cfg, checkpoint)."""
    from torch.torch_version import TorchVersion
    with torch.serialization.safe_globals([TorchVersion]):
        ck = torch.load(path, map_location="cpu", weights_only=True)
    return ck["model"], ck.get("model_cfg") or {}, ck


def plan_stages(a, train):
    """Training stages from TOML [[train.stage]], or one stage from --epochs/--lr and [train].freeze."""
    base = {"freeze": list(train.get("freeze", [])), "freeze_old_inputs": bool(train.get("freeze_old_inputs", False))}
    st = train.get("stage") or []
    if not st:
        return [dict(base, epochs=a.epochs, lr=a.lr)]
    out = []
    for i, x in enumerate(st):
        if "epochs" not in x:
            raise ValueError("[[train.stage]] thứ %d thiếu epochs" % (i + 1))
        out.append({"epochs": int(x["epochs"]), "lr": float(x.get("lr", a.lr)),
                    "freeze": list(x.get("freeze", base["freeze"])),
                    "freeze_old_inputs": bool(x.get("freeze_old_inputs", base["freeze_old_inputs"]))})
    if a.quick:
        out = [dict(x, epochs=1) for x in out]
    return out


PLATEAU_DEFAULTS = {"patience": 2, "factor": 0.3, "epochs": 3, "rounds": 2}


def plateau_cfg(train, quick=False):
    """TOML [train.plateau] -> dict or None (ordinary early stopping if absent).

    When the last stage reaches patience epochs without improvement, run up to `rounds`
    additional rounds. Each reloads the best weights, uses last-stage lr * factor**round,
    cosine decay for `epochs` without warmup, and the same freezing. Stop after a non-improving round.
    """
    p = train.get("plateau")
    if not p:
        return None
    bad = set(p) - set(PLATEAU_DEFAULTS)
    if bad:
        raise ValueError("[train.plateau] có khoá lạ %s (được: %s)" % (sorted(bad), sorted(PLATEAU_DEFAULTS)))
    out = dict(PLATEAU_DEFAULTS, **p)
    out = {"patience": int(out["patience"]), "factor": float(out["factor"]), "epochs": int(out["epochs"]),
           "rounds": int(out["rounds"])}
    if not (0 < out["factor"] < 1) or min(out["patience"], out["epochs"], out["rounds"]) < 1:
        raise ValueError("[train.plateau]: cần 0 < factor < 1, patience/epochs/rounds >= 1")
    if quick:
        out.update(patience=1, epochs=1, rounds=1)
    return out


def restart_rounds(last, plateau):
    """Additional rounds after the last stage, retaining that stage's freezing settings."""
    if not plateau:
        return []
    return [dict(last, epochs=plateau["epochs"], lr=last["lr"] * plateau["factor"] ** (r + 1), restore=True)
            for r in range(plateau["rounds"])]


def epoch_matches(core, extra, k, seed, ep):
    """Epoch ep matches: all primary matches plus k external matches rotating through a fixed
    seed-specific permutation, giving balanced usage across epochs. k=0 or k>=external count uses all.
    """
    core, extra = list(core), list(extra)
    if not extra or not k or k >= len(extra):
        return core + extra
    order = np.random.default_rng(1000 + int(seed)).permutation(len(extra))
    pick = sorted(int(order[(ep * k + j) % len(extra)]) for j in range(k))
    return core + [extra[i] for i in pick]


def main(argv=None):
    from pipeline import split as team_split
    from src.core.blend import check_out_dir
    a = parse_args(argv)
    from src import steps as stp
    spec = {}
    if a.exp:
        from src.exp import load_spec
        spec = load_spec(a.exp)
    step_specs = spec.get("step", [])
    layout = feats.get_layout([x for x in step_specs if x["use"].startswith("feature.")])
    lay_d = layout.to_dict()
    loss_steps = stp.make_all(step_specs, "loss")
    stages = plan_stages(a, spec.get("train", {}))
    a.epochs = sum(x["epochs"] for x in stages)
    torch.set_num_threads(max(1, a.threads))
    set_seed(a.seed)
    roles = team_split.roles()
    train_ids = list(a.train or roles["train"])
    extra = [n for n in (a.extra_train or []) if n not in train_ids]
    bad = [n for n in extra if n not in team_split.EXTERNAL_TRAIN]
    if bad:
        raise SystemExit("--extra-train chỉ nhận %s, có %s" % (team_split.EXTERNAL_TRAIN, bad))
    dev_id = a.val or roles["dev"][0]
    if a.final:
        os.environ[team_split.FINAL_RETRAIN_ENV] = "1"     # Inherited by degradation worker processes.
        if dev_id not in train_ids:
            train_ids = train_ids + [dev_id]
    for n in train_ids + extra:
        team_split.guard(n, "train")
    team_split.guard(dev_id, "select")
    if a.quick:
        train_ids, extra = train_ids[:1], extra[:1]
    core_ids = list(train_ids)
    if extra:                                          # cache/external/<name>.npz (Metrica auto-downloads; PFF / SkillCorner must be available).
        from pipeline.external import ensure_external
        ensure_external(extra)
        train_ids = train_ids + extra

    keep = [n for n in extra if n in set(a.extra_keep or [])]      # Use external matches every epoch without rotation.
    rot = [n for n in extra if n not in set(keep)]

    def epoch_ids(ep):
        return epoch_matches(core_ids + keep, rot, a.extra_per_epoch, a.seed, ep)
    shared.release()
    val_files = shared.dev_files("select", dev_id)[:1 if a.quick else None]
    tag = a.tag or ("s%d" % a.seed) + ("_quick" if a.quick else "")
    out_dir = check_out_dir(a.out or ROOT / "artifacts" / "nap3" / ("switch_" + tag))
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)

    model_cfg = dict(MODEL_DEFAULTS, **dict(NAP3_DEFAULTS, n_player_feat=layout.n_xp, n_frame_feat=layout.n_xf))
    init_info = None
    init_state = None
    if a.init:
        init_state, cfg2, ck0 = load_init(a.init)
        model_cfg.update({k: v for k, v in cfg2.items() if k in MODEL_DEFAULTS or k in NAP3_MODEL_KEYS})
    if a.player_attn:                                  # TOML [model] player_attn = true; disabled by default.
        model_cfg["player_attn"] = True
    if a.par_tcn:
        model_cfg["par_tcn"] = int(a.par_tcn)
    if a.par_player:
        model_cfg["par_player"] = True
    model = build_model(model_cfg).to(dev)
    new_xp, new_xf = list(range(layout.n_xp)), list(range(layout.n_xf))
    if init_state is not None:
        sx, sf = source_names(init_state, ck0)
        init_info = init_by_names(model, init_state, sx, sf, layout.xp_names, layout.xf_names)
        init_info["path"] = str(a.init)
        new_xp = [layout.xp_names.index(n) for n in init_info["new_xp"]]
        new_xf = [layout.xf_names.index(n) for n in init_info["new_xf"]]
    loss_cfg = dict(LOSS_DEFAULTS)
    sw_cfg = {"w_switch": a.w_switch, "pos_weight": a.pos_weight, "sigma_early": a.sigma_early,
              "sigma_late": a.sigma_late, "p_rev": a.p_rev, "rev_guard_s": SWITCH_DEFAULTS["rev_guard_s"],
              "near_weight": a.near_weight, "near_steps": a.near_steps}
    rev_guard_steps = int(round(sw_cfg["rev_guard_s"] * 25 / a.stride))
    train_cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}
    started_at = now_iso()
    release = shared.release_hash()
    print("train:", train_ids, "| dev/select:", dev_id, "x", len(val_files), "| init:",
          init_info and {k: init_info[k] for k in ("path", "widened", "missing", "new_xp", "new_xf")},
          "| step:", [x["use"] for x in step_specs], "| out:", out_dir, flush=True)

    t0 = time.time()
    vals = prepare_many(val_files, a.prep_workers, a.max_frames, lay_d)
    print("chuẩn bị dev/select %.0fs" % (time.time() - t0), flush=True)

    smooth = None
    if a.smooth_ball:
        from .ball_smooth import check_cfg
        smooth = check_cfg(json.loads(a.smooth_ball))
        print("làm mượt bóng bản train:", smooth, flush=True)

    def fresh_paths(ep):
        ext = set(extra)                               # Repeat external matches extra_repeat times with distinct degradation realizations (offset run).
        return [("fresh", n, ep, a.seed + 100000 * r) + ((smooth,) if smooth else ())
                for n in epoch_ids(ep) for r in range(a.extra_repeat if n in ext else 1)] + list(a.pseudo)

    paths = fresh_paths(0)
    t0 = time.time()
    train_data = prepare_many(paths, a.prep_workers, a.max_frames, lay_d)
    print("chuẩn bị train epoch 1: %.0fs" % (time.time() - t0), flush=True)
    sampler = ChunkSampler(train_data, a.chunk, a.stride, a.oversample)
    steps = sampler.n_batches(a.batch)
    if a.max_chunks:
        steps = min(steps, -(-a.max_chunks // a.batch))
    plateau = plateau_cfg(spec.get("train", {}), a.quick)
    plan = stages + restart_rounds(stages[-1], plateau)
    pat = plateau["patience"] if plateau else a.patience
    opt = sched = None
    freeze_log = []
    gen = torch.Generator().manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    kept, history, best, bad = [], [], -1.0, 0
    best_state, prev_gain, epoch = None, True, 0
    cur = paths
    for si, stg in enumerate(plan):
        restore = bool(stg.get("restore"))
        if restore:                                            # Additional training rounds from the best weights at a lower learning rate.
            if best_state is None or (si > len(stages) and not prev_gain):   # Always run the first round.
                print("dừng: vòng học thêm trước không cải thiện", flush=True)
                break
            model.load_state_dict(best_state)
            bad = 0
        start_best = best
        groups, finfo = frz.apply(model, stg["freeze"], stg["freeze_old_inputs"], new_xp, new_xf, stg["lr"], a.wd)
        if not groups:
            raise SystemExit("giai đoạn %d đóng băng hết tham số" % (si + 1))
        opt = torch.optim.AdamW(groups, lr=stg["lr"], weight_decay=a.wd)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(stg["epochs"] * steps, 0 if restore else steps))
        freeze_log.append(dict(finfo, stage=si + 1, epochs=stg["epochs"], lr=stg["lr"], from_epoch=epoch + 1,
                               restore=restore))
        print("giai đoạn %d%s: %d epoch, lr %g, đóng băng %s%s, học %d/%d tham số" % (
            si + 1, " (nạp lại bản tốt nhất %.4f)" % best if restore else "", stg["epochs"], stg["lr"],
            finfo["freeze"] or "không", " + cột vào cũ" if finfo["freeze_old_inputs"] else "", finfo["trainable"],
            finfo["total"]), flush=True)
        for _ in range(stg["epochs"]):
            epoch += 1
            te = time.time()
            paths = fresh_paths(epoch - 1)
            if paths != cur:
                train_data, cur = prepare_many(paths, a.prep_workers, a.max_frames, lay_d), paths
                sampler = ChunkSampler(train_data, a.chunk, a.stride, a.oversample)
            t_prep = time.time() - te
            model.train()
            sums, nb = {}, 0
            for batch in sampler.batches(rng, a.batch, a.max_chunks):
                batch = {k: v.to(dev) for k, v in batch.items()}
                batch = augment3(batch, gen, p_rev=a.p_rev, rev_guard_steps=rev_guard_steps, layout=layout)
                out = model(batch["xp"], batch["xf"])
                loss, parts = compute_losses(out, batch, loss_cfg)
                tgt = switch_target(batch["y"], a.sigma_early, a.sigma_late)
                l_sw = switch_loss(out["switch"], tgt, switch_mask(batch["y"], batch["w"]), a.pos_weight)
                total = loss + a.w_switch * l_sw
                if a.near_weight > 0:
                    l_near = near_change_loss(out["had"], batch["y"], batch["w"], a.near_steps)
                    total = total + a.near_weight * l_near
                    parts["near"] = float(l_near.item())
                for st in loss_steps:
                    l_st = st.loss(out, batch)
                    total = total + float(st.weight) * l_st
                    parts[st.name] = float(l_st.item())
                opt.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                parts = dict(parts, switch=float(l_sw.item()), total=float(total.item()))
                for k, v in parts.items():
                    sums[k] = sums.get(k, 0.0) + v
                nb += 1
            t_train = time.time() - te - t_prep
            model.eval()
            m = evaluate_epoch(model, vals, a.chunk, a.stride, dev, EPOCH_GRID, a.eval_workers)
            rec = {"epoch": epoch, "loss": {k: v / max(nb, 1) for k, v in sums.items()}, "val": m,
                   "train_data": [item_name(p) for p in cur],
                   "time_s": {"prep": round(t_prep, 1), "train": round(t_train, 1),
                              "total": round(time.time() - te, 1)}, "lr": sched.get_last_lr()[0]}
            history.append(rec)
            with open(out_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            print("ep %2d | loss %.3f (switch %.3f) | select spellF1 decode_switch %.4f±%.4f (scale %.2g) · "
                  "src.nap1.decode %.4f | lượt %.0f | T %.2f T_s %.2f | %.0fs (chuẩn bị %.0f, học %.0f)" % (
                      epoch, rec["loss"]["total"], rec["loss"]["switch"], m["spell_f1"], m["spell_f1_sd"],
                      m["decode_switch"]["scale"], m["nap1_decode_spell_f1"], m["n_pred_spells"], m["T"],
                      m["T_switch"], rec["time_s"]["total"], t_prep, t_train), flush=True)
            ck = {"model": model.state_dict(), "model_cfg": model_cfg, "feat_version": feats.FEAT_VERSION,
                  "layout": lay_d, "steps": step_specs, "freeze": freeze_log, "exp": spec.get("name"),
                  "loss_cfg": loss_cfg, "switch_cfg": sw_cfg, "train_cfg": train_cfg, "init": init_info,
                  "T": m["T"], "T_switch": m["T_switch"], "decode_switch": m["decode_switch"],
                  "T_source": {"subset": "dev/select", "files": m["files"]}, "epoch": epoch, "seed": a.seed,
                  "val_metrics": m, "matches": {"train": train_ids, "dev": dev_id}, "release_hash": release,
                  "versions": versions(), "started_at": started_at, "saved_at": now_iso()}
            path = out_dir / ("ep%03d.pt" % epoch)
            torch.save(ck, path)
            kept.append((epoch if a.final else m["spell_f1"], str(path)))   # Final training: retain the last epochs.
            kept.sort(key=lambda t: -t[0])
            for _, p in kept[a.keep:]:
                if os.path.exists(p):
                    os.remove(p)
            kept = kept[:a.keep]
            if m["spell_f1"] > best + 1e-6:
                best, bad = m["spell_f1"], 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            if bad >= pat and si >= len(stages) - 1 and not a.final:          # Early stopping only in the last stage and additional rounds.
                print("dừng sớm: %d epoch không cải thiện%s" % (bad, "" if si + 1 >= len(plan) else ", học thêm từ bản tốt nhất"),
                      flush=True)
                break
        prev_gain = best > start_best + 1e-6

    final = None
    if kept and not a.no_final_tune:
        final = final_tune([p for _, p in kept], vals, a, out_dir)
    summary = {"tag": tag, "started_at": started_at, "finished_at": now_iso(),
               "matches": {"train": train_ids, "dev": dev_id}, "release_hash": release, "init": init_info,
               "best_ckpt": kept[0][1] if kept else None, "best_val_spell_f1": best,
               "kept": [{"val_spell_f1": s, "path": p} for s, p in kept], "final_tune": final,
               "config": train_cfg, "model_cfg": model_cfg, "loss_cfg": loss_cfg, "switch_cfg": sw_cfg,
               "layout": lay_d, "steps": step_specs, "freeze": freeze_log, "exp": spec.get("name"),
               "epochs_run": len(history), "versions": versions()}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print("xong. tốt nhất (1 checkpoint, lưới nhỏ) %.4f -> %s" % (best, summary["best_ckpt"]), flush=True)
    return summary


def final_tune(ckpt_paths, vals, a, out_dir):
    """Ensemble retained checkpoints, search the full decode_switch grid on dev/select,
    and write decode_switch.json for default use by src.core.infer_switch.
    """
    from .infer_switch import load_checkpoint
    models = [load_checkpoint(p) for p in ckpt_paths]
    outs_per = [[forward_full(m, d["xp"], d["xf"], d["half"], ck["train_cfg"]["chunk"], ck["train_cfg"]["stride"])
                 for d in vals] for m, ck in models]
    probs = []
    for fi, d in enumerate(vals):
        probs.append(combine([op[fi] for op in outs_per], [ck["T"] for _, ck in models],
                             [ck["T_switch"] for _, ck in models], d["half"]))
    grid = dict(dsw.DEFAULT_GRID, gate_scale=[0.5, 1.0], window=[0, 3])
    rows, nap1 = score_grid(probs, vals, dsw.grid_params(grid), a.eval_workers)
    gate = [_decode_job((E, S, C, K, d["half"], d["graded"], d["y_raw"], None, "gate"))[0]
            for (E, C, K, S), d in zip(probs, vals)]
    res = {"ckpts": ckpt_paths, "best": rows[0], "top": rows[:10], "nap1_decode": nap1,
           "gate_decode": {"spell_f1": float(np.mean(gate)), "per_file": [round(g, 4) for g in gate]},
           "n_grid": len(rows)}
    (out_dir / "decode_switch.json").write_text(json.dumps({"params": rows[0]["params"], "tune": res},
                                                           indent=1, ensure_ascii=False))
    print("ensemble %d checkpoint: decode_switch %.4f %s · gate %.4f · src.nap1.decode %.4f · lượt %.0f" % (
        len(models), rows[0]["spell_f1"], rows[0]["params"], res["gate_decode"]["spell_f1"], nap1["spell_f1"],
        rows[0]["spells"]), flush=True)
    return res


if __name__ == "__main__":
    main()
