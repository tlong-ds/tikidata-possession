"""Run a stage-3 experiment from TOML and produce artifacts with a baseline comparison.

    python -m src.exp run exps/<name>.toml [--quick] [--device cuda] [--zip OUT.zip]
    python -m src.exp show <name>             # print eval.md
    python -m src.exp steps                   # list available steps
    python -m src.exp setup-kaggle            # copy Kaggle data and load the dataset baseline

TOML (see exps/*.toml, AGENTS.md):
    name = "nap1_dead"
    base = "baseline"            # initialize from each seed's best checkpoint in artifacts/exp/<base>
    [train] seeds, epochs (0 = reuse base checkpoints), lr, patience, freeze, freeze_old_inputs,
            [[train.stage]] ..., [train.plateau] (additional rounds from the best weights)
    [[step]] use = "<type>.<name>", parameters ...

Artifacts: artifacts/exp/<name>/ = spec.toml, run.json, ckpt/s<seed>/, decode_switch.json,
probs/select/, codes/select/, eval.json, eval.md, logs/. All choices use dev/select only.
"""

import argparse
import datetime as dt
import glob
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "artifacts" / "exp"
TRAIN_DEFAULTS = {"seeds": [1, 2, 3], "epochs": 10, "lr": 3e-4, "patience": 4, "freeze": [],
                  "freeze_old_inputs": False, "stage": []}
MODEL_KEYS = {"chunk": "--chunk", "player_attn": "--player-attn", "par_tcn": "--par-tcn", "par_player": "--par-player"}     # [model] -> src.core.train_switch flags
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,40}$")


# ---------------------------------------------------------------------------------------------- spec

def base_path(base):
    """Base directory: artifacts/exp/<name> or models/<name> (git-packaged model, e.g. for fine-tuning)."""
    return ROOT / base if str(base).startswith("models/") else EXP_DIR / base


def base_steps(base):
    """Actual experiment steps from run.json, including inherited steps, or from its spec.toml."""
    d = base_path(base)
    if (d / "run.json").is_file():
        st = json.loads((d / "run.json").read_text()).get("steps")
        if st is not None:
            return st
    if (d / "spec.toml").is_file():
        return load_spec(d / "spec.toml", inherit=False)["step"]
    return None


def load_spec(path, inherit=True):
    """Read experiment TOML. inherit prepends base feature/loss/etc. steps absent from this spec.
    Disable with `inherit = false`. Thus "base = baseline" means baseline + this experiment's steps.
    """
    import tomllib
    from . import steps as stp
    with open(path, "rb") as f:
        d = tomllib.load(f)
    unknown = set(d) - {"name", "base", "note", "train", "step", "inherit", "model"}
    if unknown:
        raise ValueError("%s: khoá lạ %s" % (path, sorted(unknown)))
    if not NAME_RE.match(str(d.get("name", ""))):
        raise ValueError("%s: name phải là chữ thường/số/_/- (vd. nap1_dead)" % path)
    d.setdefault("base", "baseline")
    d["train"] = dict(TRAIN_DEFAULTS, **d.get("train", {}))
    bad = set(d["train"]) - set(TRAIN_DEFAULTS) - {"wd", "batch", "p_rev", "plateau", "smooth_ball", "extra_train", "extra_per_epoch", "extra_keep", "extra_repeat", "fold", "final", "decode_from"}
    if bad:
        raise ValueError("%s: [train] có khoá lạ %s" % (path, sorted(bad)))
    if d["train"].get("extra_train"):
        from pipeline.split import EXTERNAL_TRAIN
        from pipeline.external import expand
        ext = d["train"]["extra_train"] = expand(d["train"]["extra_train"]) if isinstance(d["train"]["extra_train"], list) else d["train"]["extra_train"]
        if not isinstance(ext, list) or not set(ext) <= set(EXTERNAL_TRAIN):
            raise ValueError("%s: [train] extra_train phải là danh sách con của %s" % (path, EXTERNAL_TRAIN))
    if d["train"].get("extra_keep"):
        from pipeline.external import expand
        kp = d["train"]["extra_keep"] = expand(list(d["train"]["extra_keep"]))
        if not set(kp) <= set(d["train"].get("extra_train") or []):
            raise ValueError("%s: [train] extra_keep phải nằm trong extra_train" % path)
    if d["train"].get("smooth_ball"):
        from .core.ball_smooth import check_cfg
        check_cfg(d["train"]["smooth_ball"])
    pl = set(d["train"].get("plateau") or {}) - {"patience", "factor", "epochs", "rounds"}
    if pl:
        raise ValueError("%s: [train.plateau] có khoá lạ %s" % (path, sorted(pl)))
    d["model"] = dict(d.get("model") or {})
    bad = set(d["model"]) - set(MODEL_KEYS)
    if bad:
        raise ValueError("%s: [model] có khoá lạ %s (có: %s)" % (path, sorted(bad), sorted(MODEL_KEYS)))
    d["step"] = list(d.get("step", []))
    d["inherited"] = []
    if inherit and d.get("inherit", True) and d["base"] not in ("", "scratch"):
        bs = base_steps(d["base"])
        if bs:
            own = {s["use"] for s in d["step"]}
            d["inherited"] = [s for s in bs if s["use"] not in own]
            d["step"] = d["inherited"] + d["step"]
    stp.make_all(d["step"])                     # Validate step names and parameters immediately.
    d["path"] = str(path)
    return d


def model_flags(model):
    """TOML [model] -> train_switch CLI flags: booleans emit a flag when true; numbers emit flag + value."""
    out = []
    for k, v in sorted(model.items()):
        if isinstance(v, bool):
            out += [MODEL_KEYS[k]] if v else []
        else:
            out += [MODEL_KEYS[k], str(int(v))]
    return out


def git_info():
    def g(*a):
        try:
            return subprocess.run(["git", "-C", str(ROOT)] + list(a), capture_output=True, text=True,
                                  check=True).stdout.strip()
        except Exception:
            return None
    return {"commit": g("rev-parse", "HEAD"), "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(g("status", "--porcelain", "--untracked-files=no"))}


def now():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def say(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------------------------- data

def select_files(quick=False):
    from src.core.blend import data_files
    files = data_files("select")
    return files[:1] if quick else files


def best_ckpt(run_dir):
    """Best checkpoint in a training directory, selected on dev/select."""
    d = Path(run_dir)
    s = d / "summary.json"
    if s.is_file():
        j = json.loads(s.read_text())
        if j.get("best_ckpt"):
            p = d / Path(j["best_ckpt"].replace("\\", "/")).name
            if p.is_file():
                return p
    cks = sorted(d.glob("ep*.pt"))
    if not cks:
        raise FileNotFoundError("không có checkpoint trong %s" % d)
    return cks[-1]


def run_ckpts(run_dir):
    """Retained checkpoints in a training directory."""
    from src.core.blend import nap2_checkpoints
    return nap2_checkpoints(str(run_dir))


# ---------------------------------------------------------------------------------------------- steps

def train(spec, out, a):
    """Train each seed with src.core.train_switch in subprocesses, logging to files; return run directories."""
    tr = spec["train"]
    seeds = tr["seeds"][:1] if a.quick else tr["seeds"]
    runs = []
    base = spec["base"]
    base_dir = None if base in ("", "scratch") else base_path(base)
    for seed in seeds:
        run_dir = out / "ckpt" / ("s%d" % seed)
        if int(tr["epochs"]) == 0 and not tr["stage"]:
            src = base_dir / "ckpt" / ("s%d" % seed)
            shutil.copytree(src, run_dir, dirs_exist_ok=True)
            say("  s%d: không train, dùng checkpoint của %s" % (seed, spec["base"]))
            runs.append(run_dir)
            continue
        cmd = [sys.executable, "-u", "-m", "src.core.train_switch", "--exp", spec["path"], "--seed", str(seed),
               "--epochs", str(tr["epochs"]), "--lr", str(tr["lr"]), "--patience", str(tr["patience"]),
               "--out", str(run_dir), "--no-final-tune", "--device", a.device, "--threads", str(a.threads),
               "--prep-workers", str(a.workers), "--eval-workers", str(a.workers)]
        for k in ("wd", "batch", "p_rev"):
            if k in tr:
                cmd += ["--" + k.replace("_", "-"), str(tr[k])]
        if tr.get("smooth_ball"):
            cmd += ["--smooth-ball", json.dumps(tr["smooth_ball"])]
        if tr.get("extra_train"):
            cmd += ["--extra-train"] + list(tr["extra_train"])
        if tr.get("extra_per_epoch"):
            cmd += ["--extra-per-epoch", str(int(tr["extra_per_epoch"]))]
        if tr.get("extra_keep"):
            cmd += ["--extra-keep"] + list(tr["extra_keep"])
        if int(tr.get("extra_repeat", 1)) > 1:
            cmd += ["--extra-repeat", str(int(tr["extra_repeat"]))]
        if tr.get("final"):                            # Final training: include dev in training and retain the last epochs.
            cmd += ["--final"]
        cmd += model_flags(spec.get("model") or {})
        init = None
        if base_dir is not None:
            init = best_ckpt(base_dir / "ckpt" / ("s%d" % seed))
            cmd += ["--init", str(init)]
        if a.quick:
            cmd += ["--quick"]
        if run_dir.exists():
            shutil.rmtree(run_dir)
        log = out / "logs" / ("train_s%d.log" % seed)
        t0 = time.time()
        with open(log, "w") as f:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
        if r.returncode != 0:
            tail = "".join(open(log).readlines()[-15:])
            raise SystemExit("train s%d lỗi (xem %s):\n%s" % (seed, log, tail))
        s = json.loads((run_dir / "summary.json").read_text())
        m = [json.loads(l) for l in open(run_dir / "metrics.jsonl")]
        first = m[0]["val"]["spell_f1"] if m else float("nan")
        best_ep = max(m, key=lambda r: r["val"]["spell_f1"])["epoch"] if m else 0
        say("  s%d: %d epoch, tốt nhất ep%d %.4f (epoch 1: %.4f), từ %s, %.1f phút" % (
            seed, len(m), best_ep, s["best_val_spell_f1"], first, init.parent.name + "/" + init.name if init
            else "đầu", (time.time() - t0) / 60))
        runs.append(run_dir)
    return runs


def _probs_codes(models, files, decode_steps, quick=False):
    """Run checkpoints on files -> (vals, probs [(E, C, K, S)])."""
    from src.core import feats
    from src.core.infer_switch import layout_of, run_models
    lay = layout_of([ck for _, ck in models])
    vals, probs = [], []
    for f in files:
        d = feats.prepare_file(f, layout=lay)
        E, C, K, S = run_models(d, models)
        for st in decode_steps:
            E, S, C, K = st.adjust(E, S, C, K, d)
        vals.append(d)
        probs.append((E, C, K, S))
    return vals, probs


def ensemble(out, runs, files, decode_steps, a, decode_from=None):
    """Ensemble checkpoints across seeds, tune stage 4 on dev/select, write probs/, codes/, decode_switch.json.
    decode_from model directory bypasses tuning and copies its decoder (final training includes dev).
    """
    import numpy as np
    from src.core import decode_switch as dsw
    from src.core.blend import save_probs
    from src.core.infer_switch import decode_all, load_checkpoint
    from src.core.train_switch import score_grid
    ckpts = [c for r in runs for c in run_ckpts(r)]
    models = [load_checkpoint(c) for c in ckpts]
    t0 = time.time()
    vals, probs = _probs_codes(models, files, decode_steps, a.quick)
    grid = {"scale": [0.1, 0.3], "lambda_emit": [2.0]} if a.quick else \
        dict(dsw.DEFAULT_GRID, gate_scale=[0.5, 1.0], window=[0, 3])
    for st in decode_steps:
        grid = st.grid(grid)
    if decode_from:
        fixed = json.loads((base_path(decode_from) / "decode_switch.json").read_text())["params"]
        rows, nap1 = score_grid(probs, vals, [dsw.ensure(fixed)], a.workers)
        rows[0]["params"], rows[0]["fixed_from"] = fixed, str(decode_from)
    else:
        rows, nap1 = score_grid(probs, vals, dsw.grid_params(grid), a.workers)
    best = rows[0]
    params = dsw.ensure(best["params"])
    (out / "decode_switch.json").write_text(json.dumps(
        {"params": best["params"], "tune": {"ckpts": [str(c) for c in ckpts], "best": best, "top": rows[:10],
                                            "nap1_decode": nap1, "n_grid": len(rows)}}, indent=1))
    pdir, cdir = out / "probs" / "select", out / "codes" / "select"
    pdir.mkdir(parents=True, exist_ok=True)
    cdir.mkdir(parents=True, exist_ok=True)
    for f, d, (E, C, K, S) in zip(files, vals, probs):
        name = Path(f).name
        save_probs(pdir / name, E, "exp:" + out.name, C=np.clip(C, 0, 1), K=np.clip(K, 0, 1), S=np.clip(S, 0, 1),
                   half=d["half"])
        codes = decode_all(E, S, C, K, d["half"], d["graded"], params)
        np.savez_compressed(cdir / name, codes=np.asarray(codes, np.int8), probs=np.exp(E).astype(np.float32),
                            p_dead=np.exp(E[:, 2]).astype(np.float32))
    say("  gộp %d checkpoint: decode_switch %.4f (lưới %d ô), %.1f phút" % (
        len(ckpts), best["spell_f1"], len(rows), (time.time() - t0) / 60))
    return {"n_ckpt": len(ckpts), "spell_f1": best["spell_f1"], "params": best["params"]}


def blend(spec, out, files, ens, a):
    """Blend steps: export sources and tune on dev/select; use only if they outperform the ensemble alone."""
    from src.core import blend as bl
    from src import steps as stp
    bsteps = stp.make_all(spec["step"], "blend")
    if not bsteps:
        return None
    sources = {"switch": out / "probs" / "select"}
    grid = {"ref": "switch", "mode": "team_dead"}
    for st in bsteps:
        # Self-training steps (pathcrf) inherit this run's device and mode.
        if "quick" in st.defaults:
            st.params["quick"] = bool(a.quick)
        if "device" in st.defaults:
            st.params["device"] = a.device
        sdir = out / "blend_src" / st.name
        if not all((sdir / Path(f).name).is_file() for f in files):
            t0 = time.time()
            sdir.mkdir(parents=True, exist_ok=True)
            st.export(files, sdir, a.workers)
            say("  xuất nguồn %s: %.1f phút" % (st.name, (time.time() - t0) / 60))
        sources[st.name] = sdir
        for kind, ws in st.grid().items():
            grid.setdefault(kind, {}).update(ws)
    table, chosen = bl.tune(files, sources, grid, workers=a.workers)
    res = bl.apply(files, sources, chosen["spec"], out / "codes_blend" / "select", "select", ref="switch")
    use = res["spell_f1"] > ens["spell_f1"] + 1e-9
    say("  trộn %s: %.4f (trọng số %s) %s ensemble một mình %.4f -> %s" % (
        "+".join(s.name for s in bsteps), res["spell_f1"], chosen["label"], ">" if use else "<=",
        ens["spell_f1"], "dùng bản trộn" if use else "giữ ensemble"))
    return {"spell_f1": res["spell_f1"], "spec": chosen["spec"], "label": chosen["label"], "used": bool(use),
            "table_top": table[:10]}


def evaluate(spec, out, files, base_name):
    """Paired dev/select baseline comparison + situation breakdown -> eval.json, eval.md."""
    import numpy as np
    from src.core import evaltools as ev
    name = out.name
    final = out / ("codes_blend" if (out / "codes_blend").is_dir() and _blend_used(out) else "codes") / "select"
    preds = {}
    base_dir = EXP_DIR / base_name / "codes" / "select" if base_name and base_name != name else None
    if base_dir is not None and all((base_dir / Path(f).name).is_file() for f in files):
        preds[base_name] = ev.load_pred_dir(base_dir, files)
    preds[name] = ev.load_pred_dir(final, files)
    try:
        preds["smooth50"] = ev.baseline_preds(files)
    except KeyError:                                   # Cross-validation dev realizations have no organiser baseline predictions.
        pass
    res = ev.compare(files, preds)
    per = []
    for f in files:
        z = ev.load_eval(f)
        p = ev.load_pred(final, Path(f).name)
        per.append(ev.breakdown(z, p["codes"], p.get("team_prob")))
    bd = ev.mean_breakdown(per)
    me = res["models"][name]
    vs = me["vs_first"]["spellF1"] if me["vs_first"] and base_dir is not None else None
    if vs is None:
        verdict = "mốc (không so)"
    elif vs["n"] > 1 and vs["lo"] > 0:
        verdict = "TỐT HƠN %s" % base_name
    elif vs["n"] > 1 and vs["hi"] < 0:
        verdict = "KÉM HƠN %s" % base_name
    else:
        verdict = "CHƯA RÕ (khoảng tin cậy chứa 0)"
    sp = np.asarray(me["spellF1"])
    return {"name": name, "base": base_name if base_dir is not None else None, "files": res["files"],
            "spellF1": me["spellF1"], "macroF1": me["macroF1"], "mean": float(sp.mean()),
            "sd": float(sp.std(ddof=1)) if sp.size > 1 else 0.0, "vs_base": vs, "verdict": verdict,
            "compare": res, "breakdown": bd, "codes": str(final.relative_to(out))}


def _blend_used(out):
    p = out / "run.json"
    return p.is_file() and bool((json.loads(p.read_text()).get("blend") or {}).get("used"))


def write_md(out, run, e):
    from src.core import evaltools as ev
    L = ["# %s" % e["name"], ""]
    L.append("**%s**: spellF1 dev/select %.4f ± %.4f" % (e["verdict"], e["mean"], e["sd"]) +
             ("; so %s %+.4f [%+.4f, %+.4f], thắng %d/%d bản" % (e["base"], e["vs_base"]["mean"], e["vs_base"]["lo"],
                                                                 e["vs_base"]["hi"], e["vs_base"]["wins"],
                                                                 e["vs_base"]["n"]) if e["vs_base"] else ""))
    L += ["", "```", ev.format_compare(e["compare"]), "```", "", "| bản | " + " | ".join(
        e["compare"]["models"]) + " |", "|---|" + "---|" * len(e["compare"]["models"])]
    for i, f in enumerate(e["files"]):
        L.append("| %s | " % Path(f).stem + " | ".join("%.4f" % m["spellF1"][i] for m in e["compare"]["models"].values())
                 + " |")
    L += ["", "## Lỗi theo tình huống (trung bình các bản)", "", "```", ev.format_breakdown(e["breakdown"]), "```", "",
          "## Chạy", ""]
    L.append("- commit `%s` (%s)%s, %s, thiết bị %s" % ((run["git"]["commit"] or "?")[:9], run["git"]["branch"],
                                                      " CÓ SỬA CHƯA COMMIT" if run["git"]["dirty"] else "",
                                                      run["started"], run["device"]))
    L.append("- step: %s" % (", ".join(s["use"] for s in run["steps"]) or "không"))
    for t in run.get("train", []):
        L.append("- %s" % t)
    if run.get("ensemble"):
        L.append("- gộp %d checkpoint, decode_switch %.4f, tham số %s" % (
            run["ensemble"]["n_ckpt"], run["ensemble"]["spell_f1"], run["ensemble"]["params"]))
    if run.get("blend"):
        b = run["blend"]
        L.append("- trộn: %.4f (%s), %s" % (b["spell_f1"], b["label"], "dùng" if b["used"] else "không dùng"))
    L.append("- thời gian: " + ", ".join("%s %.1f phút" % (k, v / 60) for k, v in run["time_s"].items()))
    (out / "eval.md").write_text("\n".join(L) + "\n")


def train_lines(runs):
    out = []
    for r in runs:
        s = json.loads((Path(r) / "summary.json").read_text())
        fr = s.get("freeze") or []
        out.append("%s: %s epoch, tốt nhất %.4f, khởi tạo %s, %s" % (
            Path(r).name, s.get("epochs_run", "?"), s.get("best_val_spell_f1", float("nan")),
            ((s.get("init") or {}).get("path") or "từ đầu").split("artifacts/")[-1],
            "; ".join("gđ%d%s %d ep lr %g đóng băng %s%s (học %d/%d)" % (
                x["stage"], " (học thêm từ bản tốt nhất)" if x.get("restore") else "", x["epochs"], x["lr"],
                x["freeze"] or "không", " +cột cũ" if x["freeze_old_inputs"] else "",
                x["trainable"], x["total"]) for x in fr) or "không đóng băng"))
    return out


# blend_src: omit reproducible large per-file probabilities/codes; retain self-training step models and logs (pathcrf).
BLEND_KEEP_MODEL = {"model.pt", "history.jsonl", "summary.json", "args.json"}


def zip_keep(rel):
    """rel is relative to the experiment directory. Keep everything outside blend_src."""
    parts = Path(rel).parts
    if "blend_src" not in parts:
        return True
    name = parts[-1]
    if name.endswith(".log") or (name.endswith(".json") and "_infer" in name):
        return True
    return len(parts) >= 2 and parts[-2].endswith("_model") and name in BLEND_KEEP_MODEL


def zip_dir(out, path):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob("*")):
            if f.is_file() and zip_keep(f.relative_to(out)):
                z.write(f, f.relative_to(out.parent.parent.parent))      # artifacts/exp/<name>/...
    os.replace(tmp, path)
    return path


def kaggle_fresh_cache(input_dir=None, env=None):
    """On Kaggle, if NAP3_FRESH_CACHE is unset, point it at attached input fresh_cache/ directories
    (read-only output of ltl1603/mtah-nap3-fresh-gen / kaggle_run fresh-gen), then writable
    /tmp/fresh_cache. Return the configured value or None.
    """
    from src.nap2 import shared
    env = os.environ if env is None else env
    input_dir = INPUT if input_dir is None else input_dir
    if env.get(shared.FRESH_CACHE_ENV) or not os.path.isdir(input_dir):
        return None
    dirs = [d for d in find_inputs("fresh_cache", input_dir) if os.path.isdir(d)]
    dirs += [os.path.dirname(f) for f in find_inputs("release_tag.txt", input_dir)]
    if not dirs:
        return None
    env[shared.FRESH_CACHE_ENV] = os.pathsep.join(dirs + ["/tmp/fresh_cache"])
    return env[shared.FRESH_CACHE_ENV]


def fold_dev_dir(fold, n=5):
    """Cross-validation: create n frozen degradations of the fold's dev match with feed.realize_dev,
    indices 0..n-1 as in official dev/select, at cache/folddev/fold<k>/select/<match>_rNN.npz.
    Generate once, deterministically. Return the parent directory.
    """
    import numpy as np
    from pipeline import feed, split
    from src.nap2 import shared
    name = split.roles(fold=fold)["dev"][0]
    d = ROOT / "cache" / "folddev" / ("fold%d" % fold)
    (d / "select").mkdir(parents=True, exist_ok=True)
    for i in range(n):
        p = d / "select" / ("%s_r%02d.npz" % (name, i))
        if not p.is_file():
            v, t = feed.realize_dev(name, i, shared.release())
            tmp = p.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, **shared._as_dict(v, t))
            tmp.rename(p)
    return d


def use_fold(fold):
    """Enable a cross-validation fold for this process and training subprocesses: match roles and dev directory."""
    from pipeline import split
    from src.nap2 import shared
    os.environ[split.FOLD_ENV] = str(int(fold))
    d = fold_dev_dir(int(fold))
    os.environ["NAP2_DEV_DIR"] = str(d)
    shared.DEV_DIR = d
    return d


def cmd_run(a):
    spec = load_spec(a.spec)
    if spec["train"].get("fold") is not None:
        say("kiểm chéo fold %s: dev = %s" % (spec["train"]["fold"], use_fold(spec["train"]["fold"])))
    fc = kaggle_fresh_cache()
    if fc:
        say("bộ nhớ đệm bản làm bẩn: %s" % fc)
    if spec["base"] not in ("", "scratch") and not (base_path(spec["base"]) / "ckpt").is_dir():
        raise SystemExit("chưa có artifacts/exp/%s/ckpt: python -m src.kaggle_run pull-baseline (hoặc chạy thí nghiệm %s trước)"
                         % (spec["base"], spec["base"]))
    name = spec["name"] + ("_quick" if a.quick else "")
    out = Path(a.out) if a.out else EXP_DIR / name
    if out.exists() and not a.keep:
        shutil.rmtree(out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    shutil.copy2(a.spec, out / "spec.toml")
    run = {"name": name, "spec": spec["path"], "base": spec["base"], "steps": spec["step"], "model": spec["model"],
           "git": git_info(),
           "started": now(), "device": a.device, "host": platform.node(), "quick": a.quick, "time_s": {}}
    try:
        import torch
        if a.device == "cuda" and torch.cuda.is_available():
            run["device"] = "cuda:" + torch.cuda.get_device_name(0)
    except Exception:
        pass
    files = select_files(a.quick)
    say("== %s: base %s, step %s%s, %d file dev/select" % (
        name, spec["base"], [s["use"] for s in spec["step"]] or "không",
        " (thừa kế từ base: %s)" % [s["use"] for s in spec["inherited"]] if spec["inherited"] else "", len(files)))
    try:
        return _run_steps(spec, out, run, files, a)
    finally:
        if a.zip:                                   # Package even after failure so logs remain available.
            say("đã nén", zip_dir(out, a.zip))


def _run_steps(spec, out, run, files, a):
    from src import steps as stp
    t = time.time()
    say("1/4 train")
    runs = train(spec, out, a)
    run["time_s"]["train"] = time.time() - t
    run["train"] = train_lines(runs)
    t = time.time()
    say("2/4 gộp + chỉnh khối 4")
    run["ensemble"] = ensemble(out, runs, files, stp.make_all(spec["step"], "decode"), a,
                               decode_from=spec["train"].get("decode_from"))
    run["time_s"]["ensemble"] = time.time() - t
    t = time.time()
    say("3/4 trộn")
    run["blend"] = blend(spec, out, files, run["ensemble"], a)
    run["time_s"]["blend"] = time.time() - t
    (out / "run.json").write_text(json.dumps(run, indent=1, ensure_ascii=False))
    say("4/4 chấm so với %s" % spec["base"])
    e = evaluate(spec, out, files, spec["base"])
    run["finished"] = now()
    (out / "run.json").write_text(json.dumps(run, indent=1, ensure_ascii=False))
    (out / "eval.json").write_text(json.dumps(e, indent=1, ensure_ascii=False, default=float))
    write_md(out, run, e)
    say("")
    say((out / "eval.md").read_text())
    return 0


# ---------------------------------------------------------------------------------------------- Kaggle

INPUT = "/kaggle/input"
DATA_ROOTS = ("cache", "train_data", "nap2_init")      # Not in git; obtain from the mtah-nap3 dataset.


def find_inputs(pattern, input_dir=INPUT):
    return sorted(glob.glob(os.path.join(input_dir, "**", pattern), recursive=True))


def find_bundle(input_dir=INPUT, work="/tmp/bundle"):
    """mtah-nap3 dataset code/ directory; extract into `work` if Kaggle leaves the zip intact."""
    for p in find_inputs("code/cache", input_dir):
        if os.path.isdir(p):
            return os.path.dirname(p)
    zips = [z for z in find_inputs("*.zip", input_dir) if os.path.basename(z) == "mtah-nap3.zip"]
    if not zips:
        raise SystemExit("Gắn dataset mtah-nap3 (dữ liệu train + dev/select) vào notebook")
    if not os.path.isdir(os.path.join(work, "code", "cache")):
        zipfile.ZipFile(zips[0]).extractall(work)
    return os.path.join(work, "code")


def copy_data(bundle_code, repo=".", roots=DATA_ROOTS):
    """Copy bundle data files missing from the repository; code always comes from git."""
    n = 0
    for root in roots:
        for src in glob.glob(os.path.join(bundle_code, root, "**", "*"), recursive=True):
            if not os.path.isfile(src):
                continue
            dst = os.path.join(repo, os.path.relpath(src, bundle_code))
            if os.path.exists(dst):
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            n += 1
    return n


def _safe_member(name):
    return name.startswith("artifacts/") and not name.endswith("/") and ".." not in name.split("/")


def load_previous(input_dir=INPUT, repo="."):
    """Load artifacts/ from a result dataset (zip or Kaggle-extracted directory) without overwriting; return sources."""
    used = []
    for z in find_inputs("*.zip", input_dir):
        with zipfile.ZipFile(z) as zf:
            names = [n for n in zf.namelist() if _safe_member(n)]
            if not names:
                continue
            for n in names:
                if not os.path.exists(os.path.join(repo, n)):
                    zf.extract(n, repo)
        used.append(z)
    for d in find_inputs("artifacts", input_dir):
        if os.path.isdir(d) and "/code/artifacts" not in d and d not in used:
            for src in glob.glob(os.path.join(d, "**", "*"), recursive=True):
                rel = os.path.relpath(src, os.path.dirname(d))
                dst = os.path.join(repo, rel)
                if os.path.isfile(src) and not os.path.exists(dst):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)
            used.append(d)
    return used


def copy_external(input_dir=INPUT, repo="."):
    """Private mtah-nap3-external dataset npz files (PFF-*, SKC-*) -> cache/external/, without overwriting."""
    n = 0
    for pat in ("PFF-*.npz", "SKC-*.npz", "*_pseudo.npz"):
        for src in find_inputs(pat, input_dir):
            sub = ("pseudo",) if src.endswith("_pseudo.npz") else ()
            dst = os.path.join(repo, "cache", "external", *sub, os.path.basename(src))
            if not os.path.exists(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                n += 1
    return n


def cmd_setup_kaggle(a):
    b = find_bundle(a.input)
    say("dữ liệu: %s, chép %d file vào repo" % (b, copy_data(b, str(ROOT))))
    say("trận ngoài (PFF / SkillCorner): chép %d file vào cache/external" % copy_external(a.input, str(ROOT)))
    used = load_previous(a.input, str(ROOT))
    say("kết quả trước: %s" % (used or "không có"))
    return 0


# ---------------------------------------------------------------------------------------------- other

def cmd_show(a):
    p = EXP_DIR / a.name / "eval.md"
    if not p.is_file():
        raise SystemExit("chưa có %s" % p)
    print(p.read_text())
    return 0


def cmd_steps(a):
    from src import steps as stp
    for k, cls in sorted(stp.discover().items()):
        doc = (sys.modules[cls.__module__].__doc__ or "").strip().splitlines()
        print("%-22s %s  tham số %s" % (k, doc[0] if doc else "", cls.defaults or "{}"))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.exp", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="chạy một thí nghiệm")
    p.add_argument("spec")
    p.add_argument("--quick", action="store_true", help="1 hạt giống, 1 epoch mỗi giai đoạn, 1 file dev (thử đường ống)")
    p.add_argument("--device", default="cuda" if os.path.isdir("/kaggle") else "cpu")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out")
    p.add_argument("--keep", action="store_true", help="không xoá thư mục kết quả cũ")
    p.add_argument("--zip", help="nén thư mục kết quả ra file này")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("show", help="in eval.md")
    p.add_argument("name")
    p.set_defaults(fn=cmd_show)
    p = sub.add_parser("steps", help="các step đang có")
    p.set_defaults(fn=cmd_steps)
    p = sub.add_parser("setup-kaggle", help="(trên Kaggle) chép dữ liệu và nạp kết quả trước vào repo")
    p.add_argument("--input", default=INPUT)
    p.set_defaults(fn=cmd_setup_kaggle)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
