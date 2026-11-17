"""Stage-3 submission: label the competition match with the packaged model, write CSV and validate.

    python -m src.submit run                      # models/final_metrica -> evaluation/results/nap3.csv
    python -m src.submit run --check-twice        # 1 and 4 threads; require identical hashes
    python -m src.submit pack baseline            # artifacts/exp/<name> -> models/<name>, commit weights
    python -m src.submit run --model DIR_ENS      # DIR_ENS/ensemble.json; see ENSEMBLE_DOC
    python -m src.submit run --model A B --weights 2 1 --decode-params A/decode_switch.json

No training or label access. Input: cache/degradation/competition/BAR-ATH_tactical_view.npz,
built from data/ by `python -m pipeline.prepare --inference`, called automatically if missing.
Model: models/<name>/ checkpoints, using their saved feature layouts and decode_switch.json.
CPU deterministic algorithms reproduce the exact file from the data + repository.
Pinned environment: requirements-submit.txt.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
DEFAULT_OUT = ROOT / "evaluation" / "results" / "nap3.csv"
# Selected model: bt04_metrica = cg_ball_touch04 (contest geometry, ball-touch lag 0.4 s, wide2 decoder) plus three
# public Metrica training matches: 0.6605 dev/select, 0.6349 on target-like smoothed-ball dev (best; +0.011 over
# cg_ball_touch04, 5/5 wins). models/baseline remains the experiment initialization.
DEFAULT_MODEL = "final_metrica"
DEFAULT_INPUT = ROOT / "cache" / "degradation" / "competition" / "BAR-ATH_tactical_view.npz"   # = shared.COMPETITION
ENSEMBLE_FILE = "ensemble.json"
ENSEMBLE_DOC = """ensemble.json: {"members": [{"model": "artifacts/exp/cg_ball_touch04", "weight": 1}, ...],
  "how": "prob" | "log", "decode_switch": {"params": {...SwitchParams}}}
Mỗi thành viên là một thư mục có ckpt/s*/ep*.pt (feature theo layout riêng của nó); đường dẫn tương đối
được tìm theo thứ tự: thư mục chứa ensemble.json, gốc repo, models/<tên>. Gộp: C, K, S trung bình có trọng
số; E: 'prob' = trung bình xác suất rồi lấy log, 'log' = trung bình log-xác suất rồi chuẩn hoá lại."""


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def model_ckpts(model_dir):
    cks = sorted(Path(model_dir).glob("ckpt/s*/ep*.pt"))
    if not cks:
        raise SystemExit("không có checkpoint trong %s/ckpt/s*/ep*.pt (python -m src.submit pack <tên>)" % model_dir)
    return cks


def combine_members(parts, weights=None, how="prob", half=None):
    """parts: per-model [(E,C,K,S)] with equal frame counts -> combined (E,C,K,S).
    Nonnegative weights are normalized, equal by default. how='prob' averages P(H,A,D) then logs;
    'log' averages log probabilities then renormalizes. Outside halves, preserve
    src.core.infer_switch.combine conventions: E=log(1/3), C=0.5, K=S=0.
    Deterministic member-order summation.
    """
    import numpy as np
    if not parts:
        raise ValueError("cần ít nhất một mô hình")
    w = np.ones(len(parts)) if weights is None else np.asarray(weights, np.float64)
    if w.shape != (len(parts),) or (w < 0).any() or w.sum() <= 0:
        raise ValueError("trọng số phải không âm, đúng %d số, tổng > 0" % len(parts))
    w = w / w.sum()

    def mix(i):
        acc = None
        for wk, p in zip(w, parts):
            x = np.asarray(p[i], np.float64) * wk
            acc = x if acc is None else acc + x
        return acc

    if how == "prob":
        acc = None
        for wk, p in zip(w, parts):
            x = np.exp(np.asarray(p[0], np.float64)) * wk
            acc = x if acc is None else acc + x
        E = np.log(np.clip(acc, 1e-8, 1.0))
    elif how == "log":
        Z = mix(0)
        m = Z.max(1, keepdims=True)
        E = Z - (m + np.log(np.exp(Z - m).sum(1, keepdims=True)))
        E = np.maximum(E, np.log(1e-8))
    else:
        raise ValueError("how phải là 'prob' hoặc 'log', nhận %r" % (how,))
    C, K, S = mix(1), mix(2), mix(3)
    if half is not None:
        out_h = np.asarray(half) <= 0
        E[out_h] = np.log(1 / 3)
        C[out_h] = 0.5
        K[out_h] = 0.0
        S[out_h] = 0.0
    return E, C, K, S


def _resolve_member(path, base=None):
    p = Path(path)
    cands = [p] if p.is_absolute() else ([Path(base) / p] if base else []) + [ROOT / p, MODELS / p]
    for c in cands:
        if c.is_dir():
            return c
    raise SystemExit("không tìm thấy thư mục mô hình %r (đã thử %s)" % (str(path), ", ".join(map(str, cands))))


def load_ensemble(model_dir):
    """Directory with ensemble.json -> (members [(dir,weight)], how, SwitchParams dict); otherwise None."""
    f = Path(model_dir) / ENSEMBLE_FILE
    if not f.is_file():
        return None
    d = json.loads(f.read_text())
    mem = [(_resolve_member(m["model"], model_dir), float(m.get("weight", 1.0))) for m in d["members"]]
    dec = d.get("decode_switch", d.get("params"))
    if dec is None:
        raise SystemExit("%s thiếu decode_switch" % f)
    return mem, d.get("how", "prob"), dec.get("params", dec)


def member_probs(model_dir, raw):
    """One model checkpoint directory + raw-data dict -> ((E,C,K,S), prepared, info)."""
    from src.core import feats
    from src.core.infer_switch import layout_of, load_checkpoint, run_models
    cks = model_ckpts(model_dir)
    models = [load_checkpoint(str(p)) for p in cks]
    lay = layout_of([ck for _, ck in models])
    d = feats.prepare_raw(raw, with_targets=False, layout=lay)
    return run_models(d, models), d, {"model": _rel(model_dir), "n_ckpt": len(cks),
                                      "steps": [s["use"] for s in (lay or {}).get("features", [])]}


def _rel(p):
    p = Path(p).absolute()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def pre_kickoff_dead(codes, d, raw):
    """Graded pre-kickoff frames -> D (src/core/kickoff.py). Kickoff uses automatic Smart Tagging
    Kick Off (Start) windows from organiser data plus View ball positions. Missing tagging is an error.
    """
    import numpy as np
    sys.path.insert(0, str(ROOT))
    import games
    from src.core import kickoff
    xml = games.require(["smart_tagging"])[0]
    fps = float(raw["fps"])
    windows = kickoff.kickoff_windows(xml, fps)
    if set(windows) != {1, 2}:
        raise SystemExit("Smart Tagging %s: cần đúng một Kick Off (Start) mỗi hiệp, có %s" % (xml, sorted(windows)))
    codes, info = kickoff.apply(np.asarray(codes), d["half"], d["graded"], np.asarray(raw["B"]), windows, fps)
    return codes, info


def label_ensemble(members, how, params, input_path, threads=1):
    """members [(dir,weight)] -> label()-compatible result, combining frame probabilities before decoding."""
    import numpy as np
    import torch
    torch.set_num_threads(max(1, threads))
    torch.use_deterministic_algorithms(True)
    from src.nap2 import shared
    from src.core import decode_switch as dsw
    from src.core.infer_switch import decode_all
    params = dsw.ensure(params)
    raw = shared.load_file(str(input_path))
    parts, infos, d = [], [], None
    for m, w in members:
        pr, dm, inf = member_probs(m, raw)
        if d is not None and not np.array_equal(dm["half"], d["half"]):
            raise SystemExit("các mô hình cho half khác nhau")
        d = dm
        parts.append(pr)
        infos.append(dict(inf, weight=w))
    E, C, K, S = combine_members(parts, [w for _, w in members], how, d["half"])
    codes = decode_all(E, S, C, K, d["half"], d["graded"], params)
    codes, ko = pre_kickoff_dead(codes, d, raw)
    info = {"model": "ensemble(%s)" % ",".join(i["model"] for i in infos), "how": how, "members": infos,
            "n_ckpt": sum(i["n_ckpt"] for i in infos), "steps": [i["steps"] for i in infos],
            "decode_switch": params.to_dict(), "kickoff": ko, "frames": int(len(codes))}
    return np.asarray(codes, np.int8), np.asarray(d["graded"], bool), info


def label(model_dir, input_path, threads=1):
    """-> (int8 [T] codes 0=H/1=A/2=D/3=X, bool [T] graded, info); ensemble.json enables multiple models."""
    ens = load_ensemble(model_dir)
    if ens is not None:
        codes, graded, info = label_ensemble(ens[0], ens[1], ens[2], input_path, threads)
        info["ensemble"] = _rel(Path(model_dir))
        return codes, graded, info
    import numpy as np
    import torch
    torch.set_num_threads(max(1, threads))
    torch.use_deterministic_algorithms(True)
    from src.nap2 import shared
    from src.core import decode_switch as dsw
    from src.core import feats
    from src.core.infer_switch import decode_all, layout_of, load_checkpoint, run_models
    cks = model_ckpts(model_dir)
    models = [load_checkpoint(str(p)) for p in cks]
    lay = layout_of([ck for _, ck in models])
    params = dsw.ensure(json.loads((Path(model_dir) / "decode_switch.json").read_text())["params"])
    raw = shared.load_file(str(input_path))
    d = feats.prepare_raw(raw, with_targets=False, layout=lay)
    E, C, K, S = run_models(d, models)
    codes = decode_all(E, S, C, K, d["half"], d["graded"], params)
    codes, ko = pre_kickoff_dead(codes, d, raw)
    info = {"model": str(Path(model_dir).relative_to(ROOT)) if Path(model_dir).is_relative_to(ROOT) else str(model_dir),
            "n_ckpt": len(cks), "steps": [s["use"] for s in (lay or {}).get("features", [])],
            "decode_switch": params.to_dict(), "kickoff": ko, "frames": int(len(codes))}
    return np.asarray(codes, np.int8), np.asarray(d["graded"], bool), info


def write_csv(path, codes, graded):
    from src.nap2.infer import write_submission
    return write_submission(str(path), codes, graded)


def validate(path):
    r = subprocess.run([sys.executable, str(ROOT / "evaluation" / "validate_submission.py"), str(path)],
                       capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def _model_dir(name):
    return Path(name) if Path(name).is_dir() else MODELS / name


def cmd_run(a):
    names = a.model if isinstance(a.model, list) else [a.model]
    out = Path(a.out)
    if not Path(a.input).exists() and Path(a.input) == Path(DEFAULT_INPUT):
        print("chưa có đầu vào trận đề, dựng từ data/ (python -m pipeline.prepare --inference) ...")
        from pipeline import prepare
        if prepare.main(["--inference"]):
            raise SystemExit("không dựng được đầu vào trận đề")
    if len(names) > 1:                 # CLI ensemble shortcut: write temporary ensemble.json beside the output.
        if a.weights is not None and len(a.weights) != len(names):
            raise SystemExit("--weights cần đúng %d số" % len(names))
        dirs = [_model_dir(n) for n in names]
        dec = a.decode_params or str(dirs[0] / "decode_switch.json")
        dec = json.loads(Path(dec).read_text()) if Path(dec).is_file() else json.loads(dec)
        spec = {"members": [{"model": str(d.resolve()), "weight": w}
                            for d, w in zip(dirs, a.weights or [1.0] * len(dirs))],
                "how": a.how, "decode_switch": {"params": dec.get("params", dec)}}
        model = out.with_name(out.stem + ".ensemble")
        model.mkdir(parents=True, exist_ok=True)
        (model / ENSEMBLE_FILE).write_text(json.dumps(spec, indent=1))
    else:
        model = _model_dir(names[0])
    t0 = time.time()
    codes, graded, info = label(model, a.input, a.threads)
    write_csv(out, codes, graded)
    h = sha256(out)
    print("đã ghi %s (%d khung, %.1f phút, %d luồng) sha256 %s" % (out, info["frames"], (time.time() - t0) / 60,
                                                                  a.threads, h))
    print("mô hình %s: %d checkpoint, cột thêm %s" % (info["model"], info["n_ckpt"], info["steps"] or "không"))
    if a.check_twice:
        other = out.with_name(out.stem + ".check.csv")
        codes2, graded2, _ = label(model, a.input, 4 if a.threads == 1 else 1)
        write_csv(other, codes2, graded2)
        h2 = sha256(other)
        other.unlink()
        if h2 != h:
            raise SystemExit("KHÔNG tái lập: chạy lại với số luồng khác ra sha256 %s" % h2)
        print("tái lập: chạy lại với %d luồng ra đúng file (sha256 giống)" % (4 if a.threads == 1 else 1))
    code, text = validate(out)
    print(text)
    if code:
        raise SystemExit("validate_submission.py báo lỗi")
    (out.with_suffix(".json")).write_text(json.dumps(dict(info, sha256=h, csv=str(out.name)), indent=1))
    return 0


def cmd_pack(a):
    """artifacts/exp/<name> -> models/<name>: checkpoints, decode_switch.json and small eval/run metadata for git."""
    src = ROOT / "artifacts" / "exp" / a.name
    dst = MODELS / (a.to or a.name)
    if not (src / "decode_switch.json").is_file():
        raise SystemExit("chưa có %s/decode_switch.json (pull-baseline / fetch trước)" % src)
    if dst.exists():
        shutil.rmtree(dst)
    for run in sorted((src / "ckpt").glob("s*")):
        (dst / "ckpt" / run.name).mkdir(parents=True)
        for f in sorted(run.glob("ep*.pt")):
            shutil.copy2(f, dst / "ckpt" / run.name / f.name)
        if (run / "summary.json").is_file():          # best_ckpt: initialization for experiments using this model as their base.
            shutil.copy2(run / "summary.json", dst / "ckpt" / run.name / "summary.json")
    for f in ("decode_switch.json", "eval.md", "eval.json", "run.json", "spec.toml"):
        if (src / f).is_file():
            shutil.copy2(src / f, dst / f)
    size = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()) / 1e6
    print("đã đóng gói %s -> %s (%d checkpoint, %.1f MB)" % (src.relative_to(ROOT), dst.relative_to(ROOT),
                                                          len(model_ckpts(dst)), size))
    return 0


def main(argv=None):
    from src.nap2 import shared
    ap = argparse.ArgumentParser(prog="python -m src.submit", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="gán nhãn trận đề, ghi CSV, tự kiểm")
    p.add_argument("--model", nargs="+", default=[DEFAULT_MODEL],
                   help="tên trong models/ hoặc đường dẫn thư mục (có thể là thư mục có ensemble.json); "
                        "nhiều tên = gộp nhanh")
    p.add_argument("--weights", nargs="+", type=float, help="trọng số khi --model có nhiều tên (mặc định bằng nhau)")
    p.add_argument("--how", default="prob", choices=("prob", "log"), help="cách gộp E khi --model có nhiều tên")
    p.add_argument("--decode-params", help="decode_switch khi --model có nhiều tên (mặc định của mô hình đầu)")
    p.add_argument("--input", default=str(shared.COMPETITION))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--check-twice", action="store_true", help="chạy lại với số luồng khác, so sha256")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("pack", help="đóng gói một thí nghiệm vào models/")
    p.add_argument("name")
    p.add_argument("--to", help="tên thư mục trong models/ (mặc định = name)")
    p.set_defaults(fn=cmd_pack)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
