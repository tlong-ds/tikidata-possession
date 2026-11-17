"""Audit training logs and frozen evaluation predictions, then rebuild the analysis figure.

Collect: python docs/figures/build_analysis.py --source-root PATH --predictions PATH
Redraw from the committed summary: python docs/figures/build_analysis.py
"""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SUMMARY = ROOT / "docs/training_analysis.json"


def summary(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "sd": float(values.std(ddof=1))}


def collect(source, predictions):
    from evaluation import possession_metrics as pm
    from pipeline.quality import segments
    from pipeline.split import guard
    from src.core.evaltools import breakdown, paired_ci

    training = []
    for seed in (1, 2, 3):
        directory = source / f"artifacts/exp/bt04_metrica/ckpt/s{seed}"
        rows = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
        assert [r["epoch"] for r in rows] == list(range(1, 11))
        training.append({"seed": seed, "epochs": [
            {"epoch": r["epoch"], "loss": r["loss"],
             "monitoring_spellF1": r["val"]["spell_f1"]} for r in rows],
            "retained_epochs": sorted(int(p.stem[2:]) for p in
                                      (source / f"models/bt04_metrica/ckpt/s{seed}").glob("ep*.pt"))})

    info = json.loads((source / "artifacts/exp/bt04_metrica/ckpt/s1/summary.json").read_text())
    counts = {"DFL": np.zeros(3, dtype=np.int64), "Metrica": np.zeros(3, dtype=np.int64)}
    matches = []
    for name in info["matches"]["train"]:
        guard(name, "train")
        provider = "Metrica" if name.startswith("METRICA") else "DFL"
        directory = "external" if provider == "Metrica" else "dfl"
        with np.load(source / "cache" / directory / f"{name}.npz") as z:
            labels = z["x_y_true"]
            valid = (z["half"] > 0) & np.isin(labels, (0, 1, 2))
            row = np.bincount(labels[valid], minlength=3)
        counts[provider] += row
        matches.append({"match": name, "source": provider, "label_counts_H_A_D": row.tolist()})
    corpus = {provider: {"frames": int(row.sum()), "hours": float(row.sum() / 25 / 3600),
                         "label_counts_H_A_D": row.tolist(),
                         "label_shares_H_A_D": (row / row.sum()).tolist()}
              for provider, row in counts.items()}

    scores = json.loads((ROOT / "docs/evaluation_results.json").read_text())
    evaluated = []
    for index, filename in enumerate(scores["versions"]):
        with np.load(source / "train_data/dev/report" / filename) as z:
            arrays = {key: z[key] for key in ("y", "half", "B", "fps") if key in z}
        with np.load(predictions / filename) as z:
            codes = z["codes"]
        y, half = arrays["y"], arrays["half"]
        cat = lambda array: np.concatenate([np.r_[array[a:b], pm.X] for a, b in segments(half)])
        gt = cat(np.where(y >= 0, y, pm.X).astype(np.int8))
        pred = cat(codes)
        result = pm.evaluate(gt, pred, bootstrap_n=0)
        assert np.isclose(result["spell_f1_mean"], scores["systems"]["bt04_metrica"]["spellF1"][index],
                          rtol=0, atol=1e-12)
        assert np.isclose(result["macro_f1"], scores["systems"]["bt04_metrica"]["macroF1"][index],
                          rtol=0, atol=1e-12)
        runs = pm.valid_runs(gt)
        true_spells = pm.possession_spells(gt, runs)
        pred_spells = pm.possession_spells(pm.fill_abstentions(pred, runs), runs)
        flags = pm.matched_gt_flags(pm.spell_overlaps(true_spells, pred_spells), len(true_spells), 0.5)
        short = np.asarray([(b-a) <= 50 for a, b, _ in true_spells])
        diagnostic = breakdown(arrays, codes=codes)
        for row in diagnostic["rows"]:
            row.pop("acc_prob")
        evaluated.append({"version": filename, "per_class": result["per_class"],
                          "spell_by_iou": {str(k): v for k, v in result["spell_f1"].items()},
                          "true_possessions": result["n_gt_spells"],
                          "predicted_possessions": result["n_pred_spells"],
                          "short_possessions": int(short.sum()),
                          "short_recall_iou_05": float(flags[short].mean()),
                          "long_recall_iou_05": float(flags[~short].mean()),
                          "breakdown": diagnostic})
    per_class = {label: {metric: summary([r["per_class"][label][metric] for r in evaluated])
                         for metric in ("precision", "recall", "f1")} for label in ("H", "A", "D")}
    by_iou = {str(iou): summary([r["spell_by_iou"][str(iou)]["f1"] for r in evaluated])
              for iou in pm.SPELL_IOUS}
    paired = {metric: paired_ci(scores["systems"]["organisers_baseline"][metric],
                                scores["systems"]["bt04_metrica"][metric])
              for metric in ("spellF1", "macroF1")}
    data = {"stage": "Selected configuration, before final retraining",
            "training": training, "training_corpus": corpus, "training_matches": matches,
            "evaluation": evaluated, "per_class_summary": per_class, "spell_by_iou_summary": by_iou,
            "vs_nearest_player_baseline": paired}
    SUMMARY.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    return data


def draw(data):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.10))
    colors = ("#315f8b", "#087f79", "#b66a31")
    axis = axes[0]
    versions = data["evaluation"]
    short_counts = {r["short_possessions"] for r in versions}
    long_counts = {r["true_possessions"] - r["short_possessions"] for r in versions}
    assert len(short_counts) == len(long_counts) == 1
    labels = (f"≤2 s\nn={next(iter(short_counts))}", f">2 s\nn={next(iter(long_counts))}")
    means = [100 * np.mean([r[key] for r in versions])
             for key in ("short_recall_iou_05", "long_recall_iou_05")]
    errors = [100 * np.std([r[key] for r in versions], ddof=1)
              for key in ("short_recall_iou_05", "long_recall_iou_05")]
    axis.bar(labels, means, yerr=errors, width=0.55, color=colors[:2], capsize=3)
    axis.set(title="(a) Recall by duration", ylabel="Recall at IoU 0.5 (%)", ylim=(0, 108))
    for index, (value, error) in enumerate(zip(means, errors)):
        axis.text(index, value + error + 3, f"{value:.1f}%", ha="center", fontsize=7)
    axis = axes[1]
    labels = ("H", "A", "D")
    means = [data["per_class_summary"][label]["f1"]["mean"] for label in labels]
    errors = [data["per_class_summary"][label]["f1"]["sd"] for label in labels]
    axis.bar(labels, means, yerr=errors, width=0.55, color=colors, capsize=3)
    axis.axhline(np.mean(means), color="#667784", linestyle="--", linewidth=1)
    axis.set(title="(b) Frame classification", ylabel="Per-class F1", ylim=(0, 1.08))
    for index, value in enumerate(means):
        axis.text(index, value+0.035, f"{value:.3f}", ha="center", fontsize=7)
    axis.text(0.03, 0.03, f"macroF1 = {np.mean(means):.4f}", transform=axis.transAxes, fontsize=7,
              bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.5})
    axis = axes[2]
    thresholds = [float(k) for k in data["spell_by_iou_summary"]]
    means = [r["mean"] for r in data["spell_by_iou_summary"].values()]
    errors = [r["sd"] for r in data["spell_by_iou_summary"].values()]
    axis.errorbar(thresholds, means, yerr=errors, color=colors[0], marker="o", markersize=3,
                  linewidth=1.6, capsize=3)
    axis.set(title="(c) Possession recovery", xlabel="Temporal IoU threshold",
             ylabel="Interval F1 at threshold", ylim=(0, 1.08))
    axis.set_xticks(thresholds)
    axis.text(0.03, 0.03, f"spellF1 = {np.mean(means):.4f}", transform=axis.transAxes, fontsize=7)
    for axis in axes:
        axis.grid(axis="y", color="#e5eaf0", linewidth=0.6)
        axis.set_axisbelow(True)
    fig.tight_layout(w_pad=1.7, pad=0.8)
    for output in (ROOT / "docs/figures/training_analysis.svg", ROOT / "submission/figures/training_analysis.pdf"):
        fig.savefig(output, facecolor="white", metadata={"Creator": "TikiData"})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--predictions", type=Path)
    args = parser.parse_args()
    if bool(args.source_root) != bool(args.predictions):
        parser.error("Supply --source-root and --predictions together")
    data = collect(args.source_root, args.predictions) if args.source_root else json.loads(SUMMARY.read_text())
    draw(data)
    print("Analysis summary and vector figure ready.")


if __name__ == "__main__":
    main()
