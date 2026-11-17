# Baseline benchmark on external DFL data

Run the organizer nearest-player rule and all four fixed smoothing variants on clean observations and five independently degraded realizations per match:

```sh
.venv/bin/python -m pipeline.benchmark_baseline \
  --profile cache/degradation/validation-20261106/target.json \
  --params cache/degradation/validation-20261106/dfl.json \
  --out cache/degradation/new-baseline-benchmark
```

The default matches are training candidates J03WMX, J03WOH, J03WOY, J03WPY, J03WQQ, and development J03WN1. The CLI and scoring API reject reserved J03WR9. No competition labels are read. Seeds default to 17, 43, 101, 211, 509. Degradation runs once per complete match; predictions use only the degraded observations. Supervision comes from the original external DFL `y_true` and its valid mask, when available.

`baseline_predictions` supports both raw `Match` inputs (normalized coordinates scaled by pitch) and centered metric `View` inputs (already metres). Both use the fixed slot-to-team contract. No velocities, source speeds, identities or labels are used. A missing ball produces no nearest-player vote. Organizer centered majority smoothing can still return a team from neighboring visible frames, and it never bridges halves. This preserves the reference algorithm rather than imposing an additional missing-ball rule after smoothing.

`metrics.json` contains organizer possession-metric point estimates, including macro-F1 and mean spellF1 over IoU 0.3–0.7. Bootstrap draws are disabled, so bootstrap fields are null, not confidence intervals. The transition diagnostic uses a fixed 50-frame tolerance. Explicit X separators in temporary scorer arrays prevent adjacent cached halves being merged; exported predictions retain original zero-based cache frame indices. Invalid supervision is X and unscored.

`scores.csv` and `scores.json` provide per-match/per-condition/per-seed/per-variant scores and differences from the same match's clean baseline. `REPORT.md` reports degraded means and sample SD across seeds, keeping development distinct from training. Each match's prediction NPZ contains four variant arrays plus `frame_indices` and `half`; codes are 0=H, 1=A, 2=D. Export readback and frozen input hashes are checked. No supervised labels are included in prediction NPZs.

All variants are fixed before evaluation. Published `baseline_smooth50` remains the reference; this run does not select smoothing settings. These results measure sensitivity to observation quality and football/label-domain differences. They do not establish a benefit from degraded training, since this baseline has no learned parameters. Metrica remains diagnostic-only and is excluded from this benchmark.
