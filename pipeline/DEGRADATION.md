# Source-aware observation degradation

`load_match()` → `degrade_match()` on complete halves → future `View` → training windows.
Existing caches, labels, model implementations, and organizer scorer are unchanged. Generated
artifacts belong in ignored `cache/degradation/`; the CLI refuses overwrites and other destinations.
Only DFL and Metrica are supported. PFF awaits a loader.

The current profile version is `tracking-quality-2`. Rebuild v1 profiles and recalibrate into a new
output directory; old artifacts remain untouched and are rejected by the v2 loader. V1 and v2
aggregate distances are not directly comparable: v2 corrects empirical tail weights and uncertainty.
The absolute coverage, duration, and improvement thresholds have not been relaxed.

## Commands

Run from the repository root with the pipeline NumPy environment. The examples explicitly select
**illustrative training matches**, not an established team train/validation split. Replace the names
with your actual training-fold manifest before fitting production parameters. Never calibrate using
validation or test observations.

```bash
python -m pipeline.degradation_cli profile \
  --target BAR-ATH_tactical --out cache/degradation/v2/target.json

python -m pipeline.degradation_cli calibrate \
  --profile cache/degradation/v2/target.json \
  --train-matches DFL-MAT-J03WMX --seed 2027 \
  --out cache/degradation/v2/dfl.json

python -m pipeline.degradation_cli calibrate \
  --profile cache/degradation/v2/target.json \
  --train-matches Sample_Game_1 --seed 2027 \
  --out cache/degradation/v2/metrica.json

python -m pipeline.degradation_cli report \
  --profile cache/degradation/v2/target.json --params cache/degradation/v2/dfl.json \
  --matches DFL-MAT-J03WMX DFL-MAT-J03WN1 --out cache/degradation/v2/dfl-report.json

python -m pipeline.degradation_cli export \
  --profile cache/degradation/v2/target.json --params cache/degradation/v2/metrica.json \
  --matches Sample_Game_2 --seed 17 --out cache/degradation/v2/metrica-sample

python -m pipeline.degradation_example \
  --profile cache/degradation/v2/target.json --params cache/degradation/v2/dfl.json \
  --match DFL-MAT-J03WMX --epoch 1

python -m unittest discover -s tests -v
```

A failed quality gate exits **2**, after saving the profile, best parameters, or report for inspection.
An unsuccessful calibration is usable for research/debugging, **not a validated quality match**.
Never silently relax the gates. Profile construction audits half 1 against half 2 and reverses this
before fitting the final profile from both halves; its sidecar is `<profile>.halves.json`.
Reports always use seeds **17, 43, 101, 211, 509**, per match, with across-seed summary statistics.
The calibration budget is exactly **64 global candidates + 16 local refinements**, with common random
numbers across candidates and a retained best candidate even on failure. Selection minimizes the
number of failed gates first, then the equally weighted family distance; an improved average alone
cannot replace a candidate that satisfies more gates. The optional Python
`progress` callback does not change results.

## Model/target separation

```python
from pipeline.load import load_match
from pipeline.quality import load_profile
from pipeline.degrade import degrade_match, epoch_seed

clean = load_match("DFL-MAT-J03WMX")
# params is the frozen JSON from training-only calibration; profile loaded from the command above.
sample = degrade_match(clean, profile, params,
                       seed=epoch_seed(2027, epoch, clean.name))
model_input = sample.observations
loss_targets = sample.targets
provenance = sample.audit
```

Dương, Phúc, Đức Anh can run `pipeline.degradation_example` directly. Increment `epoch` for training;
keep it at zero for evaluation. Generate a single complete-half realization before extracting any
windows so overlapping windows share corruption. Cache that realization for the epoch if needed.
The function does not mutate input arrays and returns independent arrays for supervision.

Observation metadata is allowlisted to pitch, fps, coordinate convention, and synthetic-camera status.
Extras contain only `camera_center[T,2]`, `camera_valid[T]`, and `camera_extent[T,2]`. Clean player
identity tables, events, `Bz`, labels, and clean ball positions never reach the model.
`targets['labels']` preserves `y_true`, `touch_true`, or Metrica's **weak** `y_draft`; validity masks
exclude unknown labels and half 0. `targets['ball']` preserves all three original ball channels,
with per-channel and coordinate-valid masks. Original extras/caches remain available separately.
Audit retains seeds, source/profile/config hashes, masks, original-slot provenance and duplicates.
Export writes separate `observations.npz`, `targets.npz`, `audit_masks.npz`, and `audit.json` files.

## Geometry and synthetic assumptions

The homography maps normalized pitch coordinates to pixels. Profiling inverts it at image center
and four image corners, rejecting singular matrices and unbounded footprints. The shipped tactical
video's 1920×1080 size comes from `DATA.md`; other homography inputs must declare
`meta['image_size'] = [width, height]`. No synthetic homography is created.

External camera centers are explicitly **synthetic**: source player centroid plus target residuals.
Residuals, footprint extents, and counts are resampled jointly in within-half blocks; one-second
residual blending reduces seam jumps. Audit records every block seam. Invalid centroids/camera
samples produce invalid centers. Extents can be unavailable even when the center is valid.

The future `View` must transform centers with exactly the same geometry as players and ball:
for a 180-degree direction normalization use `(1-x, 1-y)`, then multiply by `(pitch_length,
pitch_width)` if using metres. `quality.convert_camera_center` implements that convention and
preserves NaNs. Apply the corresponding transform to supervised ball coordinates separately.
Never apply pixel-homography geometry to a synthetic center.

Camera-relative ranks and empirical target counts share one dropout budget. The selected players
persist until counts or source availability change; camera geometry chooses whom to hide or restore
at those transitions. Within-team allocation samples full empirical occupancy lifetimes, including
the long tail, instead of imposing the median as a maximum. It avoids joining different players in
consecutive frames of the same slot whenever a safe destination is available. Full slot exhaustion
still preserves the collision-free allocation and provenance. Excess-count episodes permit limited copies of currently visible tracks; this is
an assumption about observed excess counts, **not an estimated duplicate rate**. Missing source
positions are never reconstructed. Audit maps copies to the original observed source slot.

Ball masking first computes an exact additional-missing-frame budget from source and target
coverage. For existing gaps, it spends that budget extending short gaps toward target quantiles,
without filling or merging them. Remaining budget uses stratified empirical missing/visible episodes.
Conditional player-count/centroid rates alter episode spacing, which is then normalized to preserve
the budget. The audit records requested, applied, and existing-gap-extension frame counts. Already
sparser ball sources get no further ball dropout. Their gap-duration mismatch can therefore remain
unattainable under the no-extra-dropout constraint and must remain a failed gate. Player sources below target mean counts similarly get no count dropout.
Speed masking is conditional on coordinates and never synthesizes absent speed. Correlated position
perturbation amplitudes are searched only when source roughness is below target, with a per-match
guard. Off-pitch observations are retained, without clipping. The implementation does not infer true
measurement-error variance, identity-switch rate, or wrong-team frequency.

To run explicit stress studies, copy the parameter dictionary and set `preset` to
`stress_wrong_team` (occasional cross-team slot swaps) or `stress_jumps` (coordinate bursts).
They are excluded from quality-match claims and force a failed report. Default `quality` uses neither.

## Statistics, gates, and limitations

Profiling reads tracking, homography, metadata and graded/half boundaries only. It never imports EDA
or reads public competition labels/events. Profiles include input hashes, version, empirical
sequences, counts, censored-run flags, sample sizes, and uncertainty. Temporal operations stop at
half boundaries; displacement/residual measurements require contiguous finite coordinates. Run
lengths touching boundaries are retained as censored and excluded from complete-run quantiles.

Families are ball coverage/durations, player counts/dynamics, coupled missingness, slot continuity,
speed availability, motion, and camera. Joint team-count histograms are retained; count margins and
team-count difference also enter distance. Count autocorrelation uses valid local windows at 1, 5,
and 30 seconds. Motion uses adjacent displacement and three-frame midpoint residuals. Camera-ball
offsets are conditional on visible ball and may be biased. Long gaps mean at least ten seconds.

Distances use absolute rate errors and exact Wasserstein distances between stored empirical
sketches, with log1p seconds for durations. Each sample carries its actual empirical mass; an extreme
is not assigned half a quantile-grid interval. Sketches use at most 8192 deterministic sampled values;
original sample counts, exact means, full-data minima/maxima and descriptive quantiles remain
available. Distances are normalized by target spread (with a numerical floor) and
averaged with equal family weights. Calibration includes all supported synthetic camera diagnostics;
clean-to-degraded reduction includes only diagnostics observable in both source versions, so the
absence of a clean external camera channel cannot artificially inflate improvement. Unsupported diagnostics are listed; losing previously available
support is an explicit failure. Synthetic camera fit is reported even when a clean external camera
comparison is unavailable.

Uncertainty uses 128 deterministic bootstrap replicates. Rates resample 30-second block totals and
valid denominators together, then divide their sums; this preserves conditional sample weights and
centers the bootstrap on the same empirical population. Distributions resample equal-mass empirical
sketches, with the same support size, and use exact sorted-sample Wasserstein distances. Missing-gap
median/p90 CIs use complete-run bootstrap. Distribution/run dependence can understate uncertainty; the reports explicitly say so.
Mean CIs are not quantile CIs. Censoring may bias complete-run estimates; both counts are reported.
These are engineering diagnostics, not formal proofs of distribution equivalence.

Acceptance requires ball visibility within 3 percentage points, mean player count within one,
ball-speed missingness within 5 points (explicit source-missingness exemption), median/p90 missing
ball durations within 25%, and 25% aggregate reduction outside estimated **target-only** variability.
Source/degraded uncertainty cannot exempt the aggregate gate; it is used only in the separate
family non-worsening comparisons. Reports also calculate an irreducible aggregate floor for wholly
unavailable source speed channels. If that floor exceeds the required acceptance ceiling, they mark
the gate as provably unattainable under the current compared support and no-synthetic-speed rule.
This explains the failure; it does not exempt the gate or change the required 25% reduction. Every supported
family has a separate non-worsening gate. Good averages cannot override failed gates. Half-to-half
checks use direct gates, without asking an unchanged half to improve relative to itself.

Secondary diagnostics run the organizer nearest-player rule adapted to in-memory arrays and the
actual organizer `smooth` implementation for all four variants, separately within each half.
They report counts/transitions and external-source macro-F1 where labels exist, clearly identifying
Metrica weak labels. They are never part of calibration objectives and never read competition GT.
Matching these observables cannot eliminate football-style or possession-definition domain shift.

## Verification on 2026-11-06

All 27 focused tests pass, including exact ball budgets under conditional dropout, persistent player
selection, empirical fragment tails, no adjacent cross-player splicing when spare slots exist,
correct empirical tail mass, weighted block bootstrap, and the target-only aggregate exemption.
Both calibrations evaluated 80 candidates, using only `DFL-MAT-J03WMX` and `Sample_Game_1` respectively.

Frozen five-seed reports cover those training matches plus held-out `DFL-MAT-J03WN1` and
`Sample_Game_2`, including the four organizer baseline variants as secondary diagnostics:

| Source/match | Passed seeds | Remaining failed gates |
| --- | ---: | --- |
| DFL-MAT-J03WMX | 5/5 | None |
| DFL-MAT-J03WN1 (held out) | 5/5 | None |
| Sample_Game_1 | 0/5 | Aggregate reduction |
| Sample_Game_2 (held out) | 0/5 | Aggregate reduction, median missing-ball duration |

DFL aggregate distance decreases by about 75% on both tested matches under the corrected v2 metrics.
This is evidence for the tested matches, not a guarantee for all DFL matches or downstream models.
Metrica Game 1's immutable speed-family contribution alone yields a minimum distance of 0.80959,
above its required ceiling of 0.80556, even if the other compared metrics could match perfectly.
Game 2 already has lower ball coverage than target; no extra dropout is applied, so its shorter
median gap remains a failed gate. No threshold was relaxed and Metrica reports still exit 2.

The target half-to-half audit still fails missing-ball p90 in both directions and speed missingness
in one direction. Existing v1 artifacts are retained. Current local artifacts are
`cache/degradation/v2/{dfl,metrica}-report-final.json`, `verification.json`, and `dfl-sample/`.
The DFL sample export and epoch-seed example were executed and read back successfully.

## Restart-aware ball dropout (release v3, `cache/degradation/restart-v3/`)

**Why.** In the competition match the ball is visible only 0.307 of the time (95% CI 0.185–0.389) in the 8 s
before a restart (Smart Tagging `SET PIECES` start), against 0.593 overall. Degraded DFL had no such coupling,
so a model trained on it could not learn that a missing ball often means a stoppage. The cue is stable under
±2 s anchor shifts (0.30–0.33), which matters because the tags last about 35 s and their timing is noisy.

**What.** Profile metric `ball_visible_pre_restart` (`quality.restart_frames`, `quality.pre_restart_mask`;
ATD tags, DFL set-piece events, Metrica `SET PIECE`). `degrade._ball_dropout` hides, for an exact share of the
restarts, the whole pre-restart window with one gap whose length is drawn (stratified) from the target's
missing-run distribution and which ends up to 4 s after the anchor. Windows are protected from the generic
dropout and the preset frames are charged to the same total budget, so overall visibility is unchanged.
`restart_strength` in the params switches it on; old profiles (no metric) and `restart_strength` absent or 0
behave bit-for-bit as before. No labels are read; only provider event times.

**Pitfall found on the way.** A first version hid a short block inside each window. It reproduced the dip but
shortened the missing runs (median 5.6 s vs 10.7 s target) and failed every row. Whole-window gaps with
stratified lengths fixed it. `QUALITY VERSION` was deliberately not bumped (that would orphan v1 artifacts).

**Result.** Calibrated on J03WMX only (`dfl-strat.json`, `restart_strength` 0.917): 35/35 match/seed rows pass,
including the new gate (tolerance 0.10); mean pre-restart visibility is close to the 0.307 target. Degraded DFL
sits slightly above the target at every anchor shift (about +0.03 to +0.06), inside the tolerance.
The View config refits to identical velocity settings (`pipeline.refit_view_config`).

**Not done / limits.** `validate_preprocessing` has not been re-run for v3 (it is wired to the frozen `v2`
folder); the 35-row quality audit above was run with `degradation_cli report`. Metrica params were not
re-derived for the new profile. The tag-start anchor is a proxy for the kick, not a measurement of it.

Use it with `feed.load_release('cache/degradation/CURRENT-v3.json')` and `devset-v2`. `CURRENT.json` and
`devset-v1` still point at the v1 release, so existing numbers do not move silently.
