# Shared View and frozen preprocessing validation

This phase checks tracking quality and the model input interface. It does not establish downstream model benefit and does not build or train models.

## Reproduce the audit

Use the project virtual environment with `requirements-pipeline.txt` installed:

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pipeline.validate_preprocessing --out cache/degradation/validation-new-run
.venv/bin/python -m pipeline.validate_preprocessing --verify cache/degradation/validation-new-run/manifest.json
```

The output must be new and under ignored `cache/degradation/`. The runner freezes v2 inputs, all input-cache hashes, implementation/test hashes, installed dependency versions, seeds and roles before computation. It verifies original hashes at completion and records generated artifact hashes. Exit 2 means a quality gate or orientation/integration gate failed; a hash mismatch also exits nonzero. Read `REPORT.md`, `summary.json`, `integration.json`, and individual match quality reports. Calibration J03WMX is separate from the six non-calibration matches. All 35 match/seed combinations must pass for a broad DFL quality claim. Failed matches remain in the report, with unchanged parameters.

`quality_report(..., include_baselines=False)` and `degradation_cli report/export --skip-baselines` bypass baseline diagnostics entirely. Default behavior remains unchanged. This audit disables them on every match, including reserved J03WR9.

## Model input contract

`resolve_geometry(raw_match)` runs before degradation removes raw metadata. DFL uses `left_group`. Other sources use both team extrema over the opening 60 seconds of each half, with paired team differences and 95% median moving-block bootstrap intervals (5-second blocks, 1000 replicates, seed 2027). Each estimator requires 30 seconds of paired finite observations and six supported blocks. Both intervals must exclude zero and agree on direction. These are explicit implementation settings, not organizer requirements.

Unresolved geometry raises `UnresolvedOrientation`. A recorded override is accepted, for example `{"2": {"attack_sign": -1, "reason": "verified evidence reference"}}`; the sign is group 1's attack direction in original coordinates. Never invent evidence. The validation CLI accepts `--orientation-overrides` with a JSON mapping match names to these per-half objects.

`fit_view_config([load_match("DFL-MAT-J03WMX")], target_profile)` fits only on the calibration match. It uses the larger exact 99.9th percentile of adjacent implied player speeds from target tracking (graded frames) and calibration DFL tracking. The target cache observation hash must match the frozen profile. Savitzky–Golay settings minimize mean half-level median absolute speed error on identical eligible frames for all ten candidates. Ties prefer shorter windows, then lower order. Configuration includes candidate scores, source hashes and numerical-library versions.

`make_view(observations, geometry, config)` exposes:

- `P[T,46,6]`: centered metric x/y, vx/vy, team, presence. Teams are fixed: slots 0–23 = 0, slots 24–45 = 1.
- `B[T,3]`: centered metric x/y, presence.
- `cam[T,2]`, `camera_valid`, `half`, `frame_valid`, `graded`, and original zero-based cache `frame_indices`.
- Allowlisted metadata: fps, pitch, original per-half attack sign and team encoding.

Coordinates are `(xy - .5) * pitch`, with both axes negated when group 1 attacks toward −x. Frame 0/invalid-half coordinates and unavailable coordinate pairs are NaN. Missing positions have zero presence; team channels remain fixed. Velocity can be NaN while position is present. Segments split on gaps, half boundaries and adjacent speeds above the frozen threshold. No position is smoothed, clipped or deleted due to a jump. Half-window edges and undersized segments remain velocity-invalid. No identities, labels, events, clean target ball or provenance appear in View.

`transform_targets` separately transforms supervised ball coordinates using exactly the same geometry and preserves team/possession labels. Rotation never swaps labels. `iter_windows` slices a single complete-match realization within half boundaries and retains original frame indices. Partial trailing windows are omitted; length and stride default to 250 and 125 frames.

## Executable epoch handoff

```sh
.venv/bin/python -m pipeline.degradation_example \
  --profile cache/degradation/validation-20261106/target.json \
  --params cache/degradation/validation-20261106/dfl.json \
  --view-config cache/degradation/view-speed-units-v2-20261106/run/view-config.json \
  --match DFL-MAT-J03WOH --epoch 0 \
  --out cache/degradation/example-woh-epoch0
```

Use `view.npz` plus `view-meta.json` for model inputs. `windows.npy` stores start/end-exclusive slices into that same epoch; do not independently degrade windows. `targets.npz` is supervision only. `audit-sample/` and `geometry-audit.json` are offline audit material. Exports are read back and compared before success. Keep base seed 2027 and epoch 0 fixed for evaluation; change epoch for training realizations. Config hash is checked before use.

DFL is the candidate training source subject to reported gates. Metrica integration success does not erase its frozen quality failures or make it an approved training source. Original target half-to-half failures remain documented limitations.

## Later model experiment — not executed

Training candidates: J03WMX, J03WOH, J03WOY, J03WPY, J03WQQ. Development: J03WN1. Reserved final evaluation: J03WR9. Keep the reserved role even if its label-free preprocessing audit finds a blocker.

Compare clean and degraded training with identical architectures, training budgets and paired seeds. Select models on development only; freeze the complete protocol before final evaluation. Score external labels with organizer possession metrics, reporting paired-seed outcomes and uncertainty. Competition public labels must not enter model selection. No downstream benefit claim follows from preprocessing tests alone.

## Release limitations discovered on 2026-11-06

The frozen audit passed 35/35 DFL match/seed quality checks. This does not validate source speed units: the quality speed gates primarily measure availability. On calibration J03WMX, the median ratio of cached source speed to adjacent position-derived speed is approximately 3.58 (finite within-half steps with implied speed between 0.5 and 8 m/s). The selected derivative's mean half-median absolute error against the unchanged cached speed is approximately 3.678 in the cache's declared units. This suggests a possible km/h versus m/s mismatch but is not a verified unit conversion. The frozen selection and caches were retained. Verify source-unit provenance before relying on the speed-reference calibration; any correction belongs to a separately recorded calibration cycle.

ATD half 2 fails the required extrema orientation confidence check. Its leftmost-player median-difference interval includes zero while the rightmost-player estimate is negative. An evidence-backed per-half override is required before an ATD View can be released. No override was guessed.


## Confirmed speed-unit correction (View configuration version 2)

The dataset authors explicitly define XML `S` as km/h in **Data Records / Position data**: [Bassek et al. (2025), DOI 10.1038/s41597-025-04505-y](https://www.nature.com/articles/s41597-025-04505-y). The current loader copies `S` unchanged for players and ball; the generic raw-cache schema's m/s description therefore had a DFL exception.

`speed_reference_m_s` converts a separate player-speed reference by 1/3.6 during fitting. It retains NaNs, never edits caches, records unit provenance, and honors an explicit `speed_unit="m/s"` to prevent double conversion in future caches. Unsupported declarations fail explicitly. View derivatives are already in m/s and are not divided again.

Refit on J03WMX only using:

```sh
.venv/bin/python -m pipeline.recalibrate_view_speed \
  --prior cache/degradation/validation-20261106 \
  --out cache/degradation/new-speed-calibration
```

The current corrected configuration and evidence are in `cache/degradation/view-speed-units-v2-20261106/run/`. The earlier speed-unit caveat above is historical and is resolved by this conversion. Original configs/exports remain reproducible artifacts, but future exports should use the corrected configuration. The original 35/35 quality results are retained: the unchanged quality implementation tests speed missingness, not speed-reference accuracy. ATD orientation and downstream model benefit remain separate, unresolved items.

The original manifest's source hashes describe the previous implementation. Before this correction, those sources were verified and archived under `cache/degradation/view-speed-units-v2-20261106/previous-source/`. New source/configuration hashes are recorded in the new run; the old manifest and its frozen data/report/export artifacts are unchanged.

## Provider-backed ATD orientation release

The supplied tactical EPTS XML declares `attack_direction_first_half` for both teams. `pipeline.release_views.provider_overrides` verifies that these are opposing directions and that the XML team order agrees with the cache's 24/22 slot contract. It records first-half direction directly and second-half direction by the halftime reversal. The recorded override is derived from supplied metadata; it does not use possession labels or manually annotated frames. The original statistical orientation failure is retained in the new export's audit.

Rebuild all source handoffs with the corrected velocity configuration:

```sh
.venv/bin/python -m pipeline.release_views \
  --prior cache/degradation/validation-20261106 \
  --view-config cache/degradation/view-speed-units-v2-20261106/run/view-config.json \
  --atd-metadata data/tactical/BAR-ATH_tactical_atd.xml \
  --out cache/degradation/new-view-release
```

The current release is `cache/degradation/view-release-20261106/`. Its ATD `view.npz` and `windows.npy` are model observations and half-bounded window ranges; `view-meta.json` holds allowlisted geometry metadata. `geometry-audit.json` and `statistical-orientation-check.json` are audit-only. External-source handoffs keep their existing model/target/audit separation. Earlier quality results and limitations remain in the original audit. No model benefit has been tested.
