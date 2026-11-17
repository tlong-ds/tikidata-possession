> Just want to hand data to teammates? You don't need this page: see [train_data/README.md](../train_data/README.md).

# Split → frozen dev set → feed → score

One path from cache to model, one split, one set of frozen development realizations, one scorer.
Model code (nap1, nap2, anything new) imports **only** `pipeline.feed` and `pipeline.split`. Do not
call `degrade_match`, `make_view`, `create_view` or `prepare()` of another branch directly.

```
build_cache ─► split ─► release pointer ─► devset ─► feed ─► model ─► evaluate_dev
 (cache/)     split-v1    CURRENT.json    devset-v1   train_view      predictions
```

## One-time setup (already run once; outputs are private, under ignored `cache/`)

```sh
.venv/bin/python -m pipeline.build_cache all
.venv/bin/python -m pipeline.split --out cache/splits/split-v1.json
.venv/bin/python -m pipeline.feed \
  --profile cache/degradation/<run>/target.json --dfl-params cache/degradation/<run>/dfl.json \
  --view-config cache/degradation/<run>/view-config.json \
  --orientation-overrides cache/degradation/<run>/orientation-overrides.json   # writes CURRENT.json
.venv/bin/python -m pipeline.devset --realizations 10 --out cache/degradation/devset-v1
```

`CURRENT.json` pins the four release files by sha256 and checks that profile, degradation params and View
config belong together. To change the release, write a new pointer (`--out`) and rebuild the dev set; never
edit files in place.

## Roles (`pipeline/split.py`)

| Role | Matches | May be used for |
|---|---|---|
| train | J03WMX, J03WOH, J03WOY, J03WPY, J03WQQ | training |
| dev | J03WN1 | selection (`select` realizations) and reporting (`report` realizations) |
| reserved | J03WR9 | nothing until the protocol is frozen |
| target | BAR-ATH_tactical | inference only |
| diagnostic | Sample_Game_1–3 | diagnostics only |

`split.roles(fold=k)` gives leave-one-match-out folds over the six non-reserved matches (dev rotates).
Fold 0 develops on J03WMX, which the View config and degradation parameters were fitted on (label-free
statistics only); that fold is flagged `calibration_overlap`.
`split.guard(name, purpose, fold=...)` raises `ValueError` for any other use.

## API (`pipeline/feed.py`)

```python
from pipeline import feed, split
rel = feed.load_release()                                         # hash-verified
view, targets = feed.train_view('DFL-MAT-J03WOH', epoch=e, run=seed)   # fresh degraded realization
for name, rid, view, targets in feed.dev_views('cache/degradation/devset-v1', subset='select'):
    ...                                                           # frozen, content-hash verified
view = feed.target_view()                                         # competition match
```

- `View`: centered metric `P[T,46,6]` (x, y, vx, vy, team, present), `B[T,3]`, `cam`, `half`, `graded`, `frame_indices`.
  Group 1 always attacks toward +x. `vx/vy` are m/s. `targets` hold supervision only (never feed them as inputs).
- Raw `Match.P[..., 2]` is m/s for ATD, km/h for DFL and NaN for Metrica: **do not read it**.
- `View.graded` is the published grading mask for ATD and `half > 0` for sources without one.
- Training seeds are `realization_seed('train', run, epoch, match)`; dev seeds use purpose `dev`. They cannot collide.
- Dev subsets: `select` (checkpoint, temperature, threshold choices), `report` (final numbers only), `clean`.
  The dev set stores seeds and View hashes, not arrays; `dev_views` regenerates deterministically and
  refuses to continue if a hash differs (changed code/release files, or different numpy/scipy).

## Scoring (`pipeline/evaluate_dev.py`)

Write one file per realization: `<predictions>/<match>/<id>.npz` with `codes` (int8, length = cache frames,
0=H 1=A 2=D) for every `(match, id)` that `dev_views` yields, then

```sh
.venv/bin/python -m pipeline.evaluate_dev --devset cache/degradation/devset-v1 \
  --predictions runs/my_model --out cache/degradation/dev-my_model       # --subset select for tuning
```

It scores the model and the fixed `baseline_smooth50` on the **same** realizations and reports the paired
difference (mean ± SD across realizations, wins) per development match and across matches. Judge a change
by its paired difference relative to the between-match SD, not by a single seed. The competition public
window stays a pipeline check only (`evaluation/run_evaluation.py`).

## Moving nap1 / nap2 onto this path

- **nap1**: replace `nap1.view.create_view(match)` by `feed.train_view/dev_views/target_view`, then wrap the
  result in nap1's `View` the way `nap1-experiment/cache/nap1-evaluation-20261107/evaluate.py:shared` already does.
  Stop using `nap1.view` flip heuristics and `Params.max_step_m/savgol_window` (the release fixes window 21, order 3).
- **nap2**: `nap2.train` accepts `--train/--val`; pass `split.roles()['train']` and `['dev']`, and use the
  shared preprocessing adapter (`nap2/shared_preprocessing.py` in the nap2 worktree, currently uncommitted).
  **Native `split_for_fold(fold=0)` trains on the reserved J03WR9.** Replace it with `split.roles(fold=k)`.
- Both: select with `subset='select'`, report with `subset='report'`, and never tune on the public window.

## Releases

- **v1** — `cache/degradation/CURRENT.json` + `devset-v1`: the validated 2026-11-06 release (no restart cue).
- **v3** — `cache/degradation/CURRENT-v3.json` + `devset-v2`: adds restart-aware ball dropout
  ([DEGRADATION.md](DEGRADATION.md#restart-aware-ball-dropout-release-v3-cachedegradationrestart-v3)).
  Pass `release=feed.load_release('cache/degradation/CURRENT-v3.json')` to `train_view` / `dev_views` and
  `--release` to `pipeline.devset` / `pipeline.evaluate_dev`. A dev set only works with the release it was built from.
Compare models only within one release; switch the whole team at once.
