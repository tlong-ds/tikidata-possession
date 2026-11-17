# TikiData — ball possession for More than a Hack 2027 (Phase 1)

For every frame of the BAR–ATH tactical video we label who has the ball: `H`, `A`, `D` (dead ball) or `X`
(not graded). A model trained on other matches with provider labels (DFL/Sportec open data, Metrica sample
data), made robust to the target's automatic tracking by a label-free degradation of the clean source
tracking, then decoded by a switch-gated 4-state Viterbi decoder.

- Method, data, possession rules, limitations: **[docs/TECHNICAL.md](docs/TECHNICAL.md)**
  (PDF: [submission/technical.pdf](submission/technical.pdf)).
- Challenge spec from the organisers, unchanged: [CHALLENGE.md](CHALLENGE.md), [DATA.md](DATA.md),
  [EVALUATION.md](EVALUATION.md), [SUBMISSION.md](SUBMISSION.md), [RULES.md](RULES.md), [FAQ.md](FAQ.md),
  [OVERVIEW.md](OVERVIEW.md).

No match data is in this repository. Put the organisers' data bundle in `./data` (see
[data/README.md](data/README.md)) or point `MTAH_DATA_DIR` at it.

## Reproduce the submitted file

Linux, Python 3.12; CPU inference, no GPU or training data required.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-submit.txt
export MTAH_DATA_DIR=/path/to/provided-data      # or unpack the bundle into ./data
python -m src.submit run --model final_metrica --check-twice --out TikiData.csv
sha256sum TikiData.csv
```

Expected SHA-256: `aca26c78203fdd4ecf233a89754ec4dd2c3bb4db9373625279e85bcc582b59fd`.

`src.submit run` builds the target input from `data/` and the two aggregate configs in `configs/inference/`,
runs the 9 committed checkpoints of `models/final_metrica/` with deterministic algorithms, sets the graded
frames before each half's kickoff to D (`src/core/kickoff.py`), then writes and validates the CSV.
`--check-twice` reruns with another thread count and refuses a different hash. The final ensemble averages
epochs 8, 9 and 10 for each of seeds 1, 2 and 3. `submission/build.sh` copies the CSV and builds the PDF
(requires `latexmk` and a LaTeX installation):

```bash
bash submission/build.sh
```

## Files to submit

The organisers require three deliverables: the possession CSV, the public repository URL, and a short
technical document ([SUBMISSION.md](SUBMISSION.md)). The upload pair produced above is:

```text
submission/
  TikiData.csv
  TikiData.pdf
```

Upload these two files to the submission form and provide `https://github.com/tlong-ds/tikidata-possession`.
The CSV has exactly the header `frame,label` and 199,319 rows, with frames 0 through 199318 in order
and labels `H`, `A`, `D` or `X`. The build validates the exact CSV being packaged.

The organisers prescribe the deliverables and file names, while the source-code directory layout is
up to each team. `evaluation/results/` is the default input directory for their local scorer; its
`--results` option accepts a different directory. Keep the provided data bundle's original `tactical/`
and `broadcast/` names when unpacking it under `data/` or `MTAH_DATA_DIR` ([DATA.md](DATA.md)).

## Retrain the submitted model

`models/final_metrica` is `exps/final_metrica.toml`: the recipe chosen on the development match
(`exps/bt04_metrica.toml`), retrained from `models/baseline` with the development match added, decoder
parameters copied from `models/bt04_metrica`.

```bash
pip install -r requirements-train.txt
python -m pipeline.prepare --check              # what is missing in train_data/ and cache/, and the next step
python -m pipeline.prepare                      # DFL raw data -> train_data/{train,dev,test}; Metrica downloaded at a pinned commit
python -m src.exp run exps/final_metrica.toml --quick    # 1 seed, 1 epoch per stage: checks the pipeline (~minutes, CPU)
python -m src.exp run exps/final_metrica.toml --device cuda   # full run: 3 seeds x 10 epochs
python -m src.submit pack final_metrica         # artifacts/exp/final_metrica -> models/final_metrica
```

Training data: DFL/Sportec open data (Bassek et al., *Scientific Data* 2025, CC BY 4.0) in `train_data/raw/idsse/`,
and the Metrica Sports sample data. `pipeline.prepare` also needs the degradation release (the label-free noise
profile of the target tracking and the fitted DFL degradation parameters). It is derived from the competition
tracking, so it is not published here; it is rebuilt from `data/` with the tools in
[pipeline/DEGRADATION.md](pipeline/DEGRADATION.md) and [pipeline/VIEW.md](pipeline/VIEW.md). Training runs
will not match the committed weights bit for bit (GPU, fresh noise per epoch); the submission is reproduced
from the committed weights.

## Layout

| Path | What |
|---|---|
| `src/submit.py` | submission file from `models/final_metrica` |
| `src/exp.py`, `exps/*.toml` | training runner and the three recipes behind the submitted model |
| `src/core/` | features, network, training, inference, decoder (`decode_switch`), pre-kickoff rule (`kickoff`) |
| `src/steps/` | the six plug-in steps the model uses: `feature.nap1_dead`, `feature.nap1_pitch`, `feature.contest_geometry`, `feature.ball_touch`, `loss.near_change`, `decode.wide` |
| `src/nap1/`, `src/nap2/` | stage-1 signals (ball Kalman filter, pitch control, dead ball) and stage-2 network parts reused by stage 3 |
| `pipeline/` | loading DFL / Metrica / ATD, label-free target profiling and degradation, train/dev/test export |
| `models/` | `final_metrica` (submitted, 9 checkpoints), `baseline` (initialisation), `bt04_metrica` (decoder parameters) |
| `configs/inference/` | aggregate View configs for the target (no labels, no frame indices) |
| `games.py`, `evaluation/`, `starter/` | organisers' game constants, scorer, validator, public 5-minute GT and the starter baseline the pipeline reuses |
| `tests/` | `python -m pytest -q tests` (data-dependent tests need `cache/` and `train_data/`) |

## Data rules we followed

- No label, ground truth or human-produced data about BAR–ATH is used; the target contributes only label-free
  statistics (noise profile, ball-touch lag, ball smoothing) and the organisers' automatic Smart Tagging
  (restart anchors for the profile, the two kickoff tags).
- The public 5-minute ground truth was used only to check the pipeline end to end, never to train, tune or select.
- DFL match J03WR9 was held out and never opened.

## Team

Huyen Trang Ngo, Hoang Phuc Ngo, Thanh Long Ly, Khanh Duong Pham, Duc Anh Pham. Code under the
[LICENSE](LICENSE) of the organisers' starter kit; development history in
[tlong-ds/more-than-a-hack-2027](https://github.com/tlong-ds/more-than-a-hack-2027).
