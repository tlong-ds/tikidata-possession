# src: team code

Stage-3 model: the stage-2 network with extra input columns and a "switch" head (when possession changes team),
decoded by the 4-state Viterbi decoder of stage 1. Every improvement is a plug-in **step**, every training run a
**TOML** recipe. Run from the repository root, no install: `python -m src.exp ...`, `python -m src.submit ...`.

```
exps/*.toml ──> src/exp.py run ──> core/train_switch ──> core/infer_switch ──> core/decode_switch ──> artifacts/exp/<name>/
                     ▲                 │ core/feats, net, freeze, switch_target
           src/steps/*.py ─────────────┘ (feature / loss / decode)
src/submit.py: models/final_metrica -> core/infer_switch -> core/decode_switch -> core/kickoff -> CSV
```

| Path | What |
|---|---|
| `submit.py` | `run`: submission file from `models/final_metrica` (CPU, deterministic, sha256 check); `pack`: artifacts -> `models/` |
| `exp.py` | `run <toml>`: train each seed, pool checkpoints, tune or copy decoder parameters, score dev/select |
| `core/feats.py` | 11 + 21 input columns, plus the columns of feature steps |
| `core/net.py` | `Nap3Net`; initialisation by column name, new columns start at 0 so the model starts equal to its base |
| `core/freeze.py` | freezing blocks / old input columns |
| `core/train_switch.py`, `switch_target.py` | training (fresh degradation every epoch), switch-head targets and loss |
| `core/infer_switch.py`, `decode_switch.py` | ensemble inference, switch-gated decoder |
| `core/kickoff.py` | graded frames before each half's kickoff -> D (Smart Tagging kickoff window + ball position) |
| `core/blend.py`, `evaltools.py`, `ball_smooth.py` | probability files, paired comparison per realisation, ball smoothing |
| `steps/` | `feature.nap1_dead`, `feature.nap1_pitch`, `feature.contest_geometry`, `feature.ball_touch`, `loss.near_change`, `decode.wide`; registry in `steps/__init__.py` |
| `nap1/` | stage 1: View, ball Kalman filter and touch detection, pitch control, dead ball, scoring, Viterbi decoder, fitted parameters in `configs/` |
| `nap2/` | stage 2: data files and configs (`shared`), features, dataset, network, losses, calibration, inference |

Outside `src/`: `pipeline/` (loading, splitting, label-free degradation), `evaluation/` (organisers' scorer),
`starter/` (organisers' baseline, reused by the pipeline).
