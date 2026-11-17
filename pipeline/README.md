# pipeline/ — đưa mọi nguồn về định dạng ATD

    python -m pipeline.build_cache all      # ~1 phút; ra cache/ (đã gitignore) + cache/MANIFEST.json
    python -m pipeline.checks               # đối chiếu với số đã công bố (deck, eda/dfl_stats.json, baseline)

    from pipeline.load import load_match
    m = load_match("BAR-ATH_tactical")      # hoặc "DFL-MAT-J03WMX", "Sample_Game_3", ...
    m.P  # [T,46,3] x,y (0-1), speed ; m.B [T,3] ; m.half ; m.graded (chỉ ATD) ; m.extras ; m.meta

Nguyên tắc: ATD trận đề đọc **nguyên văn** (float64, không đổi mét, không lật hướng, không lọc) để
tái tạo đúng 4 baseline trên 5 phút công khai. Chỉ DFL / Metrica được chuyển về định dạng của ATD.
Định dạng: `schema.py`. Dòng ATD bị cắt được ghi vào `meta["bad_frames"]`, không sửa file gốc.
Không đọc `evaluation/gt/*` ở bất kỳ bước nào.

Source-aware observation degradation, target profiling, calibration, and reports:
[DEGRADATION.md](DEGRADATION.md). Entry point: `python -m pipeline.degradation_cli --help`.
Run it after loading complete matches and before View/window extraction; supervision stays separate.

Shared centered metric View, velocity fitting, frozen validation and executable window handoff:
[VIEW.md](VIEW.md). `python -m pipeline.validate_preprocessing --help` reproduces the audit;
`python -m pipeline.degradation_example --help` exports one deterministic epoch.

`python -m pipeline.release_views --help` builds a consistent release with corrected DFL speed
reference calibration and provider-backed ATD orientation. Current local release:
`cache/degradation/view-release-20261106/` (private, ignored). See [VIEW.md](VIEW.md).

External DFL clean/degraded baseline benchmarking and organizer possession metrics:
[BASELINE.md](BASELINE.md). Entry point: `python -m pipeline.benchmark_baseline --help`.

## Quick start

```sh
.venv/bin/python -m pipeline.prepare --check               # fresh clone: what is missing, and the next step
.venv/bin/python -m pipeline.prepare                       # builds cache, writes train_data/{train,dev,test}, verifies
.venv/bin/python -m pipeline.score_export my_preds/ select # score predictions against train_data/dev/select
```

What each folder is for, the file format and the rules: [train_data/README.md](../train_data/README.md).
Raw datasets now live in `train_data/raw/`. Advanced use (leave-one-match-out scoring, on-the-fly
realizations, rebuilding releases): [FEED.md](FEED.md).

`cache/` is not in git. The release `python -m pipeline.export` needs (`CURRENT-v3.json`, the 4 files it pins and
`devset-v2`, ~10 MB) travels as `release-v3.zip`: `pipeline/release_bundle.py` packs and verifies it, `pipeline/prepare.py`
unpacks it. Details: [train_data/README.md](../train_data/README.md).

Superseded artifacts under `cache/degradation/` (kept for reproducibility, do not use for new work):
`target-v1*`, `target-verified*`, `dfl-final`/`dfl-verified`, `metrica-*final/verified`,
`validation-20261106/*-handoff` (window-5 View config), `handoff-woh-epoch1-20261106`.
