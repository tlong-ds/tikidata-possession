import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src import exp, steps
from src.core import feats, freeze
from src.core.net import build_model, init_by_names
from src.core.train_switch import augment3, plan_stages

EXPS = sorted((Path(exp.ROOT) / "exps").glob("*.toml"))


def test_registry_finds_every_step_file():
    reg = steps.discover()
    files = [p.stem for p in (Path(steps.__file__).parent).glob("*.py") if not p.name.startswith("_")]
    assert len(files) == len({c.__module__ for c in reg.values()})
    assert {"loss.near_change", "feature.nap1_dead", "feature.nap1_pitch", "feature.contest_geometry", "feature.ball_touch", "decode.wide"} <= set(reg)


def test_make_rejects_unknown_step_and_param():
    with pytest.raises(KeyError):
        steps.make({"use": "feature.khong_co"})
    with pytest.raises(ValueError):
        steps.make({"use": "loss.near_change", "wieght": 1})
    st = steps.make({"use": "loss.near_change", "weight": 2})
    assert st.weight == 2 and st.steps == 12 and st.spec() == {"use": "loss.near_change", "weight": 2, "steps": 12}


@pytest.mark.parametrize("path", EXPS, ids=[p.stem for p in EXPS])
def test_every_experiment_file_loads(path):
    d = exp.load_spec(path)
    assert exp.NAME_RE.match(d["name"]) and d["name"] == path.stem


def test_load_spec_rejects_unknown_keys(tmp_path):
    p = tmp_path / "x.toml"
    p.write_text('name = "x"\n[train]\nepoch = 3\n')
    with pytest.raises(ValueError, match="khoá lạ"):
        exp.load_spec(p)


def test_layout_appends_named_columns_and_maps_pairs():
    lay = steps.Layout([{"use": "feature.nap1_dead"}, {"use": "feature.nap1_pitch"}])
    assert lay.n_xp == len(feats.XP_NAMES) == feats.N_PLAYER_FEAT3
    assert lay.n_xf == len(feats.XF_NAMES) + 2 == feats.N_FRAME_FEAT3 + 2
    assert lay.xf_names[-2:] == ["nap1_dead.p_dead", "nap1_pitch.p_home_minus_half"]
    assert lay.team_negate_xf == [lay.n_xf - 1]
    assert lay.team_pairs_xf[:4] == [tuple(p) for p in feats.TEAM_PAIRS_XF]
    with pytest.raises(ValueError):
        steps.Layout.from_dict({"features": [], "xf_names": ["khác"]})
    with pytest.raises(ValueError):
        steps.Layout([{"use": "loss.near_change"}])


def _model(layout):
    torch.manual_seed(0)
    return build_model({"n_player_feat": layout.n_xp, "n_frame_feat": layout.n_xf}).eval()


def test_init_by_names_keeps_outputs_when_columns_are_added():
    base = steps.Layout()
    new = steps.Layout([{"use": "feature.nap1_dead"}])
    m0, m1 = _model(base), _model(new)
    info = init_by_names(m1, m0.state_dict(), base.xp_names, base.xf_names, new.xp_names, new.xf_names)
    assert info["new_xf"] == ["nap1_dead.p_dead"] and not info["missing"]
    xp = torch.randn(2, 16, 46, base.n_xp)
    xf = torch.randn(2, 16, base.n_xf)
    extra = torch.randn(2, 16, 1) * 5
    with torch.no_grad():
        a, b = m0(xp, xf), m1(xp, torch.cat([xf, extra], -1))
    for k in a:
        assert torch.allclose(a[k], b[k], atol=1e-6), k
    back = _model(base)                                          # Dropped columns still load with unchanged outputs.
    info = init_by_names(back, m1.state_dict(), new.xp_names, new.xf_names, base.xp_names, base.xf_names)
    assert info["dropped_xf"] == ["nap1_dead.p_dead"]


def _step(model, groups):
    opt = torch.optim.AdamW(groups, lr=1e-2, weight_decay=0.1)
    out = model(torch.randn(2, 8, 46, model.cfg["n_player_feat"]), torch.randn(2, 8, model.cfg["n_frame_feat"]))
    sum(v.float().pow(2).mean() for v in out.values()).backward()
    opt.step()


def test_freeze_blocks_and_old_input_columns():
    lay = steps.Layout([{"use": "feature.nap1_dead"}])
    m = _model(lay)
    m.train()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    new_xf = [lay.n_xf - 1]
    groups, info = freeze.apply(m, ["encoder.phi.2", "temporal", "had"], True, [], new_xf, 1e-2, 0.1)
    assert 0 < info["trainable"] < info["total"]
    _step(m, groups)
    after = m.state_dict()
    d_enc = m.encoder.d_out
    for k in ("encoder.phi.2.weight", "had.weight", "proj.0.bias", "encoder.phi.0.weight"):
        assert torch.equal(before[k], after[k]), k
    w0, w1 = before["proj.0.weight"], after["proj.0.weight"]
    assert torch.equal(w0[:, :d_enc + lay.n_xf - 1], w1[:, :d_enc + lay.n_xf - 1])      # Old columns remain frozen.
    assert not torch.equal(w0[:, -1], w1[:, -1])                                          # New columns learn.
    assert not torch.equal(before["switch.weight"], after["switch.weight"])
    groups, info = freeze.apply(m, [], False, [], new_xf)                                 # Later stage: unfreeze everything.
    assert info["trainable"] == info["total"]
    before = {k: v.clone() for k, v in m.state_dict().items()}
    _step(m, groups)
    assert not torch.equal(before["had.weight"], m.state_dict()["had.weight"])
    with pytest.raises(ValueError, match="không có khối"):
        freeze.apply(m, ["khong_co"])


def test_plan_stages():
    a = SimpleNamespace(epochs=10, lr=3e-4, quick=False)
    st = plan_stages(a, {"freeze": ["encoder"], "stage": [{"epochs": 3, "lr": 1e-3}, {"epochs": 7, "freeze": []}]})
    assert [(x["epochs"], x["lr"], x["freeze"]) for x in st] == [(3, 1e-3, ["encoder"]), (7, 3e-4, [])]
    assert plan_stages(a, {})[0] == {"epochs": 10, "lr": 3e-4, "freeze": [], "freeze_old_inputs": False}
    a.quick = True
    assert [x["epochs"] for x in plan_stages(a, {"stage": [{"epochs": 3}, {"epochs": 7}]})] == [1, 1]


def test_augment_negates_team_columns_of_steps():
    lay = steps.Layout([{"use": "feature.nap1_pitch"}])
    B, W = 4, 6
    batch = {"xp": torch.zeros(B, W, 46, lay.n_xp), "xf": torch.zeros(B, W, lay.n_xf),
             "y": torch.zeros(B, W, dtype=torch.long), "team": torch.zeros(B, W, dtype=torch.long),
             "ball": torch.zeros(B, W, 2), "w": torch.ones(B, W)}
    batch["xf"][..., -1] = 0.3
    out = augment3(batch, torch.Generator().manual_seed(0), p_swap=1.0, p_flip=0.0, layout=lay)
    assert torch.allclose(out["xf"][..., -1], torch.full((B, W), -0.3))
    out = augment3(batch, torch.Generator().manual_seed(0), p_swap=0.0, p_flip=1.0, layout=lay)
    assert torch.allclose(out["xf"][..., -1], torch.full((B, W), 0.3))


DEV = sorted((Path(exp.ROOT) / "train_data" / "dev" / "select").glob("*.npz"))


@pytest.mark.skipif(not DEV, reason="cần train_data/dev/select")
@pytest.mark.parametrize("use", ["feature.nap1_dead", "feature.nap1_pitch", "feature.contest_geometry"])
def test_feature_steps_on_real_dev_file(use):
    d = feats.prepare_file(DEV[0], max_frames=3000, layout=[{"use": use}])
    st = steps.make({"use": use})
    assert d["xf"].shape[1] == feats.N_FRAME_FEAT3 + len(st.xf_names)
    assert d["xp"].shape[2] == feats.N_PLAYER_FEAT3 + len(st.xp_names)
    assert torch.isfinite(torch.as_tensor(d["xp"])).all()
    assert torch.isfinite(torch.as_tensor(d["xf"])).all()


def test_copy_data_and_load_previous(tmp_path):
    bundle, repo, inp = tmp_path / "b" / "code", tmp_path / "repo", tmp_path / "input"
    for p in ("cache/dfl/m.npz", "train_data/README.md", "nap3/old.py"):
        (bundle / p).parent.mkdir(parents=True, exist_ok=True)
        (bundle / p).write_text("bundle")
    (repo / "train_data").mkdir(parents=True)
    (repo / "train_data" / "README.md").write_text("git")
    assert exp.copy_data(str(bundle), str(repo)) == 1
    assert (repo / "train_data/README.md").read_text() == "git" and not (repo / "nap3/old.py").exists()
    (inp / "arts" / "nap3_outputs" / "artifacts" / "exp" / "baseline").mkdir(parents=True)
    (inp / "arts" / "nap3_outputs" / "artifacts" / "exp" / "baseline" / "eval.json").write_text("{}")
    (inp / "data" / "code" / "artifacts" / "nap3" / "x_quick").mkdir(parents=True)
    (inp / "data" / "code" / "artifacts" / "nap3" / "x_quick" / "a.pt").write_text("old")
    with zipfile.ZipFile(inp / "arts" / "baseline.zip", "w") as z:
        z.writestr("artifacts/exp/baseline/run.json", "{}")
        z.writestr("../evil.txt", "x")
    used = exp.load_previous(str(inp), str(repo))
    assert (repo / "artifacts/exp/baseline/eval.json").is_file() and (repo / "artifacts/exp/baseline/run.json").is_file()
    assert not (repo / "artifacts/nap3/x_quick").exists() and not (tmp_path / "evil.txt").exists()
    assert len(used) == 2


def test_find_bundle_extracts_a_kept_zip(tmp_path):
    inp = tmp_path / "input" / "mtah-nap3"
    inp.mkdir(parents=True)
    with zipfile.ZipFile(inp / "mtah-nap3.zip", "w") as z:
        z.writestr("code/cache/dfl/m.npz", "x")
    out = exp.find_bundle(str(tmp_path / "input"), work=str(tmp_path / "work"))
    assert out == str(tmp_path / "work" / "code") and (tmp_path / "work/code/cache/dfl/m.npz").is_file()


def test_experiment_inherits_base_steps(tmp_path, monkeypatch):
    monkeypatch.setattr(exp, "EXP_DIR", tmp_path / "exp")
    base = tmp_path / "exp" / "baseline"
    base.mkdir(parents=True)
    (base / "run.json").write_text(json.dumps({"steps": [{"use": "feature.nap1_dead"},
                                                         {"use": "loss.near_change", "weight": 0.5, "steps": 12}]}))
    p = tmp_path / "x.toml"
    p.write_text('name = "x"\n[[step]]\nuse = "feature.nap1_pitch"\n[[step]]\nuse = "loss.near_change"\nweight = 1.0\n')
    d = exp.load_spec(p)
    assert [s["use"] for s in d["step"]] == ["feature.nap1_dead", "feature.nap1_pitch", "loss.near_change"]
    assert d["step"][-1]["weight"] == 1.0 and [s["use"] for s in d["inherited"]] == ["feature.nap1_dead"]
    p.write_text('name = "x"\ninherit = false\n[[step]]\nuse = "feature.nap1_pitch"\n')
    assert [s["use"] for s in exp.load_spec(p)["step"]] == ["feature.nap1_pitch"]
    p.write_text('name = "x"\nbase = "khong_co"\n')
    assert exp.load_spec(p)["step"] == []
