"""[train.plateau]: on patience exhaustion, reload the best weights, lower lr and train further."""
import pytest

from src import exp
from src.core.train_switch import PLATEAU_DEFAULTS, plateau_cfg, restart_rounds


def test_plateau_off_by_default():
    assert plateau_cfg({}) is None and restart_rounds({"lr": 3e-4}, None) == []


def test_plateau_defaults_and_quick():
    assert plateau_cfg({"plateau": {"factor": 0.5}}) == dict(PLATEAU_DEFAULTS, factor=0.5)
    assert plateau_cfg({"plateau": {"rounds": 3}}, quick=True) == dict(PLATEAU_DEFAULTS, rounds=1, patience=1, epochs=1)


@pytest.mark.parametrize("bad", [{"factor": 1.0}, {"factor": 0}, {"rounds": 0}, {"lr": 1e-4}])
def test_plateau_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        plateau_cfg({"plateau": bad})


def test_restart_rounds_lower_lr_and_keep_freeze():
    last = {"epochs": 7, "lr": 3e-4, "freeze": ["temporal"], "freeze_old_inputs": False}
    r = restart_rounds(last, plateau_cfg({"plateau": {"factor": 0.3, "epochs": 2, "rounds": 2}}))
    assert [x["lr"] for x in r] == pytest.approx([9e-5, 2.7e-5])
    assert all(x["restore"] and x["epochs"] == 2 and x["freeze"] == ["temporal"] for x in r)


def test_load_spec_accepts_plateau_and_rejects_typos(tmp_path):
    p = tmp_path / "x.toml"
    p.write_text('name = "x"\nbase = "scratch"\n[train.plateau]\npatience = 2\n')
    assert exp.load_spec(p)["train"]["plateau"] == {"patience": 2}
    p.write_text('name = "x"\nbase = "scratch"\n[train.plateau]\npatiance = 2\n')
    with pytest.raises(ValueError):
        exp.load_spec(p)
