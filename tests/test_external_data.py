import json

import numpy as np
import pytest

from pipeline import split
from pipeline.external import expand, registry
from pipeline.external import pff
from src.core.train_switch import epoch_matches


def test_registry_and_split():
    assert len(registry.PFF) == 64 and len(set(registry.PFF)) == 64
    assert len(registry.SKILLCORNER) == len(registry.SKILLCORNER_IDS)
    assert set(registry.ALL) == set(split.EXTERNAL_TRAIN)
    for n in ("PFF-3812", "SKC-1874553", "METRICA-SG1"):
        split.guard(n, "train")
        with pytest.raises(ValueError):
            split.guard(n, "select")
    assert registry.source_of("PFF-10502") == "pff"
    with pytest.raises(ValueError):
        registry.source_of("DFL-MAT-J03WN1")


def test_expand_groups():
    out = expand(["METRICA", "PFF-3812", "PFF"])
    assert out[:3] == registry.METRICA and out[3] == "PFF-3812" and len(out) == 3 + 64


def test_split_hash_unchanged():
    # Adding external matches must not change split-v1.
    assert split.default_split()["split_hash"] == split.default_split()["split_hash"]
    assert "PFF-3812" not in json.dumps(split.default_split())


def test_epoch_matches_rotation():
    core = ["A", "B"]
    extra = ["X%d" % i for i in range(10)]
    assert epoch_matches(core, extra, 0, 1, 3) == core + extra
    assert epoch_matches(core, extra, 20, 1, 3) == core + extra
    seen = []
    for ep in range(5):
        ids = epoch_matches(core, extra, 4, 1, ep)
        assert ids[:2] == core and len(ids) == 6 and len(set(ids)) == 6
        seen += ids[2:]
    counts = {x: seen.count(x) for x in extra}
    assert set(counts.values()) == {2}                     # 5 epochs * 4 = 20, using each match exactly twice.
    assert epoch_matches(core, extra, 4, 1, 2) == epoch_matches(core, extra, 4, 1, 2)
    assert epoch_matches(core, extra, 4, 2, 0) != epoch_matches(core, extra, 4, 1, 0)


def test_pff_interp_gaps():
    ts = np.array([0.0, 0.1, 0.2, 1.0, 1.1])
    v = np.array([[0.0], [1.0], [2.0], [10.0], [11.0]])
    tg = np.array([0.0, 0.05, 0.2, 0.6, 1.05, 1.2])
    out = pff._interp(tg, ts, v)[:, 0]
    assert out[0] == 0.0 and abs(out[1] - 0.5) < 1e-9 and out[2] == 2.0
    assert np.isnan(out[3])                                   # Gap 0.8 s >0.25 s.
    assert abs(out[4] - 10.5) < 1e-9 and np.isnan(out[5])    # Outside the range.


def _ev(gt, t, home=True, sp="O", pe=None, period=1, **extra):
    return {"startTime": t, "endTime": t, "gameEvents": {"gameEventType": gt, "period": period, "homeTeam": home,
                                                           "setpieceType": sp, "outType": "T"},
            "possessionEvents": dict({"possessionEventType": pe}, **extra)}


def test_pff_events_to_labels(tmp_path):
    from pipeline.external import metrica
    evs = [_ev("FIRSTKICKOFF", 10.0, True, "K", "PA"), _ev("OTB", 12.0, False, "O", "PA"),
           _ev("OTB", 14.0, True, "O", "CH", challengeWinnerPlayerId=7),
           _ev("OUT", 16.0, None, None), _ev("OTB", 20.0, False, "T", "PA"), _ev("END", 30.0, None, None)]
    p = tmp_path / "ev.json"
    p.write_text(json.dumps(evs))
    grids = [(1, 10.0, 0, 25 * 25)]
    ev = pff.read_events(str(p), grids, team_of={7: 100}, home_id=100)
    assert list(ev["type"]) == ["SET PIECE", "PASS", "RECOVERY", "BALL OUT", "SET PIECE", "END"]
    period = np.ones(25 * 25, np.int8)
    y, half, info = metrica.event_labels(ev, period)
    f = lambda t: int(round((t - 10.0) * 25))
    assert y[f(11)] == 0 and y[f(13)] == 1 and y[f(15)] == 0          # Challenge won by home -> H.
    assert y[f(18)] == 2 and y[f(21)] == 1                             # Ball out -> D; away throw-in -> A.
