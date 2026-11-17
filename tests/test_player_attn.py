"""player_attn (src.core.net.AttnPlayerEncoder) and the TOML [model] table."""

import pytest
import torch

from src import exp
from src.core import feats, freeze
from src.core.net import Nap3Net, init_by_names


def _inputs(B=2, W=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    xp = torch.randn(B, W, 46, feats.N_PLAYER_FEAT3, generator=g)
    xp[..., 4] = (torch.arange(46) >= 24).float()
    xp[..., 5] = 1.0
    xp[:, :, 40:, 5] = 0.0
    xf = torch.randn(B, W, feats.N_FRAME_FEAT3, generator=g)
    return xp, xf


def _attn_from(base):
    m = Nap3Net({"player_attn": True})
    missing, unexpected = m.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and missing and all(k.startswith("encoder.attn.") for k in missing)
    return m


def test_starts_identical_to_baseline():
    torch.manual_seed(0)
    base = Nap3Net().eval()
    m = _attn_from(base).eval()
    xp, xf = _inputs()
    with torch.no_grad():
        a, b = base(xp, xf), m(xp, xf)
    for k in a:
        torch.testing.assert_close(a[k], b[k])


def test_init_by_names_reports_new_block_as_missing():
    base = Nap3Net()
    m = Nap3Net({"player_attn": True})
    names = list(feats.XP_NAMES), list(feats.XF_NAMES)
    info = init_by_names(m, base.state_dict(), *names, *names)
    assert info["missing"] and all(k.startswith("encoder.attn.") for k in info["missing"])


def _trained(seed=1):
    torch.manual_seed(seed)
    m = Nap3Net({"player_attn": True}).eval()
    with torch.no_grad():
        for p in m.encoder.attn.parameters():
            p.normal_(0, 0.3)                                   # Nonzero attention is required to test the mask.
    return m


def test_absent_players_do_not_change_output():
    m = _trained()
    xp, xf = _inputs()
    xp2 = xp.clone()
    xp2[:, :, 40:, :4] = 9.0                                    # Arbitrarily modify absent-slot data.
    xp2[:, :, 40:, 6:] = -3.0
    with torch.no_grad():
        out1, out2 = m(xp, xf), m(xp2, xf)
    torch.testing.assert_close(out1["had"], out2["had"])
    xp3 = xp.clone()
    xp3[:, :, :5, 0] += 1.0                                     # A present player must affect the output.
    with torch.no_grad():
        assert not torch.allclose(out1["had"], m(xp3, xf)["had"])


def test_frame_without_players_is_finite():
    m = _trained()
    xp, xf = _inputs()
    xp[:, :3, :, 5] = 0.0
    with torch.no_grad():
        out = m(xp, xf)
    assert all(torch.isfinite(v).all() for v in out.values())


def test_stage_one_freeze_trains_attention_and_switch_only():
    m = Nap3Net({"player_attn": True})
    groups, info = freeze.apply(m, ["encoder.phi.2", "temporal", "had", "touch", "team", "ball"], True, [], [])
    on = {n for n, p in m.named_parameters() if p.requires_grad}
    assert on and all(n.startswith(("encoder.attn.", "switch.")) for n in on)
    assert any(n.startswith("encoder.attn.") for n in on) and any(n.startswith("switch.") for n in on)


def test_model_table_in_toml(tmp_path):
    p = tmp_path / "x.toml"
    p.write_text('name = "x"\nbase = "scratch"\n[model]\nchunk = 256\nplayer_attn = true\n')
    spec = exp.load_spec(p)
    assert exp.model_flags(spec["model"]) == ["--chunk", "256", "--player-attn"]
    assert exp.model_flags({"player_attn": False}) == []
    p.write_text('name = "x"\nbase = "scratch"\n[model]\nlayers = 2\n')
    with pytest.raises(ValueError, match=r"\[model\]"):
        exp.load_spec(p)


def test_train_switch_flags():
    from src.core.train_switch import parse_args
    a = parse_args(["--chunk", "256", "--player-attn"])
    assert a.chunk == 256 and a.player_attn
    assert not parse_args([]).player_attn and parse_args([]).chunk == 128
