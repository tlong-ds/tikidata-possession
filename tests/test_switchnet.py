"""Switchnet tests: extra channels, switch targets, stage-2 initialization, augmentation, decode_switch and inference."""
import numpy as np
import pytest
import torch

from src.nap2.features import N_FRAME_FEAT, N_PLAYER_FEAT
from src.nap2.model import Nap2Net

from src.core import decode_switch as dsw
from src.core import feats
from src.core.infer_switch import combine, fit_switch_temperature, forward_full
from src.core.net import Nap3Net, init_from_nap2
from src.core.switch_target import asym_kernel, change_impulses, near_change, switch_mask, switch_target
from src.core.train_switch import augment3

H, A, D = 0, 1, 2


# ---------------------------------------------------------------- additional channels

def test_ball_timing_since_until_gap():
    seen = np.array([1, 0, 0, 0, 1, 1, 0], bool)
    since, until, gap = feats.ball_timing(seen, fps=1)
    assert since.tolist()[:6] == [0, 1, 2, 3, 0, 0]
    assert until.tolist()[:6] == [0, 3, 2, 1, 0, 0]
    assert gap.tolist()[:6] == [0, 3, 3, 3, 0, 0]
    assert since[6] == 1 and until[6] == feats.CAP_S       # Never reappears: cap at 60 s.


def test_run_age_counts_from_slot_start():
    pres = np.array([[0], [1], [1], [1], [0]], bool)
    age, left = feats.run_age(pres, fps=1)
    assert age[:, 0].tolist() == [0, 0, 1, 2, 0]
    assert left[:, 0].tolist() == [0, 2, 1, 0, 0]


def test_lost_units_pair_last_in_first_out():
    n = np.array([11, 10, 9, 10, 11, 11])
    u = feats.lost_units(n)
    assert sorted(map(tuple, u.tolist())) == [(1, 4), (2, 3)]
    cnt, mean_s, act = feats.missing_stats(n, fps=1, recent_s=10)
    assert act.tolist() == [0, 1, 2, 1, 0, 0]
    assert mean_s[2] == pytest.approx((1 + 0) / 2)


def test_feature_constants_are_consistent():
    assert feats.N_PLAYER_FEAT3 == N_PLAYER_FEAT + 2 and feats.N_FRAME_FEAT3 == N_FRAME_FEAT + 14
    cols = [c for p in feats.TEAM_PAIRS_XF for c in p]
    assert min(cols) >= N_FRAME_FEAT and max(cols) < feats.N_FRAME_FEAT3


# ---------------------------------------------------------------- switch targets

def test_change_impulses_include_restarts_and_skip_first_step():
    y = torch.tensor([[H, H, A, A, D, D, A, H, -1, H]])
    imp = change_impulses(y)
    assert imp[0, :, 0].nonzero().flatten().tolist() == [7]        # A->H counts; -1->H does not.
    assert imp[0, :, 1].nonzero().flatten().tolist() == [2, 6]     # H->A and D->A (restart) count.
    assert imp[0, 0].sum() == 0
    m = switch_mask(y, torch.ones(1, 10))
    assert m[0, 0] == 0 and m[0, 8] == 0 and m[0, 9] == 0 and m[0, 1] == 1


def test_switch_target_is_asymmetric():
    k = asym_kernel(2.0, 1.0, radius=4)
    assert k[4] == 1 and k[5] > k[3]                                # Wider before the change than after.
    y = torch.tensor([[H] * 10 + [A] * 10])
    t = switch_target(y, 2.0, 1.0)[0, :, 1]
    assert t[10] == pytest.approx(1.0)
    assert t[9] == pytest.approx(np.exp(-0.5 / 4))                  # One step early, sigma=2.
    assert t[11] == pytest.approx(np.exp(-0.5))                     # One step late, sigma=1.
    assert t[8] > t[12]


def test_switch_target_follows_team_swap():
    y = torch.tensor([[H, H, A, A, D, H, H, A]])
    swapped = torch.where((y >= 0) & (y <= 1), 1 - y, y)
    assert torch.equal(switch_target(swapped), switch_target(y).flip(-1))


def test_near_change_window():
    y = torch.tensor([[H, H, H, A, A, A, A]])
    assert near_change(y, 1)[0].tolist() == [False, False, True, True, True, False, False]


# ---------------------------------------------------------------- network

def test_nap3net_from_nap2_reproduces_nap2_heads():
    torch.manual_seed(0)
    n2 = Nap2Net().eval()
    n3 = Nap3Net().eval()
    info = init_from_nap2(n3, n2.state_dict())
    assert set(info["missing"]) == {"switch.weight", "switch.bias"}
    assert set(info["widened"]) == {"encoder.phi.0.weight", "proj.0.weight"}
    xp = torch.randn(2, 16, 46, feats.N_PLAYER_FEAT3)
    xp[..., 4] = (torch.arange(46) >= 24).float()
    xp[..., 5] = 1.0
    xf = torch.randn(2, 16, feats.N_FRAME_FEAT3)
    with torch.no_grad():
        o2 = n2(xp[..., :N_PLAYER_FEAT], xf[..., :N_FRAME_FEAT])
        o3 = n3(xp, xf)
    for k in ("had", "touch", "team", "ball"):
        torch.testing.assert_close(o3[k], o2[k])
    assert o3["switch"].shape == (2, 16, 2)


def test_absent_players_still_ignored():
    n3 = Nap3Net().eval()
    xp = torch.randn(1, 8, 46, feats.N_PLAYER_FEAT3)
    xp[..., 4] = (torch.arange(46) >= 24).float()
    xp[..., 5] = 1.0
    xp[:, :, 40:, 5] = 0.0
    xp[:, :, 40:, :5] = 0.0
    xf = torch.randn(1, 8, feats.N_FRAME_FEAT3)
    xp2 = xp.clone()
    xp2[:, :, 40:, 9:] = 7.0                                        # Additional channels of an absent slot.
    with torch.no_grad():
        torch.testing.assert_close(n3(xp, xf)["had"], n3(xp2, xf)["had"])


# ---------------------------------------------------------------- augmentation

def _batch(B=6, W=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    xp = torch.randn(B, W, 46, feats.N_PLAYER_FEAT3, generator=g)
    xp[..., 4] = (torch.arange(46) >= 24).float()
    xp[..., 5] = 1.0
    y = torch.randint(0, 3, (B, W), generator=g).to(torch.int8)
    return {"xp": xp, "xf": torch.randn(B, W, feats.N_FRAME_FEAT3, generator=g), "y": y,
            "touch": torch.rand(B, W, generator=g), "team": torch.randint(-1, 2, (B, W), generator=g).to(torch.int8),
            "ball": torch.randn(B, W, 2, generator=g), "ball_ok": torch.ones(B, W, dtype=torch.bool),
            "near": torch.zeros(B, W, dtype=torch.bool), "w": torch.ones(B, W)}


def test_augment3_swaps_team_columns_with_team_swap():
    b = _batch(B=16)
    gen = torch.Generator().manual_seed(3)
    g2 = torch.Generator()
    g2.set_state(gen.get_state())
    expect = torch.rand(16, generator=g2) < 0.5                     # Swap mask sampled as in nap2.
    out = augment3(b, gen, p_swap=0.5, p_flip=0.0)
    flipped_group = (out["xp"][..., 4] != b["xp"][..., 4]).any(-1).any(-1)
    assert torch.equal(flipped_group, expect) and expect.any() and (~expect).any()
    for i in range(len(b["y"])):
        for c1, c2 in feats.TEAM_PAIRS_XF:
            if flipped_group[i]:
                torch.testing.assert_close(out["xf"][i, :, c1], b["xf"][i, :, c2])
            else:
                torch.testing.assert_close(out["xf"][i, :, c1], b["xf"][i, :, c1])
    assert not out["rev"].any()


def test_augment3_time_reverse_flips_and_guards():
    b = _batch(B=4)
    gen = torch.Generator().manual_seed(1)
    out = augment3(b, gen, p_swap=0.0, p_flip=0.0, p_rev=1.0, rev_guard_steps=1)
    assert out["rev"].all()
    torch.testing.assert_close(out["xp"][..., 2], -b["xp"][..., 2].flip(1))
    torch.testing.assert_close(out["xp"][..., feats.XP_AGE], b["xp"][..., feats.XP_LEFT].flip(1))
    torch.testing.assert_close(out["xf"][..., feats.XF_SINCE], b["xf"][..., feats.XF_UNTIL].flip(1))
    assert torch.equal(out["y"], b["y"].flip(1))
    guard = near_change(out["y"], 1)
    assert (out["w"][guard] == 0).all() and (out["w"][~guard] == 1).all()


# ---------------------------------------------------------------- decode_switch

def _toy(T=400):
    y = np.r_[np.zeros(150), np.ones(30), np.zeros(220)].astype(np.int8)    # One short A possession.
    half = np.ones(T, np.int8)
    P = np.full((T, 3), 0.05)
    P[:, 0] = 0.6
    P[:, 1] = 0.35                                                            # E favors H outside the A possession.
    P[150:180, 0], P[150:180, 1] = 0.45, 0.50                                # During A possession, E only slightly favors A.
    P /= P.sum(1, keepdims=True)
    S = np.zeros((T, 2))
    S[148:152, 1] = 0.95                                                      # Switch to A.
    S[178:182, 0] = 0.95                                                      # Switch back to H.
    return y, half, np.log(P), S


def test_decode_switch_creates_switch_that_gate_cannot():
    y, half, E, S = _toy()
    meta = {"half": half, "fps": 25}
    p = dsw.SwitchParams(scale=0.5, lambda_emit=1.0)
    c = dsw.decode(E, S, meta, p)
    assert (c[155:175] == A).all() and (c[:140] == H).all() and (c[190:] == H).all()
    nap1_like = dsw.decode(E, np.zeros_like(S), meta, p)
    assert (nap1_like == H).all()
    gate = dsw.decode_gate(E, S, meta)
    assert (gate == H).mean() > 0.95                                          # The touch gate cannot create the switch.


def test_decode_switch_outside_halves_and_grid():
    y, half, E, S = _toy()
    half[:10] = 0
    graded = np.ones(len(half), bool)
    graded[:5] = False
    c = dsw.decode(E, S, {"half": half, "graded": graded}, dsw.SwitchParams())
    assert (c[:5] == 3).all() and (c[5:10] == D).all()
    ps = dsw.grid_params({"scale": [0.1, 0.2], "lambda_emit": [1.0]})
    assert [p.scale for p in ps] == [0.1, 0.2]
    assert dsw.ensure({"scale": 0.3, "junk": 1}).scale == 0.3


# ---------------------------------------------------------------- inference

def test_forward_full_keeps_switch_and_combine_formats():
    torch.manual_seed(0)
    model = Nap3Net().eval()
    T = 300
    half = np.r_[np.zeros(20), np.ones(140), np.full(140, 2)].astype(np.int8)
    xp = np.random.default_rng(0).normal(size=(T, 46, feats.N_PLAYER_FEAT3)).astype(np.float32)
    xp[..., 4] = (np.arange(46) >= 24)
    xp[..., 5] = 1
    xf = np.random.default_rng(1).normal(size=(T, feats.N_FRAME_FEAT3)).astype(np.float32)
    o = forward_full(model, xp, xf, half, chunk=32, stride=2)
    assert set(o) == {"had", "touch", "team", "ball", "switch"}
    assert o["switch"].shape == (T, 2) and (o["switch"][:20] == -10).all()
    E, C, K, S = combine([o], [1.2], [1.0], half)
    assert np.allclose(np.exp(E).sum(1), 1, atol=1e-5) and (S[:20] == 0).all()
    from src.core.blend import check_probs
    check_probs(E, np.clip(C, 0, 1), np.clip(K, 0, 1), np.clip(S, 0, 1), half)


def test_decode_params_file_uses_params_key(tmp_path):
    import json
    from src.core.infer_switch import _decode_params_for
    f = tmp_path / "decode_switch.json"
    f.write_text(json.dumps({"params": {"scale": 0.37, "window": 3}, "tune": {"best": {}}}))
    p = _decode_params_for(["unused.pt"], str(f))
    assert p.scale == 0.37 and p.window == 3
    assert _decode_params_for(["unused.pt"], '{"scale": 0.11}').scale == 0.11


def test_fit_switch_temperature_recovers_scale():
    rng = np.random.default_rng(0)
    true_logit = rng.normal(0, 2, 20000)
    t = (rng.uniform(size=20000) < 1 / (1 + np.exp(-true_logit))).astype(float)
    T = fit_switch_temperature(true_logit * 3, t, np.ones(20000, bool))
    assert T == pytest.approx(3.0, rel=0.15)


def test_near_change_loss_only_counts_steps_near_changes():
    from src.core.train_switch import near_change_loss
    y = torch.tensor([[H] * 10 + [A] * 10])
    logits = torch.zeros(1, 20, 3)
    logits[0, :, H] = 5.0                                   # Correct on H, wrong on A.
    far = near_change_loss(logits, y, torch.ones(1, 20), steps=0)
    near = near_change_loss(logits, y, torch.ones(1, 20), steps=2)
    assert far > 4 and near > 0                             # Step 10 (A) is wrong.
    none = near_change_loss(logits, torch.full((1, 20), H), torch.ones(1, 20), steps=2)
    assert float(none) == 0.0
