"""Stage-2 tests (plan section 8). Run from the repository root:
    python -m pytest tests/test_nap2.py -q

Mostly synthetic data; final tests need shared train_data/dev and cache/nap2/train
(see src/nap2/shared.py) and skip automatically when absent.
"""

import glob
import os
import types

import numpy as np
import pytest
import torch

from src.nap1.decode import decode
from src.nap2.calibrate import fit_temperature
from src.nap2.dataset import augment
from src.nap2.features import N_FRAME_FEAT, N_PLAYER_FEAT
from src.nap2.infer import forward_full
from src.nap2.model import build_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _rand_inputs(B=2, W=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    xp = torch.randn(B, W, 46, N_PLAYER_FEAT, generator=g)
    pres = (torch.rand(B, W, 46, generator=g) < 0.5).float()
    grp = torch.zeros(46)
    grp[24:] = 1
    xp = xp * pres.unsqueeze(-1)                  # Absent slot: all features are zero.
    xp[..., 4] = grp * pres
    xp[..., 5] = pres
    xf = torch.randn(B, W, N_FRAME_FEAT, generator=g)
    return xp, xf


@pytest.mark.parametrize("temporal", ["bigru", "tcn"])
def test_shapes(temporal):
    torch.manual_seed(0)
    m = build_model({"temporal": temporal}).eval()
    xp, xf = _rand_inputs()
    o = m(xp, xf)
    assert o["had"].shape == (2, 16, 3) and o["touch"].shape == (2, 16)
    assert o["team"].shape == (2, 16, 2) and o["ball"].shape == (2, 16, 2)


def test_slot_permutation_invariance():
    """Permuting slots while retaining their team columns must preserve outputs."""
    torch.manual_seed(0)
    m = build_model().eval()
    xp, xf = _rand_inputs()
    perm = torch.randperm(46)
    with torch.no_grad():
        a, b = m(xp, xf), m(xp[:, :, perm], xf)
    for k in a:
        assert torch.allclose(a[k], b[k], atol=1e-5), k


def test_absent_slots_ignored():
    """Arbitrary values in presence=0 slots must not affect outputs (mask correctness)."""
    torch.manual_seed(0)
    m = build_model().eval()
    xp, xf = _rand_inputs()
    junk = xp.clone()
    absent = junk[..., 5] == 0
    junk[absent] = torch.randn(int(absent.sum()), N_PLAYER_FEAT)
    junk[..., 5][absent] = 0
    with torch.no_grad():
        a, b = m(xp, xf), m(junk, xf)
    assert torch.allclose(a["had"], b["had"], atol=1e-5)


def test_seed_reproducible():
    torch.manual_seed(123)
    m1 = build_model()
    torch.manual_seed(123)
    m2 = build_model()
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(p1, p2)


def _batch():
    xp, xf = _rand_inputs(B=8, W=10)
    y = torch.randint(0, 3, (8, 10)).to(torch.int8)
    return {"xp": xp, "xf": xf, "y": y, "team": torch.randint(-1, 2, (8, 10)).to(torch.int8),
            "ball": torch.randn(8, 10, 2), "touch": torch.rand(8, 10), "ball_ok": torch.ones(8, 10, dtype=torch.bool),
            "near": torch.zeros(8, 10, dtype=torch.bool), "w": torch.ones(8, 10)}


def test_augment_swap_labels_and_involution():
    b = _batch()
    g = torch.Generator().manual_seed(0)
    a = augment(b, g, p_swap=1.0, p_flip=1.0)
    # Team swap: H<->A, D unchanged; swap touching team, negate present-player x and invert its group column.
    assert torch.equal(a["y"][b["y"] == 2], b["y"][b["y"] == 2])
    assert torch.equal(a["y"][b["y"] == 0], torch.ones_like(b["y"][b["y"] == 0]))
    pres = b["xp"][..., 5] > 0
    assert torch.allclose(a["xp"][..., 4][pres], 1 - b["xp"][..., 4][pres])
    assert torch.allclose(a["xp"][..., 0], -b["xp"][..., 0])
    # Applying twice restores the original.
    a2 = augment(a, torch.Generator().manual_seed(0), p_swap=1.0, p_flip=1.0)
    for k in ("xp", "xf", "y", "team", "ball"):
        assert torch.allclose(a2[k].float(), b[k].float()), k


def test_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0)
    n = 30000
    logits = rng.normal(0, 1.5, (n, 3))
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y = (rng.random(n)[:, None] > p.cumsum(1)).sum(1)   # Labels sampled from the probabilities themselves are calibrated.
    T = fit_temperature(logits * 3.0, y)                # Make logits three times overconfident.
    assert 2.6 < T < 3.4


def test_decode_needs_touch_to_switch():
    """Frame evidence favors A for 30 frames: A then H touches enable switching to A and back;
    without touches, the IOHMM gate retains H.
    """
    T = 400
    p = np.tile([0.80, 0.15, 0.05], (T, 1))
    p[200:230] = [0.30, 0.65, 0.05]
    E = np.log(p)
    half = np.ones(T, np.int8)
    C = np.zeros((T, 2))
    K = np.zeros(T)
    C[200], K[200] = [0.0, 1.0], 1.0
    C[230], K[230] = [1.0, 0.0], 1.0
    prm = {"eps_gate": 1e-3, "lambda_emit": 1.0, "L_alive": 7.0, "L_dead": 25.0, "touch_window": 5,
           "deadball_switch_gate": True}                     # Fixed values, independent of fitted configuration.
    y_t = decode(E, C, K, {"half": half}, prm)
    assert (y_t[:195] == 0).all() and (y_t[205:225] == 1).all() and (y_t[240:] == 0).all()
    y_n = decode(E, np.zeros((T, 2)), np.zeros(T), {"half": half}, prm)
    assert (y_n == 0).all()


def test_forward_full_covers_every_frame():
    torch.manual_seed(0)
    m = build_model().eval()
    T = 700
    half = np.r_[np.zeros(30), np.ones(300), np.zeros(20), 2 * np.ones(350)].astype(np.int8)
    xp = np.zeros((T, 46, N_PLAYER_FEAT), np.float16)
    xf = np.zeros((T, N_FRAME_FEAT), np.float32)
    o = forward_full(m, xp, xf, half, chunk=32, stride=2)
    inh = half > 0
    assert (o["touch"][inh] != -10.0).all() and (o["touch"][~inh] == -10.0).all()


def test_no_ground_truth_access():
    """No stage-2 file may read competition labels."""
    for f in glob.glob(os.path.join(ROOT, "src", "nap2", "*.py")):
        if os.path.basename(f) == "make_kaggle_bundle.py":   # Mentioned only to BLOCK packaging.
            continue
        txt = open(f, encoding="utf-8").read()
        assert "possession_gt" not in txt and "evaluation/gt" not in txt.replace("\\", "/").replace(
            "Không đọc evaluation/gt", ""), f


def _shared_files():
    from src.nap2 import shared
    dev = sorted(glob.glob(str(shared.DEV_DIR / "select" / "*.npz")))
    train = sorted(glob.glob(str(shared.TRAIN_DIR / "*.npz")))
    return dev, train


@pytest.mark.skipif(not all(_shared_files()), reason="chưa có dữ liệu chung (train_data/dev, cache/nap2/train)")
def test_shared_files_prepare():
    from src.nap2.dataset import prepare_file
    dev, train = _shared_files()
    d = prepare_file(dev[0], max_frames=6000)
    assert d["xp"].shape == (6000, 46, N_PLAYER_FEAT) and d["xf"].shape == (6000, N_FRAME_FEAT)
    assert set(np.unique(d["y"])) <= {-1, 0, 1, 2} and "baseline" in d and "y_raw" in d
    t = prepare_file(train[0], max_frames=6000)
    assert "touch_true" in t and t["touch"].max() > 0          # Training files include touch labels.
    assert t["xp"][..., 5].mean() * 46 < 22.5                  # Degraded inputs average fewer than 22 players.


def _has_release():
    from src.nap2 import shared
    from pipeline import split
    from pipeline.schema import CACHE_DIR
    n = split.roles()["train"][0]
    return shared.RELEASE.is_file() and os.path.isfile(os.path.join(CACHE_DIR, "dfl", n + ".npz"))


@pytest.mark.skipif(not _has_release(), reason="chưa có file cấu hình làm bẩn v3 hoặc cache DFL")
def test_fresh_degradation_prepare():
    """Fresh pipeline.feed degradation is reproducible by (run,epoch) and differs across epochs."""
    from src.nap2.dataset import prepare_fresh
    from pipeline import split
    n = split.roles()["train"][0]
    a, b, c = (prepare_fresh(n, e, 0, max_frames=4000) for e in (0, 0, 1))
    assert a["xp"].shape == (4000, 46, N_PLAYER_FEAT) and "touch_true" in a
    assert np.array_equal(a["xp"], b["xp"]) and not np.array_equal(a["xp"], c["xp"])
