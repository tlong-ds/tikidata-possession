"""Synthetic src.core.blend tests: file format, blends, weight grid and tune/apply."""
import json
from pathlib import Path

import numpy as np
import pytest

from src.core import blend as bl

H, A, D = 0, 1, 2


def rand_E(T, rng):
    P = rng.dirichlet(np.ones(3), T)
    return np.log(P)


# ---------------------------------------------------------------- file format

def test_save_load_roundtrip_and_checks(tmp_path):
    rng = np.random.default_rng(0)
    T = 30
    E = rand_E(T, rng)
    C = rng.uniform(size=(T, 2))
    K = rng.uniform(size=T)
    S = rng.uniform(size=(T, 2))
    half = np.r_[np.zeros(5), np.ones(25)].astype(np.int8)
    p = bl.save_probs(tmp_path / 'x.npz', E, 'nap1', C=C, K=K, S=S, half=half)
    d = bl.load_probs(p)
    assert d['source'] == 'nap1'
    np.testing.assert_allclose(d['E'], E)
    np.testing.assert_array_equal(d['half'], half)
    with pytest.raises(ValueError, match='log-xác suất'):
        bl.check_probs(np.zeros((T, 3)))                      # Logits, not log probabilities.
    with pytest.raises(ValueError, match='đi cùng'):
        bl.check_probs(E, C=C)
    with pytest.raises(ValueError):
        bl.check_probs(E, S=S + 1)


# ---------------------------------------------------------------- blending operations

def test_blend_probs_single_source_is_identity_and_mix_is_mean():
    rng = np.random.default_rng(1)
    E1, E2 = rand_E(50, rng), rand_E(50, rng)
    half = np.ones(50, np.int8)
    half[:3] = 0
    out = bl.blend_probs([(E1, 1.0), (E2, 0.0)], half)
    np.testing.assert_allclose(out[3:], E1[3:])
    assert np.allclose(out[:3], np.log(1 / 3))
    mix = bl.blend_probs([(E1, 3.0), (E2, 1.0)])           # Normalize weights by their sum.
    np.testing.assert_allclose(np.exp(mix), .75 * np.exp(E1) + .25 * np.exp(E2), atol=1e-7)


def test_team_dead_roundtrip_and_separate_weights():
    rng = np.random.default_rng(2)
    E1, E2 = rand_E(40, rng), rand_E(40, rng)
    q1, d1 = bl.team_dead(E1)
    np.testing.assert_allclose(bl.from_team_dead(q1, d1), E1, atol=1e-6)
    out = bl.blend_team_dead([(E1, .8, .65), (E2, .2, .35)])
    q, pD = bl.team_dead(out)
    q2, d2 = bl.team_dead(E2)
    np.testing.assert_allclose(q, .8 * q1 + .2 * q2, atol=1e-6)
    np.testing.assert_allclose(pD, .65 * d1 + .35 * d2, atol=1e-6)
    with pytest.raises(ValueError):
        bl.blend_team_dead([(E1, -1, 1), (E2, 1, 1)])


def test_blend_touch_gate_evidence_and_outside_halves():
    T = 6
    half = np.array([0, 1, 1, 1, 1, 0], np.int8)
    K1, C1 = np.full(T, .5), np.tile([.8, .2], (T, 1))
    K2, C2 = np.ones(T), np.tile([.1, .9], (T, 1))
    K, C = bl.blend_touch([(K1, C1, .5), (K2, C2, .5)], half)
    assert K.tolist() == [0, 1, 1, 1, 1, 0]
    np.testing.assert_allclose(C[1], [.5 * .5 * .8 + .5 * .1, .5 * .5 * .2 + .5 * .9])
    assert C[0].tolist() == [.5, .5]
    # K_i=None: C_i already contains evidence, e.g. switch-head S.
    S = np.tile([.3, .6], (T, 1))
    K, C = bl.blend_touch([(None, S, 2.0)], half)
    np.testing.assert_allclose(C[2], [.6, 1.0])           # Clip at 1.


def test_blend_sources_requires_touch_evidence():
    rng = np.random.default_rng(3)
    T = 20
    srcs = {'a': {'E': rand_E(T, rng), 'C': None, 'K': None, 'S': None},
            'b': {'E': rand_E(T, rng), 'C': np.full((T, 2), .5), 'K': np.ones(T), 'S': None}}
    spec = {'mode': 'team_dead', 'team': {'a': .5, 'b': .5}, 'dead': {'b': 1.0}, 'touch': {'a': 1.0}}
    with pytest.raises(ValueError, match='touch'):
        bl.blend_sources(srcs, spec)
    spec['touch'] = {'b': 1.0}
    E, C, K = bl.blend_sources(srcs, spec)
    assert E.shape == (T, 3) and C.shape == (T, 2) and K.shape == (T,)
    with pytest.raises(KeyError):
        bl.blend_sources(srcs, dict(spec, team={'zzz': 1.0}))


# ---------------------------------------------------------------- grid and plateau

def test_expand_grid_gives_reference_the_rest():
    grid = {'mode': 'team_dead', 'ref': 'nap2', 'team': {'nap1': [0, .2, .6]}, 'dead': {'nap1': [0, .5]}}
    specs = bl.expand_grid(grid, ['nap2', 'nap1'])
    assert len(specs) == 6
    for spec, idx in specs:
        assert spec['team']['nap2'] + spec['team']['nap1'] == pytest.approx(1)
        assert spec['dead']['nap2'] + spec['dead']['nap1'] == pytest.approx(1)
    over = bl.expand_grid({'ref': 'r', 'team': {'a': [.7], 'b': [.6]}}, ['r', 'a', 'b'])
    assert over == []                                     # Sum >1: skip.
    with pytest.raises(ValueError):
        bl.expand_grid({'ref': 'nap2', 'team': {'nap2': [.1]}}, ['nap2'])


def test_parse_values():
    assert bl.parse_values('0:0.2:0.1') == [0, .1, .2]
    assert bl.parse_values('0,0.35') == [0, .35]


def test_pick_plateau_prefers_centre_of_flat_region():
    rows = [{'idx': [i], 'spell_f1': v} for i, v in enumerate([.50, .530, .531, .530, .529, .40])]
    assert bl.pick_plateau(rows, tol=.002)['idx'] == [2]
    rows = [{'idx': [i], 'spell_f1': v} for i, v in enumerate([.52, .530, .5295, .5290, .5285, .40])]
    # Plateau indices 1..4 are within 0.002 of 0.530; centroid 2.5 -> 2 or 3, breaking ties by score.
    assert bl.pick_plateau(rows, tol=.002)['idx'] == [2]
    assert bl.pick_plateau([{'idx': None, 'spell_f1': .9}, {'idx': [0], 'spell_f1': .5}])['idx'] is None


def test_mix_hold_holds_last_touch_inside_half():
    p1 = np.array([.1, .9, .2, .3, .8, .1])
    K = np.array([0, 1, 0, 0, 1, 0.])
    half = np.array([1, 1, 1, 2, 2, 2], np.int8)
    out = bl.mix_hold(p1, K, half, alpha=1.0)
    # Half 1: retain p1 before the first touch, then hold touch p1. Do not carry it into half 2.
    np.testing.assert_allclose(out, [.1, .9, .9, .3, .8, .8])


# ---------------------------------------------------------------- tune / apply on synthetic files

def fake_dev(tmp_path, n_files=2, T=1500, seed=0):
    """Synthetic dev/select directory with sources 'good' (nearly correct) and 'noisy'."""
    rng = np.random.default_rng(seed)
    data = tmp_path / 'dev' / 'select'
    data.mkdir(parents=True)
    srcs = {'good': tmp_path / 'good', 'noisy': tmp_path / 'noisy'}
    files = []
    for i in range(n_files):
        half = np.r_[np.ones(T // 2), np.full(T - T // 2, 2)].astype(np.int8)
        y = ((np.arange(T) // 150) % 2).astype(np.int8)
        y[(np.arange(T) % 150) > 140] = D
        f = data / ('DFL-MAT-J03WN1_r%02d.npz' % i)
        np.savez(f, y=y, half=half, graded=half > 0)
        files.append(f)
        onehot = np.eye(3)[y]
        for name, noise in (('good', .15), ('noisy', .6)):
            P = (1 - noise) * onehot + noise * rng.dirichlet(np.ones(3), T)
            bl.save_probs(srcs[name] / f.name, np.log(P / P.sum(1, keepdims=True)), name,
                          C=np.c_[P[:, 0], P[:, 1]] / (P[:, :2].sum(1, keepdims=True) + 1e-9),
                          K=np.zeros(T), half=half)
    return files, {k: str(v) for k, v in srcs.items()}


def test_tune_and_apply_on_fake_files(tmp_path, monkeypatch):
    pytest.importorskip('scipy')
    files, srcs = fake_dev(tmp_path)
    monkeypatch.setattr(bl, 'guard_files', lambda *a, **k: None)
    grid = {'mode': 'team_dead', 'ref': 'good', 'team': {'noisy': [0, .5]}, 'dead': {'noisy': [0]}}
    table, chosen = bl.tune(files, srcs, grid, 'select', params='nap1', workers=1)
    base = next(r for r in table if r['baseline'])
    assert base['spec']['team'] == {'good': 1.0}
    assert len(table) == 2 and chosen['spell_f1'] == table[0]['spell_f1']
    with pytest.raises(ValueError, match='select'):
        bl.tune(files, srcs, grid, 'report')
    out = tmp_path / 'out'
    res = bl.apply(files, srcs, chosen['spec'], out, 'select', params='nap1', ref='good')
    assert (out / files[0].name).is_file()
    codes = np.load(out / files[0].name)['codes']
    assert codes.dtype == np.int8 and set(np.unique(codes)) <= {0, 1, 2}
    assert res['spell_f1'] == pytest.approx(chosen['spell_f1'])
    json.loads(str(np.load(out / files[0].name)['blend']))


def test_guard_files_blocks_test_and_wrong_folder(tmp_path):
    with pytest.raises(ValueError, match='dự trữ'):
        bl.guard_files([], 'test')
    f = tmp_path / 'report' / 'DFL-MAT-J03WN1_r05.npz'
    with pytest.raises(ValueError, match='không phải'):
        bl.guard_files([f], 'select')


def test_check_out_dir_refuses_data_dirs():
    from pipeline.schema import CACHE_DIR, TRAIN_DATA_DIR
    for bad in (TRAIN_DATA_DIR, CACHE_DIR, Path(TRAIN_DATA_DIR) / 'dev' / 'x'):
        with pytest.raises(ValueError):
            bl.check_out_dir(bad)


# ---------------------------------------------------------------- optional real data

ART = Path(__file__).resolve().parents[1] / 'artifacts' / 'blend'


@pytest.mark.skipif(not (ART / 'nap1_select').is_dir(), reason='chưa export nấc 1 (python -m src.core.blend export-nap1)')
def test_real_pure_nap1_reproduces_select_score():
    files = bl.data_files('select')
    srcs = {'nap1': str(ART / 'nap1_select')}
    res = bl.evaluate_specs(files, srcs, [bl.pure_spec('nap1')], bl.decode_params('nap1'))
    # Stage-1 dev/select = 0.3157, measured November 12.
    assert res[0, :, 0].mean() == pytest.approx(0.3157, abs=5e-4)
