"""Synthetic src.core.evaltools tests (numpy only); one real-data test is marked slow."""
import math
from pathlib import Path

import numpy as np
import pytest

from src.core import evaltools as et
from pipeline.schema import TRAIN_DATA_DIR

H, A, D = 0, 1, 2


def fake_z(n=400, seed=0):
    """Synthetic two-half match with regular ownership changes, D intervals and alternating ball visibility."""
    rng = np.random.default_rng(seed)
    half = np.r_[np.ones(n // 2), np.full(n - n // 2, 2)].astype(np.int8)
    y = np.empty(n, np.int8)
    for t in range(n):
        y[t] = (t // 40) % 2
    y[60:70] = D                       # D interval between two same-team possessions.
    y[150:160] = D
    B = np.zeros((n, 3), np.float32)
    B[:, :2] = rng.uniform(0, 1, (n, 2))
    B[::3, 2] = 0                      # Ball unseen every third frame.
    B[1::3, 2] = 1
    B[2::3, 2] = 1
    B[::3, :2] = np.nan
    ball_xy = np.cumsum(rng.normal(0, .002, (n, 2)), 0) + .5
    return {'y': y, 'half': half, 'graded': half > 0, 'fps': np.float64(25),
            'pitch': np.array([105., 68.]), 'B': B, 'ball_xy': ball_xy.astype(np.float32),
            'baseline': np.where(y == D, H, y).astype(np.int8), 'name': 'fake.npz'}


# ---------------------------------------------------------------- confidence intervals

def test_t_crit_table_matches_scipy():
    pytest.importorskip('scipy')
    for level in (0.90, 0.95, 0.99):
        for df in (1, 4, 9, 30):
            assert et.t_crit(level, df, use_scipy=False) == pytest.approx(et.t_crit(level, df), abs=2e-3)
    # Use the Cornish-Fisher approximation for df >30.
    assert et.t_crit(0.95, 60, use_scipy=False) == pytest.approx(et.t_crit(0.95, 60), abs=2e-3)


def test_paired_ci_values():
    a = np.array([.30, .32, .31, .29, .33])
    b = a + np.array([.02, .03, .01, .02, .02])
    r = et.paired_ci(a, b, use_scipy=False)
    d = b - a
    half = 2.776 * d.std(ddof=1) / math.sqrt(5)
    assert r['n'] == 5 and r['wins'] == 5 and r['losses'] == 0
    assert r['mean'] == pytest.approx(d.mean())
    assert r['lo'] == pytest.approx(d.mean() - half) and r['hi'] == pytest.approx(d.mean() + half)
    one = et.paired_ci([.3], [.4])
    assert one['mean'] == pytest.approx(.1) and math.isnan(one['lo'])
    with pytest.raises(ValueError):
        et.paired_ci([1, 2], [1, 2, 3])


# ---------------------------------------------------------------- situations

def test_true_changes_carry_across_dead_ball_not_across_halves():
    y = np.array([H, H, D, D, A, A, A, H, H, A], np.int8)
    half = np.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 2], np.int8)
    ch = et.true_changes(y, half)
    # H->A at frame 4 across D; exclude frame 7 at the half boundary; H->A again at frame 9.
    assert np.flatnonzero(ch).tolist() == [4, 9]
    # Same team before and after D: no team change.
    assert not et.true_changes(np.array([H, D, D, H], np.int8), np.ones(4, np.int8)).any()


def test_near_change_mask_stays_inside_half():
    y = np.array([H] * 5 + [A] * 5 + [A] * 5, np.int8)
    half = np.array([1] * 10 + [2] * 5, np.int8)
    m = et.near_change_mask(y, half, w=2)
    assert np.flatnonzero(m).tolist() == [3, 4, 5, 6, 7]


def test_spell_count_matches_definition():
    codes = np.array([H, H, D, H, A, A, D, D, A, H], np.int8)
    half = np.array([1] * 5 + [2] * 5, np.int8)
    # Half 1: H | H | A; half 2: A | A | H.
    assert et.spell_count(codes, half) == 6


def test_ball_speed_scales_normalised_coordinates():
    n = 50
    xy = np.c_[np.linspace(0, 0.1, n), np.zeros(n)]          # 0.1 * 105 m trong 49 khung
    spd = et.ball_speed(xy, np.ones(n, np.int8), fps=25, pitch=[105, 68])
    expect = 0.1 * 105 / (n - 1) * 25
    assert np.allclose(spd, expect, rtol=1e-6)


def test_situations_masks_are_consistent():
    z = fake_z()
    m = et.situations(z)
    live = m['in_play']
    assert live.sum() == np.isin(z['y'], (H, A)).sum()
    assert not (m['ball_seen'] & m['ball_unseen']).any()
    assert ((m['ball_seen'] | m['ball_unseen']) == (z['half'] > 0)).all()
    assert not (m['near_change'] & m['far_from_change']).any()
    assert {'half_1', 'half_2', 'ball_fast', 'ball_slow'} <= set(m)


def test_breakdown_perfect_and_flipped_predictions():
    z = fake_z()
    perfect = et.breakdown(z, codes=z['y'], team_prob=(z['y'] == H).astype(float))
    row = {r['situation']: r for r in perfect['rows']}
    assert row['in_play']['acc_codes'] == 1 and row['in_play']['acc_prob'] == 1
    assert perfect['rates']['spellF1'] == pytest.approx(1.0)
    assert perfect['rates']['n_spells_pred'] == perfect['rates']['n_spells_true']
    flipped = np.where(z['y'] == H, A, np.where(z['y'] == A, H, z['y'])).astype(np.int8)
    bad = et.breakdown(z, codes=flipped)
    rb = {r['situation']: r for r in bad['rows']}
    assert rb['in_play']['acc_codes'] == 0 and rb['in_play']['err_share'] == 1
    assert bad['rates']['wrong_team'] == pytest.approx(1.0)
    shares = rb['ball_seen']['err_share'] + rb['ball_unseen']['err_share']
    assert shares == pytest.approx(1.0)
    mean = et.mean_breakdown([perfect, bad])
    assert {r['situation']: r for r in mean['rows']}['in_play']['acc_codes'] == pytest.approx(.5)


# ---------------------------------------------------------------- oracle experiments

def test_emissions_normalised_and_oracles():
    rng = np.random.default_rng(1)
    q, pD = rng.uniform(size=100), rng.uniform(size=100)
    E = et.emissions(q, pD)
    assert np.allclose(np.exp(E).sum(1), 1)
    y = rng.integers(0, 3, 100).astype(np.int8)
    qt = et.with_true_team(q, y)
    assert (qt[y == H] > .99).all() and (qt[y == A] < .01).all() and np.allclose(qt[y == D], q[y == D])
    dt = et.with_true_dead(pD, y)
    assert (dt[y == D] > .99).all() and (dt[y != D] < .01).all()


def test_true_switch_touch_marks_changes_only():
    y = np.array([H, H, A, A, D, H], np.int8)
    K, C = et.true_switch_touch(y, np.ones(6, np.int8))
    assert np.flatnonzero(K).tolist() == [2, 5]
    assert C[2].tolist() == [0, 1] and C[5].tolist() == [1, 0] and C[0].tolist() == [.5, .5]


def test_oracle_suite_true_team_helps():
    pytest.importorskip('scipy')
    z = fake_z(n=600, seed=3)
    rng = np.random.default_rng(3)
    q = np.clip((z['y'] == H) * .6 + .2 + rng.normal(0, .25, len(z['y'])), .01, .99)
    pD = np.where(z['y'] == D, .7, .1)
    K = np.zeros(len(q))
    C = np.full((len(q), 2), .5)
    out = et.oracle_suite(z, q, pD, C, K)
    from pipeline.score_export import score
    sc = {k: score(z, v)[0] for k, v in out.items()}
    assert set(sc) == {'base', 'true_team', 'true_dead', 'true_switch'}
    assert sc['true_team'] >= sc['base']


# ---------------------------------------------------------------- files and CLI

def write_fake_root(tmp_path, n_copies=2):
    folder = tmp_path / 'dev' / 'select'
    folder.mkdir(parents=True)
    files = []
    for i in range(n_copies):
        z = fake_z(seed=i)
        f = folder / ('DFL-MAT-J03WN1_r%02d.npz' % i)
        np.savez(f, **{k: v for k, v in z.items() if k != 'name'})
        files.append(f)
    return files


def test_score_compare_and_pred_dir_roundtrip(tmp_path):
    files = write_fake_root(tmp_path)
    assert et.data_files('select', root=tmp_path) == files
    truth = {f.name: np.load(f)['y'] for f in files}
    pred = et.write_pred_dir(tmp_path / 'pred', {k: {'codes': v, 'team_prob': (v == H).astype(float)}
                                                 for k, v in truth.items()})
    assert set(et.load_pred_dir(pred, files)) == {f.name for f in files}
    assert 'team_prob' in et.load_pred(pred, files[0].name)
    s = et.score_codes(files, et.load_pred_dir(pred, files))
    assert s.shape == (2, 2) and np.allclose(s[:, 0], 1)
    res = et.compare(files, {'base': et.baseline_preds(files), 'truth': truth})
    assert res['models']['truth']['vs_first']['spellF1']['wins'] == 2
    with pytest.raises(FileNotFoundError):
        et.load_pred_dir(tmp_path / 'pred', [tmp_path / 'missing.npz'])


def test_data_files_refuses_test_without_final(tmp_path):
    with pytest.raises(PermissionError):
        et.data_files('test', root=tmp_path)
    assert et.match_of('DFL-MAT-J03WN1_r03.npz') == 'DFL-MAT-J03WN1'
    assert et.match_of('DFL-MAT-J03WMX_e1.npz') == 'DFL-MAT-J03WMX'


def test_cli_compare_breakdown_and_test_refusal(tmp_path, capsys):
    files = write_fake_root(tmp_path)
    et.write_pred_dir(tmp_path / 'pred', {f.name: np.load(f)['y'] for f in files})
    out_json = tmp_path / 'cmp.json'
    assert et.main(['compare', '--data-root', str(tmp_path), '--pred', 'smooth50=@baseline',
                    '--pred', 'truth=%s' % (tmp_path / 'pred'), '--json', str(out_json)]) == 0
    text = capsys.readouterr().out
    assert 'smooth50' in text and 'truth' in text and out_json.exists()
    assert et.main(['breakdown', '--data-root', str(tmp_path), '--pred', str(tmp_path / 'pred')]) == 0
    assert 'in_play' in capsys.readouterr().out
    with pytest.raises(SystemExit) as e:
        et.main(['compare', '--subset', 'test', '--data-root', str(tmp_path), '--pred', 'x=@baseline'])
    assert e.value.code == 2


# ---------------------------------------------------------------- real data (slow)

@pytest.mark.skipif(not (Path(TRAIN_DATA_DIR) / 'dev' / 'select').is_dir(), reason='cần train_data/dev/select')
def test_real_select_baseline_score():
    files = et.data_files('select')
    s = et.score_codes(files, et.baseline_preds(files))
    # smooth50 dev/select = 0.2113, measured November 12.
    assert s[:, 0].mean() == pytest.approx(0.2113, abs=5e-4)
