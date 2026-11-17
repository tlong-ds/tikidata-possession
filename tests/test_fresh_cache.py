"""Training-degradation cache: src.nap2.shared.fresh_train + NAP3_FRESH_CACHE."""

import os

import numpy as np
import pytest

from pipeline import feed, split
from src.nap2 import shared

TRAIN = split.roles()["train"][0]
DEV = split.roles()["dev"][0]


@pytest.fixture
def fake(tmp_path, monkeypatch):
    calls = []
    rel = tmp_path / "CURRENT-v3.json"
    rel.write_text('{"files": {"a": "x"}}')
    monkeypatch.setattr(shared, "RELEASE", rel)
    monkeypatch.setattr(shared, "release", lambda: None)

    def train_view(name, epoch, run=0, release=None):
        split.guard(name, "train")
        calls.append((name, epoch, run))
        rng = np.random.default_rng(hash((name, epoch, run)) % 2 ** 32)
        return rng, None

    def as_dict(v, t):
        T = 50
        return {"P": v.normal(size=(T, 46, 6)).astype(np.float32), "B": v.normal(size=(T, 3)).astype(np.float32),
                "half": np.ones(T, np.int8), "fps": 25.0, "y": v.integers(-1, 3, T).astype(np.int8)}
    monkeypatch.setattr(feed, "train_view", train_view)
    monkeypatch.setattr(shared, "_as_dict", as_dict)
    return calls


def _same(a, b):
    assert set(a) == set(b)
    for k in a:
        if k != "name":
            assert np.array_equal(np.asarray(a[k]), np.asarray(b[k])), k
    assert a["name"] == b["name"]


def test_no_env_means_no_cache(fake, tmp_path, monkeypatch):
    monkeypatch.delenv(shared.FRESH_CACHE_ENV, raising=False)
    shared.fresh_train(TRAIN, 3, 1)
    shared.fresh_train(TRAIN, 3, 1)
    assert len(fake) == 2 and not list(tmp_path.rglob("*.npz"))


def test_second_call_reads_the_cache(fake, tmp_path, monkeypatch):
    monkeypatch.setenv(shared.FRESH_CACHE_ENV, str(tmp_path / "c"))
    a = shared.fresh_train(TRAIN, 3, 1)
    b = shared.fresh_train(TRAIN, 3, 1)
    assert fake == [(TRAIN, 3, 1)]
    _same(a, b)
    assert a["name"] == "%s_r1_e03" % TRAIN
    files = list((tmp_path / "c").rglob("*.npz"))
    assert [f.name for f in files] == ["%s_r1_e03.npz" % TRAIN] and files[0].parent.name == shared.release_tag()
    c = shared.fresh_train(TRAIN, 3, 1, max_frames=10)
    assert c["P"].shape[0] == 10 and len(fake) == 1
    shared.fresh_train(TRAIN, 4, 1)
    shared.fresh_train(TRAIN, 3, 2)
    assert len(fake) == 3


def test_readonly_dir_first_then_writable(fake, tmp_path, monkeypatch):
    ro, rw = tmp_path / "ro", tmp_path / "rw"
    (ro / shared.release_tag()).mkdir(parents=True)
    os.chmod(ro / shared.release_tag(), 0o500)
    try:
        monkeypatch.setenv(shared.FRESH_CACHE_ENV, os.pathsep.join([str(ro), str(rw)]))
        shared.fresh_train(TRAIN, 0, 1)
        assert (rw / shared.release_tag() / ("%s_r1_e00.npz" % TRAIN)).is_file()
        shared.fresh_train(TRAIN, 0, 1)
        assert len(fake) == 1
    finally:
        os.chmod(ro / shared.release_tag(), 0o700)


def test_corrupt_file_is_regenerated_and_release_change_misses(fake, tmp_path, monkeypatch):
    monkeypatch.setenv(shared.FRESH_CACHE_ENV, str(tmp_path / "c"))
    d = tmp_path / "c" / shared.release_tag()
    d.mkdir(parents=True)
    (d / ("%s_r1_e00.npz" % TRAIN)).write_bytes(b"not a zip")
    shared.fresh_train(TRAIN, 0, 1)
    assert len(fake) == 1
    shared.RELEASE.write_text('{"files": {"a": "y"}}')                 # Different release -> different directory.
    shared.fresh_train(TRAIN, 0, 1)
    assert len(fake) == 2


def test_cache_hit_still_refuses_non_train_match(fake, tmp_path, monkeypatch):
    monkeypatch.setenv(shared.FRESH_CACHE_ENV, str(tmp_path / "c"))
    d = tmp_path / "c" / shared.release_tag()
    d.mkdir(parents=True)
    np.savez_compressed(d / ("%s_r1_e00.npz" % DEV), half=np.ones(3, np.int8))
    with pytest.raises(ValueError):
        shared.fresh_train(DEV, 0, 1)


def test_kaggle_input_autodetect(tmp_path):
    from src import exp
    (tmp_path / "mtah-nap3-fresh" / "fresh_cache").mkdir(parents=True)
    env = {}
    assert exp.kaggle_fresh_cache(str(tmp_path), env) == os.pathsep.join(
        [str(tmp_path / "mtah-nap3-fresh" / "fresh_cache"), "/tmp/fresh_cache"])
    assert exp.kaggle_fresh_cache(str(tmp_path), {shared.FRESH_CACHE_ENV: "x"}) is None    # Preserve an existing setting.
    assert exp.kaggle_fresh_cache(str(tmp_path / "none"), {}) is None


def test_flat_dir_with_release_marker(fake, tmp_path, monkeypatch):
    flat = tmp_path / "flat"
    flat.mkdir()
    monkeypatch.setenv(shared.FRESH_CACHE_ENV, str(tmp_path / "c"))
    a = shared.fresh_train(TRAIN, 0, 1)
    (tmp_path / "c" / shared.release_tag() / ("%s_r1_e00.npz" % TRAIN)).rename(flat / ("%s_r1_e00.npz" % TRAIN))
    (flat / "release_tag.txt").write_text("khac\n")
    monkeypatch.setenv(shared.FRESH_CACHE_ENV, str(flat))
    assert shared.fresh_cache_dirs() == []                            # Different release: ignore the flat directory.
    (flat / "release_tag.txt").write_text(shared.release_tag() + "\n")
    assert shared.fresh_cache_dirs() == [flat]
    _same(a, shared.fresh_train(TRAIN, 0, 1))
    assert len(fake) == 1
    from src import exp
    assert exp.kaggle_fresh_cache(str(tmp_path), {}).split(os.pathsep)[0] == str(flat)
