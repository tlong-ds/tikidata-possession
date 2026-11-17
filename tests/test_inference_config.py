"""The committed inference config (configs/inference/) is enough to build the competition View."""
import json
from pathlib import Path

import pytest

from pipeline import feed


def test_inference_release_loads_and_verifies():
    rel = feed.inference_release()
    assert rel.profile is None and rel.params is None
    assert rel.config['window'] and rel.config['polyorder'] is not None
    assert 'BAR-ATH_tactical' in rel.overrides


def test_inference_config_matches_release_when_present():
    current = Path(feed.CACHE_DIR) / 'degradation' / 'CURRENT-v3.json'
    if not current.exists():
        pytest.skip('no release on this machine')
    pinned = json.loads(current.read_text())['sha256']
    ours = json.loads(feed.INFERENCE_POINTER.read_text())['sha256']
    for k in ('view_config', 'orientation_overrides'):
        assert ours[k] == pinned[k]


def test_changed_file_is_refused(tmp_path):
    src = feed.INFERENCE_POINTER.parent
    for f in src.iterdir():
        (tmp_path / f.name).write_bytes(f.read_bytes())
    vc = tmp_path / 'view-config.json'
    vc.write_text(vc.read_text().replace('"window": 21', '"window": 23'))
    with pytest.raises(ValueError):
        feed.inference_release(tmp_path / 'inference.json')
