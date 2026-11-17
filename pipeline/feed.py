"""Model-facing feed: the only supported way for model code to obtain preprocessed data.

    from pipeline import feed, split
    rel = feed.load_release()                                   # verified CURRENT.json
    view, targets = feed.train_view('DFL-MAT-J03WOH', epoch=3, run=0)       # fresh degraded realization
    for rid, view, targets in feed.dev_views(devset_dir):       # frozen dev realizations (hash-checked)
        ...
    view = feed.target_view()                                   # competition match (inference only)

Every path goes: cache -> resolve_geometry (raw metadata) -> degrade (train / dev only) -> make_view.
Models must consume View arrays (centered metres, vx/vy in m/s, fixed team channels). Never read
raw `Match.P[..., 2]`: it is m/s for ATD, km/h for DFL and NaN for Metrica (SOURCE_SPEED_UNITS).

Roles come from `pipeline.split`; reserved/target/diagnostic matches are refused for training.
"""
import argparse
import json
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from .degrade import degrade_match, realization_seed
from .degradation_example import prepare_epoch
from .load import load_match
from .quality import digest, load_profile
from .schema import CACHE_DIR, ROOT
from .split import TARGET, guard
from .view import View, iter_windows, make_view, resolve_geometry, transform_targets

SOURCE_SPEED_UNITS = {'atd': 'm/s', 'dfl': 'km/h (raw XML S)', 'metrica': 'absent (NaN)'}
CURRENT = Path(CACHE_DIR) / 'degradation' / 'CURRENT.json'
RELEASE_KEYS = ('profile', 'dfl_params', 'view_config', 'orientation_overrides')
INFERENCE_POINTER = Path(ROOT) / 'configs' / 'inference' / 'inference.json'


@dataclass
class Release:
    profile: dict
    params: dict
    config: dict
    overrides: dict
    hashes: dict
    path: str


def _file_sha(path):
    from .split import _sha
    return _sha(path)


def write_current(profile, dfl_params, view_config, orientation_overrides, out=None):
    """Record one verified, hashed release. Paths are stored relative to the repository root."""
    out = Path(out or CURRENT)
    paths = dict(zip(RELEASE_KEYS, (profile, dfl_params, view_config, orientation_overrides)))
    rel = {k: str(Path(p).resolve().relative_to(Path(ROOT).resolve())) for k, p in paths.items()}
    ptr = {'version': 'current-v1', 'files': rel, 'sha256': {k: _file_sha(Path(ROOT) / v) for k, v in rel.items()}}
    release = _read_release(ptr, str(out))           # validates cross-file consistency before writing
    ptr['release_hash'] = digest({k: v for k, v in ptr.items() if k != 'release_hash'})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ptr, indent=2))
    return release


def _read_release(ptr, path):
    root = Path(ROOT)
    for k in RELEASE_KEYS:
        p = root / ptr['files'][k]
        if not p.exists() or _file_sha(p) != ptr['sha256'][k]:
            raise ValueError('Release file changed or missing: %s (%s)' % (k, ptr['files'][k]))
    profile = load_profile(root / ptr['files']['profile'])
    params = json.loads((root / ptr['files']['dfl_params']).read_text())
    config = json.loads((root / ptr['files']['view_config']).read_text())
    overrides = json.loads((root / ptr['files']['orientation_overrides']).read_text())
    if config.get('config_hash') != digest({k: v for k, v in config.items() if k != 'config_hash'}):
        raise ValueError('View config hash mismatch')
    if params.get('profile_hash') != profile['profile_hash'] or config.get('target_profile_hash') != profile['profile_hash']:
        raise ValueError('Profile / degradation params / View config belong to different target profiles')
    return Release(profile, params, config, overrides, dict(ptr['sha256']), path)


def load_release(path=None):
    path = Path(path or CURRENT)
    ptr = json.loads(path.read_text())
    if ptr.get('release_hash') != digest({k: v for k, v in ptr.items() if k != 'release_hash'}):
        raise ValueError('Release pointer hash mismatch: ' + str(path))
    return _read_release(ptr, str(path))


def _targets_from_clean(m):
    labels = {k: np.array(v, copy=True) for k, v in m.extras.items() if k in ('y_true', 'y_draft', 'touch_true')}
    active = m.half > 0
    bv = np.isfinite(m.B[:, :2]).all(-1)
    return {'labels': labels,
            'label_valid': {k: np.isfinite(v) & (v >= 0) & active for k, v in labels.items()},
            'ball': m.B.copy(), 'ball_valid': np.isfinite(m.B).copy(), 'ball_coordinate_valid': bv}


def clean_view(name, release=None):
    """Undegraded View of an external DFL match (for clean-vs-degraded sensitivity only)."""
    release = release or load_release()
    m = load_match(name)
    g = resolve_geometry(m, release.overrides.get(name))
    return make_view(m, g, release.config), transform_targets(_targets_from_clean(m), m.half, g)


def train_view(name, epoch, run=0, split=None, fold=None, release=None):
    """A fresh degraded realization of a training match -> (View, targets). Seeds never collide with dev."""
    guard(name, 'train', split, fold)
    release = release or load_release()
    m = load_match(name)
    view, targets, _, _ = prepare_epoch(m, release.profile, release.params, release.config,
                                        overrides=release.overrides.get(name),
                                        seed=realization_seed('train', run, epoch, name))
    return view, targets


def realize_dev(name, index, release=None, purpose='dev'):
    """Deterministic dev realization `index` of a match -> (View, targets)."""
    release = release or load_release()
    m = load_match(name)
    view, targets, _, _ = prepare_epoch(m, release.profile, release.params, release.config,
                                        overrides=release.overrides.get(name),
                                        seed=realization_seed(purpose, 0, index, name))
    return view, targets


def view_hash(view):
    import hashlib
    h = hashlib.sha256()
    for f in fields(view):
        if f.name == 'meta':
            continue
        a = np.ascontiguousarray(getattr(view, f.name))
        h.update(str((f.name, a.shape, a.dtype)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def inference_release(path=INFERENCE_POINTER):
    """The two small files the competition View needs, committed under configs/inference/.

    Inference reads only the View config (velocity smoothing) and the orientation overrides (attack direction
    from the supplied EPTS XML). The degradation profile and DFL params of the full release are for training
    only, so someone with the data bundle and this repository can build the inference input without them.
    Same sha256 as the files pinned by cache/degradation/CURRENT-v3.json (checked by tests/test_inference_config.py).
    """
    path = Path(path)
    ptr = json.loads(path.read_text())
    out = {}
    for k in ('view_config', 'orientation_overrides'):
        p = path.parent / ptr['files'][k]
        if _file_sha(p) != ptr['sha256'][k]:
            raise ValueError('Inference config changed: %s (%s)' % (k, p))
        out[k] = json.loads(p.read_text())
    config = out['view_config']
    if config.get('config_hash') != digest({k: v for k, v in config.items() if k != 'config_hash'}):
        raise ValueError('View config hash mismatch')
    return Release(None, None, config, out['orientation_overrides'], dict(ptr['sha256']), str(path))


def target_view(release=None):
    """The competition match View (inference only; no labels exist for it).

    Without `release`, uses the committed inference config (no private release bundle needed).
    """
    guard(TARGET, 'infer')
    release = release or inference_release()
    m = load_match(TARGET)
    g = resolve_geometry(m, release.overrides.get(TARGET))
    return make_view(m, g, release.config)


def dev_views(devset_dir, split=None, fold=None, subset='report', release=None):
    """Yield (match, realization_id, View, targets) for the frozen dev realizations of one subset.

    subset: 'select' (checkpoint / temperature / threshold choices), 'report' (final numbers) or
    'clean' (undegraded sensitivity reference). Content hashes recorded at build time are verified.
    """
    from .devset import load_devset
    ds = load_devset(devset_dir)
    release = release or load_release()
    if ds['release_hash'] != json.loads(Path(release.path).read_text()).get('release_hash'):
        raise ValueError('Devset was built from a different release than the one loaded')
    from .split import roles
    dev_names = roles(split, fold)['dev']
    for name in dev_names:
        if name not in ds['matches']:
            raise ValueError('Devset has no entry for %s' % name)
        guard(name, 'dev', split, fold)
        for entry in ds['matches'][name]:
            if subset == 'clean' and entry['id'] != 'clean':
                continue
            if subset != 'clean' and entry['subset'] != subset:
                continue
            if entry['id'] == 'clean':
                view, targets = clean_view(name, release)
            else:
                view, targets = realize_dev(name, entry['index'], release)
            if view_hash(view) != entry['view_sha256']:
                raise ValueError('Frozen dev realization changed: %s %s (library versions? code edits?)' % (name, entry['id']))
            yield name, entry['id'], view, targets


def windows(view, length=250, stride=125):
    return iter_windows(view, length, stride)


def main(argv=None):
    ap = argparse.ArgumentParser(description='Write cache/degradation/CURRENT.json (the one release models use)')
    for k in RELEASE_KEYS:
        ap.add_argument('--' + k.replace('_', '-'), required=True)
    ap.add_argument('--out')
    a = ap.parse_args(argv)
    out = Path(a.out or CURRENT)
    if out.exists():
        raise FileExistsError('Preserving existing release pointer: %s (write a new one with --out)' % out)
    rel = write_current(a.profile, a.dfl_params, a.view_config, a.orientation_overrides, out)
    print('release ok: config window=%s order=%s -> %s' % (rel.config['window'], rel.config['polyorder'], out))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
