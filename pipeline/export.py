"""Write the processed splits into train_data/ (see train_data/README.md). Reading them needs only numpy.

    .venv/bin/python -m pipeline.export        # fills train_data/train, train_data/dev, train_data/test

train_data/raw/ (the original datasets) is never modified. Nothing under train_data/ may be committed or published.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import feed, release_bundle, split
from .benchmark_baseline import baseline_predictions
from .schema import CACHE_DIR, TRAIN_DATA_DIR

OUTPUTS = ('train', 'dev', 'test')
# Two levels below cache/: `load_match` globs cache/<source>/<name>.npz and must never see this file.
COMPETITION = Path(CACHE_DIR) / 'degradation' / 'competition' / 'BAR-ATH_tactical_view.npz'


def save(path, view, targets=None, baseline=False):
    d = dict(P=view.P.astype(np.float32), B=view.B.astype(np.float32), cam=view.cam.astype(np.float32),
             half=view.half, graded=view.graded, fps=view.meta['fps'], pitch=np.array(view.meta['pitch']))
    if targets is not None:
        y = targets['labels']['y_true'].astype(np.int8)
        d['y'] = np.where(targets['label_valid']['y_true'], y, -1).astype(np.int8)
        d['ball_xy'] = targets['ball_xy'].astype(np.float32)
    if baseline:
        d['baseline'] = baseline_predictions(view)['baseline_smooth50']
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **d)
    return path.name


def export_competition(release=None, path=COMPETITION):
    """Write the unlabelled inference View of the competition match (needs cache/atd; release optional).

    Without `release` the committed configs/inference/ files are used (see feed.inference_release)."""
    save(Path(path), feed.target_view(release))
    return Path(path)


def export(root, release, devset, n_train=2, n_test=5, competition=None):
    root = Path(root)
    busy = [str(root / d) for d in OUTPUTS if (root / d).exists() and any((root / d).iterdir())]
    if busy:
        raise FileExistsError('Already exported: %s. Delete these folders to export again.' % ', '.join(busy))
    roles = split.roles()
    files = {'train': [], 'dev/select': [], 'dev/report': [], 'test': []}
    for name in roles['train']:                                   # fresh noise per copy
        for e in range(n_train):
            v, t = feed.train_view(name, e, release=release)
            files['train'].append(save(root / 'train' / ('%s_e%d.npz' % (name, e)), v, t))
    for subset in ('select', 'report'):                           # frozen copies of the dev match
        for name, rid, v, t in feed.dev_views(devset, subset=subset, release=release):
            files['dev/' + subset].append(save(root / 'dev' / subset / ('%s_%s.npz' % (name, rid)), v, t, baseline=True))
    for name in roles['reserved']:                                # held-out final check
        split.guard(name, 'final')
        for i in range(n_test):
            v, t = feed.realize_dev(name, i, release, purpose='test')
            files['test'].append(save(root / 'test' / ('%s_r%02d.npz' % (name, i)), v, t, baseline=True))
    if competition:                                               # inference input, kept outside train_data
        export_competition(release, competition)
    (root / 'manifest.json').write_text(json.dumps({
        'roles': {'train': roles['train'], 'dev': roles['dev'], 'test': roles['reserved']}, 'files': files,
        'release_hash': json.loads(Path(release.path).read_text()).get('release_hash')}, indent=2))
    return files


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--v1', action='store_true', help='older release without the restart-aware ball dropout')
    ap.add_argument('--train-realizations', type=int, default=2, help='noisy copies per training match')
    a = ap.parse_args(argv)
    d = Path(CACHE_DIR) / 'degradation'
    pointer = d / ('CURRENT.json' if a.v1 else 'CURRENT-v3.json')
    devset = d / ('devset-v1' if a.v1 else 'devset-v2')
    problems = release_bundle.check_release(pointer, devset)
    problems += ['missing %s (source cache)' % p for p in release_bundle.check_sources()]
    if problems:                                                  # one readable list instead of a stack trace
        raise SystemExit('Cannot export yet:\n  %s\nRun `python -m pipeline.prepare` (it explains each fix and does '
                         'what it can).' % '\n  '.join(problems))
    release = feed.load_release(pointer)
    files = export(TRAIN_DATA_DIR, release, devset, a.train_realizations, competition=COMPETITION)
    print({k: len(v) for k, v in files.items()}, '-> train_data/; competition match ->', COMPETITION)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
