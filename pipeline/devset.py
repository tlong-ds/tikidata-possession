"""Freeze development realizations: K degraded Views per non-reserved DFL match, plus a clean reference.

    python -m pipeline.devset --release cache/degradation/CURRENT.json --realizations 10 \
        --out cache/degradation/devset-v1 [--split cache/splits/split-v1.json]

What is frozen is the *recipe and its result*, not the arrays: the manifest records, for every
(match, realization), the namespaced seed index, its subset tag and the sha256 of the realized View.
`pipeline.feed.dev_views` regenerates each View deterministically and refuses to continue if the hash
differs (changed code, release files, numpy/scipy). That keeps the dev set at a few KB instead of GBs.

Subsets: the first half of the realizations is `select` (checkpoint / temperature / threshold choices),
the rest is `report` (final numbers only). The reserved match is never realized.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import scipy

from .degradation_cli import output_path
from .degradation_example import prepare_epoch
from .degrade import realization_seed
from .feed import clean_view, load_release, view_hash
from .load import load_match
from .quality import digest
from .split import guard, load_split


def dev_match_names(split):
    names = {split['official']['dev']} | {f['dev'] for f in split['folds']}
    return sorted(names)


def build(split, release, k, out_dir):
    n_select = (k + 1) // 2
    matches = {}
    for name in dev_match_names(split):
        guard(name, 'dev', split, next((f['fold'] for f in split['folds'] if f['dev'] == name), None))
        print('devset:', name, flush=True)
        m = load_match(name)
        entries = []
        view, _ = clean_view(name, release)
        entries.append({'id': 'clean', 'index': None, 'subset': 'clean', 'view_sha256': view_hash(view)})
        for i in range(k):
            view, _, _, _ = prepare_epoch(m, release.profile, release.params, release.config,
                                          overrides=release.overrides.get(name),
                                          seed=realization_seed('dev', 0, i, name))
            entries.append({'id': 'r%02d' % i, 'index': i, 'subset': 'select' if i < n_select else 'report',
                            'view_sha256': view_hash(view)})
        matches[name] = entries
    release_hash = json.loads(Path(release.path).read_text()).get('release_hash')
    ds = {'version': 'devset-v1', 'split_hash': split['split_hash'], 'release_hash': release_hash,
          'realizations': k, 'select': list(range(n_select)), 'report': list(range(n_select, k)),
          'numpy': np.__version__, 'scipy': scipy.__version__, 'matches': matches}
    ds['devset_hash'] = digest({a: b for a, b in ds.items() if a != 'devset_hash'})
    (Path(out_dir) / 'devset.json').write_text(json.dumps(ds, indent=2))
    return ds


def load_devset(path):
    p = Path(path)
    p = p / 'devset.json' if p.is_dir() else p
    ds = json.loads(p.read_text())
    if ds.get('devset_hash') != digest({a: b for a, b in ds.items() if a != 'devset_hash'}):
        raise ValueError('Devset manifest hash mismatch: ' + str(p))
    if ds['numpy'] != np.__version__ or ds['scipy'] != scipy.__version__:
        print('warning: devset built with numpy %s / scipy %s, running %s / %s' % (
            ds['numpy'], ds['scipy'], np.__version__, scipy.__version__))
    return ds


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--release')
    ap.add_argument('--split')
    ap.add_argument('--realizations', type=int, default=10)
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    if a.realizations < 2:
        ap.error('need at least 2 realizations (one select, one report)')
    out = output_path(a.out)
    out.mkdir()
    ds = build(load_split(a.split), load_release(a.release), a.realizations, out)
    print('devset ok: %d matches x (1 clean + %d realizations); hash %s' % (
        len(ds['matches']), a.realizations, ds['devset_hash'][:12]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
