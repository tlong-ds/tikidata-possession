"""Fixed data split: the single source of truth for which match may be used for what.

    python -m pipeline.split --out cache/splits/split-v1.json

Official roles (frozen in the 2026-11-06 audit): train J03WMX/OH/OY/PY/QQ, development J03WN1,
reserved final evaluation J03WR9 (never trained on, never used for selection). The competition match
is an inference target only. Metrica is diagnostic-only. On top of the official split there are
leave-one-match-out folds over the six non-reserved DFL matches, so a result can be averaged over more
than one development match.

Model code must not hard-code match lists: call `roles(...)` and `guard(name, purpose)`.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from .quality import digest
from .schema import CACHE_DIR

VERSION = 'split-v1'
DFL = ['DFL-MAT-' + s for s in ('J03WMX', 'J03WN1', 'J03WOH', 'J03WOY', 'J03WPY', 'J03WQQ', 'J03WR9')]
CALIBRATION = 'DFL-MAT-J03WMX'     # View config and degradation parameters were fitted here (label-free)
RESERVED = 'DFL-MAT-J03WR9'
TARGET = 'BAR-ATH_tactical'
DIAGNOSTIC = ['Sample_Game_1', 'Sample_Game_2', 'Sample_Game_3']
# Extra training data from other competitions (pipeline/external/metrica.py: Metrica sample games converted to the
# DFL cache contract, cache/external/). Opt-in only: an experiment lists them in [train] extra_train. They are NOT
# part of roles()['train'] or of default_split(), so split-v1 and its split_hash stay frozen, and they can never be
# used for dev / select / report / score / final.
from .external.registry import ALL as EXTERNAL_TRAIN   # Metrica, PFF WC 2022, SkillCorner (pipeline/external/registry.py)
FINAL_RETRAIN_ENV = 'NAP3_FINAL_RETRAIN'
FOLD_ENV = 'NAP3_FOLD'          # Cross-validation: roles() / guard() default to this fold (dev is another DFL match, never J03WR9). # =1: final training includes dev (src.core.train_switch --final).
PURPOSES = ('train', 'dev', 'select', 'report', 'score', 'infer', 'diagnostic', 'final')


def _sha(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(block), b''):
            h.update(chunk)
    return h.hexdigest()


def cache_path(name):
    source = 'atd' if name == TARGET else 'metrica' if name in DIAGNOSTIC else 'dfl'
    return Path(CACHE_DIR) / source / (name + '.npz')


def default_split(hashes=False):
    """The v1 split. hashes=True also records sha256 of every cache file (slow: reads ~GBs)."""
    usable = [m for m in DFL if m != RESERVED]
    official = {'train': [m for m in usable if m != 'DFL-MAT-J03WN1'], 'dev': 'DFL-MAT-J03WN1',
                'reserved': RESERVED, 'target': TARGET, 'diagnostic': list(DIAGNOSTIC)}
    folds = [{'fold': i, 'dev': m, 'train': [x for x in usable if x != m],
              'calibration_overlap': m == CALIBRATION} for i, m in enumerate(usable)]
    s = {'version': VERSION, 'dfl': list(DFL), 'calibration_match': CALIBRATION,
         'official': official, 'folds': folds,
         'note': 'Fitting of View config / degradation parameters used label-free statistics of J03WMX only.'}
    if hashes:
        names = DFL + [TARGET] + DIAGNOSTIC
        s['cache_sha256'] = {n: _sha(cache_path(n)) for n in names if cache_path(n).exists()}
    s['split_hash'] = digest({k: v for k, v in s.items() if k != 'split_hash'})
    return s


def load_split(path=None, verify=True):
    """Load a saved split (or the built-in default when path is None); verify its hashes."""
    if path is None:
        return default_split()
    s = json.loads(Path(path).read_text())
    if s.get('split_hash') != digest({k: v for k, v in s.items() if k != 'split_hash'}):
        raise ValueError('Split file hash mismatch: ' + str(path))
    if s.get('version') != VERSION:
        raise ValueError('Unsupported split version: ' + str(s.get('version')))
    if verify:
        for name, expected in s.get('cache_sha256', {}).items():
            p = cache_path(name)
            if not p.exists() or _sha(p) != expected:
                raise ValueError('Cache file changed since split was made: ' + name)
    return s


def roles(split=None, fold=None):
    """{'train': [...], 'dev': [...], 'reserved': [...], 'target': [...], 'diagnostic': [...]}."""
    s = split or default_split()
    if fold is None and os.environ.get(FOLD_ENV, '') != '':
        fold = int(os.environ[FOLD_ENV])
    if fold is None:
        o = s['official']
        return {'train': list(o['train']), 'dev': [o['dev']], 'reserved': [o['reserved']],
                'target': [o['target']], 'diagnostic': list(o['diagnostic'])}
    f = next((x for x in s['folds'] if x['fold'] == fold), None)
    if f is None:
        raise ValueError('No such fold: %r' % (fold,))
    return {'train': list(f['train']), 'dev': [f['dev']], 'reserved': [s['official']['reserved']],
            'target': [s['official']['target']], 'diagnostic': list(s['official']['diagnostic'])}


def guard(name, purpose, split=None, fold=None):
    """Raise ValueError unless `name` may be used for `purpose` under this split.

    train: training matches only. dev/select/report: the development match only. score: any non-reserved
    DFL match (for sensitivity studies). infer: the competition match. diagnostic: Metrica.
    final: the reserved match, only after the protocol is frozen (callers must not use this casually).
    """
    if purpose not in PURPOSES:
        raise ValueError('Unknown purpose %r' % purpose)
    if purpose == 'train' and name in EXTERNAL_TRAIN:
        return
    r = roles(split, fold)
    if purpose == 'train' and os.environ.get(FINAL_RETRAIN_ENV) == '1' and name in r['dev']:
        return                       # Final training after configuration selection: include dev in training, never the reserved match.
    allowed = {'train': r['train'], 'dev': r['dev'], 'select': r['dev'], 'report': r['dev'],
               'score': r['train'] + r['dev'], 'infer': r['target'], 'diagnostic': r['diagnostic'],
               'final': r['reserved']}[purpose]
    if name not in allowed:
        raise ValueError('%s may not be used for %r (allowed: %s)' % (name, purpose, ', '.join(allowed)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', required=True)
    ap.add_argument('--no-hash', action='store_true', help='skip cache file hashes (not for frozen splits)')
    a = ap.parse_args(argv)
    out = Path(a.out).resolve()
    if not out.is_relative_to((Path(CACHE_DIR) / 'splits').resolve()):
        raise ValueError('Splits must be written under ' + str(Path(CACHE_DIR) / 'splits'))
    if out.exists():
        raise FileExistsError('Preserving existing split: ' + str(out))
    out.parent.mkdir(parents=True, exist_ok=True)
    s = default_split(hashes=not a.no_hash)
    out.write_text(json.dumps(s, indent=2))
    print('official: train=%d dev=%s reserved=%s folds=%d hash=%s' % (
        len(s['official']['train']), s['official']['dev'], s['official']['reserved'], len(s['folds']),
        s['split_hash'][:12]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
