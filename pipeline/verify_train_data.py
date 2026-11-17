"""Check that a train_data/ folder is complete, unmodified and has the layout every model expects.

    .venv/bin/python -m pipeline.verify_train_data            # verify against train_data/checksums.json
    .venv/bin/python -m pipeline.verify_train_data --write    # (maintainer) record sha256 of every file

Send `checksums.json` and `manifest.json` together with the processed folders. A teammate who runs this
once knows they hold exactly the same bytes as everyone else, so scores are comparable.
Needs only numpy. Exit code 0 = OK, 1 = problem.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .schema import TRAIN_DATA_DIR

REQUIRED = {'P': (3, np.float32), 'B': (2, np.float32), 'cam': (2, np.float32), 'half': (1, np.int8),
            'graded': (1, np.bool_), 'y': (1, np.int8), 'ball_xy': (2, np.float32)}
TEAM_KEYS = {'dev/select', 'dev/report', 'test'}          # these also carry the organizer baseline


def sha256(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(block), b''):
            h.update(chunk)
    return h.hexdigest()


def _paths(root, manifest):
    for group, names in manifest['files'].items():
        for n in names:
            yield group, n, Path(root) / group / n


def write_checksums(root):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    sums = {'%s/%s' % (g, n): sha256(p) for g, n, p in _paths(root, manifest)}
    (root / 'checksums.json').write_text(json.dumps(
        {'release_hash': manifest['release_hash'], 'sha256': sums}, indent=2))
    return len(sums)


def verify(root, check_hash=True):
    """Return a list of human-readable problems (empty list = OK)."""
    root = Path(root)
    problems = []
    if not (root / 'manifest.json').exists():
        return ['manifest.json missing in %s' % root]
    manifest = json.loads((root / 'manifest.json').read_text())
    sums = None
    if check_hash:
        if not (root / 'checksums.json').exists():
            problems.append('checksums.json missing (ask the maintainer, or run with --write if you built it yourself)')
        else:
            sums = json.loads((root / 'checksums.json').read_text())
            if sums.get('release_hash') != manifest['release_hash']:
                problems.append('checksums.json and manifest.json come from different releases')
    roles = manifest['roles']
    seen = {}
    for group, name, path in _paths(root, manifest):
        key = '%s/%s' % (group, name)
        if not path.exists():
            problems.append('missing file: %s' % key)
            continue
        match = name.rsplit('_', 1)[0]
        base = group.split('/')[0]
        allowed = roles['train'] if base == 'train' else roles['dev'] if base == 'dev' else roles['test']
        if match not in allowed:                      # a match in the wrong folder leaks labels
            problems.append('%s: match %s must not be in %s/' % (key, match, base))
        if sums is not None and sums['sha256'].get(key) != sha256(path):
            problems.append('checksum mismatch (file changed or truncated): %s' % key)
        try:
            z = np.load(path)
            T = len(z['half'])
            for k, (nd, dt) in REQUIRED.items():
                if k not in z.files:
                    problems.append('%s: key %r missing' % (key, k))
                elif z[k].shape[0] != T or z[k].ndim != nd or z[k].dtype != dt:
                    problems.append('%s: %s has shape %s dtype %s' % (key, k, z[k].shape, z[k].dtype))
            if group in TEAM_KEYS and 'baseline' not in z.files:
                problems.append('%s: baseline missing' % key)
            if 'P' in z.files and z['P'].shape[1:] != (46, 6):
                problems.append('%s: P shape %s, expected [T,46,6]' % (key, z['P'].shape))
            seen[key] = T
        except Exception as e:                        # corrupt zip, truncated download, ...
            problems.append('%s: unreadable (%s)' % (key, e))
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--root', default=str(TRAIN_DATA_DIR))
    ap.add_argument('--write', action='store_true', help='record checksums.json (maintainer only)')
    ap.add_argument('--no-hash', action='store_true', help='skip the slow sha256 check')
    a = ap.parse_args(argv)
    if a.write:
        print('wrote checksums.json for %d files' % write_checksums(a.root))
        return 0
    problems = verify(a.root, check_hash=not a.no_hash)
    for p in problems:
        print('PROBLEM', p)
    print('train_data OK' if not problems else '%d problem(s)' % len(problems))
    return 1 if problems else 0


if __name__ == '__main__':
    raise SystemExit(main())
