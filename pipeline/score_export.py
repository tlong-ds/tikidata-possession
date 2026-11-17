"""Score predictions against train_data/dev or train_data/test, paired with the baseline on the same copies.

    python -m pipeline.score_export PRED_DIR select      # tuning (train_data/dev/select)
    python -m pipeline.score_export PRED_DIR report      # final dev numbers (train_data/dev/report)
    python -m pipeline.score_export PRED_DIR test        # once, right before submitting (train_data/test)

PRED_DIR holds one .npz per data file (same file name) with an array `codes`: 0=H, 1=A, 2=D per frame.
"""
import sys
from pathlib import Path

import numpy as np

from evaluation.possession_metrics import X, evaluate
from .quality import segments
from .schema import TRAIN_DATA_DIR

FOLDERS = {'select': 'dev/select', 'report': 'dev/report', 'test': 'test'}


def score(z, codes):
    """Organizer metrics with halves separated (an X separator stops possessions bridging them)."""
    gt = np.where(z['y'] >= 0, z['y'], X).astype(np.int8)
    spans = segments(z['half'])
    cat = lambda a: np.concatenate([np.r_[a[i:j], X] for i, j in spans])
    r = evaluate(cat(gt), cat(np.asarray(codes, np.int8)), taus=(50,), bootstrap_n=0)
    return r['spell_f1_mean'], r['macro_f1']


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    pred_dir, subset = args[0], (args[1] if len(args) > 1 else 'select')
    root = Path(args[2]) if len(args) > 2 else Path(TRAIN_DATA_DIR)
    files = sorted((root / FOLDERS[subset]).glob('*.npz'))
    if not files:
        raise SystemExit('No data in %s - run python -m pipeline.export first' % (root / FOLDERS[subset]))
    a = np.array([score(np.load(f), np.load(Path(pred_dir) / f.name)['codes']) + score(np.load(f), np.load(f)['baseline'])
                  for f in files])                              # model spell, model macro, base spell, base macro
    d = a[:, 0] - a[:, 2]
    note = {'select': '   (tuning only - do not report)', 'test': '   (final check - run once)'}.get(subset, '')
    print('%s: %d noisy copies%s' % (FOLDERS[subset], len(a), note))
    print('               spellF1          macro-F1')
    print('baseline   %.4f +- %.4f   %.4f' % (a[:, 2].mean(), a[:, 2].std(ddof=1), a[:, 3].mean()))
    print('model      %.4f +- %.4f   %.4f' % (a[:, 0].mean(), a[:, 0].std(ddof=1), a[:, 1].mean()))
    print('model - baseline spellF1: %+.4f +- %.4f, better in %d/%d copies' % (d.mean(), d.std(ddof=1), (d > 0).sum(), len(d)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
