"""One development scorer: baseline and models on the same frozen realizations, paired.

    # reference numbers only
    python -m pipeline.evaluate_dev --devset cache/degradation/devset-v1 --out cache/degradation/dev-baseline
    # a model: one npz per (match, realization) holding `codes` (0=H 1=A 2=D) at <predictions>/<match>/<id>.npz
    python -m pipeline.evaluate_dev --devset ... --predictions runs/my_model --out cache/degradation/dev-my_model

Reports organizer macro-F1 and mean spellF1 per realization, the paired delta against the fixed
smooth50 baseline on the SAME realizations (mean, SD, wins), and the mean across development matches
(leave-one-match-out view). Default subset is `report`; `select` is for model-selection choices only
and is labelled as such. Never reads the competition public window or the reserved match.
"""
import argparse
from pathlib import Path

import numpy as np

from .benchmark_baseline import baseline_predictions, safe, score_external
from .degradation_cli import output_path
from .feed import dev_views, load_release
from .load import load_match
from .split import default_split
from .validate_preprocessing import write

BASELINE = 'baseline_smooth50'
METRICS = ('macro_f1', 'spell_f1_mean')


def read_codes(pred_dir, name, rid, n_frames):
    p = Path(pred_dir) / name / (rid + '.npz')
    if not p.exists():
        raise FileNotFoundError('Missing prediction file: %s' % p)
    with np.load(p, allow_pickle=False) as z:
        codes = z['codes'].astype(np.int8)
    if codes.shape != (n_frames,) or not np.isin(codes, (0, 1, 2)).all():
        raise ValueError('%s: codes must be length %d with values 0/1/2' % (p, n_frames))
    return codes


def sd(x):
    return float(np.std(x, ddof=1)) if len(x) > 1 else 0.


def evaluate(devset_dir, subset='report', predictions=None, split=None, fold=None, release=None, all_folds=False):
    """all_folds=True scores the dev match of every leave-one-match-out fold (6 matches), else one fold / official dev."""
    rows = []
    clean = {}
    folds = [f['fold'] for f in (split or default_split())['folds']] if all_folds else [fold]
    stream = (item for f in folds for item in dev_views(devset_dir, split, f, subset, release))
    for name, rid, view, _ in stream:
        clean.setdefault(name, load_match(name))
        m = clean[name]
        entry = {'match': name, 'id': rid}
        base = score_external(m, {BASELINE: baseline_predictions(view)[BASELINE]})[BASELINE]
        entry['baseline'] = {k: base[k] for k in METRICS}
        if predictions:
            codes = read_codes(predictions, name, rid, m.n_frames)
            model = score_external(m, {'model': codes})['model']
            entry['model'] = {k: model[k] for k in METRICS}
            entry['delta'] = {k: model[k] - base[k] for k in METRICS}
        rows.append(entry)
        print(name, rid, {k: round(v, 4) for k, v in entry['baseline'].items()}, flush=True)
    per_match = {}
    for name in sorted({r['match'] for r in rows}):
        g = [r for r in rows if r['match'] == name]
        s = {'n': len(g), 'baseline': {k: [float(np.mean([r['baseline'][k] for r in g])), sd([r['baseline'][k] for r in g])] for k in METRICS}}
        if predictions:
            s['model'] = {k: [float(np.mean([r['model'][k] for r in g])), sd([r['model'][k] for r in g])] for k in METRICS}
            s['delta'] = {k: {'mean': float(np.mean([r['delta'][k] for r in g])), 'sd': sd([r['delta'][k] for r in g]),
                              'wins': int(sum(r['delta'][k] > 0 for r in g))} for k in METRICS}
        per_match[name] = s
    summary = {'subset': subset, 'n_matches': len(per_match), 'rows': rows, 'per_match': per_match}
    if predictions:
        summary['across_matches_delta'] = {k: {'mean': float(np.mean([s['delta'][k]['mean'] for s in per_match.values()])),
                                               'sd_between_matches': sd([s['delta'][k]['mean'] for s in per_match.values()])}
                                           for k in METRICS}
    return summary


def report(summary, official_dev):
    banner = ['**SELECTION SUBSET: use for choosing checkpoints / thresholds only; do not report.**', ''] if summary['subset'] == 'select' else []
    lines = ['# Development evaluation', ''] + banner + [
        'Frozen realizations (subset `%s`), organizer metrics with halves separated, baseline = %s on the same inputs.' % (summary['subset'], BASELINE),
        'No competition public labels, no reserved match, no parameter fitting.', '',
        '| Match | n | baseline spellF1 | baseline macro-F1 |' + (' model spellF1 | model macro-F1 | Δ spellF1 (mean ± SD, wins) |' if 'across_matches_delta' in summary else ''),
        '|---|---:|---:|---:|' + ('---:|---:|---:|' if 'across_matches_delta' in summary else '')]
    for name, s in summary['per_match'].items():
        mark = ' (official dev)' if name == official_dev else ''
        row = '| %s%s | %d | %.4f ± %.4f | %.4f ± %.4f |' % (name, mark, s['n'], *s['baseline']['spell_f1_mean'], *s['baseline']['macro_f1'])
        if 'model' in s:
            d = s['delta']['spell_f1_mean']
            row += ' %.4f ± %.4f | %.4f ± %.4f | %+.4f ± %.4f (%d/%d) |' % (*s['model']['spell_f1_mean'], *s['model']['macro_f1'], d['mean'], d['sd'], d['wins'], s['n'])
        lines.append(row)
    if 'across_matches_delta' in summary:
        d = summary['across_matches_delta']['spell_f1_mean']
        lines += ['', 'Mean paired Δ spellF1 across %d development matches: %+.4f (SD between matches %.4f).' % (summary['n_matches'], d['mean'], d['sd_between_matches']),
                  'Only matches present in the devset are averaged; a method that beats the baseline in every match and most realizations is a real gain, a mean inside the between-match SD is not.']
    lines += ['', '± is SD across degraded realizations of the same match (observation noise), not a confidence interval.']
    return '\n'.join(lines) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--devset', required=True)
    ap.add_argument('--predictions')
    ap.add_argument('--subset', choices=('report', 'select'), default='report')
    ap.add_argument('--split')
    ap.add_argument('--all-folds', action='store_true', help='score all 6 leave-one-match-out dev matches')
    ap.add_argument('--release', help='release pointer (default cache/degradation/CURRENT.json)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    out = output_path(a.out)
    out.mkdir()
    from .split import load_split
    sp = load_split(a.split)
    summary = evaluate(a.devset, a.subset, a.predictions, sp, release=load_release(a.release), all_folds=a.all_folds)
    write(out / 'results.json', safe(summary))
    (out / 'REPORT.md').write_text(report(summary, sp['official']['dev']))
    print((out / 'REPORT.md').read_text())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
