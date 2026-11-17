"""Move the external datasets into train_data/raw/ and create train/, dev/, test/.

    python -m pipeline.organize_train_data             # do it
    python -m pipeline.organize_train_data --dry-run   # only show what would happen

train_data/ is gitignored, so every teammate runs this once on their own copy. It never overwrites
anything and is safe to run again. Afterwards, fill train/, dev/ and test/ either with
`python -m pipeline.export` (needs the cache) or by unzipping the processed folders you were sent.
Standard library only.
"""
import argparse
import shutil
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[1] / 'train_data'
KEEP = {'README.md', 'manifest.json', 'checksums.json', 'raw', 'train', 'dev', 'test'}
FOLDERS = ('raw', 'train', 'dev/select', 'dev/report', 'test')


def organize(train_data=DEFAULT, dry_run=False):
    """Returns (moved, skipped): datasets moved into raw/, and names left alone because raw/ already has them."""
    td = Path(train_data)
    if not td.is_dir():
        raise SystemExit('No folder %s - unpack the datasets into train_data/ first' % td)
    moved, skipped = [], []
    for entry in sorted(td.iterdir()):
        if entry.name in KEEP or entry.name.startswith('.'):
            continue
        if (td / 'raw' / entry.name).exists():
            skipped.append(entry.name)
            continue
        moved.append(entry.name)
        if not dry_run:
            (td / 'raw').mkdir(exist_ok=True)
            shutil.move(str(entry), str(td / 'raw' / entry.name))
    if not dry_run:
        for f in FOLDERS:
            (td / f).mkdir(parents=True, exist_ok=True)
    return moved, skipped


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--train-data', default=str(DEFAULT))
    a = ap.parse_args(argv)
    moved, skipped = organize(a.train_data, a.dry_run)
    verb = 'would move' if a.dry_run else 'moved'
    for name in moved:
        print('%s  %s  ->  raw/%s' % (verb, name, name))
    for name in skipped:
        print('left   %s  (raw/%s already exists - check it by hand)' % (name, name))
    if not moved and not skipped:
        print('nothing to move')
    if not a.dry_run:
        print('ready: raw/, train/, dev/select/, dev/report/, test/ - see train_data/README.md')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
