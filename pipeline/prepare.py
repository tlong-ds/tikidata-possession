"""One command to get a working `train_data/` (and the competition input) on a fresh clone.

    .venv/bin/python -m pipeline.prepare                        # do everything that is missing
    .venv/bin/python -m pipeline.prepare --check                # only report, change nothing
    .venv/bin/python -m pipeline.prepare --bundle release-v3.zip   # first unpack the release the maintainer sent

It walks the steps below, skips what is already fine, and when something cannot be done by itself it stops
and says which file to ask for. It is safe to run again at any point.

  1. layout       raw datasets sit in train_data/raw/ (moves them if they are loose)
  2. processed    train_data/{train,dev,test} are complete and match checksums.json  -> if yes, nothing to build
  3. release      cache/degradation/CURRENT-v3.json + the files it pins + devset-v2  (from release-v3.zip)
  4. source cache cache/dfl/*.npz, cache/atd/*.npz, rebuilt from raw data (~1-2 min)
  5. export       writes train_data/{train,dev,test}        (a few minutes, ~1.1 GB)
  6. competition  cache/degradation/competition/BAR-ATH_tactical_view.npz (inference input, no labels)

    .venv/bin/python -m pipeline.prepare --inference            # only the competition input (data/ is enough)

`--inference` is what reproducing the submission needs: it builds cache/atd/BAR-ATH_tactical.npz from data/ and
writes the competition View with the committed configs/inference/ files. No release bundle, no training data.

Steps 3-6 are only needed when 2 fails or the competition input is missing. `cache/` and `train_data/` are
gitignored and must never be committed or published.
"""
import argparse
import sys
from pathlib import Path

from . import export, organize_train_data, release_bundle, verify_train_data
from .schema import ROOT, TRAIN_DATA_DIR

TD = Path(TRAIN_DATA_DIR)
DEFAULT_BUNDLE = Path(ROOT) / 'release-v3.zip'


class Stop(Exception):
    """A step that cannot proceed; the message says what to do."""


def say(tag, msg):
    print('[%s] %s' % (tag, msg), flush=True)


def loose_datasets():
    """Dataset folders still directly under train_data/ (the old layout)."""
    moved, skipped = organize_train_data.organize(TD, dry_run=True) if TD.is_dir() else ([], [])
    return moved


def processed_problems():
    """Problems with train_data/{train,dev,test}; [] means ready. Checksums are checked when present."""
    if not (TD / 'manifest.json').exists():
        return ['train_data/manifest.json missing (train/, dev/, test/ not filled yet)']
    return verify_train_data.verify(TD, check_hash=(TD / 'checksums.json').exists())


def raw_problems():
    """What is missing from the raw data that the source cache is built from."""
    from . import io_dfl
    problems = []
    if not io_dfl.match_ids():
        problems.append('no DFL data in %s: put the idsse folder in train_data/raw/idsse/' % io_dfl.DFL_DIR)
    try:
        import games
        games.require([k for k in games.asset_paths() if k != 'video'])
    except Exception as e:                       # games.require raises one readable message
        problems.append('competition ATD data missing in data/ (%s)' % str(e).splitlines()[0])
    return problems


def build_sources(missing):
    from . import build_cache
    names = {Path(m).parent.name for m in missing}
    if 'atd' in names:
        build_cache.main(['atd'])
    if 'dfl' in names:
        build_cache.main(['dfl'])


def run_inference(check=False):
    """Competition input only: cache/atd from data/, then the View from configs/inference/ (no release)."""
    from . import build_cache
    if export.COMPETITION.exists():
        say('ok', 'competition input: present (%s)' % export.COMPETITION.relative_to(ROOT))
        return 0
    atd = [m for m in release_bundle.check_sources() if '/atd/' in m]
    if atd:
        try:
            import games
            games.require([k for k in games.asset_paths() if k != 'video'])
        except Exception as e:
            raise Stop('competition data missing in data/ (%s). Unpack the organisers\' bundle into ./data or '
                       'set MTAH_DATA_DIR.' % str(e).splitlines()[0])
        if check:
            say('todo', 'source cache: cache/atd will be built from data/ (~1 min)')
            return 1
        say('run', 'source cache: building cache/atd from data/ ...')
        build_cache.main(['atd'])
    if check:
        say('todo', 'competition input: will be written from configs/inference/')
        return 1
    export.export_competition()
    say('done', 'competition input: wrote %s' % export.COMPETITION.relative_to(ROOT))
    return 0


def run(check=False, bundle=None, n_train=2):
    todo = []                                                    # commands the user still has to run (check mode)

    # 1. layout
    loose = loose_datasets()
    if loose:
        if check:
            todo.append('python -m pipeline.organize_train_data')
            say('todo', 'layout: %s are loose in train_data/, they belong in train_data/raw/' % ', '.join(loose))
        else:
            organize_train_data.organize(TD)
            say('done', 'layout: moved %s into train_data/raw/' % ', '.join(loose))
    else:
        say('ok', 'layout: train_data/raw/ in place')

    # 2. processed data
    problems = processed_problems()
    comp_ok = export.COMPETITION.exists()
    if not problems:
        say('ok', 'processed: train_data/{train,dev,test} complete%s' % (
            '' if (TD / 'checksums.json').exists() else ' (no checksums.json: sizes and keys checked only)'))
    else:
        shown = problems[:3] + (['... and %d more' % (len(problems) - 3)] if len(problems) > 3 else [])
        say('todo', 'processed: %s' % '; '.join(shown))
    say('ok' if comp_ok else 'todo', 'competition input: %s' % (
        'present' if comp_ok else 'missing (needed only for inference on the competition match)'))
    if not problems and comp_ok:
        print('\nAll ready. Train on train_data/train, tune on train_data/dev/select. See train_data/README.md.')
        return 0

    # 3. release
    rel = release_bundle.check_release()
    if rel:
        zp = Path(bundle) if bundle else (DEFAULT_BUNDLE if DEFAULT_BUNDLE.exists() else None)
        if zp and check:
            todo.append('python -m pipeline.prepare --bundle %s' % zp)
            say('todo', 'release: not unpacked yet; %s will be verified and unpacked' % zp.name)
        elif zp:
            try:
                written, same = release_bundle.unpack(zp)
            except (ValueError, OSError) as e:
                raise Stop('release bundle %s: %s' % (zp, e))
            say('done', 'release: unpacked %s (%d written, %d already identical)' % (zp.name, len(written), len(same)))
        else:
            todo.append('python -m pipeline.prepare --bundle release-v3.zip')
            say('todo', 'release: %s' % '; '.join(rel))
            say('todo', 'release: ask the maintainer for release-v3.zip (private, ~10 MB), save it as %s or pass '
                        '--bundle PATH' % DEFAULT_BUNDLE.name)
            if not check:
                raise Stop('the release files are missing. They are not in git; the maintainer makes the zip with '
                           '`python -m pipeline.release_bundle pack release-v3.zip`.')
    else:
        say('ok', 'release: cache/degradation/CURRENT-v3.json and devset-v2 verified')

    # 4. source cache
    missing = [m for m in release_bundle.check_sources() if problems or '/atd/' in m]   # no export: only atd is read
    if missing:
        raw = raw_problems()
        if raw:
            todo.append('fix raw data, then python -m pipeline.prepare')
            say('todo', 'source cache: %d file(s) missing and cannot be built: %s' % (len(missing), '; '.join(raw)))
            if not check:
                raise Stop('cannot build the source cache without the raw data (%s)' % '; '.join(raw))
        elif check:
            say('todo', 'source cache: %d file(s) will be built from raw data (~1-2 min)' % len(missing))
        else:
            say('run', 'source cache: building %d file(s) from raw data ...' % len(missing))
            build_sources(missing)
            left = [m for m in release_bundle.check_sources() if m in missing]
            if left:
                raise Stop('source cache still missing after build: %s' % ', '.join(left))
            say('done', 'source cache built')
    else:
        say('ok', 'source cache: cache/dfl and cache/atd present')

    if check:
        print('\nNot ready. Next: %s' % (todo[0] if todo else 'python -m pipeline.prepare'))
        return 1

    from . import feed
    release = feed.load_release(release_bundle.RELEASE)

    # 5. export
    if problems:
        busy = [d for d in ('train', 'dev', 'test') if (TD / d).exists() and any((TD / d).iterdir())]
        if busy:
            raise Stop('train_data/%s already has files but fails verification (%s). Delete train_data/train, '
                       'train_data/dev and train_data/test (NOT raw/), or re-unpack the processed data, then rerun.'
                       % ('/'.join(busy), problems[0]))
        say('run', 'export: writing train_data/{train,dev,test} (a few minutes) ...')
        files = export.export(TD, release, release_bundle.DEVSET.parent, n_train)
        say('done', 'export: %s' % {k: len(v) for k, v in files.items()})
        if not (TD / 'checksums.json').exists():
            n = verify_train_data.write_checksums(TD)
            say('done', 'checksums.json written for %d files (built locally)' % n)

    # 6. competition input
    if not export.COMPETITION.exists():
        export.export_competition(release)
        say('done', 'competition input: wrote %s' % export.COMPETITION.relative_to(ROOT))

    # verify
    final = verify_train_data.verify(TD, check_hash=(TD / 'checksums.json').exists())
    for p in final:
        say('FAIL', p)
    if final:
        print('\n%d problem(s) after export. If every checksum differs, your numpy/scipy versions probably differ '
              'from the maintainer\'s: run `pip install -r requirements-pipeline.txt` in the project .venv.' % len(final))
        return 1
    print('\ntrain_data OK. Train on train_data/train, tune on train_data/dev/select. See train_data/README.md.')
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--check', action='store_true', help='report only; change nothing')
    ap.add_argument('--bundle', help='release-v3.zip from the maintainer (default: ./release-v3.zip if present)')
    ap.add_argument('--train-realizations', type=int, default=2, help='noisy copies per training match (default 2)')
    ap.add_argument('--inference', action='store_true',
                    help='only the competition input for inference (needs data/, not the release or train_data)')
    a = ap.parse_args(argv)
    try:
        if a.inference:
            return run_inference(a.check)
        return run(a.check, a.bundle, a.train_realizations)
    except Stop as e:
        print('\nSTOP: %s' % e, file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
