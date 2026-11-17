"""Pack / unpack / check the small set of files that `pipeline.export` needs and git does not carry.

`cache/` is gitignored, so a fresh clone has no release. Exporting needs exactly:

  * the release pointer `cache/degradation/CURRENT-v3.json` and the 4 files it pins (profile of the
    competition match, DFL degradation parameters, View config, orientation overrides),
  * the frozen dev-set manifest `cache/degradation/devset-v2/devset.json`,
  * the source cache `cache/dfl/*.npz` and `cache/atd/BAR-ATH_tactical.npz` (rebuilt from the raw data by
    `build_cache`; it is NOT in the bundle).

Everything else under `cache/degradation/` (the ~1 GB of reports, older releases, handoffs) is not needed.

    python -m pipeline.release_bundle pack release-v3.zip     # maintainer: ~10 MB zip, send it privately
    python -m pipeline.release_bundle unpack release-v3.zip   # teammate: verifies sha256, writes into cache/
    python -m pipeline.release_bundle check                   # is everything export needs in place?

Normally you do not call this directly: `python -m pipeline.prepare` does it. The zip holds data derived from
the competition match's tracking: share it with the team only, never commit or publish it.
"""
import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path

from .schema import CACHE_DIR, ROOT, TRAIN_DATA_DIR

ROOT_P = Path(ROOT)
RELEASE = Path(CACHE_DIR) / 'degradation' / 'CURRENT-v3.json'
DEVSET = Path(CACHE_DIR) / 'degradation' / 'devset-v2' / 'devset.json'
CHECKSUMS = Path(TRAIN_DATA_DIR) / 'checksums.json'
BUNDLE_KEY = 'BUNDLE.json'
FORMAT = 'release-bundle-v1'


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def _rel(path):
    try:
        return Path(path).resolve().relative_to(ROOT_P.resolve()).as_posix()
    except ValueError:                                   # outside the repository: show it as given
        return str(path)


def _devset_file(devset):
    p = Path(devset)
    return p / 'devset.json' if p.is_dir() else p


def required_files(release=RELEASE, devset=DEVSET):
    """Repo-relative paths a teammate needs for this release (pointer, the files it pins, dev-set manifest)."""
    ptr = json.loads(Path(release).read_text())
    return [_rel(release)] + sorted(ptr['files'].values()) + [_rel(_devset_file(devset))]


def check_release(release=RELEASE, devset=DEVSET):
    """Human-readable problems with the release files (empty list = ready). Does not read any match data."""
    release, devset = Path(release), _devset_file(devset)
    problems, usable = [], False
    if not release.exists():
        problems.append('missing %s (release pointer)' % _rel(release))
    else:
        from . import feed
        try:
            feed.load_release(release)
            usable = True
        except (ValueError, KeyError, OSError) as e:
            problems.append('release %s is not usable: %s' % (_rel(release), e))
    if not devset.exists():
        problems.append('missing %s (frozen dev set)' % _rel(devset))
    elif usable and json.loads(devset.read_text()).get('release_hash') != json.loads(release.read_text())['release_hash']:
        problems.append('%s was built from a different release than %s' % (_rel(devset), _rel(release)))
    return problems


def sources_needed():
    """Source-cache files export reads: the 7 DFL matches and the competition match."""
    from .split import DFL, TARGET, cache_path
    return [cache_path(n) for n in DFL + [TARGET]]


def check_sources():
    """Source-cache files that are missing (they are rebuilt from raw data, see prepare.py)."""
    return [_rel(p) for p in sources_needed() if not p.exists()]


def pack(out, release=RELEASE, devset=DEVSET, with_checksums=True):
    release, devset = Path(release), _devset_file(devset)
    problems = check_release(release, devset)
    if problems:
        raise SystemExit('Cannot pack, the release on this machine is not consistent:\n  ' + '\n  '.join(problems))
    files = required_files(release, devset)
    ptr_hash = json.loads(release.read_text())['release_hash']
    if with_checksums and CHECKSUMS.exists() and json.loads(CHECKSUMS.read_text()).get('release_hash') == ptr_hash:
        files.append(_rel(CHECKSUMS))                  # lets the teammate prove their export matches yours
    info = {p: {'sha256': _sha(ROOT_P / p), 'size': (ROOT_P / p).stat().st_size} for p in files}
    manifest = {'format': FORMAT, 'release': _rel(release), 'release_hash': ptr_hash, 'files': info}
    out = Path(out)
    if out.exists():
        raise SystemExit('%s already exists, delete it first' % out)
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr(BUNDLE_KEY, json.dumps(manifest, indent=2))
        for p in files:
            z.write(ROOT_P / p, p)
    return manifest, out.stat().st_size


def _safe(name):
    p = Path(name)
    ok = (not p.is_absolute() and '..' not in p.parts and
          (p.parts[:1] == ('cache',) or p.as_posix() == 'train_data/checksums.json'))
    if not ok:
        raise ValueError('bundle contains a path outside cache/: %r' % name)
    return ROOT_P / p


def unpack(bundle, force=False, devset=DEVSET):
    """Verify every file against BUNDLE.json, then write it. Returns (written, unchanged).

    Existing identical files are left alone; an existing file with other content is an error unless
    `force`, because a silently different release would make every teammate's scores incomparable.
    """
    with zipfile.ZipFile(bundle) as z:
        if BUNDLE_KEY not in z.namelist():
            raise ValueError('%s is not a release bundle (no %s)' % (bundle, BUNDLE_KEY))
        manifest = json.loads(z.read(BUNDLE_KEY))
        if manifest.get('format') != FORMAT:
            raise ValueError('unsupported bundle format %r' % manifest.get('format'))
        blobs, conflicts = {}, []
        for name, meta in manifest['files'].items():
            dest = _safe(name)
            data = z.read(name)
            if hashlib.sha256(data).hexdigest() != meta['sha256']:
                raise ValueError('%s is corrupt or truncated (sha256 mismatch), download it again' % name)
            blobs[name] = (dest, data)
            if dest.exists() and _sha(dest) != meta['sha256'] and not force:
                conflicts.append(name)
        if conflicts:
            raise ValueError('these files already exist with different content (use --force to replace them):\n  '
                             + '\n  '.join(conflicts))
        written, unchanged = [], []
        for name, (dest, data) in blobs.items():
            if dest.exists() and _sha(dest) == manifest['files'][name]['sha256']:
                unchanged.append(name)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + '.part')
            tmp.write_bytes(data)
            os.replace(tmp, dest)
            written.append(name)
    problems = check_release(ROOT_P / manifest['release'], devset)
    if problems:
        raise ValueError('bundle unpacked but the release is still not usable:\n  ' + '\n  '.join(problems))
    return written, unchanged


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('pack', help='maintainer: write the zip')
    p.add_argument('out')
    p.add_argument('--no-checksums', action='store_true', help='leave train_data/checksums.json out')
    u = sub.add_parser('unpack', help='teammate: verify and extract the zip into cache/')
    u.add_argument('bundle')
    u.add_argument('--force', action='store_true')
    sub.add_parser('check', help='report what is missing')
    a = ap.parse_args(argv)
    if a.cmd == 'pack':
        manifest, size = pack(a.out, with_checksums=not a.no_checksums)
        for name, m in manifest['files'].items():
            print('%9.1f KB  %s' % (m['size'] / 1024, name))
        print('wrote %s (%.1f MB), release %s. Send it to the team privately; never commit it.' % (
            a.out, size / 1e6, manifest['release_hash'][:12]))
        return 0
    if a.cmd == 'unpack':
        try:
            written, unchanged = unpack(a.bundle, a.force)
        except (ValueError, zipfile.BadZipFile) as e:
            raise SystemExit('ERROR: %s' % e)
        print('wrote %d file(s), %d already identical. Next: python -m pipeline.prepare' % (len(written), len(unchanged)))
        return 0
    problems = check_release()
    missing = check_sources()
    for p in problems:
        print('MISSING', p)
    for p in missing:
        print('MISSING', p, '(built from raw data by: python -m pipeline.prepare)')
    print('ready' if not (problems or missing) else '%d problem(s)' % (len(problems) + len(missing)))
    return 1 if (problems or missing) else 0


if __name__ == '__main__':
    raise SystemExit(main())
