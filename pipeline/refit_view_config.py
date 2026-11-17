"""Refit the View config against a new target profile (J03WMX only, as in the frozen cycle).

    python -m pipeline.refit_view_config --profile cache/degradation/restart-v3/target.json \
        --out cache/degradation/restart-v3/view-config.json

`fit_view_config` embeds the profile hash, so any new target profile needs a new config. The speed
threshold and Savitzky-Golay settings are label-free statistics; the output is compared with the
prior config and the differences are printed, never silently accepted.
"""
import argparse
import json
from pathlib import Path

from .degradation_cli import output_path
from .load import load_match
from .quality import load_profile
from .view import fit_view_config


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--profile', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--compare', help='prior view-config.json to diff against')
    a = ap.parse_args(argv)
    out = output_path(a.out)
    config = fit_view_config([load_match('DFL-MAT-J03WMX')], load_profile(a.profile))
    out.write_text(json.dumps(config, indent=2, allow_nan=False))
    print('window=%s polyorder=%s max_speed=%.4f' % (config['window'], config['polyorder'], config['max_speed_m_s']))
    if a.compare:
        old = json.loads(Path(a.compare).read_text())
        same = all(old[k] == config[k] for k in ('window', 'polyorder', 'max_speed_m_s', 'common_eligible_count'))
        print('same velocity settings as prior config:', same)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
