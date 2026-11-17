"""Reproduce the frozen preprocessing audit; exit 2 for any quality/integration failure."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
from pathlib import Path
from .schema import CACHE_DIR
from .load import load_match, available
from .quality import load_profile
from .quality_report import quality_report, FIXED_SEEDS
from .view import fit_view_config, resolve_geometry, make_view, iter_windows, UnresolvedOrientation
from .degradation_example import prepare_epoch, export_handoff
from .degradation_cli import output_path
import numpy as np

from .split import default_split
_SPLIT=default_split()
MATCHES=list(_SPLIT['dfl'])
ROLES={m: ('calibration/training' if m==_SPLIT['calibration_match'] else 'development' if m==_SPLIT['official']['dev'] else 'reserved_final_evaluation' if m==_SPLIT['official']['reserved'] else 'training_candidate') for m in MATCHES}


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def verify_hashes(manifest):
    failures=[p for p,h in manifest['files'].items() if not Path(p).exists() or sha(p)!=h]
    if failures: raise ValueError('Frozen artifact hash mismatch: '+', '.join(failures))


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False))


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--out'); ap.add_argument('--verify'); ap.add_argument('--orientation-overrides',help='JSON mapping match names to per-half recorded overrides')
    a=ap.parse_args(argv)
    if a.verify:
        verify_hashes(json.loads(Path(a.verify).read_text())); print('All frozen hashes verified'); return 0
    if not a.out: ap.error('--out required')
    out=output_path(a.out); out.mkdir()
    overrides=json.loads(Path(a.orientation_overrides).read_text()) if a.orientation_overrides else {}
    frozen=Path(CACHE_DIR)/'degradation/v2'
    files=list(Path('pipeline').glob('*.py'))+list(Path('tests').glob('*.py'))+[Path('requirements-pipeline.txt')]
    files+=list(Path(CACHE_DIR).glob('*/*.npz'))+list(frozen.glob('*.json'))
    manifest={'files':{str(p.resolve()):sha(p) for p in files},'python':platform.python_version(),
              'dependencies':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
              'seeds':list(FIXED_SEEDS),'roles':ROLES,'baselines':False,'epoch_seed_base':2027,
              'target_role':'label-free tracking statistics only','model_benefit':'not yet tested','orientation_overrides':overrides}
    write(out/'manifest.json',manifest)
    for name in ('target.json','dfl.json','metrica.json','target.json.halves.json','metrica-report-final.json'):
        shutil.copyfile(frozen/name,out/name)
    profile=load_profile(out/'target.json'); params=json.loads((out/'dfl.json').read_text())
    summaries={}
    for name in MATCHES:
        print('Quality:',name,flush=True)
        report=quality_report([load_match(name)],profile,params,include_baselines=False)
        write(out/(name+'-quality.json'),report)
        summaries[name]=report['per_match_summary'][name]
        print(name,summaries[name],flush=True)
    print('Fitting View on J03WMX only',flush=True)
    config=fit_view_config([load_match(MATCHES[0])],profile); write(out/'view-config.json',config)
    integration={}
    for cache_name in available():
        name=cache_name.split('/')[-1]; m=load_match(name)
        print('View:',name,flush=True)
        try:
            if m.source=='atd':
                geometry=resolve_geometry(m,overrides.get(name)); v=make_view(m,geometry,config)
                # Export target observations with no competition labels/supervision.
                dest=out/(name+'-view'); dest.mkdir()
                from dataclasses import fields
                arrays={f.name:getattr(v,f.name) for f in fields(v) if f.name!='meta'}
                np.savez_compressed(dest/'view.npz',**arrays)
                with np.load(dest/'view.npz') as z:
                    for k,val in arrays.items(): np.testing.assert_equal(z[k],val)
                write(dest/'view-meta.json',v.meta); write(dest/'geometry-audit.json',geometry)
                result={'readback_verified':True,'windows':sum(1 for _ in iter_windows(v))}
            else:
                par=params if m.source=='dfl' else json.loads((out/'metrica.json').read_text())
                v,t,s,g=prepare_epoch(m,profile,par,config,overrides=overrides.get(name))
                result=export_handoff(out/(name+'-handoff'),v,t,s,g)
            integration[name]={'passed':True,**result}
        except UnresolvedOrientation as e:
            integration[name]={'passed':False,'error':str(e),'requires':'Recorded per-half override; no guessing'}
        write(out/'integration.json',integration)
    verify_hashes(manifest)
    manifest['files'].update({str(p.resolve()):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'})
    write(out/'manifest.json',manifest); verify_hashes(manifest)
    quality=all(r['passed'] for r in summaries.values()); integ=all(r['passed'] for r in integration.values())
    summary={'tracking_quality_validation':{'passed':quality,'calibration':{MATCHES[0]:summaries[MATCHES[0]]},
              'non_calibration':{k:v for k,v in summaries.items() if k!=MATCHES[0]}},
             'view_integration_validation':{'passed':integ,'matches':integration},
             'downstream_model_benefit':'not yet tested','original_hashes_verified':True,
             'metrica':'diagnostic-only; original failed gates and thresholds preserved in metrica-report-final.json',
             'target_limitations':'Original half-to-half failures preserved in target.json.halves.json'}
    write(out/'summary.json',summary)
    lines=['# Preprocessing validation','',f'Tracking-quality validation: {"PASS" if quality else "FAIL"} (all 35 match/seed gates required).',
           f'View integration validation: {"PASS" if integ else "BLOCKED"}.','Downstream model benefit: **not yet tested**.','',
           '| Match | Role | Quality | Failed gates |','|---|---|---|---|']
    for m,r in summaries.items(): lines.append(f'| {m} | {ROLES[m]} | {r["passed"]} | {", ".join(r["failed_gates"])} |')
    lines+=['','Calibration is reported separately in summary.json. No parameters were retuned and no audit matches excluded.',
            'Metrica remains diagnostic-only with its original failures. Target half-to-half failures remain limitations.',
            'Per-seed results and source limitations are in each match quality JSON; integration.json records every cached source.',
            'Frozen inputs, dependencies, code hashes and artifact hashes are in manifest.json.','', '## View integration']
    for m,r in integration.items(): lines.append(f'- {m}: {r}')
    lines+=['','## Later model validation (not executed)',
            'Train candidates: J03WMX, J03WOH, J03WOY, J03WPY, J03WQQ. Development: J03WN1. Reserved final evaluation: J03WR9.',
            'Compare clean versus degraded training with identical architectures, budgets and paired seeds. Use organizer possession metrics on external labels. Keep competition public labels out of model selection.',
            'Freeze architecture, budget, preprocessing and selection rules on development before opening final evaluation scores. Report paired-seed results and uncertainty.',
            'A reserved-match preprocessing blocker must be reported without replacing the test match. Failed preprocessing gates require a separately documented future calibration cycle; this audit does not authorize retuning.',
            'No models were built or trained. No label scoring was invoked in this audit.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    manifest['files'].update({str((out/n).resolve()):sha(out/n) for n in ('summary.json','REPORT.md')})
    write(out/'manifest.json',manifest)
    return 0 if quality and integ else 2


if __name__=='__main__': raise SystemExit(main())
