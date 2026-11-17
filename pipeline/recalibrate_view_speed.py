"""Refit View's speed reference on J03WMX only, preserving the original audit."""
import argparse
import json
from pathlib import Path
import numpy as np
from .load import load_match
from .quality import load_profile
from .view import fit_view_config, speed_reference_m_s, velocities
from .io_dfl import read_positions, _file
from .validate_preprocessing import sha, write
from .degradation_cli import output_path
from .degradation_example import prepare_epoch, export_handoff


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prior',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args(argv);prior=Path(a.prior);out=output_path(a.out);out.mkdir()
    prior_manifest=json.loads((prior/'manifest.json').read_text())
    # Implementation evolves; all frozen data/reports/exports must remain byte-identical.
    preserved={p:h for p,h in prior_manifest['files'].items()
               if not Path(p).is_relative_to(Path.cwd()/'pipeline') and not Path(p).is_relative_to(Path.cwd()/'tests')}
    preserved[str((prior/'manifest.json').resolve())]=sha(prior/'manifest.json')
    for p,h in preserved.items():
        if sha(p)!=h:raise ValueError('Prior artifact changed: '+p)
    m=load_match('DFL-MAT-J03WMX');original=m.P.copy()
    raw=read_positions(m.name)
    section='firstHalf';(_,person),rec=next(iter(raw[section]['persons'].items()))
    pid=m.meta['persons'].index(person)
    rows,slots=np.where((m.extras['slot_person']==pid)&(m.half[:,None]==1))
    rows,slots=rows[:1000],slots[:1000]
    index=np.searchsorted(rec['N'],m.extras['N'][rows])
    np.testing.assert_equal(rec['N'][index],m.extras['N'][rows])
    np.testing.assert_allclose(m.P[rows,slots,2],rec['s'][index],rtol=1e-6,atol=1e-6)
    profile=load_profile(prior/'target.json');config=fit_view_config([m],profile)
    old=json.loads((prior/'view-config.json').read_text())
    assert config['max_speed_m_s']==old['max_speed_m_s']
    assert config['common_eligible_count']==old['common_eligible_count']
    reference,_=speed_reference_m_s(m)
    xy=(m.P[:,:,:2]-.5)*m.pitch
    common=np.isfinite(reference)&(m.half[:,None]>0)
    for w in (5,7,11,15,21):
        for p in (2,3):common &= np.isfinite(velocities(xy,m.half,m.meta['fps'],dict(config,window=w,polyorder=p))).all(-1)
    old_v=velocities(xy,m.half,m.meta['fps'],old)
    old_err=np.abs(np.linalg.norm(old_v,axis=-1)-reference)
    old_corrected=float(np.mean([np.median(old_err[common&(m.half[:,None]==h)]) for h in (1,2)]))
    winner=next(r for r in config['candidates'] if (r['window'],r['polyorder'])==(config['window'],config['polyorder']))
    evidence={'source_url':'https://www.nature.com/articles/s41597-025-04505-y',
              'source_doi':'10.1038/s41597-025-04505-y','source_section':'Data Records / Position data',
              'finding':'XML S is km/h; io_dfl copies S unchanged. Only the fitting reference is divided by 3.6.',
              'raw_xml_cache_values_verified':len(rows),'raw_xml_sha256':sha(_file('DFL_04_03_positions_raw_observed',m.name)),
              'old_selected':{'window':old['window'],'polyorder':old['polyorder'],'mae_on_corrected_reference_m_s':old_corrected},
              'new_selected':winner,'segmentation_threshold_unchanged':True,'common_frames_unchanged':True,
              'fit_matches':[m.name],'quality_gates':'unchanged; prior 35/35 results retained, not rerun',
              'atd_orientation':'half 2 still unresolved','downstream_model_benefit':'not yet tested'}
    write(out/'view-config.json',config);write(out/'evidence.json',evidence)
    np.testing.assert_equal(m.P,original)
    params=json.loads((prior/'dfl.json').read_text())
    result=export_handoff(out/'calibration-handoff',*prepare_epoch(m,profile,params,config))
    write(out/'handoff-check.json',result)
    for p,h in preserved.items():
        if sha(p)!=h:raise ValueError('Prior artifact changed: '+p)
    report=f'''# DFL speed-unit correction\n\nConfirmed from dataset authors: XML S is km/h (DOI 10.1038/s41597-025-04505-y, Data Records / Position data). The loader copied S unchanged. Verified {len(rows)} raw/cache values.\n\nOnly the View fitting reference is converted by dividing by 3.6. Caches, frozen quality profiles, parameters, reports and exports are unchanged. J03WMX is the only fitting match; no labels or reserved-match scores used.\n\nNew selection: window {config['window']}, order {config['polyorder']}; mean half-median absolute speed error {winner['mean_half_median_mae']:.6f} m/s. Old window/order evaluated against the corrected reference: {old_corrected:.6f} m/s. The previous 3.677653 score mixed units and is not a valid m/s error.\n\nSegmentation threshold remains {config['max_speed_m_s']:.6f} m/s; all {config['common_eligible_count']} common eligible observations are unchanged. The calibration epoch handoff was exported and read back successfully.\n\nTracking-quality status: prior 35/35 DFL gates retained; not rerun because quality code/data did not change. View speed-reference blocker resolved; ATD half-2 orientation remains unresolved. Metrica remains diagnostic-only. Downstream model benefit is not yet tested.\n\nThe old manifest still describes old source hashes. Source edits are intentional; its original artifacts were verified separately and its previous implementation was archived alongside this run. Existing exports still use the old View configuration; regenerate future handoffs with this run's view-config.json.\n'''
    (out/'REPORT.md').write_text(report)
    files=list(out.rglob('*'))+list(Path('pipeline').glob('*.py'))+list(Path('tests').glob('*.py'))
    manifest={'preserved_prior_files':preserved,'files':{str(p.resolve()):sha(p) for p in files if p.is_file()},
              'raw_xml':evidence['raw_xml_sha256'],'numpy':config['numpy_version'],'scipy':config['scipy_version']}
    write(out/'manifest.json',manifest)
    print(json.dumps(evidence,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
