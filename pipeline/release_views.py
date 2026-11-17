"""Export corrected Views across caches, recording provider-backed ATD orientation."""
import argparse
from dataclasses import fields
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from .load import available,load_match
from .view import resolve_geometry,make_view,iter_windows,UnresolvedOrientation
from .quality import load_profile,digest
from .degradation_cli import output_path
from .degradation_example import prepare_epoch,export_handoff
from .validate_preprocessing import sha,write


def provider_overrides(path):
    root=ET.parse(path).getroot()
    teams=list(root.iter('Team'))
    if len(teams)!=2:raise ValueError('Expected two provider teams')
    directions=[]
    for team in teams:
        values=[p.findtext('Value') for p in team.iter('ProviderParameter') if p.findtext('Name')=='attack_direction_first_half']
        if len(values)!=1 or values[0] not in ('left_to_right','right_to_left'):
            raise ValueError('Missing or ambiguous provider direction')
        directions.append(1 if values[0]=='left_to_right' else -1)
    if directions[0]!=-directions[1]:raise ValueError('Provider team directions conflict')
    slots=list(root.iter('Player'))
    if len(slots)!=46 or any(p.get('teamId')!=teams[0].get('id') for p in slots[:24]) or any(p.get('teamId')!=teams[1].get('id') for p in slots[24:]):
        raise ValueError('Provider team order does not match cache slots')
    return {str(h):{'attack_sign':directions[0]*(1 if h==1 else -1),
                   'reason':'Supplied EPTS XML ProviderTeamsParameters/attack_direction_first_half; second half reverses ends',
                   'metadata_sha256':sha(path),'derivation':'direct first-half declaration' if h==1 else 'first-half declaration with halftime change of ends'} for h in (1,2)}


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('prior','view-config','atd-metadata','out'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args(argv);out=output_path(a.out);out.mkdir();prior=Path(a.prior)
    config=json.loads(Path(a.view_config).read_text())
    assert config['config_hash']==digest({k:v for k,v in config.items() if k!='config_hash'})
    overrides=provider_overrides(a.atd_metadata)
    write(out/'orientation-overrides.json',{'BAR-ATH_tactical':overrides})
    profile=load_profile(prior/'target.json');results={}
    protected=list(Path('cache').glob('*/*.npz'))+list(prior.rglob('*'))
    frozen={str(p.resolve()):sha(p) for p in protected if p.is_file()}
    for entry in available():
        name=entry.split('/')[-1];m=load_match(name);print('View:',name,flush=True)
        if m.source=='atd':
            try:
                statistical=resolve_geometry(m);status={'passed':True,'geometry':statistical}
            except UnresolvedOrientation as error:status={'passed':False,'error':str(error)}
            # Check declared direction against each half's statistical inference where supported.
            from dataclasses import replace
            for h in (1,2):
                try:g=resolve_geometry(replace(m,half=np.where(m.half==h,m.half,0)))
                except UnresolvedOrientation:continue
                if g['attack_sign'][str(h)]!=overrides[str(h)]['attack_sign']:raise ValueError('Provider/statistical direction conflict')
            geometry=resolve_geometry(m,overrides);v=make_view(m,geometry,config)
            dest=out/(name+'-view');dest.mkdir()
            arrays={f.name:getattr(v,f.name) for f in fields(v) if f.name!='meta'}
            np.savez_compressed(dest/'view.npz',**arrays)
            with np.load(dest/'view.npz') as z:
                for k,val in arrays.items():np.testing.assert_equal(z[k],val)
            windows=np.asarray([(int(w.frame_indices[0]),int(w.frame_indices[-1])+1) for w in iter_windows(v)],dtype=np.int64)
            np.save(dest/'windows.npy',windows);np.testing.assert_equal(np.load(dest/'windows.npy'),windows)
            for lo,hi in windows:assert len(set(v.half[lo:hi]))==1
            write(dest/'view-meta.json',v.meta);write(dest/'geometry-audit.json',geometry)
            write(dest/'statistical-orientation-check.json',status)
            results[name]={'passed':True,'readback_verified':True,'windows':len(windows),'orientation':'supplied provider metadata plus halftime reversal','statistical_check_passed':status['passed']}
        else:
            params=json.loads((prior/('dfl.json' if m.source=='dfl' else 'metrica.json')).read_text())
            results[name]={'passed':True,**export_handoff(out/(name+'-handoff'),*prepare_epoch(m,profile,params,config))}
        write(out/'integration.json',results)
    for p,h in frozen.items():
        if sha(p)!=h:raise ValueError('Preserved input changed: '+p)
    write(out/'summary.json',{'tracking_quality':'Previous 35/35 DFL quality gates retained; not rerun. Metrica diagnostic-only.',
         'view_integration':{'passed':all(r['passed'] for r in results.values()),'matches':len(results)},
         'downstream_model_benefit':'not yet tested','preserved_inputs_verified':True,'config_hash':config['config_hash']})
    (out/'REPORT.md').write_text('# Corrected View integration release\n\nAll '+str(len(results))+' cached matches exported and read back using the corrected speed-reference config. ATD orientation uses the supplied XML first-half attack directions and the halftime reversal. The opening-60s statistical half-2 failure remains recorded; no gate was relaxed. Provider directions were checked against every statistically resolved half.\n\nTracking quality: earlier DFL 35/35 pass retained, not rerun. Metrica remains diagnostic-only with original failures. Target half-transfer limitations remain. Downstream model benefit: not yet tested.\n\nSee orientation-overrides.json, integration.json and per-match geometry audits. All outputs are private ignored cache artifacts.\n')
    files=[p for p in out.rglob('*') if p.is_file()]+list(Path('pipeline').glob('*.py'))+list(Path('tests').glob('*.py'))+[Path(a.view_config),Path(a.atd_metadata)]
    write(out/'manifest.json',{'files':{str(p.resolve()):sha(p) for p in files},'preserved_inputs':frozen,'config_hash':config['config_hash']})
    return 0


if __name__=='__main__':raise SystemExit(main())
