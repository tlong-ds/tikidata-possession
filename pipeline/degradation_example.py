"""Executable handoff: load -> geometry -> complete-half degradation -> View -> windows."""
import argparse
import json
from pathlib import Path
from dataclasses import fields
import numpy as np
from .load import load_match
from .quality import load_profile, digest
from .degrade import degrade_match, epoch_seed
from .view import resolve_geometry, make_view, transform_targets, iter_windows
from .degradation_cli import output_path, export_sample


def prepare_epoch(clean, profile, params, config, epoch=0, base_seed=2027, overrides=None, seed=None):
    """seed=None keeps the legacy epoch_seed; pass `realization_seed(...)` for namespaced realizations."""
    if config.get('config_hash') != digest({k:v for k,v in config.items() if k!='config_hash'}):
        raise ValueError('View config hash mismatch')
    geometry = resolve_geometry(clean,overrides)
    if seed is None: seed = epoch_seed(base_seed,epoch,clean.name)
    sample = degrade_match(clean,profile,params,seed)
    view = make_view(sample.observations,geometry,config)
    targets = transform_targets(sample.targets,clean.half,geometry)
    return view, targets, sample, geometry


def export_handoff(path, view, targets, sample, geometry, length=250, stride=125):
    path=output_path(path); path.mkdir()
    arrays={f.name:getattr(view,f.name) for f in fields(view) if f.name!='meta'}
    np.savez_compressed(path/'view.npz',**arrays)
    (path/'view-meta.json').write_text(json.dumps(view.meta,indent=2))
    ta={k:v for k,v in targets.items() if isinstance(v,np.ndarray)}
    for group in ('labels','label_valid'):
        ta.update({group+'__'+k:v for k,v in targets[group].items()})
    np.savez_compressed(path/'targets.npz',**ta)
    windows=np.array([(int(w.frame_indices[0]),int(w.frame_indices[-1])+1) for w in iter_windows(view,length,stride)],dtype=np.int64).reshape(-1,2)
    np.save(path/'windows.npy',windows)
    export_sample(sample,path/'audit-sample')
    (path/'geometry-audit.json').write_text(json.dumps(geometry,indent=2))
    # Read-back all exported model/target arrays. Windows reference this single realization.
    for filename,expected in (('view.npz',arrays),('targets.npz',ta)):
        with np.load(path/filename,allow_pickle=False) as z:
            assert set(z.files)==set(expected)
            for k,v in expected.items(): np.testing.assert_equal(z[k],v)
    np.testing.assert_equal(np.load(path/'windows.npy'),windows)
    assert json.loads((path/'view-meta.json').read_text())==view.meta
    return {'readback_verified':True,'windows':len(windows),'frames':len(view.half)}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('profile','params','view-config','match','out'): p.add_argument('--'+key,required=True)
    p.add_argument('--epoch',type=int,default=0); p.add_argument('--base-seed',type=int,default=2027)
    p.add_argument('--window',type=int,default=250); p.add_argument('--stride',type=int,default=125)
    p.add_argument('--orientation-overrides',help='JSON per-half signs and recorded reasons')
    a=p.parse_args(argv); output_path(a.out)
    overrides=json.loads(Path(a.orientation_overrides).read_text()) if a.orientation_overrides else None
    result=prepare_epoch(load_match(a.match),load_profile(a.profile),json.loads(Path(a.params).read_text()),
                         json.loads(Path(a.view_config).read_text()),a.epoch,a.base_seed,overrides)
    print(json.dumps(export_handoff(a.out,*result,a.window,a.stride)))
    return 0


if __name__=='__main__': raise SystemExit(main())
