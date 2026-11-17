"""python -m pipeline.degradation_cli {profile,calibrate,report,export} --help"""
import argparse
import json
from pathlib import Path
import numpy as np
from .schema import CACHE_DIR, save_match
from .load import load_match
from .quality import fit_target_profile, save_profile, load_profile
from .quality_report import half_generalization, quality_report
from .degrade import calibrate_degradation, degrade_match


def output_path(path):
    root=(Path(CACHE_DIR)/'degradation').resolve()
    path=Path(path).resolve()
    if not path.is_relative_to(root): raise ValueError('Outputs must be under ignored '+str(root))
    if path.exists(): raise FileExistsError('Preserving existing artifact: '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    return path


def write_json(path,value):
    output_path(path).write_text(json.dumps(value,allow_nan=False,indent=2))


def export_sample(sample, path):
    """Three separate files, never a model cache containing clean targets."""
    path=output_path(path); path.mkdir()
    save_match(sample.observations,str(path/'observations.npz'))
    arrays={k:v for k,v in sample.targets.items() if isinstance(v,np.ndarray)}
    for group in ('labels','label_valid'):
        arrays.update({group+'__'+k:v for k,v in sample.targets[group].items()})
    np.savez_compressed(path/'targets.npz',**arrays)
    audit=dict(sample.audit); masks={k:audit.pop(k) for k in list(audit) if isinstance(audit[k],np.ndarray)}
    masks.update({'speed_'+k:v for k,v in audit.pop('speed_masks').items()})
    np.savez_compressed(path/'audit_masks.npz',**masks)
    (path/'audit.json').write_text(json.dumps(audit,allow_nan=False,indent=2))


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('profile'); p.add_argument('--target',required=True); p.add_argument('--out',required=True)
    p=sub.add_parser('calibrate'); p.add_argument('--profile',required=True); p.add_argument('--train-matches',nargs='+',required=True); p.add_argument('--seed',type=int,default=2027); p.add_argument('--out',required=True)
    for name in ('report','export'):
        p=sub.add_parser(name); p.add_argument('--profile',required=True); p.add_argument('--params',required=True)
        p.add_argument('--matches',nargs='+',required=True); p.add_argument('--out',required=True)
        p.add_argument('--skip-baselines',action='store_true')
        if name=='export': p.add_argument('--seed',type=int,default=17)
    a=ap.parse_args(argv)
    output_path(a.out)  # Preflight before expensive computation; no overwrites.
    if a.command=='profile':
        output_path(str(a.out)+'.halves.json')
        m=load_match(a.target)
        generalization=half_generalization(m)  # Check both directions BEFORE fitting final profile.
        profile=fit_target_profile(m)
        save_profile(profile,a.out)
        write_json(str(a.out)+'.halves.json',generalization)
        passed=bool(generalization['directions']) and all(r['comparison']['passed'] for r in generalization['directions'])
        print('Profile saved; half generalization:', 'PASS' if passed else 'FAIL (see .halves.json)')
        return 0 if passed else 2
    profile=load_profile(a.profile)
    if a.command=='calibrate':
        params=calibrate_degradation([load_match(n) for n in a.train_matches],profile,a.seed,
            progress=lambda i,n,score: print(f'{i}/{n}: best distance {score:.4f}',flush=True) if i%8==0 else None)
        write_json(a.out,params)
        for r in params['calibration']['reports']: print('failed gates:',r['failed_gates'])
        return 0 if params['calibration']['passed'] else 2
    params=json.loads(Path(a.params).read_text())
    matches=[load_match(n) for n in a.matches]
    if a.command=='report':
        report=quality_report(matches,profile,params,include_baselines=not a.skip_baselines); write_json(a.out,report)
        for r in report['rows']: print(r['match'],r['seed'],'failed gates:',r['comparison']['failed_gates'])
        return 0 if report['passed'] else 2
    if len(matches)!=1: raise ValueError('Export exactly one complete match at a time')
    sample=degrade_match(matches[0],profile,params,a.seed)
    export_sample(sample,a.out)
    report=quality_report(matches,profile,params,seeds=(a.seed,),include_baselines=not a.skip_baselines)
    write_json(str(Path(a.out)/'quality.json'),report)
    print('failed gates:',report['rows'][0]['comparison']['failed_gates'])
    return 0 if report['passed'] else 2


if __name__=='__main__':
    raise SystemExit(main())
