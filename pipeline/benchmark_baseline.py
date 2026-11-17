"""Organizer nearest-player baseline on external DFL clean/degraded observations."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from starter.baseline import smooth, VARIANTS
from evaluation.possession_metrics import evaluate, X
from .load import load_match
from .quality import load_profile,segments
from .quality_report import FIXED_SEEDS
from .degrade import degrade_match
from .view import View
from .schema import CACHE_DIR
from .degradation_cli import output_path
from .validate_preprocessing import sha,write

from .split import default_split,guard

_SPLIT=default_split()
TRAIN=list(_SPLIT['official']['train'])
DEV=_SPLIT['official']['dev']
ALLOWED=TRAIN+[DEV]


def baseline_predictions(observations):
    """Position-only prediction, half bounded. No supervision accepted or consulted."""
    m=observations
    is_view=isinstance(m,View)
    scale=1. if is_view else np.asarray(m.pitch)
    dist=np.sum(((m.P[:,:,:2]-m.B[:,None,:2])*scale)**2,axis=-1)
    if is_view:
        dist[m.P[:,:,5]==0]=np.inf;dist[m.B[:,2]==0]=np.inf
    dist[~np.isfinite(dist)]=np.inf
    idx=dist.argmin(1);codes=np.where(idx<24,0,1)
    codes[~np.isfinite(dist.min(1))]=-1
    result={}
    for name,window in VARIANTS:
        pred=np.full(len(m.half),2,dtype=np.int8)
        for a,b in segments(m.half):
            letters=smooth(codes[a:b],window)
            pred[a:b]=np.where(letters=='H',0,np.where(letters=='A',1,2))
        result[name]=pred
    return result


def score_external(match,predictions):
    """Insert X separators only in scorer arrays to prohibit temporal half bridging."""
    if match.source!='dfl':
        raise ValueError('Label scoring restricted to approved external DFL train/development matches')
    guard(match.name,'score')    # raises ValueError for reserved / target / diagnostic matches
    y=match.extras['y_true'];valid=(match.half>0)&np.isin(y,[0,1,2])
    if 'y_true_valid' in match.extras:valid &= match.extras['y_true_valid'].astype(bool)
    gt=np.where(valid,y,X).astype(np.int8)
    spans=segments(match.half)
    g=np.concatenate([np.r_[gt[a:b],X] for a,b in spans])
    result={}
    for name,pred in predictions.items():
        p=np.concatenate([np.r_[pred[a:b],X] for a,b in spans])
        # Organizer point estimates. Seed spread is reported separately, not bootstrap CIs.
        r=evaluate(g,p,taus=(50,),bootstrap_n=0)
        r['transitions']={str(tol):{k:v for k,v in item.items() if k not in ('pairs','missed','spurious')} for tol,item in r['transitions'].items()}
        result[name]=r
    return result


def safe(value):
    if isinstance(value,dict):return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [safe(v) for v in value]
    if isinstance(value,np.generic):value=value.item()
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--profile',required=True);ap.add_argument('--params',required=True)
    ap.add_argument('--matches',nargs='+',choices=ALLOWED,default=ALLOWED)
    ap.add_argument('--seeds',nargs='+',type=int,default=list(FIXED_SEEDS));ap.add_argument('--out',required=True)
    a=ap.parse_args(argv);out=output_path(a.out);out.mkdir()
    profile=load_profile(a.profile);params=json.loads(Path(a.params).read_text())
    inputs=[Path(a.profile),Path(a.params)]+[next(Path(CACHE_DIR).glob('dfl/'+n+'.npz')) for n in a.matches]
    frozen={str(p.resolve()):sha(p) for p in inputs}
    rows=[];full={}
    for name in a.matches:
        clean=load_match(name);full[name]={}
        for seed in [None]+a.seeds:
            condition='clean' if seed is None else 'degraded'
            key=condition if seed is None else f'degraded-seed{seed}'
            print(name,key,flush=True)
            obs=clean if seed is None else degrade_match(clean,profile,params,seed).observations
            pred=baseline_predictions(obs);metrics=score_external(clean,pred)
            dest=out/name;dest.mkdir(exist_ok=True)
            arrays=dict(pred,frame_indices=np.arange(clean.n_frames),half=clean.half)
            np.savez_compressed(dest/(key+'-predictions.npz'),**arrays)
            with np.load(dest/(key+'-predictions.npz')) as z:
                for k,v in arrays.items():np.testing.assert_equal(z[k],v)
            full[name][key]=metrics
            for variant,r in metrics.items():
                rows.append({'match':name,'role':'development' if name==DEV else 'training','condition':condition,'seed':seed,'variant':variant,
                             **{k:r[k] for k in ('macro_f1','spell_f1_mean','n_graded','n_gt_spells','n_pred_spells')}})
            write(out/'metrics.json',safe(full))
    for r in rows:
        base=next(b for b in rows if b['match']==r['match'] and b['variant']==r['variant'] and b['condition']=='clean')
        for metric in ('macro_f1','spell_f1_mean'):r['delta_'+metric]=r[metric]-base[metric]
    write(out/'scores.json',safe(rows))
    with open(out/'scores.csv','w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# External DFL baseline benchmark','','All four organizer variants are fixed in advance. Published smooth50 is the reference; this run does not select a variant. No baseline training occurs.',
           'Organizer macro-F1 and mean possession-spell F1 over IoU 0.3–0.7 are computed against external y_true. Scorer arrays insert X at half boundaries; prediction exports preserve original frame indices.',
           'No competition labels or reserved J03WR9 data were loaded. Metrica is excluded. Bootstrap intervals are disabled; degraded seed spread uses sample SD (n-1).',
           'Prediction NPZ codes: 0=H, 1=A, 2=D. Missing ball means no nearest-player vote; centered majority smoothing can still infer a team from neighboring visible frames, matching the organizer baseline.',
           'The baseline uses observed coordinates only. Clean ball and labels are used solely as external supervision for scoring, never as degraded prediction inputs.','',
           '| Match | Role | Variant | Clean macro-F1 | Degraded macro-F1 mean ± SD | Clean spellF1 | Degraded spellF1 mean ± SD |',
           '|---|---|---|---:|---:|---:|---:|']
    for name in a.matches:
        for variant,_ in VARIANTS:
            b=next(r for r in rows if r['match']==name and r['variant']==variant and r['condition']=='clean')
            ds=[r for r in rows if r['match']==name and r['variant']==variant and r['condition']=='degraded']
            def fmt(metric):
                vals=[r[metric] for r in ds];return f'{np.mean(vals):.4f} ± {np.std(vals,ddof=1) if len(vals)>1 else 0:.4f}'
            lines.append(f"| {name} | {b['role']} | {variant} | {b['macro_f1']:.4f} | {fmt('macro_f1')} | {b['spell_f1_mean']:.4f} | {fmt('spell_f1_mean')} |")
    lines+=['','These results measure baseline sensitivity to observation degradation. They do not measure whether degraded training improves a learned model. Tracking-quality and View integration status remain in their separate preprocessing reports.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    for p,h in frozen.items():
        if sha(p)!=h:raise ValueError('Input hash changed: '+p)
    files=[p for p in out.rglob('*') if p.is_file()]+list(Path('pipeline').glob('*.py'))+list(Path('tests').glob('*.py'))+[Path('starter/baseline.py'),Path('evaluation/possession_metrics.py'),Path('evaluation/run_evaluation.py')]
    write(out/'manifest.json',{'files':{str(p.resolve()):sha(p) for p in files},'inputs':frozen,
         'seeds':a.seeds,'matches':a.matches,'numpy':np.__version__,'reserved_match_scoring':'withheld','bootstrap_n':0,'model_training':False})
    return 0


if __name__=='__main__':raise SystemExit(main())
