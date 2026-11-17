"""Distribution comparisons, acceptance gates, and secondary organizer-baseline diagnostics."""
import numpy as np
from .quality import profile_match, segments

FIXED_SEEDS = (17, 43, 101, 211, 509)


def _distance(a,b):
    if not a['n'] or not b['n']: return None
    if a['kind']=='rate': return abs(a['value']-b['value'])
    # Exact 1-Wasserstein between the stored empirical sketches (including unequal sizes).
    x, y = np.asarray(a['samples']), np.asarray(b['samples'])
    support = np.sort(np.concatenate((x,y)))
    delta = np.diff(support)
    cdf_x = np.searchsorted(x,support[:-1],side='right') / len(x)
    cdf_y = np.searchsorted(y,support[:-1],side='right') / len(y)
    return float(np.sum(delta*np.abs(cdf_x-cdf_y)))


def compare_profiles(target, clean, degraded):
    families={}; unsupported=[]
    for key,t in target['metrics'].items():
        c,d=clean['metrics'][key],degraded['metrics'][key]
        dc,dd=_distance(t,c),_distance(t,d)
        if dd is None:
            unsupported.append(key); continue
        # No-camera clean sources have no observed geometry comparator. Still report synthetic fit.
        scale=max(t['scale'] or 0,.01)
        tolerance=((t['uncertainty'] or 0)+(c['uncertainty'] or 0)+(d['uncertainty'] or 0))/scale
        f=families.setdefault(t['family'],{})
        f[key]={'clean':None if dc is None else dc/scale,'degraded':dd/scale,
                'uncertainty':tolerance,'target_uncertainty':(t['uncertainty'] or 0)/scale,
                'immutable_floor':dc/scale if dc is not None and key in ('player_speed_missing','ball_speed_missing') and c['value']==1. else 0.,
                'target_n':t['n'],'degraded_n':d['n']}
    family_summary={}
    for name,metrics in families.items():
        paired=[m for m in metrics.values() if m['clean'] is not None]
        family_summary[name]={'clean':float(np.mean([m['clean'] for m in paired])) if paired else None,
                              'degraded':float(np.mean([m['degraded'] for m in paired])) if paired else None,
                              'uncertainty':float(np.mean([m['uncertainty'] for m in paired])) if paired else None,
                              'target_uncertainty':float(np.mean([m['target_uncertainty'] for m in paired])) if paired else None,
                              'immutable_floor':float(np.mean([m['immutable_floor'] for m in paired])) if paired else None,
                              'metrics':metrics}
    paired=[f for f in family_summary.values() if f['clean'] is not None]
    ac=float(np.mean([f['clean'] for f in paired])) if paired else 0
    ad=float(np.mean([f['degraded'] for f in paired])) if paired else 0
    uncertainty=float(np.mean([f['target_uncertainty'] for f in paired])) if paired else 0
    speed_floor=float(np.mean([f['immutable_floor'] for f in paired])) if paired else 0.
    gates={}
    for key in unsupported:
        if target['metrics'][key]['n'] and clean['metrics'][key]['n']:
            gates['lost_support_'+key]={'passed':False,'reason':'Supported target/source diagnostic became unobservable'}
    def rate_gate(key,tolerance,exempt=False):
        t,c,d=(p['metrics'][key]['value'] for p in (target,clean,degraded))
        if t is None or d is None:
            gates[key]={'passed':False,'status':'unsupported','reason':'Insufficient valid observations'}; return
        unattainable=exempt and c is not None and c>t+tolerance
        # Exemption does not permit further degradation of the unavailable channel.
        passed=abs(d-t)<=tolerance or unattainable and d<=c+1e-9
        gates[key]={'passed':bool(passed),'status':'source_limited' if unattainable else 'tested',
                    'target':t,'clean':c,'degraded':d,'tolerance':tolerance}
    rate_gate('ball_visible',.03); rate_gate('player_count',1.); rate_gate('ball_speed_missing',.05,True)
    # Restart coupling exists only for profiles/sources that measured it (old profiles are unaffected).
    pre=[p['metrics'].get('ball_visible_pre_restart',{}).get('value') for p in (target,clean)]
    if all(v is not None for v in pre): rate_gate('ball_visible_pre_restart',.10)
    for q,label in ((.5,'median'),(.9,'p90')):
        vals=[]; details=[]
        for p in (target,clean,degraded):
            run=p['runs']['ball_missing_runs']; x=np.asarray(run['seconds']); cens=np.asarray(run['censored'],bool)
            x=x[~cens]; val=float(np.quantile(x,q)) if len(x) else None
            ci=None
            if len(x):
                rng=np.random.default_rng(65); x=x[:10000]
                boot=[np.quantile(rng.choice(x,len(x),replace=True),q) for _ in range(128)]
                ci=np.quantile(boot,[.025,.975]).tolist()
            vals.append(val); details.append({'estimate':val,'ci95':ci,'n_complete':run['n_complete'],'n_censored':run['n_censored']})
        t,c,d=vals
        passed=t is not None and d is not None and abs(d-t)<=.25*t
        gates['ball_missing_'+label]={'passed':bool(passed),'relative_tolerance':.25,
                                      'target_clean_degraded':details}
    gates['aggregate_reduction']={'passed':bool(ac<=uncertainty or ad<=.75*ac),
                                  'required_reduction':.25,'outside_target_variability':ac>uncertainty,
                                  'clean':ac,'degraded':ad,'uncertainty':uncertainty,'uncertainty_scope':'target_only',
                                  'irreducible_speed_floor':speed_floor,'required_max_distance':.75*ac,
                                  'proven_unattainable_under_current_support':bool(ac>uncertainty and speed_floor>.75*ac),
                                  'floor_assumption':'Retain current compared diagnostic support; never synthesize unavailable speeds.'}
    for family,f in family_summary.items():
        if f['clean'] is not None:
            gates['nonworsening_'+family]={'passed':bool(f['degraded']<=f['clean']+f['uncertainty']),
                                         'clean':f['clean'],'degraded':f['degraded'],'uncertainty':f['uncertainty']}
    return {'passed':all(g['passed'] for g in gates.values()),'gates':gates,
            'failed_gates':[k for k,v in gates.items() if not v['passed']],
            'aggregate_clean':ac,'aggregate_degraded':ad,
            'objective_degraded':float(np.mean([np.mean([m['degraded'] for m in f.values()]) for f in families.values()])) if families else 0.,
            'families':family_summary,
            'unsupported_metrics':unsupported,
            'distance_method':'Absolute rates; exact empirical-sketch Wasserstein (log1p durations); target variability normalization; equal family weights.',
            'gates_are':'Engineering defaults, not organizer requirements.'}


def half_generalization(match):
    """Fit one half, audit the other, reverse; no label use and no external tuning."""
    from dataclasses import replace
    profiles={}
    for h in sorted(set(match.half)-{0}):
        half=np.where(match.half==h,match.half,0)
        profiles[int(h)]=profile_match(replace(match,half=half),include_sequences=False)
    reports=[]
    for a,p in profiles.items():
        for b,q in profiles.items():
            if a!=b:
                comparison=compare_profiles(p,q,q)
                # No intervention in this check: a reduction relative to itself is inapplicable.
                comparison['gates'].pop('aggregate_reduction')
                comparison['failed_gates']=[k for k,g in comparison['gates'].items() if not g['passed']]
                comparison['passed']=not comparison['failed_gates']
                reports.append({'fit_half':a,'audit_half':b,'comparison':comparison})
    return {'status':'tested' if reports else 'insufficient_halves','directions':reports}


def baseline_diagnostic(match, labels=None):
    """Organizer nearest-player rule and actual smoothing functions, no competition scorer/GT."""
    from starter.baseline import smooth, VARIANTS
    dist=np.sum(((match.P[...,:2]-match.B[:,None,:2])*match.pitch)**2,axis=-1)
    dist[~np.isfinite(dist)]=np.inf
    idx=dist.argmin(1); codes=np.where(idx<24,0,1)
    codes[~np.isfinite(dist.min(1))]=-1
    result={}
    for name,w in VARIANTS:
        pred=np.full(match.n_frames,'D',dtype='<U1')
        for a,b in segments(match.half): pred[a:b]=smooth(codes[a:b],w)
        active=match.half>0
        counts={k:int(np.sum(pred[active]==k)) for k in ('H','A','D')}
        entry={'counts':counts,'transitions':sum(int(np.sum(pred[a+1:b]!=pred[a:b-1])) for a,b in segments(match.half))}
        if labels is not None:
            valid=active & np.isin(labels,[0,1,2]); y=np.array(['H','A','D'])[np.clip(labels,0,2)]
            f1=[]
            for k in ('H','A','D'):
                tp=np.sum(valid&(pred==k)&(y==k)); den=np.sum(valid&(pred==k))+np.sum(valid&(y==k))
                f1.append(float(2*tp/den) if den else 0.)
            entry.update(n_labels=int(valid.sum()),macro_f1=float(np.mean(f1)))
        result[name]=entry
    return result


def quality_report(matches, target, params, seeds=FIXED_SEEDS, include_baselines=True):
    from .degrade import degrade_match
    rows=[]
    for m in matches:
        clean=profile_match(m,include_sequences=False)
        labels=m.extras.get('y_true',m.extras.get('y_draft')) if include_baselines else None
        base=baseline_diagnostic(m,labels) if include_baselines else None
        for seed in seeds:
            sample=degrade_match(m,target,params,int(seed)); degraded=profile_match(sample.observations,include_sequences=False)
            comparison=compare_profiles(target,clean,degraded)
            if comparison['gates']['aggregate_reduction']['proven_unattainable_under_current_support']:
                sample.audit['limitations'].append('Unavailable source speeds alone exceed the aggregate acceptance ceiling; gate remains failed.')
            if params.get('preset','quality')!='quality':
                comparison['passed']=False
                comparison['failed_gates'].append('stress_preset_excluded_from_quality_match')
            rows.append({'match':m.name,'seed':int(seed),'comparison':comparison,
                         'clean':clean,'degraded':degraded,'limitations':sample.audit['limitations'],
                         'baseline_clean':base,'baseline_degraded':baseline_diagnostic(sample.observations,labels) if include_baselines else None})
    per_match={}
    for name in sorted({r['match'] for r in rows}):
        group=[r for r in rows if r['match']==name]
        distances=[r['comparison']['aggregate_degraded'] for r in group]
        per_match[name]={'n_seeds':len(group),'passed':all(r['comparison']['passed'] for r in group),
                         'aggregate_clean':group[0]['comparison']['aggregate_clean'],
                         'aggregate_degraded_mean':float(np.mean(distances)),
                         'aggregate_degraded_sd':float(np.std(distances,ddof=1)) if len(group)>1 else 0.,
                         'failed_gates':sorted({g for r in group for g in r['comparison']['failed_gates']})}
    return {'passed':bool(rows) and all(r['comparison']['passed'] for r in rows),
            'include_baselines':bool(include_baselines), 'per_match_summary':per_match,
            'profile_hash':target['profile_hash'],'seeds':list(seeds),'rows':rows,
            'across_seeds':{'aggregate_clean_mean':float(np.mean([r['comparison']['aggregate_clean'] for r in rows])),
                            'aggregate_degraded_mean':float(np.mean([r['comparison']['aggregate_degraded'] for r in rows])),
                            'aggregate_degraded_sd':float(np.std([r['comparison']['aggregate_degraded'] for r in rows],ddof=1)) if len(rows)>1 else 0},
            'target_metrics':target['metrics'],'baseline_role':'Secondary only; never used by calibration. Metrica y_draft is weak supervision.',
            'synthetic_assumptions':['Synthetic camera and within-team fragment identities; excess counts interpreted as duplicates.'],
            'limitations':['Does not establish measurement-error variance or eliminate football-style / label-definition OOD.']}
