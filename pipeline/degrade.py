"""Source-aware observation corruption, applied to complete halves before window extraction."""
from dataclasses import dataclass
import copy
import numpy as np

from .schema import Match
from .quality import (VERSION, digest, fit_target_profile, profile_match, segments,
                      centroid, observation_hash, restart_frames, PRE_RESTART_SECONDS, RESTART_POST_SECONDS)


@dataclass
class DegradedSample:
    observations: Match
    targets: dict
    audit: dict


def epoch_seed(base_seed, epoch, match_name):
    """Stable across processes (unlike Python hash); use epoch=0 for frozen evaluation."""
    return int(digest([int(base_seed), int(epoch), match_name])[:16], 16)


REALIZATION_PURPOSES = ('train', 'dev', 'test')


def realization_seed(purpose, run, epoch, match_name):
    """Seed for one degraded realization. Purposes are namespaced so a training epoch can never
    reproduce a frozen development realization (and vice versa). `epoch_seed` stays for old artifacts."""
    if purpose not in REALIZATION_PURPOSES:
        raise ValueError('purpose must be one of %s' % (REALIZATION_PURPOSES,))
    return int(digest(['realization-v1', purpose, int(run), int(epoch), match_name])[:16], 16)


def _value(profile, name, default=0):
    value = profile['metrics'][name]['value']
    return default if value is None else value


def _durations(profile, key, default):
    data = profile['runs'][key]
    values = np.asarray(data['seconds'])[~np.asarray(data['censored'],bool)]
    return values[values > 0] if len(values) else np.array([default])


def _resample(profile, half, fps, rng, block_seconds):
    seq = profile['sequences']; target_half = np.asarray(seq['half'])
    choices = segments(target_half)
    if not choices: raise ValueError('Target profile has no active frames')
    index = np.zeros(len(half),int); seams = np.zeros(len(half),bool)
    block = max(1,round(block_seconds*fps))
    for a,b in segments(half):
        for start in range(a,b,block):
            lo,hi = choices[int(rng.integers(len(choices)))]
            size = min(block,b-start)
            span = max(1,int(np.ceil(size*profile['fps']/fps)))
            offset = int(rng.integers(lo,max(lo+1,hi-span+1)))
            index[start:start+size] = np.minimum(hi-1,offset+(np.arange(size)*profile['fps']/fps).astype(int))
            seams[start] = True
    return {k:np.asarray(seq[k],float)[index] for k in
            ('counts','camera_residual','camera_extent','camera_valid')}, seams


def _remap(P, half, rng, fragment_frames):
    """Sample complete occupancy lifetimes, not a hard cutoff at their median."""
    out = np.full_like(P, np.nan)
    provenance = np.full(P.shape[:2], -1, np.int16)
    valid = np.isfinite(P[..., :2]).all(-1)
    for a, b in segments(half):
        for lo, hi in ((0, 24), (24, 46)):
            intervals = []
            for slot in range(lo, hi):
                changes = np.diff(np.r_[False, valid[a:b, slot], False].astype(np.int8))
                for start, end in zip(np.flatnonzero(changes == 1)+a, np.flatnonzero(changes == -1)+a):
                    start = int(start)
                    while start < end:
                        stop = min(int(end), start+max(1, int(rng.choice(fragment_frames))))
                        intervals.append((start, stop, slot))
                        start = stop
            until = np.full(hi-lo, a-1, int)
            previous_source = np.full(hi-lo, -1, int)
            for start, end, slot in sorted(intervals):
                free = np.flatnonzero(until <= start)
                if not len(free):
                    raise RuntimeError('Slot allocator collision')
                # Do not splice two different players into consecutive frames of one slot.
                safe = free[(until[free] < start) | (previous_source[free] == slot)]
                if not len(safe):
                    safe = free  # Genuine full-capacity constraint, retained in provenance.
                dest = int(rng.choice(safe))
                until[dest], previous_source[dest] = end, slot
                out[start:end, lo+dest] = P[start:end, slot]
                provenance[start:end, lo+dest] = slot
    return out, provenance


def _persistent_selection(valid, wanted, score, half):
    """Change the visible set only on count/source-availability transitions."""
    selected = np.zeros_like(valid)
    for a, b in segments(half):
        changes = np.flatnonzero((wanted[a+1:b] != wanted[a:b-1]) |
                                 np.any(valid[a+1:b] != valid[a:b-1], axis=1)) + a+1
        cuts = np.r_[a, changes, b]
        keep = np.zeros(valid.shape[1], bool)
        for start, end in zip(cuts[:-1], cuts[1:]):
            keep &= valid[start]
            excess = int(keep.sum()-wanted[start])
            if excess > 0:
                candidates = np.flatnonzero(keep)
                keep[candidates[np.argsort(score[start, candidates])[-excess:]]] = False
            elif excess < 0:
                candidates = np.flatnonzero(valid[start] & ~keep)
                keep[candidates[np.argsort(score[start, candidates])[:-excess]]] = True
            selected[start:end] = keep
    return selected


def _allocate_frames(total, weights, minimum=0):
    """Integer allocation preserving an exact total and relative weights."""
    weights = np.asarray(weights, float)
    if not len(weights):
        return np.array([], int)
    minimum = min(minimum, total//len(weights))
    remaining = total-minimum*len(weights)
    allocation = remaining*weights/weights.sum()
    result = np.floor(allocation).astype(int)+minimum
    remainder = total-int(result.sum())
    if remainder:
        result[np.argsort(allocation-np.floor(allocation))[-remainder:]] += 1
    return result


def _stratified_durations(values, count, rng):
    # Small numbers of long episodes otherwise have high coverage/quantile variance.
    q = (np.arange(count)+rng.random(count))/count
    return rng.permutation(np.quantile(values, q))


def _ball_dropout(valid, half, counts, cen, profile, params, fps, rng, restarts=None):
    """A fixed additional-missingness budget; conditioning changes placement, not mass."""
    mask = np.zeros(len(valid), bool)
    active = half > 0
    target = _value(profile, 'ball_visible')
    available = int(np.sum(valid & active))
    # Restart-anchored dropout: the target loses the ball around stoppages far more often than elsewhere
    # (ATD: ~0.3 visible in the 8 s before a restart vs ~0.6 overall). For an exact share of the restarts
    # (chosen without replacement, so no binomial noise) the whole pre-restart window is hidden by one
    # gap whose length is drawn from the target's own missing-run distribution and which ends a few
    # seconds after the anchor (tag/event timing is noisy). Whole-window gaps keep the missing-run
    # statistics intact, unlike short blocks. All windows are protected from the generic dropout and the
    # preset frames are charged to the SAME total budget (overall visibility is unchanged).
    # Only provider-event restart times are used, never labels. Old profiles / restart_strength=0: no-op.
    protect = np.zeros(len(valid), bool)
    metric = profile['metrics'].get('ball_visible_pre_restart') or {}
    strength_r = params.get('restart_strength', 0)
    if strength_r and restarts is not None and len(restarts) and metric.get('value') is not None:
        width = max(1, round(PRE_RESTART_SECONDS*fps))
        post = max(1, round(RESTART_POST_SECONDS*fps))
        spans = segments(half)
        gap_frames = _durations(profile, 'ball_missing_runs', 1)*fps*params.get('duration_scale', 1)
        eligible, shares = [], []
        for a in sorted({int(x) for x in restarts}):
            lo = a-width
            if lo < 0 or a >= len(half) or half[a] <= 0 or not (half[lo:a] == half[a]).all():
                continue
            protect[lo:a] = True
            eligible.append(a)
            shares.append(float(valid[lo:a].mean()))
        if eligible:
            frac = float(np.clip(strength_r*(1-metric['value']/max(np.mean(shares), 1e-9)), 0, 1))
            n_hit = int(round(frac*len(eligible)))
            hits = rng.choice(np.array(eligible), n_hit, replace=False) if n_hit else []
            # Stratified quantiles (as in the generic dropout) keep the heavy tail of gap lengths stable.
            lengths = _stratified_durations(gap_frames, len(hits), rng) if n_hit else []
            for a, drawn in zip(hits, lengths):
                a = int(a)
                seg_lo, seg_hi = next((x, y) for x, y in spans if x <= a < y)
                end = min(seg_hi, a+int(rng.integers(0, post+1)))
                length = max(int(drawn), end-(a-width))
                start = max(seg_lo, end-length)
                mask[start:end] |= valid[start:end]
    preset = int(mask.sum())
    budget = max(0, round((available-target*active.sum())*params.get('ball_strength', 1))-preset)
    budget = min(budget, available)
    if not budget:
        return mask, {'requested_frames': 0, 'applied_frames': int(mask.sum()), 'extended_gap_frames': 0,
                      'restart_preset_frames': preset}
    blocks = segments(half)
    capacity = np.array([max(0, valid[a:b].sum()-target*(b-a)) for a,b in blocks])
    if not capacity.sum():
        return mask, {'requested_frames': budget, 'applied_frames': int(mask.sum()), 'extended_gap_frames': 0,
                      'restart_preset_frames': preset}
    budgets = _allocate_frames(budget, capacity)
    missing = _durations(profile, 'ball_missing_runs', 1)*fps*params.get('duration_scale', 1)
    visible = _durations(profile, 'ball_visible_runs', 1)*fps
    extended = 0
    for (a,b), amount in zip(blocks, budgets):
        amount = min(int(amount), int(valid[a:b].sum()))
        # Use a sparse source's small budget to extend existing short gaps toward target
        # quantiles before introducing more gaps. Never fill, shorten, or merge source gaps.
        changes = np.diff(np.r_[False, ~valid[a:b], False].astype(np.int8))
        gaps = [(int(lo+a),int(hi+a)) for lo,hi in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1))
                if lo>0 and hi<b-a]
        if gaps:
            lengths = np.array([hi-lo for lo,hi in gaps])
            ranks = np.argsort(np.argsort(lengths))
            goals = np.quantile(missing,(ranks+.5)/len(gaps))
            # Random order avoids systematically consuming all mass at the start of a half.
            for i in rng.permutation(len(gaps)):
                lo,hi = gaps[i]
                need = min(amount,max(0,int(round(goals[i]-lengths[i]))))
                if not need:
                    continue
                left,right = lo-1,hi
                for _ in range(need):
                    # Require one remaining visible neighbour to prevent joining gaps.
                    can_left = left>a and valid[left] and not mask[left] and not protect[left] and valid[left-1] and not mask[left-1]
                    can_right = right<b-1 and valid[right] and not mask[right] and not protect[right] and valid[right+1] and not mask[right+1]
                    if not (can_left or can_right):
                        break
                    if can_left and (not can_right or rng.random()<.5):
                        mask[left]=True; left-=1
                    else:
                        mask[right]=True; right+=1
                    amount-=1; extended+=1
        if amount <= 0:
            continue
        ids = np.flatnonzero(valid[a:b] & ~mask[a:b] & ~protect[a:b])+a
        amount = min(amount,len(ids))
        visible_budget = len(ids)-amount
        n = max(1,round(amount/max(float(missing.mean()),1)))
        n = min(n,amount,max(1,visible_budget-1))
        hidden_lengths = _allocate_frames(amount,_stratified_durations(missing,n,rng),minimum=1)
        visible_weights = _stratified_durations(visible,n+1,rng)
        # Conditional rates guide the spacing of episodes, with a final normalization that
        # preserves the requested number of newly hidden source-visible frames exactly.
        positions = ids[np.linspace(0,len(ids)-1,n+1).astype(int)]
        for i,t in enumerate(positions):
            low,high = next((l,h) for l,h in ((0,12),(12,18),(18,23),(23,47)) if l<=counts[t]<h)
            region = int(np.clip(cen[t,0]*3,0,2)) if np.isfinite(cen[t,0]) else 1
            cond = _value(profile,f'ball_given_count_{low}_{high}_region_{region}',target)
            propensity = np.clip((1-cond)/max(1-target,.01),.25,4)
            visible_weights[i] /= propensity**params.get('ball_condition_strength',1)
        visible_lengths = _allocate_frames(visible_budget,visible_weights,minimum=1)
        cursor = 0
        for i,length in enumerate(hidden_lengths):
            cursor += visible_lengths[i]
            mask[ids[cursor:cursor+length]] = True
            cursor += length
    return mask, {'requested_frames':budget,'applied_frames':int(mask.sum()),
                  'extended_gap_frames':extended,'restart_preset_frames':preset}


def _noise(shape, half, rng, amplitude):
    out=np.zeros(shape)
    if amplitude<=0: return out
    for a,b in segments(half):
        z=rng.normal(size=(b-a,)+shape[1:])
        # Finite moving average gives correlated perturbations and resets at each half.
        for lag,weight in ((0,1.),(1,.75),(2,.5),(3,.25)):
            if b-a>lag: out[a+lag:b] += z[:b-a-lag]*weight
    return out*amplitude/np.sqrt(1+.75**2+.5**2+.25**2)


def degrade_match(match, profile, params, seed):
    if match.source not in ('dfl','metrica'): raise ValueError('Only DFL and Metrica are supported')
    if profile.get('version') != VERSION: raise ValueError('Unsupported target profile')
    if params.get('profile_hash') != profile['profile_hash']: raise ValueError('Parameters belong to another target profile')
    if params.get('source') != match.source: raise ValueError('Parameters belong to another source')
    preset=params.get('preset','quality')
    if preset not in ('quality','stress_wrong_team','stress_jumps'): raise ValueError('Unknown preset')
    rng=np.random.default_rng(seed); fps=float(match.meta['fps']); half=match.half
    active=half>0; pv=np.isfinite(match.P[...,:2]).all(-1); bv=np.isfinite(match.B[:,:2]).all(-1)
    P=match.P.copy(); B=match.B.copy(); P[~pv]=np.nan; B[~bv]=np.nan
    seq,seams=_resample(profile,half,fps,rng,params.get('block_seconds',30))
    cen=centroid(P); residual=seq['camera_residual'].copy()
    # Blend each resampling seam using only residuals; source movement remains intact.
    blend=max(1,round(fps))
    for a,b in segments(half):
        for s in np.flatnonzero(seams[a:b])+a:
            if s==a or not np.isfinite(residual[s-1]).all(): continue
            e=min(s+blend,b); w=np.linspace(0,1,e-s)[:,None]
            residual[s:e]=residual[s-1]*(1-w)+residual[s:e]*w
    camera=cen+residual
    camera_valid=active & seq['camera_valid'].astype(bool) & np.isfinite(camera).all(1)
    camera[~camera_valid]=np.nan
    extent=seq['camera_extent'].copy(); extent[~camera_valid]=np.nan
    # Desired counts and camera geometry are a SINGLE dropout budget, not two masks.
    kept=pv.copy(); duplicates=np.zeros(pv.shape,bool)
    origin=np.broadcast_to(np.arange(46,dtype=np.int16),pv.shape).copy()
    source_counts=np.stack([pv[:,:24].sum(1),pv[:,24:].sum(1)],1)
    for team,(lo,hi) in enumerate(((0,24),(24,46))):
        source_mean=source_counts[active,team].mean() if active.any() else 0
        target_mean=_value(profile,'team1_count' if team==0 else 'team2_count')
        strength=params.get('player_strength',1) if source_mean>target_mean else 0
        wanted=np.rint(source_counts[:,team]+strength*(seq['counts'][:,team]-source_counts[:,team])).astype(int)
        wanted=np.minimum(source_counts[:,team],np.maximum(0,wanted))
        # Slow random preferences make disappearance persistent; geometry modulates their rank.
        score=np.zeros((len(P),hi-lo))
        hold=max(1,round(params.get('player_hold_seconds',2)*fps))
        for a,b in segments(half):
            keys=rng.random(((b-a+hold-1)//hold,hi-lo))
            score[a:b]=np.repeat(keys,hold,axis=0)[:b-a]
        scale=np.where(np.isfinite(extent)&(extent>0),extent,.8)
        relative=np.abs((P[:,lo:hi,:2]-camera[:,None,:])/scale[:,None,:]).sum(-1)
        score+=params.get('geometry_weight',.5)*np.nan_to_num(relative,nan=0)
        score[~pv[:,lo:hi]]=np.inf
        kept[:,lo:hi] = _persistent_selection(pv[:,lo:hi], wanted, score, half)
    kept[~active]=False; P[~kept]=np.nan
    # Duplicates are a synthetic interpretation of excess counts, never a measured duplicate rate.
    # They copy a currently observed source track; no source gap is filled with a fabricated position.
    for a,b in segments(half):
        block=max(1,round(fps))
        for s in range(a,b,block):
            e=min(s+block,b)
            if seq['counts'][s].sum()<=22: continue
            for team,(lo,hi) in enumerate(((0,24),(24,46))):
                need=max(0,int(seq['counts'][s,team])-max(11,int(kept[s,lo:hi].sum())))
                available=np.flatnonzero(kept[s:e,lo:hi].all(0))+lo
                free=np.flatnonzero((~pv[s:e,lo:hi]).all(0))+lo
                for dest,src in zip(rng.permutation(free)[:need],rng.choice(available,min(need,len(free)),replace=True) if len(available) else []):
                    P[s:e,dest]=P[s:e,src]; duplicates[s:e,dest]=True; origin[s:e,dest]=src
    source_visible=float(bv[active].mean()) if active.any() else 0
    target_visible=_value(profile,'ball_visible')
    count=np.isfinite(P[...,:2]).all(-1).sum(1)
    restarts=restart_frames(match)
    ball_mask, ball_budget = _ball_dropout(bv, half, count, centroid(P), profile, params, fps, rng, restarts)
    B[ball_mask | ~active]=np.nan
    # Conditional speed missingness is separately calibrated; never synthesize speed.
    speed_masks={}
    for label,array,clean in (('player',P,match.P),('ball',B,match.B)):
        valid=np.isfinite(array[...,:2]).all(-1)
        cv=np.isfinite(clean[...,:2]).all(-1)
        use=cv & active.reshape((-1,)+(1,)*(cv.ndim-1))
        existing=float((~np.isfinite(clean[...,2][use])).mean()) if use.any() else 1
        target=_value(profile,label+'_speed_missing',existing)
        probability=max(0,(target-existing)/max(1-existing,1e-12))
        mask=(rng.random(valid.shape)<probability)&valid
        array[...,2][mask]=np.nan; speed_masks[label]=mask
    # Roughness budget is frozen from training. Per-match guard prevents worsening rougher sources.
    noise_amplitudes={}
    for label,array in (('player',P),('ball',B)):
        amplitude=params.get(label+'_noise_m',0)
        if amplitude:
            xy=(match.P if label=='player' else match.B)[...,:2]*match.pitch
            residuals=np.linalg.norm(xy[1:-1]-(xy[:-2]+xy[2:])/2,axis=-1)
            same=(half[:-2]>0)&(half[:-2]==half[1:-1])&(half[1:-1]==half[2:])
            residuals=residuals[same]; finite=residuals[np.isfinite(residuals)]
            target=profile['metrics'][label+'_roughness']['quantiles']
            if not len(finite) or not target or np.median(finite)>=target[50]: amplitude=0
        noise_amplitudes[label]=float(amplitude)
        array[...,:2]+=_noise(array[...,:2].shape,half,rng,amplitude)/match.pitch
    if 'fragment_seconds' in params:
        fragment_frames=np.array([max(1,round(params['fragment_seconds']*fps))])
    else:
        fragment_frames=np.maximum(1,np.rint(_durations(profile,'slot_occupancy_runs',30)*fps*params.get('fragment_scale',1))).astype(int)
    P,mapping=_remap(P,half,rng,fragment_frames)
    # Remap duplicate provenance back to the source slot that supplied its coordinates.
    rows,cols=np.where(mapping>=0)
    duplicate_out=np.zeros_like(duplicates)
    duplicate_out[rows,cols]=duplicates[rows,mapping[rows,cols]]
    mapping[rows,cols]=origin[rows,mapping[rows,cols]]
    stress_mask=np.zeros(P.shape[:2],bool)
    if preset=='stress_wrong_team':
        for a,b in segments(half):
            for s in range(a,b,max(1,round(10*fps))):
                if rng.random()<.1:
                    e=min(b,s+round(fps)); P[s:e,[0,24]]=P[s:e,[24,0]]
                    mapping[s:e,[0,24]]=mapping[s:e,[24,0]]
                    duplicate_out[s:e,[0,24]]=duplicate_out[s:e,[24,0]]; stress_mask[s:e,[0,24]]=True
    if preset=='stress_jumps':
        for a,b in segments(half):
            for s in range(a,b,max(1,round(30*fps))):
                if rng.random()<.2:
                    e=min(b,s+round(fps)); P[s:e,:,:2]+=.1; stress_mask[s:e]=True
    labels={k:np.array(v,copy=True) for k,v in match.extras.items()
            if k in ('y_true','y_draft','touch_true') or k.endswith('_valid') and k.startswith(('y_','touch_'))}
    targets={'labels':labels,'label_valid':{k:np.array(match.extras.get(k+'_valid', np.isfinite(v)&(v>=0)&active),copy=True) for k,v in labels.items() if not k.endswith('_valid')},
             'ball':match.B.copy(),'ball_valid':np.isfinite(match.B).copy(),
             'ball_coordinate_valid':bv.copy()}
    obs=Match(match.name,match.source,P,B,half.copy(),
              {'fps':fps,'pitch':list(match.pitch),'coords':'normalized 0-1',
               'camera_kind':'synthetic'},
              None if match.graded is None else match.graded.copy(),
              {'camera_center':camera,'camera_valid':camera_valid,'camera_extent':extent,
               **({'restart_frames':restarts} if len(restarts) else {})})
    limitations=[]
    if source_visible<target_visible: limitations.append('Ball coverage already below target; no additional ball dropout.')
    for team,key in enumerate(('team1_count','team2_count')):
        if active.any() and source_counts[active,team].mean()<_value(profile,key):
            limitations.append(f'Team {team+1} mean coverage already below target; no count dropout.')
    for label,clean in (('player',match.P),('ball',match.B)):
        if not np.isfinite(clean[...,2]).any(): limitations.append(label+' speed unavailable; never synthesized.')
    audit={'version':VERSION,'seed':int(seed),'profile_hash':profile['profile_hash'],'config_hash':digest(params),
           'source_hash':observation_hash(match),'preset':preset,'limitations':limitations,
           'player_keep_before_remap':kept,'ball_dropout':ball_mask,'ball_budget':ball_budget,'speed_masks':speed_masks,
           'slot_source':mapping,'duplicate':duplicate_out,'resampling_seams':seams,'stress_mask':stress_mask,
           'coordinate_noise_m':noise_amplitudes,
           'synthetic_assumptions':['Camera residual block resampling; ranked camera-relative dropout.',
                                    'Within-team fragment reassignment; excess counts interpreted as limited duplicates.']}
    return DegradedSample(obs,targets,audit)


def calibrate_degradation(source_matches, target_profile, seed, *, progress=None):
    """Exactly 64 global candidates + 16 refinements, using explicitly supplied training matches."""
    from .quality_report import compare_profiles
    matches=list(source_matches)
    if not matches or len({m.source for m in matches})!=1: raise ValueError('Select training matches from exactly one source')
    if any(m.source not in ('dfl','metrica') for m in matches): raise ValueError('Unsupported source')
    clean=[profile_match(m,include_sequences=False) for m in matches]
    rng=np.random.default_rng(seed)
    base={'version':VERSION,'source':matches[0].source,'profile_hash':target_profile['profile_hash'],'preset':'quality',
          'player_strength':1.,'ball_strength':1.,'duration_scale':1.,'block_seconds':30.,
          'player_hold_seconds':2.,'geometry_weight':.5,
          'fragment_scale':1.,'ball_condition_strength':1.,'restart_strength':1.}
    for label in ('player','ball'):
        q=target_profile['metrics'][label+'_roughness']['quantiles']
        source_q=[c['metrics'][label+'_roughness']['quantiles'] for c in clean]
        med=np.mean([v[50] for v in source_q if v]) if any(source_q) else float('inf')
        base[label+'_noise_m']=max(0,q[50]-med) if q else 0
    best=None; history=[]
    bounds={'player_strength':(.7,1.3),'ball_strength':(.95,1.05),'duration_scale':(.8,1.2),
            'fragment_scale':(1.,5.),'ball_condition_strength':(0.,1.5),
            'block_seconds':(10.,90.),'player_hold_seconds':(.5,10.),'geometry_weight':(0.,2.),
            'restart_strength':(0.,1.)}
    for i in range(80):
        candidate=copy.deepcopy(base if i<64 or best is None else best[1])
        if i:
            for key,(lo,hi) in bounds.items():
                candidate[key]=float(rng.uniform(lo,hi) if i<64 else np.clip(candidate[key]+rng.normal(0,(hi-lo)*.08),lo,hi))
            for label in ('player','ball'): candidate[label+'_noise_m']=base[label+'_noise_m']*float(rng.uniform(0,2))
        reports=[]
        for j,(m,c) in enumerate(zip(matches,clean)):
            out=degrade_match(m,target_profile,candidate,epoch_seed(seed,0,m.name))
            d=profile_match(out.observations,include_sequences=False)
            reports.append(compare_profiles(target_profile,c,d))
        score=float(np.mean([r['objective_degraded'] for r in reports]))
        history.append({'candidate':i,'distance':score,'passed':all(r['passed'] for r in reports)})
        # Feasibility first: an aesthetically lower average cannot displace a candidate
        # satisfying more of the unchanged engineering gates.
        failures=sum(len(r['failed_gates']) for r in reports)
        rank=(failures,score)
        if best is None or rank<best[3]: best=(score,candidate,reports,rank)
        if progress is not None: progress(i+1,80,best[0])
    result=best[1]
    result['calibration']={'seed':int(seed),'budget':{'global':64,'local':16},
                           'training_matches':[{'name':m.name,'hash':c['input_hash']} for m,c in zip(matches,clean)],
                           'selection':'fewest failed gates, then equal-family distance',
                           'passed':all(r['passed'] for r in best[2]),'reports':best[2],'history':history}
    return result
