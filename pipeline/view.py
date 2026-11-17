"""Label-free, centered metric model interface. Geometry is resolved on raw caches."""
from dataclasses import dataclass, fields
import copy
import warnings
import numpy as np
import scipy
from scipy.signal import savgol_filter
from .quality import segments, camera_geometry, observation_hash, digest
from .schema import TEAM_OF_SLOT


class UnresolvedOrientation(ValueError):
    pass


@dataclass
class View:
    P: np.ndarray
    B: np.ndarray
    cam: np.ndarray
    camera_valid: np.ndarray
    half: np.ndarray
    frame_valid: np.ndarray
    graded: np.ndarray
    frame_indices: np.ndarray
    meta: dict


def resolve_geometry(match, overrides=None):
    """Overrides: {half: {'attack_sign': +/-1, 'reason': recorded evidence}}.

    Non-DFL: paired team extrema over opening 60s; 5s moving-block bootstrap,
    1000 replicates, seed 2027, 95% intervals. Require >=30s paired support
    and >=6 supported non-overlapping 5s blocks for each estimate.
    """
    overrides = {int(k): v for k, v in (overrides or {}).items()}
    result = {'pitch': list(match.pitch), 'fps': float(match.meta['fps']),
              'attack_sign': {}, 'evidence': {}}
    for h in sorted(set(match.half) - {0}):
        key = str(int(h))
        if int(h) in overrides:
            o = overrides[int(h)]
            if o.get('attack_sign') not in (-1, 1) or not o.get('reason'):
                raise ValueError('Orientation override requires sign and recorded reason')
            sign, evidence = int(o['attack_sign']), {'override': dict(o)}
        elif match.source == 'dfl':
            left = match.meta.get('left_group', {}).get({1: 'firstHalf', 2: 'secondHalf'}.get(int(h)))
            if left not in (1, 2):
                raise UnresolvedOrientation(f'{match.name} half {h}: absent left_group')
            sign, evidence = (1 if left == 1 else -1), {'left_group': left}
        else:
            fps = result['fps']; start = np.flatnonzero(match.half == h)[0]
            x = match.P[start:start + round(60*fps), :, 0].copy()
            x[match.half[start:start+len(x)] != h] = np.nan
            estimates = []
            for reducer in (np.nanmin, np.nanmax):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    d = reducer(x[:, 24:], axis=1) - reducer(x[:, :24], axis=1)
                block = max(1, round(5*fps))
                support = sum(np.isfinite(d[a:a+block]).sum() >= block/2 for a in range(0,len(d)-block+1,block))
                if np.isfinite(d).sum() < 30*fps or support < 6:
                    raise UnresolvedOrientation(f'{match.name} half {h}: insufficient opening extrema support')
                rng = np.random.default_rng(2027)
                boots = []
                for _ in range(1000):
                    starts = rng.integers(0, len(d)-block+1, size=int(np.ceil(len(d)/block)))
                    sample = np.concatenate([d[a:a+block] for a in starts])[:len(d)]
                    boots.append(np.nanmedian(sample))
                lo, hi = np.quantile(boots, [.025, .975])
                estimates.append({'median': float(np.nanmedian(d)), 'ci95': [float(lo),float(hi)],
                                  'sign': 1 if lo > 0 else -1 if hi < 0 else 0})
            if not estimates[0]['sign'] or estimates[0]['sign'] != estimates[1]['sign']:
                raise UnresolvedOrientation(f'{match.name} half {h}: extrema disagree or interval includes zero: {estimates}')
            sign, evidence = estimates[0]['sign'], {'extrema': estimates, 'bootstrap_seed': 2027, 'block_seconds': 5}
        result['attack_sign'][key] = sign
        result['evidence'][key] = evidence
    return result


def transform_xy(xy, half, geometry, inverse=False):
    xy = np.asarray(xy, float)
    signs = np.full(len(half), np.nan)
    for h in set(half) - {0}:
        sign = geometry['attack_sign'].get(str(int(h)))
        if sign not in (-1,1): raise UnresolvedOrientation(f'No geometry for half {h}')
        signs[half == h] = sign
    signs = signs.reshape((len(half),) + (1,)*(xy.ndim-1))
    pitch = np.asarray(geometry['pitch'])
    return xy*signs/pitch + .5 if inverse else (xy-.5)*pitch*signs


def velocity_segments(xy, half, fps, max_speed):
    for a,b in segments(half, np.isfinite(xy).all(-1)):
        cuts = np.r_[a, np.flatnonzero(np.linalg.norm(np.diff(xy[a:b],axis=0),axis=1)*fps > max_speed)+a+1, b]
        yield from zip(cuts[:-1],cuts[1:])


def velocities(xy, half, fps, config):
    out = np.full_like(xy, np.nan, dtype=float)
    w, p = config['window'], config['polyorder']; edge = w//2
    for slot in range(xy.shape[1]):
        for a,b in velocity_segments(xy[:,slot],half,fps,config['max_speed_m_s']):
            if b-a < w: continue
            v = savgol_filter(xy[a:b,slot],w,p,deriv=1,delta=1/fps,axis=0)
            out[a+edge:b-edge,slot] = v[edge:-edge]
    return out


def implied_speed_quantile(match):
    values = []
    half = match.half if match.graded is None else np.where(match.graded,match.half,0)
    for a,b in segments(half):
        v = np.linalg.norm(np.diff(match.P[a:b,:,:2],axis=0)*match.pitch,axis=-1)*match.meta['fps']
        values.append(v[np.isfinite(v)])
    values = np.concatenate(values) if values else np.array([])
    if not len(values): raise ValueError('No adjacent positions for threshold fit')
    return float(np.quantile(values,.999))


def speed_reference_m_s(match):
    """Convert a copy of DFL's raw XML S channel; never rescale cached arrays.

    Bassek et al., Scientific Data (2025), Position data:
    https://doi.org/10.1038/s41597-025-04505-y defines S in km/h.
    Legacy io_dfl caches copy S verbatim, despite the generic schema's m/s label.
    Explicit metadata supports future already-converted caches without double conversion.
    """
    if match.source != 'dfl':
        raise ValueError('Speed-reference calibration requires documented DFL units')
    unit = match.meta.get('speed_unit', 'km/h')
    if unit not in ('km/h', 'm/s'):
        raise ValueError('Unsupported declared speed unit: '+str(unit))
    scale = 1/3.6 if unit == 'km/h' else 1.
    return np.array(match.P[:,:,2], dtype=float, copy=True)*scale, {
        'input_unit': unit, 'output_unit': 'm/s', 'scale': scale,
        'declaration': 'metadata' if 'speed_unit' in match.meta else 'legacy io_dfl raw S contract',
        'source_doi': '10.1038/s41597-025-04505-y'}


def fit_view_config(training_matches, target_profile):
    """Frozen cycle: only J03WMX may fit settings. Target contributes label-free steps."""
    from .load import load_match
    matches = list(training_matches)
    if len(matches) != 1 or matches[0].name != 'DFL-MAT-J03WMX':
        raise ValueError('This calibration cycle requires J03WMX only')
    m = matches[0]; target = load_match(target_profile['name'])
    if observation_hash(target) != target_profile['input_hash']: raise ValueError('Target cache hash mismatch')
    qs = {'target': implied_speed_quantile(target), 'calibration': implied_speed_quantile(m)}
    reference, unit_provenance = speed_reference_m_s(m)
    config = {'version': 'view-speed-units-2', 'speed_reference': unit_provenance,
              'max_speed_m_s': max(qs.values()), 'speed_quantiles_999': qs}
    xy = (m.P[:,:,:2]-.5)*m.pitch
    candidates = [(w,p) for w in (5,7,11,15,21) for p in (2,3)]
    common = np.isfinite(reference) & (m.half[:,None]>0)
    # Establish identical eligible frames before computing scores; keep memory bounded.
    for w,p in candidates:
        v = velocities(xy,m.half,m.meta['fps'],dict(config,window=w,polyorder=p))
        common &= np.isfinite(v).all(-1)
    scores=[]
    for w,p in candidates:
        v = velocities(xy,m.half,m.meta['fps'],dict(config,window=w,polyorder=p))
        err = np.abs(np.linalg.norm(v,axis=-1)-reference)
        hs = [int(h) for h in sorted(set(m.half)-{0})]
        if any(not (common & (m.half[:,None]==h)).any() for h in hs):
            raise ValueError('No common eligible speed frames in a calibration half')
        medians = [float(np.median(err[common & (m.half[:,None]==h)])) for h in hs]
        scores.append({'window': w, 'polyorder': p, 'mean_half_median_mae': float(np.mean(medians)), 'half_medians':medians})
    winner = min(scores,key=lambda r:(r['mean_half_median_mae'],r['window'],r['polyorder']))
    config.update(window=winner['window'],polyorder=winner['polyorder'], candidates=scores,
                  common_eligible_count=int(common.sum()), training_matches=[m.name],
                  calibration_hash=observation_hash(m),target_profile_hash=target_profile['profile_hash'],
                  numpy_version=np.__version__,scipy_version=scipy.__version__)
    config['config_hash']=digest(config)
    return config


def make_view(observations, geometry, config):
    m = observations
    if list(m.pitch) != list(geometry['pitch']) or float(m.meta['fps']) != geometry['fps']:
        raise ValueError('Geometry does not match observation pitch/fps')
    pv = np.isfinite(m.P[:,:,:2]).all(-1) & (m.half[:,None]>0)
    bv = np.isfinite(m.B[:,:2]).all(-1) & (m.half>0)
    xy = transform_xy(m.P[:,:,:2],m.half,geometry); xy[~pv] = np.nan
    P = np.full((m.n_frames,46,6),np.nan)
    P[:,:,:2] = xy; P[:,:,2:4] = velocities(xy,m.half,m.meta['fps'],config)
    P[:,:,4] = TEAM_OF_SLOT; P[:,:,5] = pv
    B = np.column_stack((transform_xy(m.B[:,:2],m.half,geometry),bv)); B[~bv,:2] = np.nan
    cam, cv, _ = camera_geometry(m); cv &= m.half>0
    cam = transform_xy(cam,m.half,geometry); cam[~cv] = np.nan
    return View(P,B,cam,cv,m.half.copy(),m.half>0,
                # Sources without a published grading table (DFL, Metrica): every in-half frame counts.
                (m.half>0) if m.graded is None else m.graded.copy(),
                np.arange(m.n_frames), {'fps':geometry['fps'],'pitch':list(geometry['pitch']),
                                       'attack_sign':dict(geometry['attack_sign']), 'team_encoding':'0=group1,1=group2'})


def transform_targets(targets, half, geometry):
    out = copy.deepcopy(targets)
    out['ball_xy'] = transform_xy(targets['ball'][:,:2],half,geometry)
    out['ball_coordinate_valid'] = np.isfinite(out['ball_xy']).all(-1)
    # Raw ball and source speed remain only in the audit sample, not this handoff.
    out.pop('ball'); out.pop('ball_valid',None)
    return out


def iter_windows(view, length=250, stride=125):
    if length <= 0 or stride <= 0: raise ValueError('Positive length/stride required')
    for a,b in segments(view.half):
        for start in range(a,b-length+1,stride):
            yield View(**{f.name: (copy.deepcopy(view.meta) if f.name=='meta' else getattr(view,f.name)[start:start+length]) for f in fields(View)})
