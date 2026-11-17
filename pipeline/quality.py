"""Label-free tracking quality profiling. No EDA, events, or ground-truth imports."""
import hashlib
import json
from pathlib import Path

import numpy as np

VERSION = 'tracking-quality-2'
QUANTILES = np.linspace(0, 1, 101)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def observation_hash(m):
    h = hashlib.sha256()
    for a in (m.P, m.B, m.half, m.graded, m.extras.get('H'), m.extras.get('H_valid'),
              m.extras.get('camera_center'), m.extras.get('camera_valid'), m.extras.get('camera_extent')):
        if a is not None:
            a = np.ascontiguousarray(a)
            h.update(str((a.shape, a.dtype)).encode()); h.update(a.tobytes())
    h.update(json.dumps({'pitch': m.pitch, 'fps': m.meta['fps'],
                         'image_size': m.meta.get('image_size')}, sort_keys=True).encode())
    return h.hexdigest()


def segments(half, valid=None):
    """Contiguous valid intervals; never bridge half 0, gaps, or half changes."""
    valid = (half > 0) if valid is None else ((half > 0) & valid)
    cuts = np.r_[0, np.flatnonzero((half[1:] != half[:-1]) | (valid[1:] != valid[:-1])) + 1, len(half)]
    return [(int(a), int(b)) for a, b in zip(cuts[:-1], cuts[1:]) if b > a and valid[a]]


def runs(x, half, state=True):
    """(seconds-independent frame lengths, boundary-censored flags)."""
    lengths, censored = [], []
    for a, b in segments(half):
        cuts = np.r_[a, np.flatnonzero(x[a+1:b] != x[a:b-1]) + a + 1, b]
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            if bool(x[lo]) == state:
                lengths.append(hi-lo); censored.append(lo == a or hi == b)
    return np.asarray(lengths, float), np.asarray(censored, bool)


PRE_RESTART_SECONDS = 8
RESTART_POST_SECONDS = 4      # synthetic gaps may extend this far past an anchor (timing of tags/events is noisy)


def restart_frames(m):
    """Label-free restart anchors: target Smart Tagging `SET PIECES` starts, or provider set-piece events.

    DFL event frames are already lag-corrected (`io_dfl.event_frames`); tag starts are a noisy proxy for
    the kick (the tags last ~35 s), so consumers use a window of several seconds, never a single frame.
    """
    e = m.extras
    if 'restart_frames' in e:
        return np.asarray(e['restart_frames'], dtype=np.int64)
    if 'tbl_tag_code' in e:
        return np.asarray(e['tbl_tag_start_frame'])[np.asarray(e['tbl_tag_code']) == 'SET PIECES'].astype(np.int64)
    if 'tbl_ev_setpiece' in e:
        return np.asarray(e['tbl_ev_frame'])[np.asarray(e['tbl_ev_setpiece']).astype(bool)].astype(np.int64)
    if 'tbl_ev_type' in e:
        return np.asarray(e['tbl_ev_start_frame'])[np.asarray(e['tbl_ev_type']) == 'SET PIECE'].astype(np.int64)
    return np.zeros(0, np.int64)


def pre_restart_mask(anchors, half, fps, seconds=PRE_RESTART_SECONDS):
    """Union of the `seconds` before each anchor; windows that would cross a half boundary are dropped."""
    mask = np.zeros(len(half), bool)
    width = max(1, round(seconds * fps))
    for a in anchors:
        a = int(a)
        lo = a - width
        if lo < 0 or a >= len(half) or half[a] <= 0 or not (half[lo:a] == half[a]).all():
            continue
        mask[lo:a] = True
    return mask


def centroid(P):
    ok = np.isfinite(P[..., :2]).all(-1)
    count = ok.sum(1)
    return np.divide(np.where(ok[..., None], P[..., :2], 0).sum(1), count[:, None],
                     out=np.full((len(P), 2), np.nan), where=count[:, None] > 0)


def camera_geometry(m):
    """Inverse homography of image centre/corners, in normalized pitch coordinates."""
    n = m.n_frames
    if 'camera_center' in m.extras:
        c = np.array(m.extras['camera_center'], copy=True)
        v = np.asarray(m.extras['camera_valid'], bool) & np.isfinite(c).all(1)
        extent = np.array(m.extras.get('camera_extent', np.full((n, 2), np.nan)), copy=True)
        c[~v] = np.nan
        return c, v, extent
    c, extent = np.full((n, 2), np.nan), np.full((n, 2), np.nan)
    if 'H' not in m.extras:
        return c, np.zeros(n, bool), extent
    # The shipped tactical video is documented in DATA.md. Other inputs must declare resolution.
    size = m.meta.get('image_size')
    if size is None and m.name == 'BAR-ATH_tactical':
        size = (1920, 1080)
    if size is None:
        raise ValueError('Homography profiling requires meta["image_size"] = [width, height]')
    H = m.extras['H']
    valid = np.asarray(m.extras.get('H_valid', np.ones(n, bool)), bool) & np.isfinite(H).all((1, 2))
    ids = np.flatnonzero(valid)
    if len(ids):
        ids = ids[np.abs(np.linalg.det(H[ids])) > 1e-12]
        w, h = size
        points = np.array([[w/2, h/2, 1], [0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]])
        q = np.einsum('nij,kj->nki', np.linalg.inv(H[ids]), points)
        with np.errstate(divide='ignore', invalid='ignore'):
            xy = q[..., :2] / q[..., 2:]
        c[ids] = xy[:, 0]
        # A footprint crossing the projective horizon has no finite bounded extent.
        bounded = np.all(q[:, 1:, 2] > 1e-10, axis=1) | np.all(q[:, 1:, 2] < -1e-10, axis=1)
        extent[ids[bounded]] = np.ptp(xy[bounded, 1:], axis=1)
    valid = np.isfinite(c).all(1)
    return c, valid, extent


def convert_camera_center(center, pitch, flip=False):
    """Future View geometry: flip BOTH axes in normalized space, then scale to metres."""
    xy = np.asarray(center).copy()
    return (1 - xy if flip else xy) * np.asarray(pitch)


def _metric(values, family, kind='distribution', units='native', bootstrap_units=None):
    x = np.asarray(values, float).ravel()
    x = x[np.isfinite(x)]
    if not len(x):
        return {'family': family, 'kind': kind, 'n': 0, 'value': None, 'ci95': None,
                'uncertainty': None, 'scale': None, 'quantiles': [], 'samples': [], 'units': units}
    n, exact_mean = len(x), float(x.mean())
    extremes = [float(x.min()), float(x.max())]
    if n > 8192:
        x = x[np.random.default_rng(971).choice(n, 8192, replace=False)]
    x = np.sort(x)
    q = np.quantile(x, QUANTILES)
    rng = np.random.default_rng(817)
    if kind == 'rate':
        # Resample block totals AND valid denominators. An almost-empty conditional block
        # must not have the same weight as a fully observed block.
        units_array = (np.asarray(bootstrap_units, float) if bootstrap_units is not None
                       else np.column_stack((x, np.ones(len(x)))))
        if len(units_array):
            indices = rng.integers(len(units_array), size=(128, len(units_array)))
            totals = units_array[indices].sum(axis=1)
            reps = totals[:, 0] / totals[:, 1]
            # Centre at the statistic of the SAME empirical population, never a sketch mean.
            center = units_array[:, 0].sum() / units_array[:, 1].sum()
            uncertainty = float(np.quantile(np.abs(reps-center), .95))
            ci = np.quantile(reps, [.025, .975]).tolist()
        else:
            ci, uncertainty = None, None
        samples = []
    else:
        # Equal-mass empirical bootstrap, with the same support size as the stored sketch.
        # Unlike endpoint quadrature, one extreme retains mass 1/n rather than ~0.5%.
        draws = rng.choice(x, (128, len(x)), replace=True)
        draws.sort(axis=1)
        uncertainty = float(np.quantile(np.abs(draws-x).mean(axis=1), .95))
        ci = np.quantile(draws.mean(axis=1), [.025, .975]).tolist()
        samples = x.tolist()
    return {'family': family, 'kind': kind, 'n': int(n), 'value': exact_mean,
            'ci95': ci, 'uncertainty': uncertainty,
            'scale': float(max(q[75]-q[25], np.std(x), 1e-3)),
            'quantiles': q.tolist(), 'samples': samples, 'extremes': extremes, 'units': units}


def _block_totals(x, half, fps):
    out = []
    step = max(1, round(30*fps))
    for a, b in segments(half):
        for start in range(a, b, step):
            z = np.asarray(x[start:min(b, start+step)])
            z = z[np.isfinite(z)]
            if z.size:
                out.append((float(z.sum()), int(z.size)))
    return out


def profile_match(m, *, include_sequences=True):
    fps = float(m.meta['fps']); active = m.half > 0
    if m.graded is not None: active &= m.graded
    half = np.where(active, m.half, 0)
    pv = np.isfinite(m.P[..., :2]).all(-1); bv = np.isfinite(m.B[:, :2]).all(-1)
    counts = np.stack([pv[:, :24].sum(1), pv[:, 24:].sum(1)], axis=1)
    total = counts.sum(1); cen = centroid(m.P)
    cam, cv, extent = camera_geometry(m)
    metrics, run_data = {}, {}
    def rate(name, x, family):
        x = np.asarray(x, float).copy(); x[~active] = np.nan
        metrics[name] = _metric(x, family, 'rate', bootstrap_units=_block_totals(x, half, fps))
    def dist(name, x, family, units='native'):
        metrics[name] = _metric(x, family, units=units)
    def durations(name, x, family, state=True):
        lens, cens = runs(x, half, state)
        lens /= fps
        run_data[name] = {'seconds': lens.tolist(), 'censored': cens.tolist(),
                          'n_complete': int((~cens).sum()), 'n_censored': int(cens.sum())}
        dist(name, np.log1p(lens[~cens]), family, 'log1p_seconds')
    rate('ball_visible', bv, 'ball')
    # Dead-ball cue: ball visibility in the seconds before restarts (None value when a source has none).
    rate('ball_visible_pre_restart', np.where(pre_restart_mask(restart_frames(m), m.half, fps), bv, np.nan), 'ball')
    durations('ball_missing_runs', bv, 'ball', False)
    durations('ball_visible_runs', bv, 'ball')
    long_mask = np.zeros(len(bv), bool)
    for a, b in segments(half, ~bv):
        if b-a >= 10*fps: long_mask[a:b] = True
    rate('ball_long_gap_share', np.where(~bv, long_mask, np.nan), 'ball')
    rate('player_count', total, 'players')
    dist('team1_count', counts[active, 0], 'players'); dist('team2_count', counts[active, 1], 'players')
    # Joint histogram is retained separately; projections capture team-count dependence in distance.
    dist('team_count_difference', (counts[:, 0]-counts[:, 1])[active], 'players')
    rate('excess_count', total > 22, 'players')
    durations('low_coverage_runs', total < 16, 'players')
    for seconds in (1, 5, 30):
        lag = max(1, round(seconds*fps)); pairs = []
        for a, b in segments(half):
            if b-a <= lag: continue
            # Window-local correlations supply uncertainty and prevent bridging boundaries.
            for s in range(a, b-lag, max(lag+1, round(60*fps))):
                e = min(b-lag, s+round(60*fps))
                x, y = total[s:e], total[s+lag:e+lag]
                if len(x)>2 and np.std(x)>0 and np.std(y)>0: pairs.append(np.corrcoef(x,y)[0,1])
        metrics[f'count_acf_{seconds}s'] = _metric(pairs, 'players', 'rate')
    for name, values, valid in [('player', m.P, pv), ('ball', m.B, bv)]:
        miss = np.where(valid, ~np.isfinite(values[..., 2]), np.nan)
        if miss.ndim == 2:
            denom = valid.sum(1)
            miss = np.divide(np.nansum(miss, 1), denom, out=np.full(len(miss), np.nan), where=denom>0)
        rate(name+'_speed_missing', miss, 'speed')
        xy = values[..., :2] * np.asarray(m.pitch)
        adj = np.linalg.norm(np.diff(xy, axis=0), axis=-1)
        ok = valid[1:] & valid[:-1]
        same = active[1:] & active[:-1] & (half[1:] == half[:-1])
        ok &= same.reshape((-1,) + (1,)*(ok.ndim-1))
        dist(name+'_displacement', adj[ok], 'motion', 'metres_per_frame')
        residual = np.linalg.norm(xy[1:-1] - (xy[:-2]+xy[2:])/2, axis=-1)
        ok3 = valid[:-2] & valid[1:-1] & valid[2:]
        ok3 &= (same[:-1]&same[1:]).reshape((-1,) + (1,)*(ok3.ndim-1))
        dist(name+'_roughness', residual[ok3], 'motion', 'metres')
    for state, label in [(True, 'slot_occupancy_runs'), (False, 'slot_absence_runs')]:
        ls, cs = [], []
        for slot in range(46):
            l,c = runs(pv[:,slot], half,state); ls.extend(l/fps); cs.extend(c)
        ls, cs = np.asarray(ls), np.asarray(cs, bool)
        run_data[label] = {'seconds':ls.tolist(), 'censored':cs.tolist(), 'n_complete':int((~cs).sum()), 'n_censored':int(cs.sum())}
        dist(label,np.log1p(ls[~cs]),'slots','log1p_seconds')
    churn = np.full(len(pv), np.nan)
    if len(pv)>1:
        same = active[1:] & active[:-1] & (half[1:] == half[:-1])
        churn[1:] = np.where(same, (pv[1:] != pv[:-1]).sum(1), np.nan)
    rate('slot_churn', churn, 'slots')
    # Indicator only, NOT an identity-switch or absolute measurement-error estimate.
    jumps = np.linalg.norm(np.diff(m.P[..., :2]*m.pitch, axis=0), axis=-1)
    if len(pv)>1:
        rate('slot_discontinuity', np.r_[np.nan, np.where(same, np.sum(jumps>3,axis=1), np.nan)], 'slots')
    for low, high in [(0,12),(12,18),(18,23),(23,47)]:
        for region in range(3):
            cond = (total>=low)&(total<high)&np.isfinite(cen[:,0])&(np.clip((cen[:,0]*3).astype(float),0,2)//1 == region)
            rate(f'ball_given_count_{low}_{high}_region_{region}', np.where(cond,bv,np.nan), 'coupling')
    has_camera = 'H' in m.extras or 'camera_center' in m.extras
    rate('camera_valid', cv if has_camera else np.full(len(cv), np.nan), 'camera')
    residual = cam-cen
    for axis in range(2):
        dist(f'camera_residual_{axis}',residual[active & cv,axis],'camera')
        dist(f'camera_extent_{axis}',extent[active & cv,axis],'camera')
        dist(f'camera_ball_offset_{axis}',(cam-m.B[:,:2])[active & cv & bv,axis],'camera')
    delta = np.linalg.norm(np.diff(cam,axis=0),axis=-1)
    same = active[1:] & active[:-1] & (half[1:] == half[:-1]) & cv[1:] & cv[:-1]
    dist('camera_change',delta[same],'camera')
    joint = np.zeros((25,23),int)
    np.add.at(joint,(counts[active,0],counts[active,1]),1)
    p = {'version':VERSION,'name':m.name,'source':m.source,'input_hash':observation_hash(m),
         'fps':fps,'n_active':int(active.sum()),'metrics':metrics,'runs':run_data,
         'joint_team_counts':joint.tolist(),
         'uncertainty_method':'128 deterministic bootstrap replicates; weighted 30s block totals/denominators for rates; equal-mass empirical bootstrap for distributions. Censored runs excluded; dependence can understate uncertainty.',
         'interpretation':['Slot occupancy is not identity-switch frequency.',
                           'Excess count is not an identified duplicate rate.',
                           'Motion tails and off-pitch coordinates are not necessarily errors.',
                           'Camera-ball offset is conditional on visible ball and may be biased.']}
    if include_sequences:
        # Full empirical sequences stay in ignored profile artifacts; no labels or clean identities.
        p['sequences'] = {'half':half.tolist(),'counts':counts.tolist(),'ball_visible':bv.tolist(),
                          'centroid':_finite_list(cen),'camera_residual':_finite_list(residual),
                          'camera_extent':_finite_list(extent),'camera_valid':cv.tolist()}
    return p


def _finite_list(a):
    a = np.asarray(a,object); a[~np.isfinite(np.asarray(a,float))] = None
    return a.tolist()


def fit_target_profile(target_match):
    p = profile_match(target_match)
    p['profile_hash'] = digest(p)
    return p


def save_profile(profile, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, allow_nan=False, separators=(',',':')))


def load_profile(path):
    p = json.loads(Path(path).read_text())
    if p['version'] != VERSION: raise ValueError('Unsupported quality profile version')
    expected = p.get('profile_hash')
    if expected and digest({k:v for k,v in p.items() if k != 'profile_hash'}) != expected:
        raise ValueError('Profile hash mismatch')
    return p
