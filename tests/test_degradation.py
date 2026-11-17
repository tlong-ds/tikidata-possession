import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

from pipeline.schema import Match, load_match_file
from pipeline.quality import (fit_target_profile, profile_match, runs, camera_geometry,
                              convert_camera_center, save_profile, load_profile)
from pipeline.degrade import degrade_match, calibrate_degradation, epoch_seed
from pipeline.quality_report import compare_profiles, quality_report
from pipeline.degradation_cli import export_sample, main


def fixture(n=400,source='dfl',target=False):
    t=np.arange(n); P=np.full((n,46,3),np.nan); B=np.ones((n,3))*.5
    for slot in list(range(11))+list(range(24,35)):
        P[:,slot,0]=.15+(slot%24)*.06+.001*np.sin(t/10)
        P[:,slot,1]=.2+(slot%5)*.1; P[:,slot,2]=2
    B[:,0]+=.001*np.sin(t/8); B[:,2]=3
    half=np.where(t<n//2,1,2).astype(np.int8)
    extras={'y_true':(t%3).astype(np.int8),'touch_true':(t%4).astype(np.int8),
            'slot_person':np.ones((n,46)),'Bz':np.ones(n),'tbl_ev_secret':np.array([123])}
    if target:
        B[t%20<8]=np.nan
        P[t%40<20,8:11]=np.nan; P[t%40<20,32:35]=np.nan
        B[t%7==0,2]=np.nan
        extras.update(camera_center=np.column_stack((.5+.01*np.sin(t/10),np.full(n,.5))),
                      camera_valid=np.ones(n,bool),camera_extent=np.ones((n,2))*.9)
    return Match('fixture',source,P,B,half,{'fps':10,'pitch':[105,68],'persons':['secret']},extras=extras)


def parameters(profile,source='dfl'):
    return {'source':source,'profile_hash':profile['profile_hash'],'player_strength':1.,
            'ball_strength':1.,'duration_scale':1.,'fragment_seconds':3.,'preset':'quality'}


class DegradationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.profile=fit_target_profile(fixture(target=True))

    def test_immutable_and_separated(self):
        m=fixture(); before=copy.deepcopy(m); s=degrade_match(m,self.profile,parameters(self.profile),17)
        np.testing.assert_equal(m.P,before.P); np.testing.assert_equal(m.B,before.B)
        np.testing.assert_equal(s.targets['ball'],m.B)
        for k in ('y_true','touch_true'): np.testing.assert_equal(s.targets['labels'][k],m.extras[k])
        self.assertEqual(set(s.observations.extras),{'camera_center','camera_valid','camera_extent'})
        self.assertNotIn('persons',s.observations.meta)
        s.targets['ball'][0]=999; np.testing.assert_equal(m.B,before.B)

    def test_reproducible_and_variable(self):
        m=fixture(); p=parameters(self.profile)
        a=degrade_match(m,self.profile,p,1); b=degrade_match(m,self.profile,p,1); c=degrade_match(m,self.profile,p,2)
        np.testing.assert_equal(a.observations.P,b.observations.P)
        np.testing.assert_equal(a.observations.B,b.observations.B)
        self.assertFalse(np.array_equal(a.audit['slot_source'],c.audit['slot_source']))
        self.assertEqual(epoch_seed(7,1,'a'),epoch_seed(7,1,'a'))
        self.assertNotEqual(epoch_seed(7,1,'a'),epoch_seed(7,2,'a'))

    def test_provenance_team_gaps_and_fields(self):
        m=fixture(); m.P[20:40,0]=np.nan; m.B[10:30]=np.nan
        s=degrade_match(m,self.profile,parameters(self.profile),3)
        rows,dests=np.where(s.audit['slot_source']>=0); src=s.audit['slot_source'][rows,dests]
        np.testing.assert_equal(src<24,dests<24)
        self.assertTrue(np.isfinite(m.P[rows,src,:2]).all())
        np.testing.assert_equal(s.observations.P[rows,dests,:2],m.P[rows,src,:2])
        self.assertTrue(np.isnan(s.observations.B[10:30]).all())
        for a in (s.observations.P,s.observations.B):
            self.assertTrue(np.isnan(a[~np.isfinite(a[...,:2]).all(-1)]).all())

    def test_censoring_and_boundaries(self):
        lengths,censored=runs(np.array([0,0,1,0,0,0],bool),np.array([1,1,1,2,2,2]),False)
        np.testing.assert_equal(lengths,[2,3]); self.assertTrue(censored.all())
        m=fixture(); m.half[150:250]=0
        s=degrade_match(m,self.profile,parameters(self.profile),4)
        self.assertTrue(np.isnan(s.observations.P[150:250]).all())
        self.assertFalse(s.observations.extras['camera_valid'][150:250].any())
        np.testing.assert_equal(s.observations.half,m.half)

    def test_short_all_missing_and_empty(self):
        for n in (0,1,2,10):
            m=fixture(n); m.P[:]=np.nan; m.B[:]=np.nan
            s=degrade_match(m,self.profile,parameters(self.profile),7)
            self.assertTrue(np.isnan(s.observations.P).all()); self.assertTrue(np.isnan(s.observations.B).all())
            self.assertFalse(s.observations.extras['camera_valid'].any())
            profile_match(s.observations,include_sequences=False)

    def test_exhausted_slots(self):
        m=fixture(60); m.P[:]=.5
        s=degrade_match(m,self.profile,{**parameters(self.profile),'player_strength':0},5)
        self.assertEqual(np.isfinite(s.observations.P[...,0]).sum(),60*46)
        self.assertFalse(s.audit['duplicate'].any())

    def test_sparse_source_no_additional_ball_mask(self):
        m=fixture(source='metrica'); m.B[:300]=np.nan; m.P[:,:,2]=np.nan; m.B[:,2]=np.nan
        s=degrade_match(m,self.profile,parameters(self.profile,'metrica'),8)
        self.assertFalse(s.audit['ball_dropout'].any())
        np.testing.assert_equal(s.observations.B,m.B)
        self.assertTrue(np.isnan(s.observations.P[:,:,2]).all())
        dense=degrade_match(fixture(),self.profile,parameters(self.profile),8)
        self.assertTrue(dense.audit['ball_dropout'].any())

    def test_camera_geometry_conversion(self):
        m=fixture(10); m.meta['image_size']=[1920,1080]
        m.extras={'H':np.tile(np.diag([1920.,1080.,1]),(10,1,1)),'H_valid':np.ones(10,bool)}
        m.extras['H_valid'][2]=False; m.extras['H'][3]=0
        c,v,e=camera_geometry(m)
        np.testing.assert_allclose(c[0],[.5,.5]); np.testing.assert_allclose(e[0],[1,1])
        self.assertFalse(v[2]); self.assertFalse(v[3])
        np.testing.assert_allclose(convert_camera_center([[.2,.3]],[100,60],True),[[80,42]])

    def test_known_distributions_and_recovery(self):
        target=fixture(4000,target=True); p=fit_target_profile(target)
        self.assertAlmostEqual(p['metrics']['ball_visible']['value'],.6)
        lens=p['runs']['ball_missing_runs']; complete=np.asarray(lens['seconds'])[~np.asarray(lens['censored'])]
        np.testing.assert_allclose(complete,.8)
        clean=fixture(4000)
        s=degrade_match(clean,p,parameters(p),17)
        d=profile_match(s.observations,include_sequences=False)
        self.assertLess(abs(d['metrics']['ball_visible']['value']-.6),.05)
        self.assertLess(abs(d['metrics']['player_count']['value']-19),1)

    def test_labels_cannot_change_profile_or_corruption(self):
        m=fixture(); other=copy.deepcopy(m); other.extras['y_true'][:]=99
        self.assertEqual(profile_match(m),profile_match(other))
        a=degrade_match(m,self.profile,parameters(self.profile),8)
        b=degrade_match(other,self.profile,parameters(self.profile),8)
        np.testing.assert_equal(a.observations.P,b.observations.P)

    def test_config_validation_and_hash_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'p.json'; save_profile(self.profile,p)
            self.assertEqual(load_profile(p),self.profile)
            obj=json.loads(p.read_text()); obj['fps']=99; p.write_text(json.dumps(obj))
            with self.assertRaises(ValueError): load_profile(p)
        with self.assertRaises(ValueError): degrade_match(fixture(),self.profile,parameters(self.profile,'metrica'),1)

    def test_unavailable_camera_is_not_a_failure_observation(self):
        p=profile_match(fixture(),include_sequences=False)
        self.assertEqual(p['metrics']['camera_valid']['n'],0)

    def test_failed_gates(self):
        m=fixture(); c=profile_match(m,include_sequences=False)
        report=compare_profiles(self.profile,c,c)
        self.assertFalse(report['passed']); self.assertIn('ball_visible',report['failed_gates'])

    def test_export_separation_and_no_overwrite(self):
        s=degrade_match(fixture(),self.profile,parameters(self.profile),1)
        with tempfile.TemporaryDirectory() as tmp, patch('pipeline.degradation_cli.CACHE_DIR',tmp):
            path=Path(tmp)/'degradation'/'sample'; export_sample(s,path)
            m=load_match_file(path/'observations.npz')
            self.assertNotIn('y_true',m.extras)
            with np.load(path/'targets.npz') as z: np.testing.assert_equal(z['labels__y_true'],s.targets['labels']['y_true'])
            with self.assertRaises(FileExistsError): export_sample(s,path)

    def test_duplicate_provenance(self):
        target=fixture(80,target=True); target.P[:]=.5
        p=fit_target_profile(target); m=fixture(80)
        sample=degrade_match(m,p,parameters(p),7)
        self.assertTrue(sample.audit['duplicate'].any())
        rows,cols=np.where(sample.audit['duplicate']); src=sample.audit['slot_source'][rows,cols]
        np.testing.assert_equal(sample.observations.P[rows,cols,:2],m.P[rows,src,:2])
        np.testing.assert_equal(cols<24,src<24)

    def test_existing_validity_masks_preserved(self):
        m=fixture(); m.extras['y_true_valid']=np.arange(m.n_frames)%2==0
        sample=degrade_match(m,self.profile,parameters(self.profile),7)
        np.testing.assert_equal(sample.targets['label_valid']['y_true'],m.extras['y_true_valid'])

    def test_motion_does_not_bridge_half(self):
        m=fixture(10); m.P[:5,:,:2]=0; m.P[5:,:,:2]=1
        m.B[:5,:2]=0; m.B[5:,:2]=1
        p=profile_match(m,include_sequences=False)
        self.assertEqual(p['metrics']['player_displacement']['value'],0)
        self.assertEqual(p['metrics']['ball_roughness']['value'],0)

    def test_offpitch_not_clipped(self):
        m=fixture(); m.P[:,:,0]-=2; m.B[:,0]+=2
        sample=degrade_match(m,self.profile,parameters(self.profile),7)
        self.assertLess(np.nanmax(sample.observations.P[:,:,0]),0)
        self.assertGreater(np.nanmin(sample.observations.B[:,0]),1)

    def test_stress_excluded_from_quality_claims(self):
        p={**parameters(self.profile),'preset':'stress_wrong_team'}
        report=quality_report([fixture(60)],self.profile,p,seeds=(17,))
        self.assertFalse(report['passed'])
        self.assertIn('stress_preset_excluded_from_quality_match',report['rows'][0]['comparison']['failed_gates'])

    def test_conditioning_cannot_change_ball_budget(self):
        from pipeline.degrade import _ball_dropout
        m=fixture(4000)
        valid=np.isfinite(m.B[:,:2]).all(1)
        valid[100:140]=False
        counts=np.where(np.arange(4000)%100<50,14,22)
        cen=np.full((4000,2),.5)
        for strength in (0.,1.,4.):
            mask,audit=_ball_dropout(valid,m.half,counts,cen,self.profile,
                                    {'ball_condition_strength':strength},10,np.random.default_rng(8))
            self.assertEqual(int(mask.sum()),audit['requested_frames'])
            self.assertFalse((mask & ~valid).any())
            self.assertAlmostEqual(np.mean(valid & ~mask),.6,places=3)

    def test_persistent_selection_does_not_shuffle_at_constant_count(self):
        from pipeline.degrade import _persistent_selection
        rng=np.random.default_rng(1)
        valid=np.ones((200,24),bool)
        selected=_persistent_selection(valid,np.full(200,11),rng.random((200,24)),np.ones(200))
        self.assertTrue((selected==selected[0]).all())
        self.assertEqual(selected[0].sum(),11)

    def test_empirical_fragment_tail_and_no_immediate_splice(self):
        from pipeline.degrade import _remap
        from pipeline.quality import runs
        m=fixture(1000)
        out,mapping=_remap(m.P,m.half,np.random.default_rng(12),np.array([5,5,5,300]))
        occupancy=np.isfinite(out[:,:,0])
        longest=max(max(runs(occupancy[:,slot],m.half)[0],default=0) for slot in range(46))
        self.assertGreater(longest,5)
        adjacent=(mapping[:-1]>=0)&(mapping[1:]>=0)&(m.half[:-1,None]==m.half[1:,None])
        self.assertTrue((mapping[:-1][adjacent]==mapping[1:][adjacent]).all())

    def test_wasserstein_retains_empirical_tail_mass(self):
        from pipeline.quality import _metric
        from pipeline.quality_report import _distance
        x=np.zeros(1000); y=x.copy(); y[-1]=1000
        a=_metric(x,'motion'); b=_metric(y,'motion')
        self.assertAlmostEqual(_distance(a,b),1.)
        self.assertEqual(b['extremes'],[0.,1000.])
        self.assertAlmostEqual(_distance(a,_metric(np.zeros(23),'motion')),0.)

    def test_weighted_block_bootstrap_centers_on_population(self):
        from pipeline.quality import _metric
        units=np.array([[0.,1000.]]*90+[[1.,1.]]*10)
        x=np.r_[np.zeros(90000),np.ones(10)]
        m=_metric(x,'coupling','rate',bootstrap_units=units)
        rng=np.random.default_rng(817)
        sums=units[rng.integers(100,size=(128,100))].sum(1)
        reps=sums[:,0]/sums[:,1]
        expected=np.quantile(np.abs(reps-x.mean()),.95)
        self.assertAlmostEqual(m['uncertainty'],expected)
        self.assertLess(m['uncertainty'],.001)

    def test_source_uncertainty_cannot_exempt_aggregate_gate(self):
        target=copy.deepcopy(self.profile)
        clean=profile_match(fixture(),include_sequences=False)
        for metric in target['metrics'].values(): metric['uncertainty']=0.
        for metric in clean['metrics'].values(): metric['uncertainty']=1000.
        report=compare_profiles(target,clean,clean)
        gate=report['gates']['aggregate_reduction']
        self.assertTrue(gate['outside_target_variability'])
        self.assertFalse(gate['passed'])
        self.assertEqual(gate['uncertainty_scope'],'target_only')

    def test_missing_speed_floor_does_not_relax_gate(self):
        m=fixture(); m.P[:,:,2]=np.nan; m.B[:,2]=np.nan
        clean=profile_match(m,include_sequences=False)
        report=compare_profiles(self.profile,clean,clean)
        gate=report['gates']['aggregate_reduction']
        self.assertGreater(gate['irreducible_speed_floor'],0)
        self.assertGreater(gate['irreducible_speed_floor'],gate['required_max_distance'])
        self.assertTrue(gate['proven_unattainable_under_current_support'])
        self.assertFalse(gate['passed'])

    def test_calibration_budget_and_training_selection(self):
        # Small full halves keep this integration test inexpensive; no reduced production budget.
        p=fit_target_profile(fixture(40,target=True))
        result=calibrate_degradation([fixture(40)],p,6)
        self.assertEqual(len(result['calibration']['history']),80)
        self.assertEqual(result['calibration']['budget'],{'global':64,'local':16})
        self.assertEqual(result['calibration']['training_matches'][0]['name'],'fixture')

if __name__=='__main__': unittest.main()
