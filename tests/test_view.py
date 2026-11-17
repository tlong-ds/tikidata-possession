import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from pipeline.view import (resolve_geometry, transform_xy, make_view, transform_targets,
                           velocities, iter_windows, UnresolvedOrientation, fit_view_config,
                           speed_reference_m_s)
from pipeline.quality import digest, fit_target_profile
from pipeline.quality_report import quality_report
from pipeline.degradation_example import prepare_epoch, export_handoff
from pipeline.degradation_cli import main
from pipeline.validate_preprocessing import sha, verify_hashes
from test_degradation import fixture, parameters


def config():
    c={'window':5,'polyorder':2,'max_speed_m_s':30}
    return dict(c,config_hash=digest(c))


def oriented(n=400):
    m=fixture(n); m.meta['left_group']={'firstHalf':1,'secondHalf':2}
    return m


class ViewTests(unittest.TestCase):
    def test_speed_reference_conversion_no_mutation_or_double_conversion(self):
        m=oriented();m.P[:,:,2]=36.;m.P[0,0,2]=np.nan
        before=m.P.copy();speed,provenance=speed_reference_m_s(m)
        np.testing.assert_allclose(speed[1:],10.)
        self.assertTrue(np.isnan(speed[0,0]));np.testing.assert_equal(m.P,before)
        self.assertEqual(provenance['input_unit'],'km/h')
        m.meta['speed_unit']='m/s'
        np.testing.assert_equal(speed_reference_m_s(m)[0],m.P[:,:,2])
        m.meta['speed_unit']='unknown'
        with self.assertRaises(ValueError):speed_reference_m_s(m)

    def test_fit_uses_converted_reference(self):
        m=oriented(100);m.name='DFL-MAT-J03WMX'
        m.P[:,:,:2]=.5;m.P[:,:,0]+=np.arange(100)[:,None]/105/10
        m.P[:,:,2]=3.6
        target=fixture(target=True)
        # Target supplies a threshold above the 1 m/s calibration fixture;
        # avoid roundoff splits exactly at a constant-speed quantile boundary.
        target.P[:,:,0]+=np.arange(target.n_frames)[:,None]/105/5
        profile=fit_target_profile(target)
        with patch('pipeline.load.load_match',return_value=target):
            c=fit_view_config([m],profile)
        self.assertLess(min(r['mean_half_median_mae'] for r in c['candidates']),1e-10)
        self.assertEqual(c['speed_reference']['scale'],1/3.6)
        self.assertEqual(c['version'],'view-speed-units-2')

    def test_known_transform_inverse_non_square_both_directions(self):
        g=resolve_geometry(oriented()); xy=np.array([[.75,.25],[.75,.25]])
        h=np.array([1,2]); out=transform_xy(xy,h,g)
        np.testing.assert_allclose(out,[[26.25,-17],[-26.25,17]])
        np.testing.assert_allclose(transform_xy(out,h,g,inverse=True),xy)

    def test_shared_geometry_and_no_mutation_or_leak(self):
        m=oriented(); m.B[:,:2]=m.P[:,0,:2]
        m.extras['camera_center']=m.B[:,:2].copy();m.extras['camera_valid']=np.ones(m.n_frames,bool)
        old=copy.deepcopy(m);g=resolve_geometry(m);v=make_view(m,g,config())
        t=transform_targets({'ball':m.B,'labels':{'y_true':m.extras['y_true']}},m.half,g)
        np.testing.assert_equal(v.P[:,0,:2],v.B[:,:2]);np.testing.assert_equal(v.cam,v.B[:,:2])
        np.testing.assert_equal(t['ball_xy'],v.B[:,:2]);np.testing.assert_equal(t['labels']['y_true'],m.extras['y_true'])
        np.testing.assert_equal(m.P,old.P);np.testing.assert_equal(m.B,old.B)
        self.assertEqual(set(vars(v)),{'P','B','cam','camera_valid','half','frame_valid','graded','frame_indices','meta'})
        self.assertEqual(set(v.meta),{'fps','pitch','attack_sign','team_encoding'})
        self.assertTrue(np.all(v.P[:,:24,4]==0));self.assertTrue(np.all(v.P[:,24:,4]==1))

    def test_constant_velocity_and_stationary(self):
        xy=np.zeros((30,2,2));xy[:,0,0]=np.arange(30)*.2
        v=velocities(xy,np.ones(30),10,config())
        np.testing.assert_allclose(v[2:-2,0,0],2,atol=1e-12)
        np.testing.assert_allclose(v[2:-2,1],0,atol=1e-12)
        self.assertTrue(np.isnan(v[:2]).all());self.assertTrue(np.isnan(v[-2:]).all())

    def test_gaps_jumps_half_and_short_segments(self):
        xy=np.zeros((40,1,2));xy[10]=np.nan;xy[20:]=100
        half=np.ones(40);half[30:]=2
        v=velocities(xy,half,10,config())
        for a,b in ((8,13),(18,22),(28,32)):
            self.assertTrue(np.isnan(v[a:b]).all())
        self.assertTrue(np.isfinite(v[14:18]).all())
        self.assertTrue(np.isnan(velocities(xy[:4],half[:4],10,config())).all())
        np.testing.assert_equal(xy[20],[[100,100]])

    def test_missing_and_partial_coordinates(self):
        m=oriented();m.P[:]=np.nan;m.B[:]=np.nan;m.P[0,0,0]=.5
        v=make_view(m,resolve_geometry(m),config())
        self.assertTrue(np.isnan(v.P[:,:,:4]).all());self.assertTrue(np.all(v.P[:,:,5]==0))
        self.assertTrue(np.isnan(v.B[:,:2]).all());self.assertTrue(np.all(v.B[:,2]==0))
        self.assertFalse(v.camera_valid.any())

    def test_orientation_unresolved_and_override(self):
        m=oriented();m.meta.pop('left_group')
        with self.assertRaises(UnresolvedOrientation):resolve_geometry(m)
        g=resolve_geometry(m,{1:{'attack_sign':-1,'reason':'fixture kickoff'},2:{'attack_sign':1,'reason':'fixture kickoff'}})
        self.assertEqual(g['attack_sign'],{'1':-1,'2':1})
        with self.assertRaises(ValueError):resolve_geometry(m,{1:{'attack_sign':1}})

    def test_extrema_agreement_and_disagreement(self):
        m=oriented(1400);m.source='metrica';m.P[:,:,:2]=np.nan
        m.P[:,0,:2]=[.1,.5];m.P[:,1,:2]=[.4,.5]
        m.P[:,24,:2]=[.6,.5];m.P[:,25,:2]=[.9,.5]
        self.assertEqual(resolve_geometry(m)['attack_sign'],{'1':1,'2':1})
        m.P[:,1,0]=.95
        with self.assertRaises(UnresolvedOrientation):resolve_geometry(m)
        m.P[:]=np.nan
        with self.assertRaises(UnresolvedOrientation):resolve_geometry(m)

    def test_windows_single_realization_indices_and_reproducibility(self):
        m=oriented();p=fit_target_profile(fixture(target=True));params=parameters(p)
        a=prepare_epoch(m,p,params,config(),epoch=2);b=prepare_epoch(m,p,params,config(),epoch=2)
        np.testing.assert_equal(a[0].P,b[0].P)
        windows=list(iter_windows(a[0],50,25));self.assertTrue(windows)
        np.testing.assert_equal(windows[0].P[25:],windows[1].P[:25])
        np.testing.assert_equal(windows[0].frame_indices[25:],windows[1].frame_indices[:25])
        for w in windows:self.assertEqual(len(set(w.half)),1)
        self.assertNotEqual(a[2].audit['seed'],prepare_epoch(m,p,params,config(),epoch=3)[2].audit['seed'])

    def test_fit_rejects_noncalibration(self):
        with self.assertRaises(ValueError):fit_view_config([oriented()],{})

    def test_config_tampering(self):
        c=config();c['window']=7
        with self.assertRaisesRegex(ValueError,'hash'):prepare_epoch(oriented(),{}, {},c)

    def test_skip_baselines_api_and_cli_exit(self):
        m=oriented();p=fit_target_profile(fixture(target=True));params=parameters(p)
        with patch('pipeline.quality_report.baseline_diagnostic',side_effect=AssertionError('label scoring forbidden')):
            r=quality_report([m],p,params,seeds=(17,),include_baselines=False)
            self.assertIsNone(r['rows'][0]['baseline_clean']);self.assertFalse(r['include_baselines'])
            with tempfile.TemporaryDirectory() as d:
                path=Path(d);(path/'params.json').write_text(json.dumps(params))
                with patch('pipeline.degradation_cli.output_path',side_effect=lambda p:Path(p)),patch('pipeline.degradation_cli.load_profile',return_value=p),patch('pipeline.degradation_cli.load_match',return_value=m):
                    code=main(['report','--profile','unused','--params',str(path/'params.json'),'--matches','fixture','--out',str(path/'report.json'),'--skip-baselines'])
                self.assertEqual(code,0 if json.loads((path/'report.json').read_text())['passed'] else 2)

    def test_export_readback_and_hash_failure(self):
        from pipeline.schema import CACHE_DIR
        m=oriented();p=fit_target_profile(fixture(target=True));args=prepare_epoch(m,p,parameters(p),config())
        with tempfile.TemporaryDirectory(dir=Path(CACHE_DIR)/'degradation') as d:
            result=export_handoff(Path(d)/'handoff',*args,length=50,stride=25)
            self.assertTrue(result['readback_verified'])
            f=Path(d)/'file';f.write_text('frozen');manifest={'files':{str(f):sha(f)}}
            verify_hashes(manifest);f.write_text('changed')
            with self.assertRaises(ValueError):verify_hashes(manifest)


if __name__=='__main__': unittest.main()
