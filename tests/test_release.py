import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch
import numpy as np
import torch
from adaptive.release import ROOT,RESULT,verify_report,verify_manifest,records,training_data
from adaptive.protocol import weight_key,parse_response

class FrozenRelease(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from adaptive.base import configure
        configure()

    def test_manifest_and_complete_transcript(self):
        report=verify_report()
        self.assertEqual(report['feedback_rows'],1000)
        self.assertEqual(report['real_R'],103.85699706606101)
        self.assertTrue(report['target_achieved'])
        self.assertEqual(report['role_totals'],{'exploit':710,'learn':290})
        self.assertAlmostEqual(report['trajectory_recomputed_R'],report['real_R'],places=9)
        self.assertFalse(report['evaluator_called'])

    def test_original_and_online_separation(self):
        for n in (0,1,19,20):
            x,y=training_data(n)
            self.assertEqual(len(y),2048+50*n)
            np.testing.assert_allclose(x.reshape(-1,3,7).sum(-1),1,atol=1e-12)
        self.assertEqual(len(records()),2048)

    def test_original_cold_start_and_all_model_hashes(self):
        from adaptive.learning import verify_original_data
        from adaptive.base import load_ensemble
        x,y=training_data(0);verify_original_data(x,y)
        for n in range(21):
            models=load_ensemble(RESULT/'models'/f'v{n:02d}')
            self.assertEqual(len(models),5)
            self.assertTrue(all(sum(p.numel() for p in m.parameters())==3692 for m in models))

    def test_transfer_bins_and_cross_building_pairs(self):
        from adaptive.acquisition_v1 import transfer_proposals,joint_pair,BINS,update_alpha
        w=np.tile([.999999,1e-6,0,0,0,0,0],(3,1)).reshape(21)
        distances=[]
        for b in BINS:
            x,meta=transfer_proposals(w,b,1.)
            self.assertTrue((x>=0).all());np.testing.assert_allclose(x.reshape(-1,3,7).sum(-1),1)
            distances.append(np.linalg.norm(x[0]-w))
        self.assertLess(distances[0],distances[1]);self.assertLess(distances[1],distances[2])
        u=np.full(21,1/7);noise=np.full((3,7),.001);noise[:,0]=-.003
        pair=joint_pair(u,(0,2),noise).reshape(2,3,7)
        np.testing.assert_allclose(pair[:,1],np.tile(u.reshape(3,7)[1],(2,1)))
        np.testing.assert_allclose(pair.mean(0),u.reshape(3,7),atol=1e-14)
        state=dict(radius=1.)
        update_alpha(state,np.arange(50),110,110)
        self.assertEqual(update_alpha(state,np.arange(50),110,110)['alpha_next'],2.)

    def test_branch_isolation_scaling_and_sum_before_abs(self):
        from adaptive.model import DistrictSurrogate
        x,_=training_data(0);m=DistrictSurrogate(100.,2026);w=torch.tensor(x[:4])
        self.assertTrue(torch.allclose(m(w),m(w*3),atol=1e-12,rtol=0))
        z=m.branch_features(w);changed=w.clone().reshape(-1,3,7);changed[:,0]=torch.roll(changed[:,0],1,-1)
        self.assertTrue(torch.equal(z[:,1:],m.branch_features(changed)[:,1:]))
        self.assertTrue(torch.equal(m(w),m.scale*(z.sum(1)+m.offset).abs().mean(-1)))

    def test_single_formal_budget_rejects_call21(self):
        from argparse import Namespace
        from adaptive.budget import claim_run,reserve_call,complete_call
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);args=Namespace(session=p/'one',test_epochs=None,resume=False)
            with patch('adaptive.budget.ROOT',p):
                claim_run(args)
                with self.assertRaises(RuntimeError):claim_run(args)
                for n in range(1,21):reserve_call(args,n,str(n));complete_call(args,n,str(n))
                with self.assertRaises(RuntimeError):reserve_call(args,21,'no')

    def test_observed_winner_and_polish_policy(self):
        b=json.loads((RESULT/'rounds/20/candidates.json').read_text())
        self.assertEqual(b['sources'][48],'exploit_final_scan')
        self.assertEqual(b['metadata'][48]['selection'],'observed_improvement')
        self.assertEqual(b['metadata'][48]['delta'],.0005)
        from adaptive.anchor import update_feedback
        state={};previous=min(r['R'] for r in records())
        for number in range(1,21):
            folder=RESULT/'rounds'/f'{number:02d}'
            batch=json.loads((folder/'candidates.json').read_text())
            scores=json.loads((folder/'stdout.json').read_text())['R']
            self.assertEqual(batch['phase'],'feedback_triggered_polish' if state.get('polish_active') else 'v1_discovery')
            self.assertEqual(batch['anchor_used_for_selection'],bool(state.get('polish_active') and state.get('anchor_enabled')))
            best=min(previous,min(scores))
            transition=update_feedback(state,batch,scores,previous,best)
            saved=json.loads((folder/'feedback.json').read_text())['v3_transition']
            self.assertEqual(transition['anchor_enabled_next'],saved['anchor_enabled_next'])
            previous=best
        self.assertIsNone(state.get('polish_trigger_round'))
        self.assertFalse(state['polish_active'])
        self.assertFalse(state['anchor_enabled'])

    def test_line_scans_and_direction_identity(self):
        from adaptive.polling import moved,scan
        class Context:
            def shadow(self,points):
                return {'raw_score':np.asarray(points)[:,0],'anchor_score':np.asarray(points)[:,0]}
        x=np.full((1,21),1/7);y=np.array([100.])
        points,metadata=scan(Context(),x,y,17,[],{weight_key(x[0].tolist())})
        self.assertEqual(len(points),20)
        self.assertTrue((points>=0).all())
        np.testing.assert_allclose(points.reshape(-1,3,7).sum(-1),1.)
        self.assertTrue(all(m['delta']>=.0005-1e-14 for m in metadata))

    def test_training_smoke_preserves_original_rows_and_weight_cap(self):
        from adaptive import base
        x,y=training_data(0)
        with tempfile.TemporaryDirectory() as temp:
            models=base.train_ensemble(x[:64],y[:64],64,Path(temp)/'fit',1,1)
            self.assertEqual(len(models),5)
            report=json.loads((Path(temp)/'fit/training.json').read_text())
            self.assertEqual(report['online_rows'],0)
            with np.load(Path(temp)/'fit/sample_weights.npz',allow_pickle=False) as z:
                self.assertTrue((z['weights']<=3).all())
            self.assertTrue(np.isfinite(base.predict(models,x[:8])).all())

    def test_portable_access_audit_allows_source_but_blocks_archives(self):
        script="from pathlib import Path\nimport sys,tempfile\nfrom adaptive.audit import install,ROOT\nsys.argv=['guard','--session',tempfile.gettempdir()+'/public-audit-test']\ninstall('public-audit-test')\nopen(ROOT/'adaptive/data/train.csv','rb').close()\nfor path in (ROOT/'results/formal-v4/online.json',Path(tempfile.gettempdir())/'online-soft-tree/solver/sessions/blocked.json'):\n try:open(path,'rb')\n except PermissionError:continue\n raise RuntimeError('Archived input not blocked')\n"
        result=subprocess.run([sys.executable,'-c',script],cwd=ROOT,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_information_training_audits_are_exact_and_pre_feedback(self):
        from adaptive.training_audit import assess
        from adaptive.protocol import digest
        from adaptive.base import load_ensemble
        for after,count in ((4,20),(8,15),(12,15),(16,10)):
            folder=RESULT/'learning-audits'/f'after-r{after:02d}'
            report=json.loads((folder/'training.json').read_text())
            self.assertEqual(report['full_rows']-report['without_information_rows'],count)
            self.assertEqual(len(load_ensemble(folder)),1)
            self.assertEqual(digest(folder/'seed2026.npz'),report['models'][0]['sha256'])
            batch=json.loads((RESULT/f'rounds/{after+1:02d}/candidates.json').read_text())
            scores=json.loads((RESULT/f'rounds/{after+1:02d}/stdout.json').read_text())['R']
            expected=json.loads((folder/f'evaluation-on-r{after+1:02d}.json').read_text())
            self.assertEqual(assess(batch,scores),expected)
            comparison=batch['information_model_comparison']
            self.assertTrue(comparison['generated_before_feedback'])
            self.assertFalse(comparison['used_for_acquisition'])
            np.testing.assert_array_equal(comparison['full_predictions'],np.asarray(batch['predictions'])[:,0])

    def test_frozen_core_files_match_original_experiment(self):
        from adaptive.protocol import digest
        origin=json.loads((ROOT/'ORIGIN.json').read_text())
        for name,sha in origin['unchanged_algorithm_files'].items():
            self.assertEqual(digest(ROOT/'adaptive'/name),sha)

    def test_response_rejects_misalignment_and_nan(self):
        for x in ({'R':[1.]},{'R':[float('nan'),1.]},{'R':[True,1.]}):
            with self.assertRaises(ValueError):parse_response(x,2)

if __name__=='__main__':unittest.main()
