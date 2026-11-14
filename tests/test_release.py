import json
from pathlib import Path
import tempfile
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
        self.assertEqual(report['real_R'],104.50394996162504)
        self.assertFalse(report['target_achieved'])
        self.assertFalse(report['evaluator_called'])

    def test_original_and_online_separation(self):
        for n in (0,1,19,20):
            x,y=training_data(n)
            self.assertEqual(len(y),2048+50*n)
            np.testing.assert_allclose(x.reshape(-1,3,7).sum(-1),1,atol=1e-12)
        self.assertEqual(len(records()),2048)

    def test_original_pretrain_and_all_model_hashes(self):
        from adaptive.learning import verify_pretrained
        from adaptive.base import load_ensemble
        x,y=training_data(0);verify_pretrained(x,y)
        for n in range(21):
            models=load_ensemble(RESULT/'models'/f'v{n:02d}')
            self.assertEqual(len(models),5)
            self.assertTrue(all(sum(p.numel() for p in m.parameters())==3692 for m in models))

    def test_transfer_bins_and_cross_building_pairs(self):
        from adaptive.acquisition import transfer_proposals,joint_pair,BINS,update_alpha
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

    def test_frozen_quota_exceptions_are_explicit(self):
        # Preserve actual history, including the late structured fallback.
        b=json.loads((RESULT/'rounds/19/candidates.json').read_text())
        self.assertEqual(len(b['weights']),50)
        self.assertEqual(len(set(weight_key(w) for w in b['weights'])),50)
        self.assertEqual(b['sources'][12],'transfer')
        self.assertEqual(b['metadata'][12]['selection'],'rotating')
        self.assertEqual(b['metadata'][12]['delta'],.002)

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

    def test_response_rejects_misalignment_and_nan(self):
        for x in ({'R':[1.]},{'R':[float('nan'),1.]},{'R':[True,1.]}):
            with self.assertRaises(ValueError):parse_response(x,2)

if __name__=='__main__':unittest.main()
