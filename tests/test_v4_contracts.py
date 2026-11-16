"""Zero-oracle validation of active roles, line identity and training diagnostics."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from adaptive import acquisition as acq
from adaptive.base import configure,train_ensemble
from adaptive.anchor import AnchorContext,update_feedback
from adaptive.information import splits,generate as info_batch,pair_pool
from adaptive.lines import identity,point_on_line,registry,boundary_candidates,transfer_metadata
from adaptive.protocol import atomic_json,weight_key,digest
from adaptive.resources import rank_cards
ROOT=Path(__file__).resolve().parents[1]


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure()
        with (ROOT/'adaptive/data/train.csv').open(encoding='utf-8-sig') as f:rows=sorted(csv.DictReader(f),key=lambda r:r['candidate_id'])
        names=json.loads((ROOT/'adaptive/recipe.json').read_text())['input_feature_order']
        cls.x=np.array([[float(r[k]) for k in names] for r in rows]);cls.y=np.array([float(r['ramping']) for r in rows])
        cls.temp=tempfile.TemporaryDirectory(prefix='v4-unit-')
        cls.models=train_ensemble(cls.x[:128],cls.y[:128],128,Path(cls.temp.name)/'fit',1,1)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_role_schedule_all20_rounds(self):
        expected={1:(30,20),6:(30,20),7:(35,15),14:(35,15),15:(40,10),19:(40,10),20:(50,0)}
        for n,pair in expected.items():self.assertEqual(splits(n)[:2],pair)
        self.assertEqual(sum(splits(n)[0] for n in range(1,21)),710)
        self.assertEqual(sum(splits(n)[1] for n in range(1,21)),290)
        for n in range(1,21):self.assertEqual(sum(splits(n)[2]),splits(n)[1])
        with self.assertRaises(ValueError):splits(21)

    def test_original_only_frozen_kernel_and_no_old_model(self):
        from adaptive.learning import verify_original_data
        verify_original_data(self.x,self.y)
        config=json.loads((ROOT/'adaptive/recipe.json').read_text())
        self.assertEqual(digest(ROOT/'adaptive/acquisition_v1.py'),config['frozen_v1_acquisition_sha256'])
        self.assertEqual(digest(ROOT/'adaptive/data/train.csv'),config['data_sha256'])
        self.assertFalse((ROOT/'pretrained_original').exists())

    def test_line_identity_context_and_scaling(self):
        w=np.full(21,1/7);key,base,total=identity(w,0,1,2)
        moved=point_on_line(w,0,1,2,.1)
        self.assertEqual(key,identity(moved,0,2,1)[0])
        self.assertEqual(key,identity(w*7,0,1,2)[0])
        changed=moved.reshape(3,7).copy();changed[1,0]+=.01;changed[1,1]-=.01
        self.assertNotEqual(key,identity(changed,0,1,2)[0])
        changed=moved.reshape(3,7).copy();changed[0,4]+=.01;changed[0,5]-=.01
        self.assertNotEqual(key,identity(changed,0,1,2)[0])
        self.assertAlmostEqual(total,2/7)

    def test_boundary_uses_only_real_matching_line_endpoints(self):
        x=self.x.copy();x[0]=np.full(21,1/7);y=self.y.copy();y[0]=100.
        w=point_on_line(x[0],0,1,2,1/7+.016)
        meta=transfer_metadata(x[0],w,0,100.,0,1,2)
        row=dict(id='r01-01',round=1,index=1,weights=w.reshape(3,7).tolist(),R=99.,acquisition=meta)
        xx=np.concatenate([x,w[None]]);yy=np.r_[y,99.]
        candidates=boundary_candidates(xx,yy,[row]);self.assertEqual(len(candidates),1)
        proposal,m=candidates[0]
        self.assertAlmostEqual(proposal[2],1/7+.008)
        self.assertEqual(m['observed_R_gap'],1.)
        self.assertNotIn('R',m)
        bad=dict(row,weights=(w.reshape(3,7)+np.array([[0]*7,[.01,-.01,0,0,0,0,0],[0]*7])).tolist())
        self.assertEqual(boundary_candidates(xx,yy,[bad]),[])
        future=dict(row,acquisition=dict(meta,center=2048))
        with self.assertRaises(RuntimeError):registry(xx,yy,[future])

    def test_information_pairs_atomic_feasible_and_matched(self):
        seen={weight_key(w.tolist()) for w in self.x};context=AnchorContext(self.models,self.x,self.y)
        pool=pair_pool(context,self.x,self.y,seen,1,np.random.default_rng(2026))
        self.assertGreater(len(pool),10)
        for points,meta in pool[:40]:
            self.assertTrue((points>=0).all());np.testing.assert_allclose(points.reshape(2,3,7).sum(-1),1.)
            self.assertNotEqual(weight_key(points[0].tolist()),weight_key(points[1].tolist()))
            self.assertEqual(identity(points[0],meta['building'],meta['i'],meta['j'])[0],meta['line_key'])
            self.assertEqual(identity(points[1],meta['building'],meta['i'],meta['j'])[0],meta['line_key'])
            self.assertGreaterEqual(meta['utility'],0.)

    def test_boundary_missing_explicitly_refills_competitive(self):
        seen={weight_key(w.tolist()) for w in self.x};selected=self.x[:30]
        points,sources,meta,report=info_batch(self.models,self.x,self.y,1,[],seen,self.x[:50],selected)
        self.assertEqual(len(points),20);self.assertEqual(sources.count('learn_pair'),12)
        self.assertEqual(sources.count('learn_boundary'),0);self.assertEqual(sources.count('learn_competitive'),8)
        self.assertEqual(report['boundary_shortfall'],4)
        pair_ids={m['pair_id'] for m in meta if 'pair_id' in m}
        self.assertEqual(len(pair_ids),6)
        self.assertTrue(all(sum(m.get('pair_id')==key for m in meta)==2 for key in pair_ids))
        self.assertFalse(any(weight_key(w.tolist()) in seen for w in points))

    def test_acquisition_roles_and_final_round_no_information(self):
        seen={weight_key(w.tolist()) for w in self.x}
        with tempfile.TemporaryDirectory() as temp:
            for n in (1,7,15,20):
                folder=Path(temp)/str(n)
                points,sources,p=acq.generate_batch(self.models,self.x,self.y,n,seen,folder,1.,2,
                       search_state={'polish_active':False,'anchor_enabled':False},online=[])
                data=json.loads((folder/'candidates.json').read_text());e,l,_=splits(n)
                self.assertEqual(data['roles'].count('exploit'),e);self.assertEqual(data['roles'].count('learn'),l)
                self.assertEqual(len(points),50);self.assertEqual(len({weight_key(w.tolist()) for w in points}),50)
                self.assertTrue((points>=0).all());np.testing.assert_allclose(points.reshape(-1,3,7).sum(-1),1.)
                self.assertFalse(any(weight_key(w.tolist()) in seen for w in points))
                self.assertEqual(data['shadow']['available_online_rows'],0)
                if n==20:self.assertFalse(any(s.startswith('learn_') for s in sources))

    def test_anchor_and_polish_criteria_still_causal(self):
        scores=np.arange(50)+100.;shadow=dict(eligible=[True]*10+[False]*40,raw_mean=(scores+10).tolist(),anchor_mean=(scores+1).tolist(),raw_score=scores.tolist(),anchor_score=scores.tolist())
        state={}
        for n in range(1,10):
            update_feedback(state,dict(round=n,shadow=shadow),scores,100.,100.)
            self.assertFalse(state['polish_active'])
        update_feedback(state,dict(round=10,shadow=shadow),scores,100.,100.)
        self.assertTrue(state['polish_active']);self.assertTrue(state['anchor_enabled'])
        self.assertLessEqual(max(state['anchor_gate']['assessed_rounds']),10)

    def test_training_audit_metrics_and_no_use_in_selection(self):
        from adaptive.training_audit import attach,assess
        # Pure metric test: all comparison predictions are stored before scores.
        batch=dict(round=5,information_model_comparison=dict(after_feedback_round=4,evaluated_round=5,seed=2026,
             generated_before_feedback=True,used_for_acquisition=False,full_predictions=[1.,2.,3.],without_information_predictions=[4.,3.,2.]))
        result=assess(batch,[1.,2.,3.]);self.assertEqual(result['full']['MAE'],0.)
        self.assertGreater(result['MAE_improvement_from_information'],0.)
        self.assertFalse(result['used_for_acquisition']);self.assertEqual(result['extra_true_evaluations'],0)
        batch['information_model_comparison']['generated_before_feedback']=False
        with self.assertRaises(RuntimeError):assess(batch,[1.,2.,3.])

    def test_budget_and_GPU_recovery_guard(self):
        from adaptive.budget import claim_run,reserve_call,complete_call
        from supervise import may_retry
        with tempfile.TemporaryDirectory() as temp:
            args=SimpleNamespace(test_epochs=None,session=Path(temp)/'one',resume=False)
            with patch('adaptive.budget.ROOT',Path(temp)):
                claim_run(args)
                for n in range(1,21):reserve_call(args,n,str(n));complete_call(args,n,str(n))
                with self.assertRaises(RuntimeError):reserve_call(args,21,'no')
        self.assertTrue(may_retry({'status':'diagnostic_training','pending':None},'CUDA out of memory',0))
        self.assertFalse(may_retry({'status':'evaluating','pending':{'round':1}},'CUDA error',0))
        args=SimpleNamespace(exclude_gpu='',gpu=None,allow_shared_gpu=True)
        sample=[dict(index=0,free=20000,used=60000,util=10,processes=2),dict(index=1,free=10000,used=0,util=0,processes=0)]
        self.assertEqual([r[4] for r in rank_cards([sample]*3,args)],[1,0])


if __name__=='__main__':unittest.main(verbosity=2)
