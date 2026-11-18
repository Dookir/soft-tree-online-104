import json
import sys
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from adaptive import base,structure_search as special,budget
from adaptive.acquisition import generate_batch,update_alpha
from adaptive.acquisition_original import update_alpha as original_update
from adaptive.protocol import weight_key
from adaptive.release import ROOT,training_data
from adaptive.model import DistrictSurrogate


class StructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base.configure();cls.x,cls.y=training_data(0)
        cls.models=[DistrictSurrogate(float(np.median(cls.y)),s).eval().requires_grad_(False) for s in range(2026,2031)]

    def test_baseline_hashes_and_original_radius_rule(self):
        sys.path.insert(0,str(ROOT.parent))
        from submission import check
        check();self.assertEqual(len(self.x),2048)
        self.assertIs(update_alpha,original_update)
        self.assertFalse(any((ROOT/name).exists() for name in ('results','examples','reference','validation')))
        self.assertFalse(any((ROOT/'adaptive'/name).exists() for name in ('multibasin.py','direction_first.py','determinism.py')))

    def test_sampling_support_concentration_and_banks(self):
        q,m,banks=special.specialized_pool(self.x,self.y,1)
        q2,m2,b2=special.specialized_pool(self.x,self.y,1)
        self.assertTrue(np.array_equal(q,q2));self.assertEqual(m,m2)
        self.assertEqual(len(q),256)
        blocks=q.reshape(-1,3,7)[:,1]
        self.assertTrue(np.all(blocks>=0));self.assertTrue(np.allclose(blocks.sum(1),1))
        for size in (3,4):
            for concentration in (1.,.3):self.assertEqual(sum(v['support_size']==size and v['concentration']==concentration for v in m),64)
        for w,v in zip(blocks,m):
            active=list(special.SUPPORT)+([] if v['extra_coordinate'] is None else [v['extra_coordinate']])
            self.assertEqual(set(np.flatnonzero(w)),set(active))
        self.assertEqual({v['extra_coordinate'] for v in m if v['support_size']==4},set(special.EXTRAS))
        self.assertTrue(all(blocks[:,j].max()>.9 and blocks[:,j].min()<.001 for j in special.SUPPORT))
        for b,ids in zip((0,2),banks):
            self.assertTrue(1<=len(ids)<=8)
            self.assertTrue(all(self.y[i]<=self.y.min()+2 for i in ids))
            w=self.x[ids].reshape(-1,3,7)[:,b]
            if len(ids)>1:self.assertGreaterEqual(np.linalg.norm(w[:,None]-w[None],axis=2)[np.triu_indices(len(ids),1)].min(),.05-1e-12)

    def test_transfers_use_original_bins_and_radius(self):
        q,m=special.transfer_pool(self.x,self.y,1.7);parent=int(np.argmin(self.y))
        for w,d in zip(q,m):
            self.assertEqual(d['center'],parent);self.assertEqual(d['building'],1)
            self.assertIn(d['donor'],special.SUPPORT);self.assertIn(d['receiver'],special.SUPPORT)
            expected=self.x[parent].reshape(3,7).copy();expected[1,d['donor']]-=d['delta'];expected[1,d['receiver']]+=d['delta']
            self.assertTrue(np.array_equal(expected.reshape(21),w))
            nominal=expected[1,d['donor']]+d['delta'] if d['bin']=='support' else {'small':.001,'medium':.01,'coarse':.08}[d['bin']]*1.7
            self.assertAlmostEqual(d['requested_delta'],nominal)
        self.assertTrue(all(d in {(v['donor'],v['receiver']) for v in m} for d in ((2,5),(5,2),(2,6),(6,2),(5,6),(6,5))))

    def test_future_and_exact_parent_history(self):
        _,m=special.transfer_pool(self.x,self.y,1.);v=m[0];parent=v['center']
        row=dict(round=18,source='specialized_transfer',R=float(self.y[parent]-.1),acquisition=v)
        self.assertTrue(special.successful_directions(self.x,self.y,[row],19))
        x=self.x.copy();x[parent,0]+=1e-15
        with self.assertRaises(ValueError):special.successful_directions(x,self.y,[row],19)
        y=self.y.copy();y[(parent+1)%len(y)]=y[parent]-1
        self.assertEqual(special.successful_directions(self.x,y,[row],19),{})
        with self.assertRaises(ValueError):special.successful_directions(self.x,self.y,[row],18)

    def test_six_slots_late_follow_and_exhaustion(self):
        seen={weight_key(w.tolist()) for w in self.x}
        q,s,m,r=special.select(self.models,self.x,self.y,1,1.,[],seen,list(self.x[:1]))
        self.assertEqual(len(q),6)
        self.assertEqual(s,['specialized_model']*2+['specialized_coverage']*2+['specialized_transfer']*2)
        self.assertEqual(m[-2]['selection'],'model_ranked_transfer')
        self.assertEqual(m[-1]['selection'],'fixed_rotation_no_model_gate')
        direction=m[-2]['direction'];parent=m[-2]['center']
        row=dict(round=18,source='specialized_transfer',R=float(self.y[parent]-.1),acquisition=m[-2])
        q,s,m,r=special.select(self.models,self.x,self.y,19,1.,[row],seen,list(self.x[:1]))
        self.assertEqual(s[:4],['specialized_model']*4)
        self.assertTrue(all(v['direction']==direction and v['selection']=='same_exact_parent_improvement' for v in m[4:]))
        pool,_,_=special.specialized_pool(self.x,self.y,1);transfers,_=special.transfer_pool(self.x,self.y,1.)
        blocked=seen|{weight_key(w.tolist()) for w in list(pool)+list(transfers)}
        q,s,m,r=special.select(self.models,self.x,self.y,1,1.,[],blocked,list(self.x[:1]))
        self.assertEqual(len(q),0);self.assertEqual(r['shortfall'],6)

    def test_twenty_round_batches_replay_and_learning_pairs(self):
        sys.path.insert(0,str(ROOT.parent));import finalize_submission as finalizer
        x=self.x.copy();y=self.y.copy();online=[];totals={'exploit':0,'learn':0}
        with tempfile.TemporaryDirectory() as tmp:
            finalizer.SESSION=Path(tmp)
            for n in range(1,21):
                seen={weight_key(w.tolist()) for w in x};folder=Path(tmp)/'rounds'/f'{n:02d}'
                q,_,_=generate_batch(self.models,x,y,n,seen,folder,radius=1.,search_steps=2,
                    search_state=dict(polish_active=n>=15,anchor_enabled=n>=15),online=online)
                batch=json.loads((folder/'candidates.json').read_text());orig=json.loads((folder/'original-pool/candidates.json').read_text())
                self.assertEqual(len(q),50);self.assertEqual(len({weight_key(w.tolist()) for w in q}),50)
                self.assertTrue(all(weight_key(w.tolist()) not in seen for w in q))
                self.assertTrue(np.all(q>=-1e-14));self.assertTrue(np.allclose(q.reshape(-1,3,7).sum(2),1,atol=1e-12))
                from adaptive.acquisition import STRATEGY, PROTECTED
                report = batch['specialization']
                reserved = 6 if n <= 6 or STRATEGY == 'boundary' else (4 if n <= 18 else 2)
                self.assertEqual(report['actual'] + report['shortfall'], reserved)
                if n >= 7:
                    self.assertTrue(all(orig['sources'][i] not in PROTECTED for i in report['replacement_slots']))
                    for source in ('specialized_donor_comparison', 'specialized_step_comparison'):
                        pair = [(p, m) for p, s, m in zip(q, batch['sources'], batch['metadata']) if s == source]
                        self.assertIn(len(pair), (0, 2))
                        if pair:
                            self.assertEqual(pair[0][1]['center'], pair[1][1]['center'])
                            for w, m in pair:
                                expected = x[m['center']].reshape(3, 7).copy()
                                expected[1, m['donor']] -= m['delta']
                                expected[1, m['receiver']] += m['delta']
                                self.assertTrue(np.array_equal(w, expected.reshape(-1)))
                            k = 'delta' if source.endswith('donor_comparison') else 'direction'
                            self.assertEqual(pair[0][1][k], pair[1][1][k])
                e=batch['role_counts']['exploit']
                self.assertEqual(batch['weights'][e:],orig['weights'][e:]);self.assertEqual(batch['metadata'][e:],orig['metadata'][e:])
                pairs={}
                for role,m in zip(batch['roles'],batch['metadata']):
                    totals[role]+=1
                    if 'pair_id' in m:pairs.setdefault(m['pair_id'],[]).append(m['pair_member'])
                self.assertTrue(all(sorted(v)==[0,1] for v in pairs.values()))
                with patch('adaptive.base.load_ensemble',return_value=self.models):finalizer.validate_specialization(batch,x,y,n,online,folder)
                if n <= 6 and batch['specialization']['sample_actual']==4:
                    expected=['specialized_model']*(2 if n<=18 else 4)+(['specialized_coverage']*2 if n<=18 else [])
                    self.assertEqual([s for s in batch['sources'] if s in ('specialized_model','specialized_coverage')],expected)
                scores=101+np.square(q).sum(1)
                online.extend(dict(id=f'r{n:02d}-{i:02d}',round=n,weights=w.reshape(3,7).tolist(),
                                   R=float(score),source=s,role=role,acquisition=m)
                    for i,(w,score,s,role,m) in enumerate(zip(q,scores,batch['sources'],batch['roles'],batch['metadata']),1))
                x=np.concatenate([x,q]);y=np.concatenate([y,scores])
        self.assertEqual(totals,dict(exploit=710,learn=290))

    def test_followup_retains_parent_after_child_becomes_best(self):
        from adaptive.paired_selection import choose, followups, histories, move
        parent = np.array([[.5,.5,0,0,0,0,0],[0,0,.55,0,0,.45,0],[.5,.5,0,0,0,0,0]]).reshape(21)
        original = self.x.copy()
        original[0] = parent
        direction = (1, 5, 6)
        child = move(original, 0, direction, .01)
        other = move(original, 0, (1, 2, 6), .02)
        x = np.vstack((original, child, other))
        y = np.r_[np.full(2048, 106.), 104.5, 105.4]
        y[0] = 105.
        m = dict(center=0, building=1, donor=5, receiver=6, delta=.01, parent_key=special.parent_key(parent))
        online = [dict(id='r18-01', round=18, source='transfer', acquisition=m, R=104.5),
                  dict(id='r18-02', round=18, source='recombination', acquisition={}, R=105.4)]
        self.assertEqual(np.argmin(y), 2048)
        history = histories(x, y, online, 19)
        self.assertIn((0, *direction), history)
        self.assertNotIn((2048, *direction), history)
        with self.assertRaises(ValueError):
            histories(x, y, online, 18)
        self.assertEqual(list(followups(x, y, online, 21, 1., history, False)), [])
        seen = {weight_key(w.tolist()) for w in x}
        for strategy, cap in (('recombination', 2), ('boundary', 6)):
            batch = choose(self.models,x,y,19,1.,online,seen,[parent],strategy,cap)
            steps = [m for w,s,m in batch if s == 'specialized_step_followup']
            self.assertEqual(len(steps), 2)
            self.assertTrue(all(m['center'] == 0 and m['reference_child_id'] == 'r18-01' for m in steps))
            self.assertLess(steps[0]['delta'], .01)
            self.assertGreater(steps[1]['delta'], .01)

    def test_synthetic_recombination_quality_and_distinct_directions(self):
        from adaptive.paired_selection import choose
        original = self.x.copy()
        x = np.vstack((original, original[:3]))
        y = np.r_[np.full(2048, 104.), 104.3, 104.7, 105.1]
        online = [dict(id=f'r{r:02d}-{i+1:02d}', round=r, source='recombination', acquisition={}, R=float(y[2048+i]))
                  for i,r in enumerate((10,11,11))]
        seen = {weight_key(w.tolist()) for w in x}
        batch = choose(self.models,x,y,12,1.,online,seen,[x[0]],'recombination',4)
        moves = [m for w,s,m in batch if s == 'specialized_recombination_transfer']
        self.assertEqual(len(moves), 2)
        self.assertTrue(all(m['center'] == 2048 for m in moves))
        self.assertNotEqual(moves[0]['direction'], moves[1]['direction'])
        self.assertTrue(all(m['center'] != 2050 for _,_,m in batch if 'center' in m))

    def test_guard_parent_and_spawn(self):
        for mode in ('direct','spawn'):
            script="""import sys,runpy
from pathlib import Path
root=Path.cwd();sys.argv=['guard','--session',str(root/'sessions/selftest')]
if MODE=='spawn':runpy.run_path(str(root/'run.py'),run_name='__mp_main__')
else:
 from adaptive.input_guard import install
 install()
(root/'adaptive/data/train.csv').read_bytes()
for path in [root/'results/formal-v4/best.json',root.parent.parent/'diagnostics/probe/request.json',root.parent.parent/'soft-tree-teacher-submission-20261116/best-result.json',root.parent.parent/'cpu-boundary-20261118/project/sessions/formal/online.json',root.parent/'evidence/formal/online.json',root/'sessions/other/online.json']:
 try:path.read_bytes()
 except PermissionError:continue
 raise RuntimeError('External input not blocked')
""".replace('MODE',repr(mode))
            p=subprocess.run([sys.executable,'-c',script],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)

    def test_budget_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=SimpleNamespace(session=Path(tmp)/'session',test_epochs=None,resume=False)
            with patch.object(budget,'ROOT',Path(tmp)):
                budget.claim_run(args)
                with self.assertRaises(RuntimeError):budget.claim_run(args)
                for n in range(1,21):budget.reserve_call(args,n,str(n));budget.complete_call(args,n,str(n))
                with self.assertRaises(RuntimeError):budget.reserve_call(args,21,'extra')


if __name__=='__main__':unittest.main()
