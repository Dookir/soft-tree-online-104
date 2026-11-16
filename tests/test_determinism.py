import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from adaptive import base
from adaptive.determinism import seed_all,settings
from adaptive.release import ROOT,training_data
from adaptive.repeatability import compare,array_digest


class Determinism(unittest.TestCase):
    @classmethod
    def setUpClass(cls):base.configure()

    def test_strict_runtime_and_global_generators(self):
        import random
        policy=settings()
        self.assertEqual((policy['threads'],policy['interop_threads']),(1,1))
        self.assertEqual(policy['default_dtype'],'torch.float64')
        self.assertTrue(policy['deterministic_algorithms'])
        for key in ('deterministic_warn_only','mkldnn_enabled','cudnn_enabled','cudnn_benchmark','matmul_tf32','cudnn_tf32'):
            self.assertFalse(policy[key],key)
        seed_all(2027);first=(random.random(),np.random.random(),torch.rand(3))
        seed_all(2027);second=(random.random(),np.random.random(),torch.rand(3))
        self.assertEqual(first[:2],second[:2]);self.assertTrue(torch.equal(first[2],second[2]))
        seed_all(2026)

    def test_conflicting_startup_environment_is_overridden(self):
        env=os.environ.copy()
        env.update(PYTHONHASHSEED='random',PYTHONUTF8='0',OMP_NUM_THREADS='8',
                   MKL_NUM_THREADS='8',CUBLAS_WORKSPACE_CONFIG=':16:8')
        done=subprocess.run([sys.executable,str(ROOT/'run.py'),'doctor'],cwd=ROOT,env=env,
                            capture_output=True,text=True,encoding='utf-8',timeout=30)
        self.assertEqual(done.returncode,0,done.stderr)
        policy=json.loads(done.stdout)['determinism']
        self.assertEqual(policy['python_hash_seed'],'2026');self.assertTrue(policy['utf8_mode'])
        self.assertEqual(policy['environment']['CUBLAS_WORKSPACE_CONFIG'],':4096:8')
        self.assertEqual(policy['environment']['OMP_NUM_THREADS'],'1')

    def test_model_training_ignores_unrelated_global_rng_state(self):
        x,y=training_data(0)
        with tempfile.TemporaryDirectory() as temp:
            fits=[]
            for number in (1,2):
                seed_all(number*991)
                folder=Path(temp)/str(number);folder.mkdir()
                base.train_one(2026,x[:64],y[:64],np.ones(64),float(np.median(y)),folder,2)
                with np.load(folder/'seed2026.npz',allow_pickle=False) as archive:
                    fits.append({k:archive[k].copy() for k in archive.files})
            self.assertTrue(all(item['exact'] for item in compare(*fits).values()))
            self.assertEqual(array_digest(fits[0]),array_digest(fits[1]))
        seed_all(2026)

    def test_signature_check_rejects_shape_mismatch_and_nan(self):
        for second in ({'probe':np.ones(3)},{'probe':np.array([1.,np.nan])}):
            with self.assertRaises(ValueError):compare({'probe':np.ones(2)},second)

    def test_legacy_training_cache_requires_fresh_output(self):
        x,y=training_data(0)
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            (folder/'training.json').write_text(json.dumps({'data_fingerprint':'legacy'}),encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError,'different numerical policy'):
                base.train_ensemble(x[:64],y[:64],64,folder,jobs=1,epoch_override=1)

    def test_conflicting_interpreter_flags_fail_without_restart_loop(self):
        for flags in (['-E'],['-X','utf8=0']):
            done=subprocess.run([sys.executable,*flags,str(ROOT/'run.py'),'doctor'],cwd=ROOT,
                                capture_output=True,text=True,timeout=10)
            self.assertNotEqual(done.returncode,0)
            self.assertIn('Strict startup requires',done.stderr)


if __name__=='__main__':unittest.main()
