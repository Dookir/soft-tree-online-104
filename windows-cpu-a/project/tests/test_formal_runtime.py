import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class FormalRuntimeTests(unittest.TestCase):
    def test_deepest_formal_paths_are_below_windows_limit(self):
        paths=[ROOT/'sessions/formal/rounds/20/original-pool/v3-pool/v1-pool/candidates.json.tmp',
               ROOT/'sessions/formal/learning-audits/after-r16/evaluation-on-r17.json.tmp']
        self.assertTrue(all(len(str(p.resolve()))<240 for p in paths))

    def test_native_calls_are_serialized_and_temp_isolated(self):
        script="""import sys,json,time
from pathlib import Path
from types import SimpleNamespace
from adaptive import native_runtime
root=Path(sys.argv[1]);root.mkdir()
native_runtime.ROOT=root;env={}
with native_runtime.native_slot(SimpleNamespace(backend='native'),env,1):
 start=time.monotonic();time.sleep(.2);end=time.monotonic()
(root/'result.json').write_text(json.dumps(dict(start=start,end=end,env=env)))
"""
        with tempfile.TemporaryDirectory() as tmp:
            children=[subprocess.Popen([sys.executable,'-c',script,str(Path(tmp)/str(i))],cwd=ROOT) for i in (1,2)]
            self.assertTrue(all(p.wait(timeout=30)==0 for p in children))
            data=[json.loads((Path(tmp)/str(i)/'result.json').read_text()) for i in (1,2)]
            ordered=sorted(data,key=lambda r:r['start'])
            self.assertGreaterEqual(ordered[1]['start'],ordered[0]['end'])
            self.assertNotEqual(data[0]['env']['TEMP'],data[1]['env']['TEMP'])


if __name__=='__main__':unittest.main()
