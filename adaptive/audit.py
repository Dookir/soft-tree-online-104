"""Deny old experiments' checkpoints/labels and log actual numerical inputs."""
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
INSTALLED=False


def install(label=None):
    global INSTALLED
    if INSTALLED:return
    if label is None:
        argv=sys.argv
        label=Path(argv[argv.index('--session')+1]).name if '--session' in argv else 'validation'
    directory=ROOT/'access-audits'/label;directory.mkdir(parents=True,exist_ok=True)
    log=directory/f'{os.getpid()}.jsonl'
    forbidden=('/online-soft-tree/solver/sessions/','/online-tabm/solver/sessions/',
               '/线上/solver/sessions/','/线上/tabm-online/sessions/',
               '/soft-tree-release/outputs/','/soft-tree-validation/',
               '/tabm-offline-search/models/','/fresh-offline/results/',
               '/experiment-archives/','/soft-tree-online-fast/sessions/',
               '/soft-tree-online-fast/calibration/','/soft-tree-online-104/sessions/',
               '/soft-tree-online-104/pretrained_original/','/soft-tree-online-104-release/results/',
               '/soft-tree-online-104-release/pretrained_original/','/releases/soft-tree-online-104-',
               '/soft-tree-online-104-v2/','/soft-tree-online-104-v3/','/soft-tree-online-104-v4/calibration/foreign/')
    def hook(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        name=os.fsdecode(args[0]);absolute=os.path.abspath(name)
        portable=absolute.replace('\\','/')
        inside=Path(absolute).is_relative_to(ROOT)
        if '--session' in sys.argv and inside and Path(absolute).is_relative_to(ROOT/'results'):
            raise PermissionError('Archived feedback/checkpoints are excluded from a fresh online run')
        if not inside and any(part in portable for part in forbidden):
            raise PermissionError('Old experiment input is forbidden: '+absolute)
        if '--session' in sys.argv and Path(absolute).is_relative_to(ROOT/'calibration') and Path(name).suffix.lower() in ('.csv','.npz','.pt','.pth'):
            raise PermissionError('Calibration checkpoints are not formal-run inputs')
        if Path(absolute)==ROOT/'baseline_reference.json' and args[1]!='w':
            argv=sys.argv
            if '--session' not in argv:raise PermissionError('Baseline comparison is post-run only')
            state=Path(argv[argv.index('--session')+1])/'state.json'
            current=json.loads(state.read_text()) if state.exists() else {}
            if current.get('status')!='complete' or current.get('completed_rounds')!=20:
                raise PermissionError('Baseline comparison is post-run only')
        if Path(name).suffix.lower() not in ('.csv','.npz','.pt','.pth','.joblib'):return
        row=json.dumps(dict(path=absolute,mode=str(args[1]),flags=args[2]),ensure_ascii=False)+'\n'
        fd=os.open(log,os.O_CREAT|os.O_APPEND|os.O_WRONLY,0o600)
        try:os.write(fd,row.encode())
        finally:os.close(fd)
    sys.addaudithook(hook);INSTALLED=True
