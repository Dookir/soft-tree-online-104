"""Formal coordinator and spawned workers may read only this experiment's data."""
import json
import os
from pathlib import Path
import sys

INSTALLED=False


def install():
    global INSTALLED
    if INSTALLED or '--session' not in sys.argv:return
    project=Path(__file__).resolve().parents[1]
    session=Path(sys.argv[sys.argv.index('--session')+1]).resolve()
    folder=project/'access-audits'/session.name;folder.mkdir(parents=True,exist_ok=True)
    logfile=folder/f'{os.getpid()}.jsonl'
    def log(row):
        fd=os.open(logfile,os.O_CREAT|os.O_APPEND|os.O_WRONLY,0o600)
        try:os.write(fd,(json.dumps(row)+'\n').encode())
        finally:os.close(fd)
    def hook(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=Path(os.path.abspath(os.fsdecode(args[0]))).resolve()
        reading=not any(c in str(args[1]) for c in ('w','a','x'))
        runtime=any(part in ('.venv','.tools') for part in path.parts)
        if reading and not runtime:
            if path.is_relative_to(project.parent.parent) and not path.is_relative_to(project):raise PermissionError('External experiment input denied')
            if path.is_relative_to(project/'sessions') and not path.is_relative_to(session):raise PermissionError('Foreign session input denied')
            if any(path.is_relative_to(project/d) for d in ('results','reference','examples','validation','calibration')):raise PermissionError('Historical/diagnostic input denied')
            if path.suffix.lower() in ('.json','.csv','.tsv','.npz','.npy','.pt','.pth','.pkl','.joblib') and not path.is_relative_to(project):raise PermissionError('External numerical input denied')
        if path.suffix.lower() in ('.json','.csv','.npz','.npy','.pt','.pth'):
            log(dict(path=str(path),mode=str(args[1]),reading=reading))
    sys.addaudithook(hook);INSTALLED=True
    log(dict(event='guard_installed',pid=os.getpid(),session=str(session)))
