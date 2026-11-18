"""Serialize original EXE calls and isolate extraction directories on Windows."""
from contextlib import contextmanager
import ctypes
from ctypes import wintypes
import hashlib
import os
from pathlib import Path
import tempfile
from .protocol import atomic_json

ROOT=Path(__file__).resolve().parents[1]


@contextmanager
def native_slot(args,env,number):
    if args.backend!='native' or os.name!='nt':
        yield
        return
    api=ctypes.WinDLL('kernel32',use_last_error=True)
    api.CreateMutexW.argtypes=[ctypes.c_void_p,wintypes.BOOL,wintypes.LPCWSTR]
    api.CreateMutexW.restype=wintypes.HANDLE
    api.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD]
    api.WaitForSingleObject.restype=wintypes.DWORD
    api.ReleaseMutex.argtypes=[wintypes.HANDLE]
    api.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=api.CreateMutexW(None,False,'Local\\CodexTeacherLab6NativeEvaluator')
    if not handle:raise ctypes.WinError(ctypes.get_last_error())
    acquired=False
    status=ROOT/'native-status.json'
    identity=hashlib.sha256(str(ROOT).encode()).hexdigest()[:12]
    temporary=Path(tempfile.gettempdir())/'CodexLab6'/identity/f'r{number:02d}'
    temporary.mkdir(parents=True,exist_ok=True)
    env.update(TMP=str(temporary),TEMP=str(temporary))
    try:
        atomic_json(status,dict(status='waiting_for_evaluator',round=number,pid=os.getpid(),temporary=str(temporary)))
        result=api.WaitForSingleObject(handle,3600000)
        if result not in (0,128):raise RuntimeError('Native evaluator mutex timeout/error: '+str(result))
        acquired=True
        atomic_json(status,dict(status='evaluating',round=number,pid=os.getpid(),temporary=str(temporary)))
        yield
    finally:
        if acquired:api.ReleaseMutex(handle)
        api.CloseHandle(handle)
        atomic_json(status,dict(status='idle',round=number,pid=os.getpid()))
