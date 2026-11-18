"""Automatic idle/spare GPU selection. Does not signal other users' processes."""
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def snapshot():
    tree=ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'],text=True,timeout=20))
    cards=[]
    for index,gpu in enumerate(tree.findall('gpu')):
        cards.append(dict(index=index,free=int(gpu.findtext('fb_memory_usage/free').split()[0]),
            used=int(gpu.findtext('fb_memory_usage/used').split()[0]),
            util=int(gpu.findtext('utilization/gpu_util').split()[0]),
            processes=len(gpu.findall('processes/process_info'))))
    return cards


def rank_cards(samples,args):
    excluded={int(v) for v in args.exclude_gpu.split(',') if v};qualified=[]
    for index in range(len(samples[0])):
        readings=[sample[index] for sample in samples]
        if index in excluded or (args.gpu is not None and args.gpu!=index):continue
        free=min(r['free'] for r in readings);peak=max(r['util'] for r in readings)
        mean=sum(r['util'] for r in readings)/len(readings)
        idle=all(r['used']<=1024 and r['processes']==0 and r['util']<=5 for r in readings)
        spare=args.allow_shared_gpu and peak<=50 and mean<=25
        if free>=8192 and (idle or spare):
            qualified.append((0 if idle else 1,mean,peak,-free,index,readings))
    return sorted(qualified)


def select_device(args):
    if args.device=='cpu':
        os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['SOFTTREE_DEVICE']='cpu'
        return dict(device='cpu')
    started=time.monotonic()
    while True:
        try:
            samples=[snapshot()]
            for _ in range(2):time.sleep(2);samples.append(snapshot())
            candidates=rank_cards(samples,args)
            error=None
        except (OSError,subprocess.SubprocessError,ValueError,ET.ParseError) as exc:
            candidates=[];error=str(exc)
        if candidates:
            idle,mean,peak,negative_free,index,readings=candidates[0]
            os.environ.update(CUDA_VISIBLE_DEVICES=str(index),SOFTTREE_DEVICE='cuda:0',
                              SOFTTREE_GPU_MEMORY_GIB='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            args.train_jobs=1
            return dict(device='cuda:0',physical_gpu=index,selection='idle' if idle==0 else 'spare_shared',
                free_MiB=-negative_free,mean_utilization=mean,peak_utilization=peak,
                memory_budget_GiB=2,readings=readings,shared_GPU_allowed=args.allow_shared_gpu,
                wait_gpu_seconds=time.monotonic()-started)
        if not args.wait_for_gpu:
            raise RuntimeError('No eligible GPU; use --wait-for-gpu or --device cpu')
        print('WAITING_FOR_IDLE_OR_SPARE_GPU: free>=8GiB; shared3-sample mean<=25%, peak<=50%; retry60s'+
              (('; query error='+error) if error else ''),file=sys.stderr,flush=True)
        time.sleep(60)
