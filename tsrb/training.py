"""Portable orchestration of the extracted formal scientific functions.

Preserves the 100-epoch loss, optimizer, batch/accumulation, and selection rules.
No training is performed by importing this module.
"""
import csv,json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets.cached import CachedDataset
from datasets.transforms import normalize
from models.segmentation import model,forward,segloss
from tsrb.relational import relloss
from tsrb.runtime import seedall,sh,rng,restore
from tsrb.metrics import evaluate
from tsrb.config import split_path,new_json,PACKAGE
from tsrb.trp import validate_q
from baselines.objectives import Objective

METHODS=['TSRB','PixelKD','CWD','CIRKD','DIST','FixedMargin','D0','U1']
def train(c,seed,init,output,method='TSRB',teacher_role=False,device='cuda:0'):
    if seed not in c['seeds']:raise ValueError('Seed outside frozen protocol')
    if teacher_role and seed!=0:raise ValueError('Dataset teacher uses seed0')
    output=Path(output)
    if output.exists():raise FileExistsError('Use a new output directory; existing experiments are protected')
    n=c['classes'];ignore=c['ignore_index'];dev=torch.device(device)
    seedall(seed)
    if teacher_role:
        torch.use_deterministic_algorithms(True,warn_only=True)
        torch.set_float32_matmul_precision('highest')
    if teacher_role and c['dataset']=='TrashCan':
        # core.model resets Torch's RNG after model construction for this teacher.
        before=torch.get_rng_state();s=model(n,init,dev);torch.set_rng_state(before)
    else:s=model(n,init,dev)
    initial=sh(s.state_dict());teacher=None
    with (PACKAGE/'manifests'/c['dataset']/'initialization.csv').open() as f:
        expected=next(r['model_state_hash'] for r in csv.DictReader(f) if int(r['seed'])==seed)
    if initial!=expected:raise ValueError('Canonical initial state hash mismatch')
    if not teacher_role and method in ('TSRB','U1','PixelKD','CWD','CIRKD','DIST'):
        teacher=model(n,init,dev)
        teacher.load_state_dict(torch.load(c['teacher'],map_location='cpu',weights_only=False)['model'],strict=True)
        teacher.eval().requires_grad_(False)
    objective=Objective(method,n,ignore,dev)
    q=None
    if method=='TSRB':
        q=np.load(c['trp']);validate_q(q,n);q=torch.from_numpy(q).double().to(dev)
    if method=='U1':q=torch.ones(n,n,dtype=torch.float64,device=dev)/(n-1);q.fill_diagonal_(0)
    td=CachedDataset(split_path(c,'inner_train'),c['data_root'],c['cache_root'],c['dataset'],True)
    dd=CachedDataset(split_path(c,'inner_dev'),c['data_root'],c['cache_root'],c['dataset'])
    assert len(td)==c['counts']['inner_train'] and len(dd)==c['counts']['inner_dev']
    generator=torch.Generator().manual_seed(seed)
    tl=DataLoader(td,2,shuffle=True,drop_last=True,num_workers=0,pin_memory=True,generator=generator)
    dl=DataLoader(dd,2,shuffle=False,num_workers=0,pin_memory=True)
    modules=[s]+objective.modules
    opt=torch.optim.AdamW([p for m in modules for p in m.parameters() if p.requires_grad],lr=.001,weight_decay=.0001,foreach=False,fused=False)
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
    output.mkdir(parents=True)
    new_json(output/'config.json',dict(c,seed=seed,method=method,teacher_role=teacher_role,initial_state_hash=initial))
    best=-1.;best_epoch=0;selection=teacher_role or c['dataset']=='SUIM'
    with (output/'history.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['epoch','train_loss','inner_dev_mIoU']);writer.writeheader()
        for epoch in range(1,101):
            td.epoch=epoch;s.train();opt.zero_grad(set_to_none=True);total=count=0
            for bi,(x,y,_) in enumerate(tl,1):
                x,y=x.to(dev),y.to(dev)
                # Teacher training used torchvision forward directly.
                if teacher_role:so=s(normalize(x));sf=None
                else:so,sf=forward(s,x)
                loss=segloss(so,y,ignore);to=tf=None
                if teacher is not None:
                    with torch.no_grad():to,tf=forward(teacher,x);tp=torch.softmax(to['out'],1)
                if method in ('TSRB','U1'):loss+=relloss(so,y,tp,q,n,ignore)
                elif not teacher_role:loss+=objective(so,sf,to,tf,y)
                if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite loss')
                (loss/4).backward()
                if bi%4==0 or bi==len(tl):opt.step();opt.zero_grad(set_to_none=True)
                total+=float(loss.detach())*len(x);count+=len(x)
            sched.step();m,_=evaluate(s,dl,dev,n,ignore,False)
            improved=selection and m['mIoU']>best
            if improved:best,best_epoch=m['mIoU'],epoch
            ck={'epoch':epoch,'model':s.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'rng':rng(generator),'initial_model_state_hash':initial,'best':best,'best_epoch':best_epoch,'method':method,'dataset':c['dataset'],'seed':seed,'memory':objective.memory.state_dict() if method=='CIRKD' else None,'channel':objective.channel.state_dict() if method=='CIRKD' else None}
            torch.save(ck,output/'last.pt')
            if improved:torch.save(ck,output/'best.pt')
            writer.writerow(dict(epoch=epoch,train_loss=total/count,inner_dev_mIoU=m['mIoU']));f.flush()
            print(f'{c["dataset"]} {method} seed={seed} epoch={epoch}/100 dev={m["mIoU"]:.6f}',flush=True)
    selected='best.pt' if selection else 'last.pt'
    new_json(output/'completion.json',dict(status='COMPLETED',epochs=100,selected_checkpoint=selected,checkpoint_epoch=best_epoch if selection else 100))
