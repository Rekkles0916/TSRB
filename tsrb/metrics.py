import torch
import numpy as np
from datasets.transforms import normalize, boundary_f_score
def cmatrix(p,t,n,ignore):
 v=t!=ignore;return np.bincount(n*t[v].astype(np.int64)+p[v].astype(np.int64),minlength=n*n).reshape(n,n)


def met(cm,bfs):
 d=np.diag(cm).astype(float);gt=cm.sum(1);pd=cm.sum(0);iou=d/np.maximum(gt+pd-d,1);dice=2*d/np.maximum(gt+pd,1);return {'mIoU':float(iou.mean()),'PixelAcc':float(d.sum()/max(cm.sum(),1)),'MeanAcc':float((d/np.maximum(gt,1)).mean()),'Boundary-F':float(np.mean(bfs)) if bfs else None,'MeanDice':float(dice.mean()),'per_class_IoU':iou.tolist(),'per_class_Dice':dice.tolist(),'confusion_matrix':cm.tolist()}


@torch.inference_mode()
def evaluate(m,dl,dev,n,ignore,boundary):
 m.eval();cm=np.zeros((n,n),np.int64);b=[];ids=[]
 for x,y,names in dl:
  p=m(normalize(x.to(dev)))['out'].argmax(1).cpu().numpy()
  for a,z,name in zip(p,y.numpy(),names):cm+=cmatrix(a,z,n,ignore);b.append(boundary_f_score(a,z)) if boundary else None;ids.append(name)
 return met(cm,b),ids

