"""Dataset-level teacher-correct alternative competition, native softmax T=1.

Preserves dataset-specific accumulation: SUIM FP32 sums per alternative per
batch (pair_risk); TrashCan CPU float64 vector sums (qbuild.py).
"""
import numpy as np
import torch

class RelationalProfile:
    def __init__(self, classes, dataset, ignore_index):
        self.n=classes;self.dataset=dataset;self.ignore=ignore_index
        self.mass=np.zeros((classes,classes),np.float64)
        self.count=np.zeros(classes,np.int64)
    @torch.no_grad()
    def update(self, probabilities, target):
        if self.dataset=='TrashCan':probabilities=probabilities.cpu();target=target.cpu()
        correct=probabilities.argmax(1).eq(target)&target.ne(self.ignore)
        for y in range(self.n):
            mask=correct&target.eq(y);n=int(mask.sum());self.count[y]+=n
            if not n:continue
            if self.dataset=='TrashCan':
                self.mass[y]+=probabilities.permute(0,2,3,1)[mask].double().sum(0).numpy()
            else:
                for c in range(self.n):
                    if c!=y:self.mass[y,c]+=float(probabilities[:,c][mask].sum())
    def finish(self):
        if (self.count==0).any():raise RuntimeError('TRP zero teacher-correct support: '+str(np.where(self.count==0)[0].tolist()))
        profile=self.mass/self.count[:,None];q=profile.copy();np.fill_diagonal(q,0)
        if self.dataset=='TrashCan':q=q/q.sum(1,keepdims=True)
        else:
            # setup.py Python summation followed by qmatrix's torch row normalization.
            for y in range(self.n):q[y]/=sum(float(q[y,c]) for c in range(self.n) if c!=y)
            t=torch.from_numpy(q)
            for y in range(self.n):t[y,y]=0;t[y]=t[y]/t[y].sum()
            q=t.numpy()
        validate_q(q,self.n)
        return profile,q,self.count.copy()

def validate_q(q,n):
    q=np.asarray(q)
    if q.shape!=(n,n) or not np.isfinite(q).all() or (q<0).any() or not np.all(np.diag(q)==0) or not np.allclose(q.sum(1),1,atol=1e-12):
        raise ValueError('TRP must be finite/nonnegative, off-diagonal, row-normalized')
