"""Canonical RCM + TRP-weighted TSRB. T=1, class macro, correct teacher pixels."""
import torch
def relloss(o,y,tp,q,n,ignore):
 p=torch.softmax(o['out'],1);valid=y.ne(ignore);safe=y.clamp(0,n-1);correct=tp.argmax(1).eq(y)&valid;py=p.gather(1,safe[:,None]).squeeze(1);ty=tp.gather(1,safe[:,None]).squeeze(1);d=torch.relu((ty[:,None]-tp)-(py[:,None]-p));w=q.to(p.dtype)[safe].permute(0,3,1,2);z=(w*d).sum(1);v=[z[correct&y.eq(k)].mean() for k in range(n) if (correct&y.eq(k)).any()];return torch.stack(v).mean() if v else p.sum()*0

