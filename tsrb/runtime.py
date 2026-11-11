import torch, random, os, hashlib
import numpy as np
def seedall(seed):
 os.environ['PYTHONHASHSEED']=str(seed);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_num_threads(4)


def sh(sd):
 h=hashlib.sha256()
 for k,v in sorted(sd.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()


def rng(g):return {'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),'numpy':np.random.get_state(),'python':random.getstate(),'loader':g.get_state()}


def restore(s,g):torch.set_rng_state(s['torch'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in s['cuda']]);np.random.set_state(s['numpy']);random.setstate(s['python']);g.set_state(s['loader'].cpu())

