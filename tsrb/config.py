import hashlib,json
from pathlib import Path

PACKAGE=Path(__file__).resolve().parents[1]
def load_config(path):
    c=json.loads(Path(path).read_text())
    ds=c['dataset'];assert ds in ('SUIM','TrashCan')
    expected={'classes':8 if ds=='SUIM' else 23,'epochs':100,'batch':2,'accumulation':4,'workers':0,'lr':.001,'weight_decay':.0001,'aux_weight':.4,'lambda_tsrb':1.,'temperature':1.,'size':512}
    for k,v in expected.items():
        if c[k]!=v:raise ValueError('Frozen protocol mismatch: '+k)
    if c['seeds']!=list(range(5 if ds=='SUIM' else 3)):raise ValueError('Frozen seed set mismatch')
    policy='best_inner_dev_strict_greater_earliest_tie' if ds=='SUIM' else 'fixed_epoch_100'
    if c['checkpoint_policy']!=policy:raise ValueError('Frozen checkpoint policy mismatch')
    return c
def split_path(c,split):return PACKAGE/'manifests'/c['dataset']/(split+'.csv')
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def new_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(data,f,indent=2,allow_nan=False)
