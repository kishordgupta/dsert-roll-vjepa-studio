import json,urllib.request
from common import *
s=json.loads((ROOT/'selection.json').read_text())
base='https://huggingface.co/datasets/jeongyh98/DSERT-RoLL/resolve/'+s['revision']+'/'
with urllib.request.urlopen(base+'split/val.txt',timeout=120) as r: seqs=r.read().decode().splitlines()
for q in seqs:
    if not q.startswith('Clear/'): continue
    p=ROOT/'data'/q/'label.pkl';p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists():
        with urllib.request.urlopen(base+q+'/label.pkl',timeout=120) as r: p.write_bytes(r.read())
    d=labels(q); print(q,d['meta']['light'],flush=True)
    if d['meta']['light']=='Normal':
        (ROOT/'synthetic-selection.json').write_text(json.dumps(dict(revision=s['revision'],sequences={'Clear':q}),indent=2))
        break
else: raise RuntimeError('No normal-light validation sequence')
