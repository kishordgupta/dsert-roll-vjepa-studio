import json, pathlib, urllib.request, time, pickle
import numpy as np
root=pathlib.Path(__file__).resolve().parent
repo='jeongyh98/DSERT-RoLL'
api='https://huggingface.co/api/datasets/'+repo
def get(url):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=120) as r: return r.read()
        except Exception:
            if attempt==4: raise
            time.sleep(2**attempt)
revision=json.loads(get(api))['sha']
base='https://huggingface.co/datasets/'+repo+'/resolve/'+revision+'/'
split=get(base+'split/val.txt').decode().splitlines()
selected={}
for seq in split:
    seq=seq.strip()
    if seq: selected.setdefault(seq.split('/')[0], seq)
(root/'selection.json').write_text(json.dumps(dict(revision=revision,sequences=selected),indent=2))
for weather,seq in sorted(selected.items()):
    dest=root/'data'/seq
    dest.mkdir(parents=True,exist_ok=True)
    raw=get(base+seq+'/label.pkl')
    (dest/'label.pkl').write_bytes(raw)
    print(weather,seq,len(raw),flush=True)
