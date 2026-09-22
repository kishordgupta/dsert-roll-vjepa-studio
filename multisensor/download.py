import json, hashlib, urllib.request, time, concurrent.futures
from common import *
s=json.loads((ROOT/'selection.json').read_text())
base='https://huggingface.co/datasets/jeongyh98/DSERT-RoLL/resolve/'+s['revision']+'/'
paths=[]; samples=[]
for weather,seq in sorted(s['sequences'].items()):
    d=labels(seq); frames=d['info'][:48]
    assert len(frames)==48
    for f in frames:
        assert len(f['sensor'])==9 and all(f['sensor'].values())
        paths.extend(f['sensor'].values())
    times=[int(f['time_stamp']) for f in frames]
    samples.append(dict(weather=weather,light=d['meta']['light'],sequence=seq,frames=48,timestamps_ns=times,frame_indices=[int(f['frame_idx']) for f in frames],pose_available=all('pose' in f for f in frames)))
def download(rel):
    p=ROOT/'data'/rel
    assert p.resolve().is_relative_to((ROOT/'data').resolve())
    p.parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(5):
        try:
            if not p.exists():
                with urllib.request.urlopen(base+rel,timeout=120) as r: raw=r.read()
                tmp=p.with_suffix(p.suffix+'.part'); tmp.write_bytes(raw); tmp.replace(p)
            raw=p.read_bytes()
            assert len(raw)>0
            return dict(path=rel,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        except Exception:
            if attempt==4: raise
            time.sleep(2**attempt)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    manifest=[]
    for i,item in enumerate(pool.map(download,sorted(set(paths)))):
        manifest.append(item)
        if i%100==0: print('downloaded',i+1,'of',len(set(paths)),flush=True)
(ROOT/'manifest.json').write_text(json.dumps(dict(dataset='jeongyh98/DSERT-RoLL',revision=s['revision'],split='val',samples=samples,files=manifest),indent=2))
print('COMPLETE',len(manifest),sum(x['bytes'] for x in manifest),flush=True)
