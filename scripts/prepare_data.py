"""Download pinned labels and the 48-frame, nine-stream demo selections."""
from pathlib import Path
import json, subprocess, sys, urllib.request
root=Path(__file__).resolve().parents[1]
data=root/'multisensor'
for selection in ['selection.json','synthetic-selection.json']:
 spec=json.loads((data/selection).read_text())
 for seq in spec['sequences'].values():
  dest=data/'data'/seq/'label.pkl'
  dest.parent.mkdir(parents=True,exist_ok=True)
  if not dest.exists():
   url='https://huggingface.co/datasets/jeongyh98/DSERT-RoLL/resolve/'+spec['revision']+'/'+seq+'/label.pkl'
   urllib.request.urlretrieve(url,dest)
for script in ['download.py','download_synthetic.py']:
 subprocess.run([sys.executable,str(data/script)],cwd=data,check=True)
print('Pinned dataset subsets downloaded. Manifests contain per-file SHA-256 checksums.')
