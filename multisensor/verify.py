import json,subprocess,hashlib,numpy as np
from common import *
out=ROOT/'outputs';rows=[]
for p in sorted(out.glob('*.mp4')):
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_entries','stream=codec_name,width,height,nb_read_frames,avg_frame_rate,duration','-of','json',str(p)]))
    subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(p),'-f','null','-'],check=True)
    assert int(probe['streams'][0]['nb_read_frames'])==48
    rows.append(dict(file=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),probe=probe))
s=json.loads((ROOT/'synthetic-selection.json').read_text());d=labels(s['sequences']['Clear']);channels={}
for key in ['livox_path','ouster_path','radar_path']:
    clouds=[np.load(ROOT/'data'/f['sensor'][key],allow_pickle=False) for f in d['info'][:48]]
    fields={}
    for k in clouds[0].dtype.names:
        a=np.concatenate([p[k] for p in clouds]);v=a[np.isfinite(a)]
        fields[k]=dict(count=len(a),finite_fraction=float(len(v)/len(a)),min=float(v.min()),max=float(v.max()),mean=float(v.mean()))
    channels[key]=fields
(out/'video_validation.json').write_text(json.dumps(rows,indent=2));(out/'raw_channel_summary.json').write_text(json.dumps(channels,indent=2))
print('VERIFIED',len(rows),'videos; 48 frames each; complete decode; all range-sensor fields summarized')
