import json,html
from common import *
out=ROOT/'outputs';m=json.loads((out/'vjepa_metrics.json').read_text());r=json.loads((out/'render_metrics.json').read_text())
rows=['# Results','', 'Pretrained V-JEPA 2.1; single sequence; 48 observations, 16 sampled encoder frames.','', '| Variant | RGB cosine distance | All-sensor late fusion distance |','|---|---:|---:|']
for v in m['variants']:rows.append('| %s | %.8f | %.8f |'%(v['variant'],v['rgb_cosine_distance'],v['all_sensor_late_fusion_cosine_distance']))
rows+=['','Repeat relative L2: '+str(m['repeat_relative_l2']),'','All-sensor baseline retains seven unchanged sensor streams. Lower distances do not establish weather robustness.','',m['method'],'',m['limits']]
(out/'report.md').write_text(chr(10).join(rows))
parts=['<!doctype html><html><head><meta charset=utf-8><title>DSERT-RoLL synthetic environments</title><style>body{background:#101827;color:#edf2f7;font:17px system-ui;max-width:1150px;margin:40px auto;padding:20px}video,img{max-width:100%;border-radius:8px}section{margin:35px 0}a{color:#8bd5ff}pre{white-space:pre-wrap;background:#1d293d;padding:20px}h1{font-size:34px}</style></head><body><h1>DSERT-RoLL synthetic environments</h1><p>48 synchronized observations from a clear daylight drive. Procedural RGB fog, rain, and night exposure; original sensors provide reference and depth. V-JEPA measures feature changes.</p>']
for name,title in [('comparison','Compare the recorded base with three synthetic variants'),('all_sensors','All nine recorded sensor streams and vehicle pose'),('fog','Synthetic fog'),('rain','Synthetic rain'),('night','Synthetic night exposure'),('original','Recorded base')]:
    parts.append('<section><h2>'+title+'</h2><video controls loop preload=metadata src='+name+'.mp4></video></section>')
parts.append('<h2>V-JEPA experiment</h2><img src=experiment.png><pre>'+html.escape(chr(10).join(rows))+'</pre>')
parts.append('<p>Scope, methods and limitations: <a href=../README.md>README</a>. <a href=vjepa_metrics.json>Model metrics</a>, <a href=render_metrics.json>Rendering metrics</a>, <a href=video_validation.json>Video validation</a>, <a href=raw_channel_summary.json>Raw sensor channel summary</a>.</p><p>Dataset: Cho, Kang, Jeong et al., DSERT-RoLL, CVPR 2026. <a href=https://huggingface.co/datasets/jeongyh98/DSERT-RoLL>Official dataset</a>, CC-BY-4.0.</p></body></html>')
(out/'index.html').write_text(''.join(parts))
print(chr(10).join(rows))
