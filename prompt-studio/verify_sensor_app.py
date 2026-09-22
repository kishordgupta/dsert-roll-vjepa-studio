import json,urllib.request,quickjs,sys
from pathlib import Path
root=Path(__file__).resolve().parent; base='http://127.0.0.1:8765'; jid=sys.argv[1] if len(sys.argv)>1 else (_ for _ in ()).throw(SystemExit('Usage: verify_sensor_app.py JOB_ID'))
j=json.load(urllib.request.urlopen(base+'/api/jobs/'+jid));print('JOB',jid,j['status'],j['message'])
if j['status']!='complete':raise SystemExit(1)
ctx=quickjs.Context();ctx.eval("var nodes={};var current=null;var $=id=>nodes[id]||(nodes[id]={load(){},setAttribute(){}});var document={querySelectorAll(){return []}};function setBusy(){};function error(){};")
s=(root/'static/app.js').read_text();ctx.eval(s[s.index('function view(name)'):s.index('async function history()')])
ctx.eval('show('+json.dumps(j)+');view("sensors");')
assert ctx.eval("$('player').src")== '/jobs/'+jid+'/synthetic_sensors.mp4'
assert ctx.eval("$('sensorDownload').hidden") is False
assert ctx.eval("$('sensorDownload').href")== '/jobs/'+jid+'/sensor_data.zip'
old=dict(j);old['result']=dict(j['result']);old['result'].pop('sensor_version')
ctx.eval('show('+json.dumps(old)+');view("sensors");')
assert ctx.eval("$('player').src")== '/jobs/'+jid+'/sensors.mp4'
assert ctx.eval("$('sensorDownload').hidden") is True
print('UI JS new + legacy job behavior PASS')
for name in ['sensor_data.zip','synthetic_sensors.mp4','video.mp4','embeddings.npz']:
 req=urllib.request.Request(base+'/jobs/'+jid+'/'+name,headers={'Range':'bytes=0-1023'})
 with urllib.request.urlopen(req) as r:
  data=r.read();assert r.status==206 and len(data)==1024
  assert data==(root/'jobs'/jid/name).read_bytes()[:1024]
  if name.endswith('.zip'):assert data[:2]==b'PK'
  print('DOWNLOAD',name,r.status,r.headers['Content-Type'],r.headers['Content-Range'])
validation=json.load(urllib.request.urlopen(base+'/jobs/'+jid+'/validation.json'))
assert validation['valid'] and validation['primary_sensor_files']==432 and validation['archive_crc_valid']
print('RESULT',json.dumps(j['result'],indent=2));print('VALIDATION',json.dumps(validation,indent=2))
(root/'sensor-upgrade-verification.json').write_text(json.dumps(dict(job_id=jid,result=j['result'],validation=validation,ui_logic_passed=True,download_ranges_passed=True),indent=2))
