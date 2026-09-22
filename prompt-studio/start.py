import os,pathlib,subprocess,time,urllib.request,json
ROOT=pathlib.Path(__file__).resolve().parent
URL='http://127.0.0.1:8765'
def running():
    try:
        with urllib.request.urlopen(URL+'/api/config',timeout=2) as r:d=json.load(r)
        return d.get('model')=='V-JEPA 2.1' and 'daylight' in d.get('sources',{})
    except Exception:return False
if not running():
    env=os.environ.copy();env.setdefault('CUDA_VISIBLE_DEVICES','0');env['TORCH_HOME']=str(ROOT.parent/'cache')
    with (ROOT/'server.log').open('ab') as log:
        p=subprocess.Popen([str(ROOT.parent/'.venv311/bin/python'),str(ROOT/'server.py')],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    (ROOT/'server.pid').write_text(str(p.pid))
    for _ in range(40):
        if running():break
        if p.poll() is not None:raise SystemExit('Studio failed to start. See '+str(ROOT/'server.log'))
        time.sleep(.5)
    else:raise SystemExit('Studio did not respond. See server.log.')
print('DSERT-RoLL Prompt Studio is ready at '+URL)
