import json,mimetypes,threading,uuid,traceback,re,time
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from urllib.parse import urlsplit,unquote
from concurrent.futures import ThreadPoolExecutor
from catalog import ROOT,DATA,SOURCES,interpret
PORT=8765;JOBS=ROOT/'jobs';LOCK=threading.Lock();POOL=ThreadPoolExecutor(max_workers=1);ACTIVE=None
def save(job):
    p=JOBS/job['id']/'status.json';p.parent.mkdir(exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(job,indent=2));tmp.replace(p)
def run(job):
    global ACTIVE
    def progress(percent,message):job.update(progress=percent,message=message,status='running');save(job)
    try:
        from engine import generate
        result=generate(job['request'],JOBS/job['id'],progress)
        job.update(status='complete',progress=100,message='Video ready',result=result);save(job)
    except Exception as exc:
        (JOBS/job['id']/'error.log').write_text(traceback.format_exc());job.update(status='failed',message='Generation failed: '+str(exc)[:300]);save(job)
    finally:
        with LOCK:ACTIVE=None
for p in JOBS.glob('*/status.json'):
    try:
        j=json.loads(p.read_text())
        if j['status'] in ['running','queued']:j.update(status='failed',message='Interrupted by server restart. Submit the prompt again.');save(j)
    except Exception:pass
class Handler(BaseHTTPRequestHandler):
    def log_message(self,fmt,*args):pass
    def valid_host(self):return self.headers.get('Host') in ['localhost:8765','127.0.0.1:8765']
    def begin_response(self,status,ctype,length):
        self.send_response(status);self.send_header('Content-Type',ctype);self.send_header('Content-Length',str(length));self.send_header('X-Content-Type-Options','nosniff');self.send_header('Cache-Control','no-store');self.send_header('Referrer-Policy','no-referrer')
        self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self'; connect-src 'self'; frame-ancestors 'none'")
    def reply(self,data,status=200):
        raw=json.dumps(data).encode();self.begin_response(status,'application/json',len(raw));self.end_headers();self.wfile.write(raw)
    def do_POST(self):
        global ACTIVE
        if not self.valid_host() or self.headers.get('Origin') not in [None,'http://localhost:8765','http://127.0.0.1:8765']:return self.reply({'error':'This app accepts same-origin local requests only.'},403)
        if self.headers.get('Content-Type','').split(';')[0]!='application/json':return self.reply({'error':'Expected JSON'},415)
        try:
            n=int(self.headers.get('Content-Length','0'))
            if not 0<n<=4096:raise ValueError('Request too large or empty')
            body=json.loads(self.rfile.read(n))
            if not isinstance(body,dict):raise ValueError('Expected a JSON object')
            request=interpret(body)
        except (ValueError,TypeError) as exc:return self.reply({'error':str(exc)},422)
        if self.path=='/api/interpret':return self.reply(request)
        if self.path!='/api/jobs':return self.reply({'error':'Not found'},404)
        if not (ROOT/'READY').exists():return self.reply({'error':'The rendering engine is being prepared.'},503)
        with LOCK:
            if ACTIVE:return self.reply({'error':'A video is already rendering. Wait for it to finish.','active':ACTIVE},409)
            jid=uuid.uuid4().hex;ACTIVE=jid
            job=dict(id=jid,status='queued',progress=0,message='Starting',created=time.time(),request=request);save(job);POOL.submit(run,job)
        return self.reply(job,202)
    def do_GET(self):
        if not self.valid_host():return self.reply({'error':'Invalid host'},403)
        path=unquote(urlsplit(self.path).path)
        if path=='/api/config':return self.reply(dict(sources=SOURCES,ready=(ROOT/'READY').exists(),active=ACTIVE,model='V-JEPA 2.1',frames=48))
        if path=='/api/jobs':
            jobs=[]
            for p in JOBS.glob('*/status.json'):
                try:jobs.append(json.loads(p.read_text()))
                except Exception:pass
            return self.reply(sorted(jobs,key=lambda x:x['created'],reverse=True)[:20])
        match=re.fullmatch('/api/jobs/([a-f0-9]{32})',path)
        if match:
            p=JOBS/match[1]/'status.json'
            return self.reply(json.loads(p.read_text())) if p.exists() else self.reply({'error':'Job not found'},404)
        files={'/':ROOT/'static/index.html','/app.js':ROOT/'static/app.js','/style.css':ROOT/'static/style.css','/reference/original.mp4':DATA/'outputs/original.mp4','/reference/all_sensors.mp4':DATA/'outputs/all_sensors.mp4'}
        p=files.get(path)
        match=re.fullmatch('/jobs/([a-f0-9]{32})/(video.mp4|comparison.mp4|sensors.mp4|metrics.json|request.json|synthetic_sensors.mp4|sensor_data.zip|validation.json|embeddings.npz)',path)
        if match:p=JOBS/match[1]/match[2]
        if p is None or not p.is_file():return self.reply({'error':'Not found'},404)
        size=p.stat().st_size;start=0;end=size-1;status=200
        if self.headers.get('Range'):
            match=re.fullmatch('bytes=([0-9]+)-([0-9]*)',self.headers['Range'])
            if not match:return self.reply({'error':'Unsupported range'},416)
            start=int(match[1]);end=min(int(match[2]) if match[2] else end,end)
            if start>end or start>=size:return self.reply({'error':'Range not satisfiable'},416)
            status=206
        self.begin_response(status,mimetypes.guess_type(p.name)[0] or 'application/octet-stream',end-start+1);self.send_header('Accept-Ranges','bytes')
        if status==206:self.send_header('Content-Range','bytes %d-%d/%d'%(start,end,size))
        self.end_headers()
        try:
            with p.open('rb') as f:
                f.seek(start);remaining=end-start+1
                while remaining:
                    raw=f.read(min(65536,remaining))
                    if not raw:break
                    self.wfile.write(raw);remaining-=len(raw)
        except (BrokenPipeError,ConnectionResetError):pass
if __name__=='__main__':
    print('DSERT Prompt Studio: http://127.0.0.1:8765',flush=True)
    ThreadingHTTPServer(('127.0.0.1',PORT),Handler).serve_forever()
