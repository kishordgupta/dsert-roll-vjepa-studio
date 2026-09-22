import os,sys,json,time,subprocess,shutil,hashlib
import numpy as np,cv2,torch
from catalog import ROOT,DATA,SOURCES
os.environ.setdefault('TORCH_HOME',str(ROOT.parent/'cache'))
sys.path.insert(0,str(DATA))
import render
from common import labels
sys.path.insert(0,str(ROOT.parent/'vjepa2'))
from src.hub.backbones import vjepa2_1_vit_base_384
from evals.hub.preprocessor import vjepa2_preprocessor
MODEL=None;TRANSFORM=None
GROUPS={'RGB':[0,1],'Event':[2,3],'Thermal':[4,5],'LiDAR':[6,7],'Radar':[8]}
def unit(a):return a/max(float(np.linalg.norm(a)),1e-12)
def distance(a,b):return float(np.clip(1-np.dot(unit(a),unit(b)),0,2))
def family(a):return {k:unit(np.mean([unit(a[i]) for i in ids],axis=0)) for k,ids in GROUPS.items()}
def fused(a):return unit(np.mean(list(family(a).values()),axis=0))
def encode_model(a):
    global MODEL,TRANSFORM
    if MODEL is None:
        MODEL,predictor=vjepa2_1_vit_base_384(pretrained=True);del predictor
        MODEL=MODEL.eval().cuda().float();TRANSFORM=vjepa2_preprocessor(crop_size=384)
    x=TRANSFORM(np.ascontiguousarray(a))[0].unsqueeze(0).cuda().float()
    with torch.inference_mode():y=MODEL(x)
    if not torch.isfinite(y).all():raise RuntimeError('Non-finite model output')
    return y.mean(1)[0].cpu().numpy()
def encode_video(path,frames,fps):
    first=frames[0];h,w=first.shape[:2]
    p=subprocess.Popen(['ffmpeg','-nostdin','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s',str(w)+'x'+str(h),'-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    try:
        for f in frames:p.stdin.write(f.tobytes())
    finally:p.stdin.close()
    if p.wait()!=0:raise RuntimeError('Video encoding failed')
def source_cache(key,progress):
    dest=ROOT/'cache'/key;dest.mkdir(exist_ok=True)
    if (dest/'complete.json').exists():return dest
    d=labels(SOURCES[key]['sequence']);frames=d['info'][:48];cal=d['meta']['calibration'];indices=np.linspace(0,47,16,dtype=int)
    ts=np.array([int(f['time_stamp']) for f in frames],np.int64);relative=(ts-ts[0])/1e9;fps=47/relative[-1]
    rgb=[];depth=[];bank=[];sensor_movies=[];coverage=[]
    for i,f in enumerate(frames):
        cams=[render.camera(f['sensor'][k]) for k in render.CAM_KEYS]
        pcs=[np.load(DATA/'data'/f['sensor'][k],allow_pickle=False) for k in render.PC_KEYS]
        deps=[render.depth_map(pcs,cal,cam,im.shape) for cam,im in zip(render.CAM_CAL[:2],cams[:2])]
        pair=[cv2.resize(im,(960,600)) for im in cams[:2]];rgb.append(np.stack(pair));depth.append(np.stack([a[0] for a in deps]));coverage.append([a[1] for a in deps])
        panels=[render.pc_panel(p,cal,n,cams[0])[0] for p,n in zip(pcs,render.PC_CAL)];sensors=cams+panels
        if i in indices:bank.append(np.stack([cv2.resize(im,(384,384))[:,:,::-1] for im in pair+sensors[2:]]))
        rows=[np.hstack([render.tile(sensors[j],render.NAMES[j]) for j in range(k*3,k*3+3)]) for k in range(3)]
        footer=np.full((52,1440,3),18,np.uint8);render.text(footer,'Recorded sensor reference | '+SOURCES[key]['sequence']+' | %.2fs'%relative[i],(12,32),.6)
        sensor_movies.append(np.vstack(rows+[footer]))
        if i%4==0:progress(5+int(i/48*30),'Preparing synchronized sensor views '+str(i+1)+'/48')
    encode_video(dest/'sensors.mp4',sensor_movies,fps)
    np.savez_compressed(dest/'frames.npz',rgb=np.stack(rgb),depth=np.stack(depth).astype('float16'),bank=np.stack(bank))
    meta=dict(fps=float(fps),frames=48,indices=indices.tolist(),timestamps_ns=ts.tolist(),depth_support=float(np.mean(coverage)),source=SOURCES[key],poses=[np.asarray(f['pose']).tolist() for f in frames])
    (dest/'complete.json').write_text(json.dumps(meta,indent=2));return dest

def apply_effects(im,depth,i,request):
    strength=request['strength'];effects=request['effects']
    if effects==['original'] or strength==0:return im.copy()
    a=im.astype('float32')/255;h,w=a.shape[:2]
    if 'warm' in effects:a=np.clip(a*np.array([1-.2*strength,1+.06*strength,1+.25*strength]),0,1)
    if 'fog' in effects:
        trans=np.exp(-.05*strength*depth.astype('float32'))[:,:,None]
        a=a*trans+np.array([.82,.84,.85])*(1-trans)
    rng=np.random.default_rng(request['seed'])
    if 'rain' in effects:
        a=a*(1-.23*strength)+np.array([.10,.085,.075])*strength;layer=np.zeros_like(a,dtype='float32')
        count=int(850*strength)
        for x,y,speed,length in zip(rng.uniform(0,w,count),rng.uniform(0,h,count),rng.uniform(24,60,count),rng.uniform(8,24,count)):
            x=int((x+i*8)%w);y=int((y+i*speed)%h)
            cv2.line(layer,(x,y),(x+int(length*.3),y+int(length)),(.35,.32,.29),1,cv2.LINE_AA)
        a=np.clip(a+cv2.GaussianBlur(layer,(3,3),.5),0,1)
    if 'snow' in effects:
        layer=np.zeros_like(a,dtype='float32');count=int(650*strength)
        for x,y,speed,radius in zip(rng.uniform(0,w,count),rng.uniform(0,h,count),rng.uniform(3,15,count),rng.integers(1,4,count)):
            xx=int((x+i*3+8*np.sin(i*.2+x))%w);yy=int((y+i*speed)%h)
            cv2.circle(layer,(xx,yy),int(radius),(.8,.8,.8),-1,cv2.LINE_AA)
        a=np.clip(a*(1-.1*strength)+cv2.GaussianBlur(layer,(3,3),.4),0,1)
    if 'night' in effects:a=np.power(np.clip(a,0,1),1+1.2*strength)*np.array([1-.7*strength,1-.79*strength,1-.84*strength])
    return np.uint8(np.rint(np.clip(a,0,1)*255))
def labeled(im,label):
    bar=np.full((40,im.shape[1],3),18,np.uint8);render.text(bar,label,(12,27),.65)
    return np.vstack([bar,im])
def generate(request,out,progress):
 from multisensor_pipeline import generate as generate_all
 return generate_all(request,out,progress)
