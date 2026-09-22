import json,sys,cv2,subprocess,hashlib
import numpy as np
from scipy.spatial import cKDTree
from common import *
OUT=ROOT/'outputs';OUT.mkdir(exist_ok=True)
CAM_KEYS=['rgb_left_path','rgb_right_path','event_left_path','event_right_path','thermal_left_path','thermal_right_path']
PC_KEYS=['livox_path','ouster_path','radar_path']
NAMES=['RGB left','RGB right','Event left','Event right','Thermal left','Thermal right','Livox','Ouster','Radar']
CAM_CAL=['RGB_L','RGB_R','Event_L','Event_R','Thermal_L','Thermal_R']
PC_CAL=['Livox','Ouster','Radar']
W,H=960,600
def text(im,s,xy=(12,28),scale=.65,color=(245,245,245)):
    cv2.putText(im,s,xy,cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)
def camera(rel):
    p=ROOT/'data'/rel
    if p.suffix=='.npz':
        with np.load(p,allow_pickle=False) as z:
            if 'event' in z:
                a=z['event'];im=np.full((704,1152,3),245,np.uint8)
                x,y=a[:,0].astype(int),a[:,1].astype(int)
                valid=(x>=0)&(x<1152)&(y>=0)&(y<704)
                im[y[valid],x[valid]]=np.where(a[valid,2,None]==1,[40,40,230],[230,60,40])
            else: im=z['image']
    else: im=cv2.imread(str(p),cv2.IMREAD_UNCHANGED)
    assert im is not None,str(p)
    if im.dtype!=np.uint8: im=np.uint8(np.clip(im,0,255))
    if im.ndim==2: im=cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
    return im[:,:,:3]
def project(pc,cal,sensor,cam,shape):
    xyz=np.column_stack([pc[k] for k in ['x','y','z']]).astype(float)
    T=np.asarray(cal[sensor][cam],float);K=np.asarray(cal[cam]['intrinsic'],float)
    camxyz=xyz@T[:3,:3].T+T[:3,3]
    uv,_=cv2.projectPoints(xyz,cv2.Rodrigues(T[:3,:3])[0],T[:3,3],K,np.zeros(4))
    uv=uv[:,0];hh,ww=shape[:2]
    v=np.isfinite(uv).all(1)&np.isfinite(camxyz).all(1)&(camxyz[:,2]>1)&(uv[:,0]>=0)&(uv[:,0]<ww)&(uv[:,1]>=0)&(uv[:,1]<hh)
    return uv[v],camxyz[v,2],v
def depth_map(pcs,cal,cam,shape):
    uvz=[project(p,cal,n,cam,shape) for p,n in zip(pcs,PC_CAL)]
    uv=np.concatenate([a[0] for a in uvz]);z=np.concatenate([a[1] for a in uvz])
    uv=uv*np.array([W/shape[1],H/shape[0]])
    yy,xx=np.mgrid[0:H:4,0:W:4];grid=np.column_stack([xx.ravel(),yy.ravel()])
    if len(z):
        dist,idx=cKDTree(uv).query(grid); dep=np.where(dist<40,np.clip(z[idx],1,120),120)
        coverage=float(np.mean(dist<40))
    else: dep=np.full(len(grid),120.);coverage=0.
    dep=cv2.resize(dep.reshape(yy.shape).astype('float32'),(W,H),interpolation=cv2.INTER_LINEAR)
    dep=cv2.GaussianBlur(dep,(0,0),15)
    return dep,coverage
def variants(im,dep,i):
    a=cv2.resize(im,(W,H)).astype('float32')/255
    trans=np.exp(-.035*dep)[:,:,None]
    fog=a*trans+np.array([.82,.84,.85])*(1-trans)
    night=np.power(np.clip(a,0,1),1.65)*np.array([.34,.27,.23])
    rain=a*.70+np.array([.10,.085,.075])
    streak=np.zeros_like(a,dtype='float32');rng=np.random.default_rng(42)
    for x,y,speed,length in zip(rng.uniform(0,W,600),rng.uniform(0,H,600),rng.uniform(24,60,600),rng.uniform(8,22,600)):
        x=int((x+i*8)%W);y=int((y+i*speed)%H)
        cv2.line(streak,(x,y),(x+int(length*.3),y+int(length)),(.35,.32,.29),1,cv2.LINE_AA)
    rain=np.clip(rain+cv2.GaussianBlur(streak,(3,3),.5),0,1)
    return {k:np.uint8(np.clip(v,0,1)*255) for k,v in [('original',a),('fog',fog),('rain',rain),('night',night)]}
def pc_panel(pc,cal,sensor,rgb):
    uv,z,valid=project(pc,cal,sensor,'RGB_L',rgb.shape)
    panel=cv2.resize(rgb,(480,300))//3
    uv=uv*np.array([480/rgb.shape[1],300/rgb.shape[0]])
    values=pc['doppler'][valid] if sensor=='Radar' else z
    scaled=np.clip((values+10)/20,0,1) if sensor=='Radar' else np.clip(values/100,0,1)
    colors=cv2.applyColorMap(np.uint8(scaled*255),cv2.COLORMAP_TURBO).reshape(-1,3)
    for xy,c in zip(uv.astype(int),colors): cv2.circle(panel,tuple(xy),1,tuple(int(v) for v in c),-1)
    return panel,len(z)
def tile(im,name):
    panel=cv2.resize(im,(480,276));bar=np.full((24,480,3),22,np.uint8);text(bar,name,(8,18),.50)
    return np.vstack([bar,panel])
def video(name,frames,fps):
    h,w=frames[0].shape[:2]
    cmd=['ffmpeg','-nostdin','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s',str(w)+'x'+str(h),'-r',str(fps),'-i','-','-an','-c:v','libx264','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/(name+'.mp4'))]
    p=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    for f in frames:p.stdin.write(f.tobytes())
    p.stdin.close();assert p.wait()==0
def run():
    s=json.loads((ROOT/'synthetic-selection.json').read_text());seq=s['sequences']['Clear'];d=labels(seq);cal=d['meta']['calibration'];frames=d['info'][:48]
    ts=np.array([int(f['time_stamp']) for f in frames],np.int64);rel=(ts-ts[0])/1e9
    fps=(len(frames)-1)/rel[-1];poses=np.array([f['pose'] for f in frames]);position=poses[:,:3,3]
    dist=np.r_[0,np.cumsum(np.linalg.norm(np.diff(position,axis=0),axis=1))]
    speed=np.r_[0,np.diff(dist)/np.diff(rel)];selected=np.linspace(0,47,16,dtype=int)
    movies={k:[] for k in ['original','fog','rain','night','comparison','all_sensors']};banks={k:[] for k in ['sensors','original','fog','rain','night']};stats=[]
    for i,f in enumerate(frames):
        cams=[camera(f['sensor'][k]) for k in CAM_KEYS];pcs=[np.load(ROOT/'data'/f['sensor'][k],allow_pickle=False) for k in PC_KEYS]
        deps=[depth_map(pcs,cal,cam,im.shape) for cam,im in zip(CAM_CAL[:2],cams[:2])]
        changed=[variants(im,dep[0],i) for im,dep in zip(cams[:2],deps)]
        panels=[pc_panel(p,cal,n,cams[0]) for p,n in zip(pcs,PC_CAL)]
        sensors=cams+[p[0] for p in panels]
        rows=[np.hstack([tile(sensors[j],NAMES[j]) for j in range(r*3,r*3+3)]) for r in range(3)]
        footer=np.full((100,1440,3),18,np.uint8)
        text(footer,'Original recorded sensors | '+seq+' | t=%.2fs'%rel[i],(14,27))
        text(footer,'Pose path %.2fm | speed %.2fm/s | radar Doppler median %.2f'%(dist[i],speed[i],float(np.median(pcs[2]['doppler']))),(14,58))
        text(footer,'LiDAR color: camera depth 0-100m | radar color: stored Doppler -10 to 10 | camera images are display values',(14,86),.55)
        mosaic=np.vstack(rows+[footer]);movies['all_sensors'].append(mosaic)
        views=[]
        for name in ['original','fog','rain','night']:
            label=('RECORDED BASE' if name=='original' else 'SYNTHETIC '+name.upper())+' | t=%.2fs'%rel[i]
            bar=np.full((40,W,3),18,np.uint8);text(bar,label,(12,27))
            frame=np.vstack([bar,changed[0][name]]);movies[name].append(frame);views.append(cv2.resize(frame,(720,480)))
        movies['comparison'].append(np.vstack([np.hstack(views[:2]),np.hstack(views[2:])]))
        if i in selected:
            banks['sensors'].append(np.stack([cv2.resize(x,(384,384))[:,:,::-1] for x in sensors]))
            for name in ['original','fog','rain','night']:banks[name].append(np.stack([cv2.resize(v[name],(384,384))[:,:,::-1] for v in changed]))
        stats.append(dict(frame=i,time_s=float(rel[i]),depth_coverage=[v[1] for v in deps],point_counts=[len(p) for p in pcs],projected_counts=[p[1] for p in panels],sensor_dtypes=[str(p.dtype) for p in pcs],pose=poses[i].tolist()))
        if i%8==0: print('rendered',i+1,flush=True)
    for name,ims in movies.items():video(name,ims,fps)
    cv2.imwrite(str(OUT/'comparison.jpg'),movies['comparison'][24]);cv2.imwrite(str(OUT/'all_sensors.jpg'),movies['all_sensors'][24])
    np.savez_compressed(OUT/'model_inputs.npz',**{k:np.stack(v) for k,v in banks.items()})
    (OUT/'render_metrics.json').write_text(json.dumps(dict(sequence=seq,revision=s['revision'],frames=48,fps=fps,duration_sample_span_s=float(rel[-1]),timestamp_note='CFR uses average interval; exact nanosecond timestamps in manifest',frame_selection=selected.tolist(),path_length_m=float(dist[-1]),stats=stats,synthetic=dict(fog_beta_per_m=.035,depth_blur_sigma_px=15,fallback_depth_m=120,nearest_depth_radius_px=40,rain_seed=42,night_gamma=1.65)),indent=2))
    print('RENDER COMPLETE',fps,flush=True)
if __name__=='__main__':run()
