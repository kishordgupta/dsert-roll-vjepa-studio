"""Deterministic research augmentations, not calibrated sensor simulation."""
import json,hashlib,copy,zipfile
from pathlib import Path
import numpy as np
import cv2
from catalog import DATA,SOURCES
STREAMS=['rgb_left','rgb_right','event_left','event_right','thermal_left','thermal_right','livox','ouster','radar']
VERSION='sensor-augmentation-1.0'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def serial(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 raise TypeError(type(x).__name__)
def params(req):
 s=req['strength'];e=req['effects']
 if e==['original']:s=0.
 return dict(strength=s,fog=s*('fog' in e),rain=s*('rain' in e),snow=s*('snow' in e),night=s*('night' in e),warm=s*('warm' in e))
def event_packet(a,p,rng):
 if not p['strength']:return a.copy(),dict(retained=len(a),added=0,changed=False)
 fog,rain,snow,night=(p[k] for k in ['fog','rain','snow','night'])
 keep=np.clip(1-.38*fog-.16*rain-.19*snow-.35*night,.15,1)
 b=a[rng.random(len(a))<keep].copy();n=int(len(a)*(.002*p['strength']+.015*rain+.012*snow+.008*night))
 noise=np.empty((n,4),dtype=a.dtype)
 noise[:,0]=rng.integers(0,1152,n);noise[:,1]=rng.integers(0,704,n);noise[:,2]=rng.integers(0,2,n)
 lo,hi=int(a[:,3].min()),int(a[:,3].max())
 noise[:,3]=rng.integers(lo,hi+1,n)
 b=np.concatenate([b,noise]);b=b[np.argsort(b[:,3],kind='stable')]
 assert np.all(b[:,0]<1152) and np.all(b[:,1]<704) and np.all(b[:,2]<=1)
 return b,dict(retained=len(b)-n,added=n,source_time_min=lo,source_time_max=hi,changed=True)
def event_image(a):
 im=np.full((704,1152,3),255,np.uint8)
 im[a[:,1],a[:,0]]=np.where(a[:,2,None]>0,np.array([0,0,255],np.uint8),np.array([255,0,0],np.uint8))
 return im
def thermal_image(a,p,rng):
 if not p['strength']:return a.copy()
 weather=p['fog']+.7*p['rain']+.8*p['snow'];v=a.astype(np.float32)
 v=(v-127.5)*(1-.22*min(weather,1.5))+127.5
 if weather>0:v=cv2.GaussianBlur(v,(3,3),.35+.65*weather)
 noise=rng.normal(0,.4*p['strength']+1.8*weather,a.shape[:2])
 return np.uint8(np.clip(np.rint(v+noise[:,:,None]),0,255))
def point_cloud(a,p,rng,kind):
 n=len(a);idx=np.arange(n,dtype=np.int32);flag=np.zeros(n,np.uint8)
 if not p['strength']:return a.copy(),idx,flag,dict(input=n,output=n,clutter=0,changed=False)
 xyz=np.column_stack([a[k] for k in ['x','y','z']]).astype(np.float64);r=np.linalg.norm(xyz,axis=1)
 valid=np.isfinite(xyz).all(1)&(r>.05)
 if kind=='radar':
  beta=.0001*p['fog']+.0015*p['rain']+.002*p['snow'];sigma=.005*p['strength']+.035*p['rain']+.02*p['snow'];clutter_rate=0.
 else:
  beta=.012*p['fog']+.003*p['rain']+.005*p['snow'];sigma=.002*p['strength']+.009*p['rain']+.012*p['snow'];clutter_rate=.025*p['fog']+.008*p['rain']+.016*p['snow']
 keep=valid&(rng.random(n)<np.exp(-beta*r))
 idx=idx[keep];b=a[keep].copy();rr=r[keep];directions=xyz[keep]/rr[:,None]
 new_r=np.maximum(.05,rr+rng.normal(0,sigma,len(b)));flag=np.zeros(len(b),np.uint8)
 if clutter_rate:
  hit=rng.random(len(b))<clutter_rate
  new_r[hit]=rng.uniform(.05,1.,hit.sum())*np.minimum(rr[hit],30.);flag[hit]=1
 new_xyz=directions*new_r[:,None]
 for j,k in enumerate(['x','y','z']):b[k]=new_xyz[:,j]
 if kind=='radar':
  b['power']=np.maximum(0,b['power']*np.exp(-2*beta*rr)*(1+rng.normal(0,.01*p['strength'],len(b))))
  b['doppler']+=rng.normal(0,.005*p['strength']+.04*p['rain']+.03*p['snow'],len(b))
 else:
  b['intensity']=np.maximum(0,b['intensity']*np.exp(-2*beta*rr)*(1+rng.normal(0,.01*p['strength'],len(b))))
  if 'range' in b.dtype.names:b['range']=np.rint(new_r*1000).astype(np.uint32)
  if 'reflectivity' in b.dtype.names:b['reflectivity']=np.rint(b['reflectivity']*np.exp(-2*beta*rr)).astype(b['reflectivity'].dtype)
 assert all(np.isfinite(b[k]).all() for k in b.dtype.names)
 return b,idx,flag,dict(input=n,output=len(b),clutter=int(flag.sum()),changed=True,attenuation_per_m=beta,range_noise_std_m=sigma)
def build_bundle(req,out,progress,rgb,depth,meta):
 import render
 from common import labels
 from engine import apply_effects,encode_video,labeled
 records=labels(SOURCES[req['source']]['sequence']);frames=records['info'][:48];cal=records['meta']['calibration']
 target=out/'sensor_data';target.mkdir();p=params(req);generated=[];views=[];comparisons=[];movies=[];items=[]
 scaled=copy.deepcopy(cal)
 for name in ['RGB_L','RGB_R']:
  k=np.array(scaled[name]['intrinsic']).copy();k[0]*=.5;k[1]*=.5;scaled[name]['intrinsic']=k;scaled[name]['shape']=np.array([960,600])
 (target/'calibration.json').write_text(json.dumps(scaled,default=serial,indent=2))
 np.save(target/'poses.npy',np.asarray([f['pose'] for f in frames]));np.save(target/'timestamps_ns.npy',np.asarray(meta['timestamps_ns'],dtype=np.int64))
 for i,f in enumerate(frames):
  ims=[];pcs=[];row=dict(frame=i,timestamp_ns=int(f['time_stamp']),streams={})
  for j,name in enumerate(STREAMS):
   source=DATA/'data'/f['sensor'][name+'_path'];dest=target/name;dest.mkdir(exist_ok=True)
   rng=np.random.default_rng(np.random.SeedSequence([req['seed'],i,j]));stats={}
   if j<2:
    raw=rgb[i,j];b=apply_effects(raw,depth[i,j],i,req);path=dest/(str(i).zfill(4)+'.png');assert cv2.imwrite(str(path),b);ims.append(b)
    changed=not np.array_equal(raw,b);stats=dict(changed=changed,resolution=[960,600],calibration_scale=.5)
   elif j<4:
    with np.load(source,allow_pickle=False) as z:raw=z['event']
    b,stats=event_packet(raw,p,rng);path=dest/(str(i).zfill(4)+'.npz');np.savez_compressed(path,event=b);ims.append(event_image(b))
   elif j<6:
    raw=cv2.imread(str(source),cv2.IMREAD_COLOR)
    if raw is None:raise RuntimeError('Cannot decode '+str(source))
    b=thermal_image(raw,p,rng);path=dest/(str(i).zfill(4)+'.png');assert cv2.imwrite(str(path),b);ims.append(b);stats=dict(changed=not np.array_equal(raw,b),non_radiometric=True)
   else:
    raw=np.load(source,allow_pickle=False);b,indices,flags,stats=point_cloud(raw,p,rng,name);path=dest/(str(i).zfill(4)+'.npy');np.save(path,b);pcs.append(b)
    np.savez_compressed(dest/(str(i).zfill(4)+'_provenance.npz'),source_index=indices,return_kind=flags)
   row['streams'][name]=dict(file=path.relative_to(target).as_posix(),sha256=sha(path),source_file=str(source),source_sha256=sha(source),stats=stats)
  panels=[render.pc_panel(pc,scaled,sensor,ims[0])[0] if len(pc) else np.zeros((300,480,3),np.uint8) for pc,sensor in zip(pcs,render.PC_CAL)]
  sensors=ims+panels
  if i in meta['indices']:generated.append(np.stack([cv2.resize(im,(384,384))[:,:,::-1] for im in sensors]))
  title='SYNTHETIC SENSOR AUGMENTATION' if p['strength'] else 'IDENTITY CONTROL'
  views.append(labeled(ims[0],title));comparisons.append(np.hstack([cv2.resize(labeled(rgb[i,0],'RECORDED SOURCE'),(720,480)),cv2.resize(labeled(ims[0],title),(720,480))]))
  tiles=[np.hstack([render.tile(sensors[j],render.NAMES[j]) for j in range(k*3,k*3+3)]) for k in range(3)]
  footer=np.full((52,1440,3),18,np.uint8);render.text(footer,title+' | '+str(i+1)+'/48 | seed '+str(req['seed']),(12,32),.6)
  movies.append(np.vstack(tiles+[footer]));items.append(row)
  if i%4==0:progress(36+int(i/48*38),'Generating all nine sensor streams '+str(i+1)+'/48')
 progress(75,'Encoding video and synchronized sensor visualization')
 for name,seq in [('video.mp4',views),('comparison.mp4',comparisons),('synthetic_sensors.mp4',movies)]:encode_video(out/name,seq,meta['fps'])
 manifest=dict(version=VERSION,synthetic=True,request=req,parameters=p,frames=48,streams=STREAMS,records=items,calibration='calibration.json',poses='poses.npy',timestamps='timestamps_ns.npy',model_role='V-JEPA 2.1 encodes visualizations of all nine generated streams. It does not decode these measurements.',timebase=dict(frame='Original DSERT-RoLL time_stamp, integer nanoseconds; common frame association.',event='Column 3 source sensor-local clock retained; units and global clock mapping unverified. Added events sampled within each original packet time interval.'),units=dict(rgb='uint8 BGR in memory, lossless PNG; 960x600, half-scale original camera intrinsics',thermal='uint8 display intensity, not temperature or radiometric data',event='uint32 Nx4: x,y,polarity(0/1),source-local timestamp; 1152x704',lidar='Original structured dtype; xyz meters; Ouster range recomputed as rounded norm(xyz)*1000 millimeters',radar='Original structured dtype; xyz meters; power and doppler retain source numeric units; physical units not independently calibrated'),methods=dict(rgb='Depth-informed fog and seeded image-space precipitation/lighting effects.',event='Contrast-dependent packet thinning plus seeded uniform background/precipitation noise events; retained events unchanged.',thermal='Weather-dependent display contrast reduction, blur and read-noise perturbations. Visible-light darkness does not become thermal darkness.',lidar='Range-dependent stochastic dropout, exponential intensity attenuation, radial Gaussian jitter and occasional nearer returns on recorded beams.',radar='Mild range-dependent dropout, relative power perturbation and radial range/Doppler noise; fog effect weaker than rain/snow.',pose='Recorded poses and timestamps preserved as references. No new motion, GPS or IMU measurements.'),limits=['Research augmentation with hand-set parameters, not a physically calibrated simulator or a trained multimodal sensor decoder.','Streams share prompt, frame timing and source geometry; stochastic weather effects are not generated from a shared physical particle field.','Event packets are not generated by continuous irradiance simulation; timestamp units are preserved without claiming absolute event-time synchronization.','Thermal outputs cannot be interpreted as temperature. Radar perturbations use native numeric units.','LiDAR line, tag and timing fields are inherited; synthetic closer returns are marked return_kind=1. All outputs belong to a synthetic augmented sample.','RGB raster downsampling requires the supplied scaled calibration. No new objects, geometry or routes.'])
 (target/'manifest.json').write_text(json.dumps(manifest,default=serial,indent=2))
 (target/'README.txt').write_text('SYNTHETIC DSERT-RoLL AUGMENTATION'+chr(10)+'Read manifest.json for schema, units, methods and scientific limits. Not V-JEPA-decoded measurements.'+chr(10))
 return np.stack(generated),manifest
