import json,time,shutil,subprocess,zipfile
from pathlib import Path
import numpy as np
import cv2
from sensor_synthesis import build_bundle,sha,STREAMS,VERSION,serial

def validate_bundle(target,manifest):
 counts={k:0 for k in STREAMS};changed={k:0 for k in STREAMS};events=points=0
 timestamps=[r['timestamp_ns'] for r in manifest['records']]
 assert len(timestamps)==48 and all(b>a for a,b in zip(timestamps,timestamps[1:]))
 for record in manifest['records']:
  for name,item in record['streams'].items():
   path=target/item['file'];assert path.is_file() and sha(path)==item['sha256'];assert sha(item['source_file'])==item['source_sha256']
   counts[name]+=1;changed[name]+=bool(item['stats']['changed'])
   if name.startswith('event'):
    with np.load(path,allow_pickle=False) as z:a=z['event']
    assert a.ndim==2 and a.shape[1]==4 and a.dtype==np.uint32 and len(a)>0
    assert np.all(a[:,0]<1152) and np.all(a[:,1]<704) and np.all(a[:,2]<=1)
    assert np.all(a[1:,3]>=a[:-1,3]);events+=len(a)
   elif name in ['livox','ouster','radar']:
    a=np.load(path,allow_pickle=False);assert len(a)>0
    assert all(np.isfinite(a[n]).all() for n in a.dtype.names);points+=len(a)
    with np.load(path.with_name(path.stem+'_provenance.npz'),allow_pickle=False) as z:
     assert len(z['source_index'])==len(a)==len(z['return_kind'])
    if name=='ouster':
     r=np.sqrt(sum(a[n].astype(float)**2 for n in ['x','y','z']))
     if manifest['parameters']['strength']:assert np.max(np.abs(a['range']-r*1000))<.6
   else:
    a=cv2.imread(str(path));assert a is not None
    if name.startswith('rgb'):assert a.shape==(600,960,3)
 assert all(n==48 for n in counts.values())
 if manifest['parameters']['strength']:assert all(n>0 for n in changed.values()),changed
 assert np.array_equal(np.load(target/'timestamps_ns.npy'),timestamps)
 assert np.load(target/'poses.npy').shape==(48,4,4)
 return dict(valid=True,primary_sensor_files=sum(counts.values()),files_per_stream=counts,changed_frames=changed,event_count=events,point_count=points,source_hashes_unchanged=True)

def generate(req,out,progress):
 import engine as e
 start=time.perf_counter();(out/'request.json').write_text(json.dumps(req,indent=2))
 cache=e.source_cache(req['source'],progress);meta=json.loads((cache/'complete.json').read_text())
 with np.load(cache/'frames.npz',allow_pickle=False) as z:rgb=z['rgb'];depth=z['depth'];bank=z['bank']
 changed,manifest=build_bundle(req,out,progress,rgb,depth,meta)
 shutil.copyfile(cache/'sensors.mp4',out/'sensors.mp4')
 progress(79,'V-JEPA 2.1: encoding all nine generated sensor streams on GPU 1')
 if (cache/'baseline.npy').exists():base=np.load(cache/'baseline.npy',allow_pickle=False)
 else:base=np.stack([e.encode_model(bank[:,j]) for j in range(9)]);np.save(cache/'baseline.npy',base)
 features=[]
 for j,name in enumerate(STREAMS):
  progress(79+j,'V-JEPA 2.1: '+name);features.append(e.encode_model(changed[:,j]))
 features=np.stack(features);assert features.shape==base.shape==(9,768) and np.isfinite(features).all()
 np.savez_compressed(out/'embeddings.npz',source=base,generated=features,streams=np.array(STREAMS))
 distances={k:e.distance(e.family(base)[k],e.family(features)[k]) for k in e.GROUPS}
 progress(89,'Checking every exported sensor file and video')
 checks=validate_bundle(out/'sensor_data',manifest);video_checks=[]
 for name in ['video.mp4','comparison.mp4','synthetic_sensors.mp4','sensors.mp4']:
  path=out/name;probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_entries','stream=codec_name,width,height,nb_read_frames,avg_frame_rate,duration','-of','json',str(path)]))
  subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(path),'-f','null','-'],check=True)
  assert int(probe['streams'][0]['nb_read_frames'])==48
  video_checks.append(dict(file=name,sha256=sha(path),probe=probe))
 result=dict(sensor_version=VERSION,frames=48,duration_s=48/meta['fps'],fps=meta['fps'],depth_support=meta['depth_support'],rgb_distance=distances['RGB'],fused_distance=e.distance(e.fused(base),e.fused(features)),family_distances=distances,sensor_files=432,sensor_archive='sensor_data.zip')
 details=dict(**result,request=req,model='V-JEPA 2.1 ViT-B 384 FP32',gpu=e.torch.cuda.get_device_name(0),method='Encode 16-frame visualizations of each of nine generated streams. Embeddings are not raw sensor decoders.',sensor_validation=checks,video_checks=video_checks,source_metadata=meta,limitations=manifest['limits'],seconds_before_archive=time.perf_counter()-start)
 (out/'metrics.json').write_text(json.dumps(details,default=serial,indent=2))
 progress(94,'Packaging video, sensor files, calibration and V-JEPA embeddings')
 archive=out/'sensor_data.zip'
 with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
  for path in sorted((out/'sensor_data').rglob('*')):
   if path.is_file():z.write(path,path.relative_to(out))
  for name in ['video.mp4','comparison.mp4','synthetic_sensors.mp4','embeddings.npz','metrics.json','request.json']:z.write(out/name,name)
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None
  assert all(not p.startswith('/') and '..' not in Path(p).parts for p in z.namelist())
 result.update(archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),seconds=time.perf_counter()-start)
 (out/'validation.json').write_text(json.dumps(dict(**checks,archive_crc_valid=True,archive_sha256=result['archive_sha256'],all_nine_streams_encoded=True),indent=2))
 return result
