import os,sys,json,time
import numpy as np,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *
sys.path.insert(0,str(ROOT.parent/'vjepa2'))
from src.hub.backbones import vjepa2_1_vit_base_384
from evals.hub.preprocessor import vjepa2_preprocessor
OUT=ROOT/'outputs'
def unit(x):return x/max(float(np.linalg.norm(x)),1e-12)
def distance(a,b):return float(np.clip(1-np.dot(unit(a),unit(b)),0,2))
model,predictor=vjepa2_1_vit_base_384(pretrained=True)
del predictor
model=model.eval().cuda().float();transform=vjepa2_preprocessor(crop_size=384)
z=np.load(OUT/'model_inputs.npz',allow_pickle=False);records=[]
def encode(a,name):
    x=transform(a.copy())[0].unsqueeze(0).cuda().float()
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    with torch.inference_mode():y=model(x)
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    assert torch.isfinite(y).all(),name
    e=y.mean(1)[0].cpu().numpy()
    records.append(dict(name=name,shape=list(y.shape),seconds=elapsed,peak_allocated_bytes=torch.cuda.max_memory_allocated(),finite=True))
    print(name,elapsed,flush=True);return e
base=np.stack([encode(z['original'][:,i] if i<2 else z['sensors'][:,i],str(i)) for i in range(9)])
repeat=encode(z['original'][:,0],'repeat_original_left')
groups={'RGB':[0,1],'Event':[2,3],'Thermal':[4,5],'LiDAR':[6,7],'Radar':[8]}
def families(a):return {k:unit(np.mean([unit(a[i]) for i in ids],axis=0)) for k,ids in groups.items()}
def fuse(a):return unit(np.mean(list(families(a).values()),axis=0))
results=[];embeddings={'original':base};f0=fuse(base);rgb0=families(base)['RGB']
for name in ['fog','rain','night']:
    a=base.copy()
    for side in range(2):a[side]=encode(z[name][:,side],name+str(side))
    embeddings[name]=a
    results.append(dict(variant=name,rgb_cosine_distance=distance(rgb0,families(a)['RGB']),all_sensor_late_fusion_cosine_distance=distance(f0,fuse(a))))
family=families(base);ablations=[]
for omitted in groups:
    remaining=unit(np.mean([v for k,v in family.items() if k!=omitted],axis=0))
    ablations.append(dict(omitted=omitted,cosine_distance=distance(f0,remaining)))
metrics=dict(model='V-JEPA 2.1 ViT-B 384',torch=torch.__version__,device=torch.cuda.get_device_name(0),dtype='float32',attention='Upstream PyTorch SDPA automatic selection',repeat_relative_l2=float(np.linalg.norm(base[0]-repeat)/np.linalg.norm(base[0])),variants=results,ablations=ablations,inferences=records,method='Each of 9 sensor visualizations encoded separately. Unit-normalized embeddings averaged within 5 sensor families, then equally averaged and normalized. Only RGB streams altered. This is an untrained late-fusion baseline, not a jointly trained multimodal model.',limits='Single 48-frame sequence; 16 frames encoded. Sensor visualizations and RGB-camera pretraining can bias comparisons. No detection accuracy, physical simulator validation, model training, or pixel generation by V-JEPA.')
(OUT/'vjepa_metrics.json').write_text(json.dumps(metrics,indent=2));np.savez_compressed(OUT/'embeddings.npz',**embeddings)
fig,ax=plt.subplots(1,2,figsize=(12,4),layout='constrained');x=np.arange(3)
ax[0].bar(x-.18,[v['rgb_cosine_distance'] for v in results],.36,label='RGB only')
ax[0].bar(x+.18,[v['all_sensor_late_fusion_cosine_distance'] for v in results],.36,label='All-sensor late fusion')
ax[0].set(xticks=x,xticklabels=[v['variant'] for v in results],ylabel='Cosine distance from recorded base',title='Synthetic RGB perturbations; other sensors fixed');ax[0].legend()
ax[1].bar(list(groups),[v['cosine_distance'] for v in ablations]);ax[1].set(ylabel='Cosine distance',title='Remove one sensor family from baseline')
fig.savefig(OUT/'experiment.png',dpi=160);plt.close(fig)
print(json.dumps(dict(repeat_relative_l2=metrics['repeat_relative_l2'],variants=results,ablations=ablations)),flush=True)
print('EVALUATION COMPLETE',flush=True)
