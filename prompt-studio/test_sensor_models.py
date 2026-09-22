import sys,numpy as np,cv2
sys.path.insert(0,'../multisensor')
from common import labels
from catalog import SOURCES,DATA
from sensor_synthesis import *
f=labels(SOURCES['daylight']['sequence'])['info'][0]
p=params(dict(strength=.7,effects=['rain','night']));zero=params(dict(strength=.7,effects=['original']))
for kind in ['livox','ouster','radar']:
 a=np.load(DATA/'data'/f['sensor'][kind+'_path'],allow_pickle=False)
 b,idx,flag,stats=point_cloud(a,p,np.random.default_rng(123),kind)
 c=point_cloud(a,p,np.random.default_rng(123),kind)[0]
 assert all(np.array_equal(b[n],c[n]) for n in b.dtype.names)
 assert not np.array_equal(a,b)
 identity=point_cloud(a,zero,np.random.default_rng(123),kind)[0];assert np.array_equal(a,identity)
 print(kind,len(a),len(b),'deterministic + identity PASS')
with np.load(DATA/'data'/f['sensor']['event_left_path']) as z:a=z['event']
b,stats=event_packet(a,p,np.random.default_rng(123));c=event_packet(a,p,np.random.default_rng(123))[0]
assert np.array_equal(b,c) and not np.array_equal(a,b)
assert np.array_equal(a,event_packet(a,zero,np.random.default_rng(123))[0]);print('event',stats,'PASS')
a=cv2.imread(str(DATA/'data'/f['sensor']['thermal_left_path']));b=thermal_image(a,p,np.random.default_rng(123))
assert not np.array_equal(a,b) and np.array_equal(a,thermal_image(a,zero,np.random.default_rng(123)))
assert np.array_equal(b,thermal_image(a,p,np.random.default_rng(123)));print('thermal deterministic + identity PASS')
