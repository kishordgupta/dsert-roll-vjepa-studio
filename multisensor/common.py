import pathlib, pickle, numpy as np
ROOT=pathlib.Path(__file__).resolve().parent
class SafeUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        allowed={('numpy','ndarray'),('numpy','dtype'),('numpy.core.multiarray','_reconstruct'),('numpy.core.multiarray','scalar'),('numpy._core.multiarray','_reconstruct'),('numpy._core.multiarray','scalar')}
        allowed.update({('numpy.core.numeric','_frombuffer'),('numpy._core.numeric','_frombuffer')})
        if (module,name) not in allowed: raise ValueError('Unsupported pickle global: '+module+'.'+name)
        return super().find_class(module,name)
def labels(seq):
    with open(ROOT/'data'/seq/'label.pkl','rb') as f: return SafeUnpickler(f).load()
