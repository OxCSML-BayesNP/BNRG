import ctypes
from model import sample_w
import numpy as np

nu = -1.5
a = 0.1
b = 2.0
n = 1000000

import time
st = time.time()
w1 = sample_w(nu, a, b, n)
print time.time() - st

st = time.time()
w2 = np.zeros(n)
ctypes.CDLL("./gigrnd_.so").gigrnd(
        ctypes.c_double(nu),
        ctypes.c_double(a),
        ctypes.c_double(b),
        ctypes.c_int(n),
        ctypes.c_void_p(w2.ctypes.data))
print time.time() - st

print (1/w1).mean(), (1/w2).mean()
