import numpy as np
import matplotlib.pyplot as plt

import numpy.ctypeslib as npct
from ctypes import c_int
from ctypes import c_double


#############
# void dma(double * z2d, int M, int N, double * scales_flucts_out, int s_min, int s_max, double min_nonzero)
# #############


# input type for the cos_doubles function
# must be a double array, with single dimension that is contiguous
array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')

# load the library, using numpy mechanisms
libfbm2D = npct.load_library("libfbm2D", ".")

# setup the return types and argument types
libfbm2D.dma.restype = None
libfbm2D.dma.argtypes = [c_double, c_int, c_double, array_1d_double]


def c_fbm2D(H, N, base_stdev=1.0):
    out_array = np.zeros((2**N + 1)**2)
    return libfbm2D.dma(H, N, base_stdev, out_array)




z2d = c_fbm2D(H=0.6, N=8)
plt.imshow(z2d)