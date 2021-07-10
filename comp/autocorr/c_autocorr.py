
import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_int
from ctypes import c_double
import matplotlib.pyplot as plt

import fract
#############
# void autocorr(double * z2d, int M, int N, double * scales_flucts_out, int s_min, int s_max, double min_nonzero)
# #############


# input type for the cos_doubles function
# must be a double array, with single dimension that is contiguous
array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')

array_1d_int = npct.ndpointer(dtype=np.int, ndim=1, flags='CONTIGUOUS')

# load the library, using numpy mechanisms
libautocorr = npct.load_library("libautocorr", "./c_autocorr")

# setup the return types and argument types
libautocorr.autocorr.restype = None
libautocorr.autocorr.argtypes = [array_1d_double, c_int, c_int, array_1d_double, array_1d_int, c_int]


# def autocorr_1(z2d):

#     xwid, ywid = z2d.shape
#     autocorr_size = max(xwid, ywid)/2


#     r_out = np.arange(0.0, np.float64(autocorr_size), 1.0)
#     autocorr_out = np.empty_like(r_out)

#     libautocorr.autocorr(z2d.ravel(), xwid, ywid, autocorr_out, autocorr_size)

#     return r_out, autocorr_out




# if __name__ == "__main__":
#     z2d = fract.fbm2D_spectral(H=0.6, N=9)
#     z2d = fract.cut_profile(z2d, 0.99)

#     res = autocorr_1(z2d)


z2d = fract.fbm2D_spectral(H=0.6, N=8)
z2d = fract.cut_profile(z2d, 0.99)


xwid, ywid = z2d.shape
autocorr_size = min(xwid, ywid)//2

r_out = np.arange(0.0, np.float64(autocorr_size), 1.0)
autocorr_out = np.zeros(r_out.shape, dtype=float)
count_out = np.zeros(r_out.shape, dtype=int)

libautocorr.autocorr(z2d.ravel(), xwid, ywid, autocorr_out, count_out, autocorr_size)

autocorr_out = autocorr_out*10000 /count_out

plt.scatter(r_out, autocorr_out)



