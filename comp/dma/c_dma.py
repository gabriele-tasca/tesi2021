
import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_int
from ctypes import c_double
import matplotlib.pyplot as plt

import fract
#############
# void dma(double * z2d, int M, int N, double * scales_flucts_out, int s_min, int s_max, double min_nonzero)
# #############


# input type for the cos_doubles function
# must be a double array, with single dimension that is contiguous
array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')

# load the library, using numpy mechanisms
libdma = npct.load_library("libdma", "./c_dma")

# setup the return types and argument types
libdma.dma.restype = None
libdma.dma.argtypes = [array_1d_double, c_int, c_int, array_1d_double, c_int, c_int, c_double]


def dma_1(z2d, min_nonzero=0.99, s_min=5, s_max="auto"):

    M, N = z2d.shape

    if s_max == "auto":
        s_max = min((M,N))//5
    else:
        # s_max = s_max
        pass

    if s_max < s_min:
        print("too few data")
        return "too few data"

    # s_min = 5
    # s_max = 21
    s_out = np.arange(start=s_min, stop=s_max).astype(float)
    f_out = np.empty_like(s_out)

    libdma.dma(z2d.ravel(), M, N, f_out, s_min, s_max, min_nonzero)

    return s_out, f_out




if __name__ == "__main__":
    # z2d = fract.fbm2D(H=0.6, N=9)
    # z2d = fract.cut_profile(z2d, 0.99)

    name = "Fellaria Est"
    z2d = np.load("/home/gaboloth/D/fisica/tesi/dati/npysquare/all/"+name+"-2d.npy")
    z2d = z2d * 0.0002
    
    # np.mean(z2d)
    # np.std(z2d)

    # shunk = z2d.shape[1]//3
    # z2d = z2d[ :, 2*shunk:3*shunk ]

    # plt.figure()
    # plt.imshow(z2d)
    # plt.show()

    # dfa_H, dfa_c, scales, flucts = fract.dfa_H(z2d)
    # plt.figure()
    # plt.scatter(scales, flucts)
    # plt.plot(scales, fract.power_law(scales, dfa_H, dfa_c), color="springgreen", label="dfa: H ="+str(dfa_H))

    # plt.xscale("log")
    # plt.yscale("log")
    # plt.legend()
    # plt.show()

    scales_out_real, flucts_out_real = dma_1(z2d)
    
    # plt.figure()
    # plt.scatter(scales_out_real, flucts_out_real )
    # plt.show()



    scales_out = scales_out_real
    flucts_out = np.sqrt(flucts_out_real)
    scales_out = scales_out[  ~np.isnan(flucts_out) ]
    flucts_out = flucts_out[  ~np.isnan(flucts_out) ]



    import matplotlib.pyplot as plt

    dma_1, dma_c, pcov = fract.autoseeded_weighted_power_law_fit(scales_out[:25], flucts_out[:25], sigmas=flucts_out[:25])
    np.min(scales_out)

    print("dma_1", dma_1)

    # ### branches
    # scale_f1 = 17
    scale_f2 = 25

    # dfa_H2, c2, pcov = fract.autoseeded_weighted_power_law_fit(scales_out[:scale_f1], flucts_out[:scale_f1], flucts_out[:scale_f1])
    dfa_H2, c2, pcov = fract.autoseeded_weighted_power_law_fit(scales_out[scale_f2:], flucts_out[scale_f2:], flucts_out[scale_f2:])
    # print("dfa_H2", dfa_H2)


    plt.figure()
    plt.scatter(scales_out, flucts_out)
    plt.plot(scales_out, fract.power_law(scales_out, dma_1, dma_c), color="springgreen", label="dma: H ="+str(dma_1))
    plt.plot(scales_out, fract.power_law(scales_out, dfa_H2, c2), color="red", label="dfa: H = "+str(dfa_H2))

    # plt.plot(scales_out, fract.power_law(scales_out, fourier_H, c2/0.8), color="purple", label="fourier: H = "+str(fourier_H))

    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.show()
