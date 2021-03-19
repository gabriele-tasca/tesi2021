import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import fract


def dma_1(z2d):
    pass




# name = "Adamello"
# z2d = np.load("/home/gaboloth/D/fisica/tesi/dati/npysquare/all/"+name+"-2d.npy")

z2d = fract.fbm2D(0.6, N=7)


s_min = 6
s_max = "auto"
min_nonzero = 0.99


M, N = z2d.shape

if s_max == "auto":
    s_max = min(N,M)//4
else:
    pass # (keep the passed s_max)


scales_flucts = np.zeros((s_max - s_min ,2))









# remove nan values (from linear regression on too few data points)
scales_flucts = scales_flucts[~np.isnan(scales_flucts).any(axis=1)]


# return scales_flucts[:,0], scales_flucts[:,1]

scales, flucts = scales_flucts[:,0], scales_flucts[:,1]

dfa_H, dfa_c, pcov = fract.autoseeded_weighted_power_law_fit(scales, flucts, sigmas=scales)


plt.figure()
plt.scatter(scales, flucts)
plt.plot(scales, fract.power_law(scales, dfa_H, dfa_c), color="springgreen", label="dfa")

plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.show()