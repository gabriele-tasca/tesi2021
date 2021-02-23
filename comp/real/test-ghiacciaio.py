import numpy as np
import matplotlib.pyplot as plt

import scipy.stats 


import fract

print("reading .npy")
z2d = np.load("/home/gaboloth/D/fisica/tesi/dati/npysquare/adamello-2d.npy")

print("calculating...")
(m,n) = z2d.shape

# old_flucts = flucts

# subselect = z2d[int(m*0.6) : int(m*0.9) , int(n*0.6) :int(n*0.9)  ]
subselect = z2d


# H, c, scales, flucts = fract.profile_var_rms_from_z2d(subselect)
H, c, scales, flucts = fract.profile_var_range_from_z2d(subselect)
# H, c, scales, flucts = fract.profile_dfa_from_z2d(subselect)
# H, freq_exp, c, scales, flucts = fract.profile_fourier_from_z2d(subselect, images=True, corr=True)

plt.scatter(scales,flucts, marker=".", color="deepskyblue")
plt.plot(scales, (10**c)*scales**H, color="purple")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("scale")
plt.ylabel("fluctuation")
plt.title("H = " + str(H))

plt.show()

# z2d[z2d == 0] = np.nan
plt.imshow(z2d, interpolation = 'none')
plt.show()

# subselect[subselect == 0] = np.nan
plt.imshow(subselect, interpolation = 'none')
plt.show()


####################à
