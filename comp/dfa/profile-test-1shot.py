import numpy as np
import matplotlib.pyplot as plt
import dfa

import scipy.stats 

# plotting single segment regression
z2d = np.load("profiles/profile-10.npy")
side = np.sqrt(z2d.size)


H, c, scales, flucts = dfa.profile_dfa_from_z2d(z2d, s_max=(side/20).astype(int))


plt.scatter(scales,flucts, marker=".", color="deepskyblue")
plt.plot(scales, (10**c)*scales**H, color="purple")
plt.xscale("log")
plt.yscale("log")
plt.title("H = " + str(H))
plt.show()
