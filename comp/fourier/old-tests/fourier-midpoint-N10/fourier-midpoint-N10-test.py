import numpy as np
import matplotlib.pyplot as plt

import fract

N = 10
stat_n = 10

first_time = True
try:
    disk_np_res = np.load("fourier-exact-N10-test.npy")
    first_time = False
except FileNotFoundError:
    first_time = True

hs = (np.arange(0.2, 1.0, 0.1))

# don't overwrite...
results = [ [] for h in hs]

means = np.zeros(np.arange(0.2, 1.0, 0.1).shape)
stdevs = np.zeros(np.arange(0.2, 1.0, 0.1).shape)


for H_i, H in enumerate(np.arange(0.2, 1.0, 0.1)):
    print(H)

    for j in range(0,stat_n):
        print("    ",j)
        z2d = fract.fbm2D_midpoint(H, N)
        res1 = 0
        try:
            # res1 = fract.fourier_H(z2d, data=False)
            det_H, det_exp, det_c, freqs, powers = fract.fourier_H(z2d, data=True)

            # print("        ",res1)
            print("        ",det_H)
        except Exception as ex:
            res1 = np.nan
            print("Exception ", ex)

        # results[H_i].append(res1)
        results[H_i].append(det_H)



np_res = np.array(results)
if first_time == False:
    disk_np_res = np.hstack(( disk_np_res, np_res  ))
elif first_time == True:
    disk_np_res = np_res

# np.save("fourier-exact-N10-test.npy", disk_np_res)


means = np.mean(disk_np_res, axis=1)
stdevs = np.std(disk_np_res, axis=1)

plt.errorbar(hs, means+0.25, yerr=stdevs)
plt.plot(hs,hs)