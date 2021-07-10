import numpy as np
import matplotlib.pyplot as plt

import fract

N = 10
contour_H = 0.8
stat_n = 6

first_time = True
try: 
    disk_np_res = np.load("alt1-box-spectral-prof-cH"+str(contour_H)+"-N"+str(N)+"-test.npy")
    first_time = False
except FileNotFoundError:
    first_time = True

hs = (np.arange(0.2, 1.0, 0.1))

# don't overwrite...
results = [ [] for h in hs]

means = np.zeros(hs.shape)
stdevs = np.zeros(hs.shape)


for H_i, H in enumerate(hs):
    print(H)

    for j in range(0,stat_n):
        print("    ",j)
        z2d = fract.fbm2D_spectral(H,N)
        z2d = fract.cut_profile(z2d, contour_H)

        try:
            detect_H1 = fract.box_counting_H(z2d, data=False)
            print("        ",detect_H1)
        except Exception as ex:
            res1 = np.nan
            print("Exception ", ex)

        results[H_i].append(detect_H1)





np_res = np.array(results)
if first_time == False:
    disk_np_res = np.hstack(( disk_np_res, np_res  ))
elif first_time == True:
    disk_np_res = np_res

# SAVING
np.save("alt1-box-spectral-prof-cH"+str(contour_H)+"-N"+str(N)+"-test.npy", disk_np_res)


means = np.mean(disk_np_res, axis=1)
stdevs = np.std(disk_np_res, axis=1)
plt.errorbar(hs, means, yerr=stdevs)
plt.plot(hs,hs)



