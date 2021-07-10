import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import fract
import pandas as pd

N = 10
stat_n = 0

first_time = True
try:
    disk_np_res = np.load("box-exact-N10-test.npy")
    first_time = False
except FileNotFoundError:
    first_time = True

hs = (np.arange(0.2, 1.0, 0.1))

# don't overwrite...
results = [ [] for h in hs]

means = np.zeros(np.arange(0.2, 1.0, 0.1).shape)
stdevs = np.zeros(np.arange(0.2, 1.0, 0.1).shape)


for H_i, H in enumerate(np.arange(0.2, 1.0, 0.1)):
    gen_params = fract.save_fbm2D_exact_generator_params(H,N)
    print(H)
    loc_result_arr = np.zeros(stat_n)

    for j in range(0,stat_n//2):
        print("    ",j)
        z2d1, z2d2 = fract.fbm2D_exact_from_generator(*gen_params)
        try:
            res1 = fract.box_counting_H(z2d1, data=False)
            loc_result_arr[2*j] = res1
        except:
            res1 = np.nan

        try:
            res2 = fract.box_counting_H(z2d2, data=False)
        except:
            res1 = np.nan
        loc_result_arr[2*j+1] = res2

        results[H_i].append(res1)
        results[H_i].append(res2)





np_res = np.array(results)
if first_time == False:
    disk_np_res = np.hstack(( disk_np_res, np_res  ))
elif first_time == True:
    disk_np_res = np_res

np.save("box-exact-N10-test.npy", disk_np_res)

print(disk_np_res.shape)

%matplotlib tk

filename = "box-exact-response.png"
means = np.mean(disk_np_res, axis=1)
stdevs = np.std(disk_np_res, axis=1)
plt.errorbar(hs, means, yerr=stdevs)
plt.xlabel("Generation H")
plt.ylabel("Detected H")
plt.plot(hs,hs)
plt.savefig(filename, bbox_inches='tight')

