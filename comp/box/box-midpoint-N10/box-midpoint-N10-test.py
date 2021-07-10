import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import fract
import pandas as pd

N = 10
stat_n = 10

first_time = True
try:
    disk_np_res = np.load("box-midpoint-N10-test.npy")
    first_time = False
except FileNotFoundError:
    first_time = True

hs = np.arange(0.2, 1.0, 0.1)

# don't overwrite...
results = [ [] for h in hs]

means = np.zeros(hs.shape)
stdevs = np.zeros(hs.shape)


for H_i, H in enumerate(hs):
    print(H)

    for j in range(0,stat_n):
        print("    ",j)
        z2d1 = fract.fbm2D_midpoint(H, N)
        # plt.imshow(z2d1)
        # plt.show()
        try:
            res1 = fract.box_counting_H(z2d1, data=False)
        except Exception as ex:
            print(ex)
            res1 = np.nan


        results[H_i].append(res1)



np_res = np.array(results)
if first_time == False:
    disk_np_res = np.hstack(( disk_np_res, np_res  ))
elif first_time == True:
    disk_np_res = np_res

np.save("box-midpoint-N10-test.npy", disk_np_res)

print(disk_np_res.shape)

%matplotlib tk

filename = "box-midpoint-response.png"
means = np.mean(disk_np_res, axis=1)
stdevs = np.std(disk_np_res, axis=1)
plt.errorbar(hs, means, yerr=stdevs)
plt.xlabel("Generation H")
plt.ylabel("Detected H")
plt.plot(hs,hs)
plt.savefig(filename, bbox_inches='tight')

