import numpy as np
import matplotlib.pyplot as plt

import fract

N = 10
stat_n = 10

first_time = True
try:
    disk_np_res = np.load("dfa-spectral-N"+str(N)+"-test.npy")
    first_time = False
except FileNotFoundError:
    first_time = True

print("first_time? ", first_time)
hs = (np.arange(0.2, 1.0, 0.1))

# don't overwrite...
results = [ [] for h in hs]

means = np.zeros(hs.shape)
stdevs = np.zeros(hs.shape)


for H_i, H in enumerate(hs):
    print(H)

    for j in range(0,stat_n):
        print("    ",j)
        z2d = fract.fbm2D_spectral(H, N)

        try:
            res1 = fract.dfa_H(z2d, data=False)
            print("        ",res1)
        except Exception as ex:
            res1 = np.nan
            print("Exception ", ex)

        results[H_i].append(res1)



np_res = np.array(results)
if first_time == False:
    disk_np_res = np.hstack(( disk_np_res, np_res  ))
elif first_time == True:
    disk_np_res = np_res

np.save("dfa-spectral-N"+str(N)+"-test.npy", disk_np_res)

print(disk_np_res.shape)

%matplotlib tk

filename = "dfa-spectral-response.png"
means = np.mean(disk_np_res, axis=1)
stdevs = np.std(disk_np_res, axis=1)
plt.errorbar(hs, means, yerr=stdevs)
plt.xlabel("Generation H")
plt.ylabel("Detected H")
plt.plot(hs,hs)
plt.savefig(filename, bbox_inches='tight')

