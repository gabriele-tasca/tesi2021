import numpy as np
import matplotlib.pyplot as plt

import fract

N = 10
stat_n = 10

first_time = True
try:
    disk_np_res = np.load("higuchi-exact-N10-test.npy")
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

    for j in range(0,stat_n//2):
        print("    ",j)
        z2d1, z2d2 = fract.fbm2D_exact_from_generator(*gen_params)
        res1 = 0
        res2 = 0
        try:
            detect_H1 = fract.higuchi_H(z2d1, data=False)
            print("        ",detect_H1)
        except Exception as ex:
            res1 = np.nan
            print("Exception ", ex)

        try:
            detect_H2 = fract.higuchi_H(z2d2, data=False)
            # print("        ",res2)
        except Exception as ex:
            res2 = np.nan
            print("Exception ", ex)


        results[H_i].append(detect_H1)
        results[H_i].append(detect_H2)





np_res = np.array(results)
if first_time == False:
    disk_np_res = np.hstack(( disk_np_res, np_res  ))
elif first_time == True:
    disk_np_res = np_res

# SAVING
np.save("higuchi-exact-N10-test.npy", disk_np_res)

print(disk_np_res.shape)

%matplotlib tk

filename = "higuchi-exact-response.png"
means = np.mean(disk_np_res, axis=1)
stdevs = np.std(disk_np_res, axis=1)
plt.errorbar(hs, means, yerr=stdevs)
plt.xlabel("Generation H")
plt.ylabel("Detected H")
plt.plot(hs,hs)
plt.savefig(filename, bbox_inches='tight')




