import numpy as np
import matplotlib.pyplot as plt

import fract

N = 10
stat_n = 4

first_time = True
try:
    disk_np_res = np.load("box-altexact-N10-test.npy")
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
            res1 = fract.box_counting_H(z2d1, data=False, log_spacing=True)
            loc_result_arr[2*j] = res1
        except:
            res1 = np.nan

        try:
            res2 = fract.box_counting_H(z2d2, data=False, log_spacing=True)
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

np.save("box-altexact-N10-test.npy", disk_np_res)


means = np.mean(disk_np_res, axis=1)
stdevs = np.std(disk_np_res, axis=1)
plt.errorbar(hs, means, yerr=stdevs)
plt.plot(hs,hs)



########### extra tests

H = 0.3
N = 10
gen_params = fract.save_fbm2D_exact_generator_params(H,N)
z2d1, z2d2 = fract.fbm2D_exact_from_generator(*gen_params)

det_H, b_exp, b_c, b_k, scales, nbox_s = fract.logplot(fract.box_counting_H(z2d1, log_spacing=True))


