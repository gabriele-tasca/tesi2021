import numpy as np
import matplotlib.pyplot as plt

import fract

N = 10
stat_n = 10

first_time = True
try:
    disk_np_res = np.load("fourier-exact-N"+str(N)+"-test.npy")
    first_time = False
except FileNotFoundError:
    first_time = True

hs = (np.arange(0.2, 1.0, 0.1))

# don't overwrite...
results = [ [] for h in hs]

means = np.zeros(np.arange(0.2, 1.0, 0.1).shape)
stdevs = np.zeros(np.arange(0.2, 1.0, 0.1).shape)


for H_i, H in enumerate(hs):
    gen_params = fract.save_fbm2D_exact_generator_params(H,N)
    print(H)

    for j in range(0,stat_n//2):
        print("    ",j)
        z2d1, z2d2 = fract.fbm2D_exact_from_generator(*gen_params)
        res1 = 0
        res2 = 0
        try:
            det_H1 = fract.fourier_H(z2d1, data=False, windowing=True, fill_with_mean=False)
            print("        ",det_H1)
        except Exception as ex:
            res1 = np.nan
            print("Exception ", ex)

        try:
            det_H2 = fract.fourier_H(z2d2, data=False, windowing=True, fill_with_mean=False)
        except Exception as ex:
            res2 = np.nan
            print("Exception ", ex)


        results[H_i].append(det_H1)
        results[H_i].append(det_H2)





np_res = np.array(results)
if first_time == False:
    disk_np_res = np.hstack(( disk_np_res, np_res  ))
elif first_time == True:
    disk_np_res = np_res

# NOT SAVING
np.save("fourier-exact-N"+str(N)+"-test.npy", disk_np_res)


print(disk_np_res.shape)

%matplotlib tk

filename = "fourier-exact-response.png"
means = np.mean(disk_np_res, axis=1)
stdevs = np.std(disk_np_res, axis=1)
plt.errorbar(hs, means, yerr=stdevs)
plt.xlabel("Generation H")
plt.ylabel("Detected H")
plt.plot(hs,hs)
plt.savefig(filename, bbox_inches='tight')








# import numpy as np
# import matplotlib.pyplot as plt

# import fract


# H = 0.8
# N = 9
# # gen_params = fract.save_fbm2D_exact_generator_params(H,N)
# # z2d, z2d2 = fract.fbm2D_exact_from_generator(*gen_params)

# z2d = fract.fbm2D_exact(H,N)

# ## plt.imshow(z2d)


# freqs, powers = fract.fourier_1(z2d)
# det_exp, det_c, pcov = fract.autoseeded_weighted_power_law_fit(freqs, powers, sigmas=powers)

# plt.scatter(freqs, powers)
# plt.plot(freqs, fract.power_law(freqs, det_exp, det_c), color="red", label="H ="+str(H))
# plt.xscale("log")
# plt.yscale("log")

# print(fract.freq_exp_to_H(det_exp))