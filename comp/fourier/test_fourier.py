import numpy as np
import matplotlib.pyplot as plt

import fourier

import fract

N = 8

stat_n = 30

start = 0.1
stop = 0.9001
step = 0.1
nhs = len(np.arange(start,stop=stop,step=step))
res = np.zeros( (nhs, 3))


for (i,H) in enumerate( np.arange(start,stop=stop,step=step) ):
    # H = start + i*step  
    stat_res = np.zeros(stat_n)
    print("H", H)
    for j in range(1, stat_n):

        # generate
        # Random.seed!(729 + i*34);
        data = fract.fbm2D(H,N=N)

        # dfa
        h_detected, freq_exp, c, freqs, powers = fourier.profile_fourier_from_z2d(data)
        stat_res[j] = h_detected
        # print("     ", j)

    av_h_detected = np.mean(stat_res)
    std_h_detected = np.std(stat_res)
    print("     av. H detected =", av_h_detected)
    
    newrow = np.transpose([H, av_h_detected, std_h_detected])
    res[i,:] = newrow

# print(res)
plt.errorbar(res[:,0], res[:,1], yerr=res[:,2]);
plt.xlabel("generation H")
plt.ylabel("detected H")
linex = [0.075,0.925]
liney = [0.075,0.925]
plt.plot(linex, liney)
plt.show()

# res10 = res

# past_reses = [res8, res9, res10]

# for i,r in enumerate(past_reses):
#     plt.errorbar(r[:,0], r[:,1], yerr=r[:,2], label=str(8+i));
#     plt.plot(linex, liney)
# plt.xlabel("generation H")
# plt.ylabel("detected H")
# linex = [0.075,0.925]
# liney = [0.075,0.925]