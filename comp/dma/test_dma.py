import numpy as np
import matplotlib.pyplot as plt

import fract
import c_dma

z2d = fract.fbm2D(H=0.6, N=8)

dmah, dmac, arr1, arr2 = fract.dma_H(z2d)

N = 8

stat_n = 50

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
        scales, flucts = c_dma.dma_1(data)
        dma_H, dma_c, pcov = fract.autoseeded_weighted_power_law_fit(scales, flucts, sigmas=flucts)

        stat_res[j] = dma_H
        # print("     ", j)

    av_h_detected = np.mean(stat_res)
    std_h_detected = np.std(stat_res)
    print("     av. H detected =", av_h_detected)
    
    newrow = np.transpose([H, av_h_detected, std_h_detected])
    res[i,:] = newrow



res = np.loadtxt("results.txt")

plt.errorbar(res[:,0], res[:,1], yerr=res[:,2]);
plt.xlabel("generation H")
plt.ylabel("detected H")
linex = [0.075,0.925]
liney = [0.075,0.925]
plt.plot(linex, liney)
plt.show()

# np.savetxt("results.txt", res)

