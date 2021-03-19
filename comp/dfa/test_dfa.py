import numpy as np
import matplotlib.pyplot as plt

import fract

N = 9

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
        data_plane, _ = np.mgrid[0:data.shape[0], 0:data.shape[1]]

        # data = data + data_plane/5.0


        # dfa
        h_detected, c, scales, flucts = fract.dfa_H(data, approx="quadratic", estimator="rms")
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



np.savetxt("results4.csv", res, delimiter=";")



# res = np.loadtxt("results.csv", delimiter=";")


