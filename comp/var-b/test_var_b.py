import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import fract
import var_b
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
        data = fract.fbm2D(H,N=N)

        h_detected, c, scales, flucts = var_b.profile_var_range_from_z2d_corrected_corners(data)
        
        popt, pcov = curve_fit(var_b.pow_law, scales, flucts)
        stat_res[j] = popt[1] 

        
        # stat_res[j] = h_detected 

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