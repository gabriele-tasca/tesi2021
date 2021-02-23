import numpy as np
import matplotlib.pyplot as plt

import scipy.stats 


import dfa

print("reading .npy")
z2d = np.load("/home/gaboloth/D/fisica/tesi/dati/npysquare/forni07-2d.npy")

print("dfa")
(m,n) = z2d.shape

# old_flucts = flucts

# subselect = z2d[int(m*0.6) : int(m*0.9) , int(n*0.6) :int(n*0.9)  ]
subselect = z2d

################
################
####################


# def profile_dfa_from_z2d_2(z2d, s_min = 6, s_max = "auto", min_nonzero = 0.9):
#     (m, n) = z2d.shape
#     x2d,y2d = np.mgrid[:m,:n]

#     s_min = 6
#     if s_max == "auto":
#         s_max = n//4
#     else:
#         s_max = s_max


#     # scales - flucts table
#     scales_flucts = np.zeros((s_max - s_min ,2))
#     # scales = np.zeros((s_max - s_min))
#     # flucts = np.zeros((s_max - s_min))

#     for j,s in enumerate(range(s_min,s_max)):
#         n_submat_n = n//s
#         n_submat_m = m//s
#         segment_fs = np.zeros(n_submat_m*n_submat_n)

#         index = 0
#         miss_count = 0


#         # hack = min(n_submat_n, n_submat_m)
#         for v in range(0, n_submat_m):
#             for w in range(0, n_submat_n):
#                 # slice out segment from 2d array
#                 x_segment = (x2d[s*v:s*(v+1), s*w:s*(w+1)])
#                 y_segment = (y2d[s*v:s*(v+1), s*w:s*(w+1)])
#                 z_segment = (z2d[s*v:s*(v+1), s*w:s*(w+1)])

#                 if np.count_nonzero(z_segment) > (s**2 * min_nonzero):
#                     f = segment_fluct2(x_segment.ravel(), y_segment.ravel(), z_segment.ravel(), s)
#                     segment_fs[index] = f
#                     index += 1
#                 else:
#                     miss_count += 1
#         # remove unfilled slots from tail
#         # print(miss_count/(index+miss_count))
#         segment_fs = np.trim_zeros(segment_fs, trim="b")
#         fluct_av_sq = np.sum(segment_fs)/(index)
#         scales_flucts[j,0] = s
#         print( "fluct_av_sq", (fluct_av_sq), "scale", s, ",", n_submat_m*n_submat_n, "submatrices of which", miss_count, "empty")
#         scales_flucts[j,1] = np.sqrt(fluct_av_sq)

#     # remove nan values
#     scales_flucts = scales_flucts[~np.isnan(scales_flucts).any(axis=1)]

    
#     # s_f_log = np.log10(scales_flucts)
#     s_log = np.log10(scales_flucts[:,0])
#     f_log = np.log10(scales_flucts[:,1])

#     # A = np.vstack([np.ones(len(s_log)), s_log]).T
#     H, c, r_value, p_value, std_err = scipy.stats.linregress(s_log, f_log)
#     print("H =  ",H)
#     print("1.5")
#     return H, c, scales_flucts[:,0], scales_flucts[:,1]


# def segment_fluct2(x, y, z, lin_s):
#     design = np.column_stack((x*x, y*y, x*y, x, y, np.full(lin_s**2, 1.0)))

#     param = np.array(np.linalg.lstsq(design, z)[0])
#     trend_z = design @ param

#     z_residue = z - trend_z
#     fluct2 = np.sum(z_residue**2)/(lin_s**2)
#     return fluct2

################
################
################

# a = np.random.rand( 5,4 )
# a[3,2] = np.nan
# a
# a[~np.isnan(a).any(axis=1)]

H, c, scales, flucts = dfa.profile_dfa_from_z2d_2(subselect, messages=True)

plt.scatter(scales,flucts, marker=".", color="deepskyblue")
plt.plot(scales, (10**c)*scales**H, color="purple")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("scale")
plt.ylabel("fluctuation")
plt.title("H = " + str(H))

plt.show()

# z2d[z2d == 0] = np.nan
plt.imshow(z2d, interpolation = 'none')
plt.show()

# subselect[subselect == 0] = np.nan
plt.imshow(subselect, interpolation = 'none')
plt.show()


####################à
