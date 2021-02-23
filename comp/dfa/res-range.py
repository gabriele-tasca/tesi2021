# import numpy as np
# import scipy.stats 
# import matplotlib.pyplot as plt

# # %matplotlib tk

# def segment_fluct2(x, y, z, lin_s):
#     design = np.column_stack((x*x, y*y, x*y, x, y, np.full(lin_s**2, 1.0)))

#     param = np.array(np.linalg.lstsq(design, z)[0])
#     trend_z = design @ param

#     z_residue = z - trend_z
#     fluct2 = np.sum(z_residue**2)/(lin_s**2)
#     return fluct2

# def segment_fluct2_plot(x, y, z, lin_s):
#     design = np.column_stack((x*x, y*y, x*y, x, y, np.full(lin_s**2, 1.0)))

#     param = np.array(np.linalg.lstsq(design, z)[0])
#     trend_z = design @ param

#     z_residue = z - trend_z
#     fluct2 = np.sum(z_residue**2)/(lin_s**2)
#     return fluct2, trend_z

# # plotting single segment regression
# # z2d = np.load("data-4.npy") 
# # m,n = z2d.shape
# # x2d,y2d = np.mgrid[:m,:n]
# # x = x2d.ravel()
# # y = y2d.ravel()
# # z = z2d.ravel()

# # fluct, trend_z = segment_fluct2_plot(x,y,z,n)

# # trend_z2d = np.reshape(trend_z, (-1,n))

# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection="3d")
# # ax.scatter(x,y,z)
# # ax.plot_surface(x2d,y2d,trend_z2d, color=(0,0,1,0.5))
# # # ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2])
# # plt.show()





# def square_grid_dfa(x, y, z, s_min = 6, s_max = "auto"):
#     n = np.sqrt(x.size).astype(int)
#     # x2d,y2d = np.mgrid[:n,:n]
#     x2d = x.reshape((n,n))
#     y2d = y.reshape((n,n))
#     z2d = z.reshape((n,n))


#     s_min = 6
#     if s_max == "auto":
#         s_max = n//4
#     else:
#         s_max = s_max

#     scales = np.zeros((s_max - s_min))
#     flucts = np.zeros((s_max - s_min))

#     for j,s in enumerate(range(s_min,s_max)):
#         m = n//s
#         segment_fs = np.zeros(m**2)
#         index = 0
#         for v in range(0, m):
#             for w in range(0, m):
#                 # slice out segment from 2d array 
#                 x_segment = (x2d[s*v:s*(v+1), s*w:s*(w+1)])
#                 y_segment = (y2d[s*v:s*(v+1), s*w:s*(w+1)])
#                 z_segment = (z2d[s*v:s*(v+1), s*w:s*(w+1)])
#                 f = segment_fluct2(x_segment.ravel(), y_segment.ravel(), z_segment.ravel(), s)
#                 segment_fs[index] = f
#                 index += 1
#         fluct_av_sq = np.sum(segment_fs)/(m**2)
#         scales[j] = s
#         flucts[j] = np.sqrt(fluct_av_sq)

#     s_log = np.log10(scales)
#     f_log = np.log10(flucts)

#     # A = np.vstack([np.ones(len(s_log)), s_log]).T
#     H, c, r_value, p_value, std_err = scipy.stats.linregress(s_log, f_log)

#     return H, c, scales, flucts
