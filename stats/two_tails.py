import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import fract

df = pd.read_csv("stats.csv", sep=";")



def to_data(name, inner_func, *args, **kwargs):
    npy_points = np.load("/home/gaboloth/D/fisica/tesi/dati/npysquare/all/"+name+"-2d.npy")
    return inner_func(npy_points, *args, **kwargs)

def fourier_H_discard(*args, **kwargs):
    fourier_H, freq_exp, f_c, freqs, powers = fract.fourier_H(*args, **kwargs)
    return fourier_H

def dfa_H_discard(*args, **kwargs):
    try:
        dfa_H, dfa_c, scales3, flucts3 = fract.dfa_H(*args, **kwargs)
        return dfa_H
    except:
        return np.nan




# df["Fourier H"] = df["Nome"].apply(to_data, inner_func=fourier_H_discard)


# df["DFA H"] = df["Nome"].apply(to_data, inner_func=dfa_H_discard)



# name = "Adamello"
# npy_points = np.load("/home/gaboloth/D/fisica/tesi/dati/npysquare/all/"+name+"-2d.npy")



def fake_function():
    pass
    # z2d = npy_points
    # (m, n) = z2d.shape
    # x2d,y2d = np.mgrid[:m,:n]

    # ### single scale test
    # s = 100

    # n_submat_n = n//s
    # n_submat_m = m//s
    # segment_fs = np.zeros(n_submat_m*n_submat_n)
    # index = 0
    # miss_count = 0
    # for v in range(0, n_submat_m):
    #     for w in range(0, n_submat_n):
    #         # slice out segment from 2d array
    #         x_segment = (x2d[s*v:s*(v+1), s*w:s*(w+1)])
    #         y_segment = (y2d[s*v:s*(v+1), s*w:s*(w+1)])
    #         z_segment = (z2d[s*v:s*(v+1), s*w:s*(w+1)])



    #         if np.count_nonzero(z_segment) > (s**2 * 0.9):

    #             x_segment = x_segment[ z_segment != 0.0]
    #             y_segment = y_segment[ z_segment != 0.0]
    #             z_segment = z_segment[ z_segment != 0.0]

    #             x, y ,z = (x_segment.ravel(), y_segment.ravel(), z_segment.ravel())
                

    #             design = np.column_stack((x*x, y*y, x*y, x, y, np.full(x.size, 1.0)))

    #             param = np.array(np.linalg.lstsq(design, z)[0])
    #             trend_z = design @ param

    #             z_residue = z - lineartrend_z
    #             f = np.sum(z_residue**2)/(s**2)

    #             segment_fs[index] = f
    #             print(v,w,": fill = ", "{:.2f}".format(np.count_nonzero(z_segment)/(s**2)), ", fluct = ", f)
                
    #             index += 1
    #         else:
    #             print(v,w,": missed")
    #             miss_count += 1

    # segment_fs = np.trim_zeros(segment_fs, trim="b")
    # if segment_fs.size > 4:
    #     fluct_av_sq = np.sum(segment_fs)/(index)
    #     print("average for scale",s,":",np.sqrt(fluct_av_sq))
    #     print( "fluct_av_sq", (fluct_av_sq), "scale", s, ",", n_submat_m*n_submat_n, "submatrices olinear "scale", s, ",", n_submat_m*n_submat_n, "submatrices of which", miss_count, "empty. fit too poor, aborted")


    # (v_plot, w_plot) = (3,1)



# fig,ax = plt.subplots(1,1)
# ovshape = z2d.shape + (4,)linear
#### single npypoints test

npy_points = fract.fbm2D(1.5, N=9)
# npy_points_2 = fract.fbm2D(0.3, N=9)
# npy_points += npy_points_2
# npy_points_plane, _ = np.mgrid[0:npy_points.shape[0], 0:npy_points.shape[1]]

# # #### flat plane
# npy_points = np.full((512,512), 0.0)

# # #### add noise
# npy_points += np.random.normal(size=npy_points.size, scale=0.2).reshape(npy_points.shape)

#### add tilted plane
# npy_points = npy_points + npy_points_plane/25.0


plt.figure()
plt.imshow(npy_points)
plt.show()


## fourier
fourier_H, freq_exp, f_c, freqs, powers = fract.fourier_H(np.copy(npy_points), fill_with_mean=True, images=True, corr=True)
print("fourier_H", fourier_H)


plt.figure()
plt.scatter(freqs, powers)
plt.plot(freqs, fract.power_law(freqs, freq_exp, f_c), color="red", label="fourier")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.show()


#### dfa
dfa_H, dfa_c, scales3, flucts3 = fract.dfa_H(npy_points, messages=False, min_nonzero=0.99)
print("dfa_H", dfa_H)


### first branch
dfa_H2, c2, pcov = fract.autoseeded_weighted_power_law_fit(scales3[:17], flucts3[:17], flucts3[:17])
print("dfa_H2", dfa_H2)


plt.figure()
plt.scatter(scales3, flucts3)
plt.plot(scales3, fract.power_law(scales3, dfa_H, dfa_c), color="springgreen", label="dfa: H ="+str(dfa_H))
plt.plot(scales3, fract.power_law(scales3, dfa_H2, c2), color="red", label="dfa: H = "+str(dfa_H2))

plt.plot(scales3, fract.power_law(scales3, fourier_H, c2/0.8), color="purple", label="fourier: H = "+str(fourier_H))

plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.title(name)
plt.show()



# ##########
# df
# df.drop(df.columns[0], axis=1, inplace=True)

# # del df["uiasfgyaiofu"]



# plt.hist(np.log10(df["Area"]), bins=20)



# a = df[ df["Area"] < 97513  ]
# plt.scatter(a["Fourier H"], a["Area Ratio"])


# plt.scatter(df["Fourier H"], df["Map Area Ratio"])
# plt.scatter(df["Fourier H"], df["Area Ratio"])
# plt.scatter(df["Height Stdev"], df["Area Ratio"])
# plt.scatter(df["Height Stdev"], df["Fourier H"])

# plt.scatter(df["Fourier H"], df["DFA H"])
# # # plt.xscale("log")
# # # plt.yscale("log")
# plt.show()


# df.to_csv("stats.csv", sep=";", index=False)