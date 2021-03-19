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



# name = "Ventina"
# npy_points = np.load("/home/gaboloth/D/fisica/tesi/dati/npysquare/all/"+name+"-2d.npy")






#### single npypoints test

npy_points = fract.fbm2D(0.7, N=9)

# #### flat plane
# npy_points = np.full((512,512), 0.0)

# # #### add noise
# npy_points += np.random.normal(size=npy_points.size, scale=0.2).reshape(npy_points.shape)

x2d, y2d = np.mgrid[0:npy_points.shape[0], 0:npy_points.shape[1]]

#### add tilted plane
# npy_points = npy_points + x2d*50


#### add plane tiles
n_plane_tiles = 5
s_plane = npy_points.shape[0] // n_plane_tiles

s = s_plane
for v in range(n_plane_tiles):
    for w in range(n_plane_tiles):
                mx = np.random.rand() *25.0
                my = np.random.rand() *25.0
                x_segment = (x2d[s*v:s*(v+1), s*w:s*(w+1)])
                y_segment = (y2d[s*v:s*(v+1), s*w:s*(w+1)])
                z_segment = (npy_points[s*v:s*(v+1), s*w:s*(w+1)])

                z_segment += x_segment*mx + y_segment*my


from scipy.ndimage import rotate
npy_points = rotate(npy_points, 30)
# plt.figure()
# plt.imshow(npy_points)
# plt.show()


## fourier
fourier_H, freq_exp, f_c, freqs, powers = fract.fourier_H(np.copy(npy_points), fill_with_mean=True, images=True, corr=True)
print("fourier_H", fourier_H)


freq_exp_2, fc2, pcov = fract.autoseeded_weighted_power_law_fit(freqs[2:20], powers[2:20], sigmas=powers[2:20])
fourier_H2 = fract.freq_exp_to_H(freq_exp_2)

plt.figure()
plt.scatter(freqs, powers)
plt.plot(freqs, fract.power_law(freqs, freq_exp, f_c), color="red", label="fourier: H ="+str(fourier_H))
plt.plot(freqs, fract.power_law(freqs, freq_exp_2, fc2), color="springgreen", label="fourier: H ="+str(fourier_H2))
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


plt.xscale("log")
plt.yscale("log")
plt.legend()
# plt.title(name)
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