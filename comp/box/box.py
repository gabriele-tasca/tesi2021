import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import fract



df = pd.read_csv("/home/gaboloth/D/fisica/tesi/stats/stats.csv", sep=";")
# df = pd.read_csv("stats.csv", sep=";")

# df["Nome"]

# name = "Cima Wanda"
# npy_points = np.load("/home/gaboloth/D/fisica/tesi/dati/npysquare/all/"+name+"-2d.npy")

# res = np.zeros(9)
# stat_n = 50
# for i,h in enumerate(np.linspace(0.1, 0.9, 9)):
#     for j in range(stat_n):
#         z2d = fract.fbm2D(H=h, N=8)
#         res[i] += (area2(z2d))/stat_n
# plt.plot(res)


N = 8
H = 0.9
z2d = fract.fbm2D_exact(H=H, N=N)


# name = "Adamello"
# z2d = np.load("/home/gaboloth/D/fisica/tesi/dati/npysquare/all/"+name+"-2d.npy")
# z2d = z2d/5.0





# box counting


xwid, ywid = z2d.shape
z_scale = 1000000
space = min(z2d.shape)
smax = space//6
smin = 2

scales = np.zeros(smax - smin)
nbox_s = np.zeros(smax - smin)

s_dim = 2

for i, s in enumerate(np.arange(smin,smax)):
    n_boxes = 0
    area = s**(s_dim)
    for i_x in range(xwid//s):
        for i_y in range(ywid//s):
            submat = z2d[ i_x*s:(i_x+1)*s +1 , i_y*s:(i_y+1)*s +1 ]*z_scale
            # if (submat[ submat != 0].size)/(submat.size) > 0.90:
            n_boxes += (1 + (np.max(submat) - np.min(submat))//s )
            ######### else: zero boxes
    nbox_s[i] = n_boxes*area
    scales[i] = s

num = 1
b_exp, b_c, b_k, pcov = fract.autoseeded_weighted_power_law_fit(scales[num:], nbox_s[num:])

# det_H = 2 - (-b_exp +1 ) 
det_H = 3 + b_exp -s_dim
fract_dim = 3 - det_H

print("fractal dimension:", fract_dim)
print("detected H:", det_H)
plt.scatter(scales, nbox_s)
plt.plot(scales, fract.power_law(scales, b_exp, b_c, b_k), color="red")
plt.xscale("log")
plt.yscale("log")
plt.title("Box counting: H = "+str(det_H))
plt.show()





# def to_data(name, inner_func, *args, **kwargs):
#     npy_points = np.load("/home/gaboloth/D/fisica/tesi/dati/npysquare/all/"+name+"-2d.npy")
#     return inner_func(npy_points, *args, **kwargs)

# def higuchi_H_discard(*args, **kwargs):
#     try:
#         det_H, _b_c, _scales, _areas = higuchi_H(*args, **kwargs)
#         print(det_H)
#         return det_H
#     except Exception as ex:
#         print(ex)
#         return np.nan

# df["Higuchi H (paper)"] = df["Nome"].apply(to_data, inner_func=higuchi_H_discard)


# a = df[ df["Area"] < 97513  ]
# plt.scatter(a["Fourier H"], a["Higuchi H (paper)"])

# plt.scatter(df["DFA H"], df["Higuchi H (paper)"])

# df.to_csv("stats.csv", sep=";", index=False)

############ test

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# import fract
# testx = np.arange(1,1000)
# spamy = 5*testx**(-2.0) 
# # spamy = np.full_like(testx, 20) 
# testy = 5*testx**(-2.2) + spamy

# num = 1
# test_exp, test_c, test_k, pcov = fract.autoseeded_weighted_power_law_fit(testx[num:], testy[num:], sigmas=testy[num:])

# test_det_H =  test_exp
# fract_dim = 3 + test_exp

# # print("fractal dimension:", fract_dim)
# print("detected H:", test_det_H)
# plt.scatter(testx, testy)
# plt.plot(testx, fract.power_law(testx, test_exp, test_c, test_k), color="red")
# plt.xscale("log")
# plt.yscale("log")
# plt.title("Box counting: H = "+str(test_det_H))
# plt.show()