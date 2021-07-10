import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import fract
from fract import *

# df = pd.read_csv("stats.csv", sep=";")





def to_data(name, inner_func, *args, **kwargs):
    npy_points = np.load("/home/gaboloth/D/fisica/tesi/dati/npysquare/all/"+name+"-2d.npy")
    try:
        res = inner_func(npy_points, *args, **kwargs)
    except Exception as exc:
        print(exc)
        res = np.nan
    return res

name = "Adamello"
print(name)
z2d = np.load("/home/gaboloth/D/fisica/tesi/dati/npysquare/all/"+name+"-2d.npy")





# static analysis
box_H, box_exp, box_c, box_k, box_X, box_Y = (box_counting_H(z2d))

hig_H, hig_exp, hig_c, hig_X, hig_Y  = (higuchi_H(z2d))

df_H, df_c, df_X, df_Y = (dfa_H(z2d, approx="quadratic"))


# creative interpretation

box_exp2, box_c2, _pcov = autoseeded_weighted_power_law_fit(box_X[0:40], box_Y[0:40])
box_exp3, box_c3, _pcov = autoseeded_weighted_power_law_fit(box_X[40:], box_Y[40:])
box_H2 = box_exp_to_H(box_exp2)
box_H3 = box_exp_to_H(box_exp3)
plt.figure()
plt.scatter(box_X, box_Y)
plt.plot(box_X, power_law(box_X, box_exp2, box_c2), color="red")
plt.plot(box_X, power_law(box_X, box_exp3, box_c3), color="green")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Scale")
plt.ylabel("Area of box covering")
plt.show()
print("small scale box_H", box_H2)
print("large scale box_H", box_H3)
# plt.savefig(name+"-box-2line.png", bbox_inches="tight")


hig_exp2, hig_c2, _pcov = autoseeded_weighted_power_law_fit(hig_X[0:10], hig_Y[0:10])
hig_exp3, hig_c3, _pcov = autoseeded_weighted_power_law_fit(hig_X[10:], hig_Y[10:])
hig_H2 = higuchi_exp_to_H(hig_exp2)
hig_H3 = higuchi_exp_to_H(hig_exp3)
plt.figure()
plt.scatter(hig_X, hig_Y)
plt.plot(hig_X, power_law(hig_X, hig_exp2, hig_c2), color="red")
plt.plot(hig_X, power_law(hig_X, hig_exp3, hig_c3), color="green")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Scale")
plt.ylabel("Higuchi Area")
plt.show()
print("small scale hig_H", hig_H2)
print("large scale hig_H", hig_H3)
plt.savefig(name+"-higuchi-2line.png", bbox_inches="tight")


df_H2, df_c2, _pcov = autoseeded_weighted_power_law_fit(df_X[0:25], df_Y[0:25])
df_H3, df_c3, _pcov = autoseeded_weighted_power_law_fit(df_X[25:], df_Y[25:])
plt.figure()
plt.scatter(df_X, df_Y)
plt.plot(df_X, power_law(df_X, df_H2, df_c2), color="red")
plt.plot(df_X, power_law(df_X, df_H3, df_c3), color="green")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Scale")
plt.ylabel("Fluctuation")
plt.show()
print("small scale df_H", df_H2)
print("large scale df_H", df_H3)
plt.savefig(name+"-dfa-2line.png", bbox_inches="tight")



# test = np.full((512,512), 25.0)
# # test = fbm2D_exact(0.9, 9)
# res1, res2 = dfa_1(test, approx="quadratic")
# resH, resc, _pcov = autoseeded_weighted_power_law_fit(res1, res2)
# plt.scatter(res1, res2)
# plt.plot(res1, power_law(res1, resH, resc), color="orange")
# print("exp ", resH)


# autocorrelation

