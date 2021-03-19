import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import fract

import pywt

df = pd.read_csv("/home/gaboloth/D/fisica/tesi/stats/stats1.csv", sep=";")

tdf = df[1:5]


def to_data(name, inner_func, *args, **kwargs):
    npy_points = np.load("/home/gaboloth/D/fisica/tesi/dati/npysquare/all/"+name+"-2d.npy")
    return inner_func(npy_points, *args, **kwargs)


# name = "Narcanello"
# npy_points = np.load("/home/gaboloth/D/fisica/tesi/dati/npysquare/all/"+name+"-2d.npy")

npy_points = fract.fbm2D(0.3, N=8)
npy_points_plane, _ = np.mgrid[0:npy_points.shape[0], 0:npy_points.shape[1]]

# add mean plane
# npy_points = npy_points + npy_points_plane/15.0


plt.imshow(npy_points)
plt.show()

(cA, (cH, cV, cD)) = pywt.dwt2(npy_points, mode="zero", wavelet="db1")

plt.imshow(cV)
plt.imshow(np.log10(cA))