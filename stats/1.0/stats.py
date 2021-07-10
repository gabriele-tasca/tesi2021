import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import fract
from fract import *

df = pd.read_csv("stats.csv", sep=";")



name = "Dosegù"
z2d = np.load("/home/gaboloth/D/fisica/tesi/dati/npysquare/all/"+name+"-2d.npy")

plt.imshow(z2d)

z2d = fbm2D(H=0.6, N=8)

z2d = fbm2D(H=0.6, N=8)
fou_H, fou_exp, fou_c, fou_X, fou_Y  = logplot(fourier_H(z2d))
print("fou H", fou_H)

z2d = fbm2D(H=0.6, N=8)
hig_H, hig_exp, hig_c, hig_X, hig_Y  = logplot(higuchi_H(z2d))
print("hig H", hig_H)

z2d = fbm2D(H=0.6, N=8)
df_H, df_c, df_X, df_Y = logplot(dfa_H(z2d))
print("dfa H", df_H)

z2d = fbm2D(H=0.6, N=8)
dma_H, dma_c, dma_X, dma_Y = logplot(dma_H(z2d))
print("dma H", dma_H)