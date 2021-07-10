import numpy as np
import matplotlib.pyplot as plt

import fract
from fract import *

gen_H = 0.2
contour_H = 0.8
N = 9

print("gen H:", gen_H)
z_full = fbm2D_midpoint(gen_H, N)

f_args = logplot(box_counting_H(z_full))
print("full square det H:", f_args[0])

z_cut = cut_profile(z_full, contour_H)

c_args = logplot(box_counting_H(z_cut))
print("profile square det H:", c_args[0])


plt.figure()
plt.imshow(z_full)
plt.show()
plt.figure()
plt.imshow(z_cut)
plt.show()