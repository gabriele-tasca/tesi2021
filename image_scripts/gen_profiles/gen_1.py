import numpy as np
import matplotlib.pyplot as plt

from fract import *

N = 10
H = 0.7
n = 1
gen_params = save_fbm2D_exact_generator_params(H, N)


z1, z2 = fbm2D_exact_from_generator(*gen_params)

contour_H = 0.95
z_cut = cut_profile(z1, contour_H)
z_cut[z_cut == 0.0] = np.nan

plt.imshow(z_cut)
plt.show()



plt.savefig(("profile-H%g-contourH%g-%i.png"%(H, contour_H, n)), bbox_inches='tight' )
n += 1