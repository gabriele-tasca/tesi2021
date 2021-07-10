import numpy as np
import matplotlib.pyplot as plt

import fract
from fract import *

N = 10
H = 0.8

z = fbm2D_exact(H,N)

gen_params = fract.save_fbm2D_exact_generator_params(H,N)

for j in range(5):
    z2d1, z2d2 = fract.fbm2D_exact_from_generator(*gen_params)

    try:
        res1 = fract.dfa_H(z2d1)
        print("        ",res1)
    except Exception as ex:
        res1 = np.nan
        print("Exception ", ex)

    try:
        res2 = fract.dfa_H(z2d2)
        # print("        ",res2)
    except Exception as ex:
        res2 = np.nan
        print("Exception ", ex)
