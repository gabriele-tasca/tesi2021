import numpy as np
import matplotlib.pyplot as plt

import fract

%matplotlib tk

N = 10
hs = [0.25, 0.75]

for H in hs:
    z = fract.fbm2D_spectral(H,N)

    plt.imshow(z)
    # plt.title(r"H = %g, %i x %i points"%(H, 2**N, 2**N))
    plt.savefig(("spectr%g.png"%H), bbox_inches='tight' )
    plt.show()