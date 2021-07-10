import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng

import fract



def fbm2D_midpoint_new(H, N=4, base_stdev=1.0):
    side = 2**N + 1
    rng = np.random.default_rng()

    vec = np.zeros( (side,side) )

    # initialize corners
    stdev = (0.5)* base_stdev
    vec[  0  ,  0  ] += rng.normal( scale=stdev )
    vec[  0  ,  side-1  ] += rng.normal( scale=stdev )
    vec[  side-1  ,  0  ] += rng.normal( scale=stdev )
    vec[  side-1  ,  side-1  ] += rng.normal( scale=stdev )

    # loop 
    for g in range(1,N):
        # gpow = 2**g
        gstep = 2**(N-g)
        stdev = (0.5)**(H*g) * base_stdev

        gsize2 = 2**(g+1)+1

        half = gstep//2
        # diagonal interpolation
        north_west = vec[  0:side-gstep:gstep, 0:side-gstep:gstep  ]
        north_east = vec[  0:side-gstep:gstep, gstep:side:gstep  ]
        south_west = vec[  gstep:side:gstep, 0:side-gstep:gstep  ]
        south_east = vec[  gstep:side:gstep, gstep:side:gstep  ]
        
        vec[  half:side:gstep, half:side:gstep  ] = (north_west + north_east + south_west + south_east)/4

        ##### interpolation 2.1

        #   left edge
        edge_up_1l = vec[  0:side-gstep:gstep, 0  ]
        edge_down_1l = vec[  gstep:side+half:gstep, 0  ]
        edge_right_1l = vec[ half:side:gstep, half]
        vec[  half:side:gstep, 0  ] = (edge_up_1l + edge_down_1l + edge_right_1l)/3

        #  right edge
        edge_up_1r = vec[  0:side-gstep:gstep, side-1 ]
        edge_down_1r = vec[  gstep:side+half:gstep, side-1  ]
        edge_left_1r = vec[ half:side:gstep, side-1-half ]
        vec[  half:side:gstep, side-1  ] = (edge_up_1r + edge_down_1r + edge_left_1r)/3

        #   bulk
        up3 = vec[  0:side-half:gstep, gstep:side-gstep:gstep  ]
        down3 = vec[  2*half:side+half:gstep, gstep:side-gstep:gstep  ]
        left3 = vec[ half:side:gstep, half:side-half-gstep:gstep]
        right3 = vec[ half:side:gstep, half+gstep:side-half:gstep]
        vec[  half:side:gstep, gstep:side-gstep:gstep  ] = (up3 + down3 + left3 + right3)/4

        ##### interpolation 2.2

        # up edge
        edge_left_2u = vec[ 0, 0:side-half:gstep ]
        edge_right_2u = vec[  0, half+half:side+half:gstep  ]
        edge_down_2u = vec[ half, half:side:gstep]
        vec[  0, half:side:gstep  ] = (edge_left_2u + edge_right_2u + edge_down_2u)/3
        
        # down edge
        edge_left_2d = vec[ side-1, 0:side-half:gstep ]
        edge_right_2d = vec[ side-1, half+half:side+half:gstep  ]
        edge_up_2d = vec[ side-half, half:side:gstep]
        vec[  side-1, half:side:gstep  ] = (edge_left_2d + edge_right_2d + edge_up_2d)/3

        # bulk
        left4 = vec[  gstep:side-gstep:gstep, 0:side-half:gstep ]
        right4 = vec[  gstep:side-gstep:gstep, half+half:side+half:gstep  ]
        up4 = vec[ half:side-half-gstep:gstep, half:side:gstep]
        down4 = vec[ half+gstep:side-half:gstep, half:side:gstep]

        vec[ gstep:side-gstep:gstep, half:side:gstep ] = (up4 + down4 + left4 + right4)/4


        # random addition (sq 1 + sq 2 bulk)
        rands = rng.normal( size=(gsize2,gsize2), scale=stdev*np.sqrt(2))
        vec[  0:side:half, 0:side:half  ] += rands


    # return vec[1:side-1,1:side-1]
    return vec


# z2d = fract.fbm2D_spectral(H=0.6, N=8)
# plt.imshow(z2d)

# detect_H,  c, const, freqs, powers = fract.dfa_H(z2d)

# print(detect_H)