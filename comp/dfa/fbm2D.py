import numpy as np
import matplotlib.pyplot as plt

def fbm2D(H, N=4, base_stdev = 1.0):
    side = 2**N + 1
    npoints = side**2
    rng = np.random.default_rng()

    vec = np.zeros( (side,side) )
    for g in range(1,N+1):
        gpow = 2**g
        gstep = (side-1) // gpow
        stdev = (0.5)**(H*g) * base_stdev

        # random additions        
        rands = rng.normal( size=(gpow+1,gpow+1))
        # rands = np.full( (gpow+1,gpow+1) , 1)
        # print(vec.shape)
        # print(rands.shape)
        
        vec[  0:side:gstep  ,  0:side:gstep  ] += rands
        # print(vec)

        # interpolation
        if g != N+1:
            delta = gstep//2
            # inted = vec[  1+delta:side:gstep, 1:side:gstep  ]
            up = vec[  0:side-delta:gstep, 0:side:gstep  ]
            down = vec[  0+2*delta:side+delta:gstep, 0:side:gstep  ]
            vec[  0+delta:side:gstep, 0:side:gstep  ] = (up + down)/2
            # print(up.shape)
            # print(down.shape)
            # print( (vec[  0+delta:side:gstep, 0:side:gstep  ]).shape)

            left = vec[  0:side:gstep, 0:side-delta:gstep]
            right = vec[  0:side:gstep,  0+2*delta:side+delta:gstep]
            vec[  0:side:gstep, 0+delta:side:gstep  ] = (left + right)/2

            
            up2 = vec[  0:side-delta:gstep, 0+delta:side:gstep  ] 
            down2 = vec[  0+2*delta:side:gstep, 0+delta:side:gstep  ]

            left2 = vec[  0+delta:side:gstep, 0:side-delta:gstep  ]
            right2 = vec[  0+delta:side:gstep, 0+2*delta:side:gstep  ]
            # print(" up2" , up2.shape)
            # print(" down2" , down2.shape)
            # print(" right2" , right2.shape)
            # print(" left2" , left2.shape)
            # print("kan ",vec[  0+delta:side:gstep, 0+delta:side:gstep  ].shape)
            vec[  0+delta:side:gstep, 0+delta:side:gstep  ] = (up2 + down2 + left2 + right2)/4

    return vec

H = 0.3
N = 13
side = 2**N + 1
npoints = side**2
vec2 = fbm2D(H, N)

# pygui(true)
p = plt.imshow(vec2)
plt.colorbar(p)
plt.title("H = "+ str(H) +" , "+str(side)+"x"+str(side)+" points")
